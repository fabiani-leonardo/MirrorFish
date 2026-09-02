#!/usr/bin/env python3
"""
Report di consumo per la richiesta di quota.

Produce i numeri da portare al colloquio: quanto consuma davvero una chiamata,
quanta parte della finestra token resta libera, e quale RPM si puo' chiedere
restando dentro il limite di token gia' concesso.

    # 1. misura dal vivo (~30 chiamate reali, ~4 min a 8 rpm)
    python scripts/quota_report.py --live --calls 30

    # 2. oppure dai dati di un run VERO gia' fatto
    python scripts/quota_report.py --db runs/pilot/run.db

Rifiuta i run stub: i token di StubLLM sono un'euristica sui caratteri, non
una misura, e presentarli come tali in un colloquio e' indifendibile.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirrorfish.config import DEFAULT_TOKEN_BUDGET, LLMConfig  # noqa: E402
from mirrorfish.llm import OpenAICompatClient  # noqa: E402
from mirrorfish.store import Store  # noqa: E402

# Prompt realistici: stessa forma e lunghezza di quelli della simulazione,
# altrimenti la misura non e' rappresentativa.
BIO = (
    "Ho 54 anni, vivo in Campania, diploma di ragioneria, lavoro come "
    "impiegato amministrativo in una piccola azienda. Leggo i giornali la "
    "mattina presto e uso i social soprattutto la sera. Sono abbastanza "
    "scettico sulla politica, ho votato in modo diverso nelle ultime tre "
    "elezioni. Mi interessa il tema della giustizia perche' mio cognato ha "
    "avuto una causa civile durata nove anni."
)
NOTES = "\n".join(
    f"- Rafforza la propria diffidenza verso la riforma dopo un intervento "
    f"che ne contesta gli effetti sull'indipendenza della magistratura ({i})."
    for i in range(8)
)
FEED = "\n".join(
    f"#{100 + i} @utente_{i:03d}: Sul referendum continuo a pensare che "
    f"separare le carriere non risolva i tempi della giustizia, che sono il "
    f"problema vero per chi ci passa davvero. ({i})"
    for i in range(8)
)

SYSTEM = (f"Sei utente_042, un cittadino italiano che usa un social network.\n\n"
          f"CHI SEI:\n{BIO}\n\nCOME LA PENSI ADESSO:\n{NOTES}\n\n"
          f"Rispondi SOLO con JSON: "
          f'{{"action": "POST"|"REPLY"|"LIKE"|"IGNORE", "content": "...", '
          f'"target_post_id": numero o null}}')
USER = (f"Oggi e' 2026-03-14.\n\nQUELLO CHE VEDI SULLA TUA HOME:\n{FEED}\n\n"
        f"Cosa fai adesso? Rispondi in JSON.")


async def measure_live(calls: int, max_tokens: int, rpm: float) -> list[dict]:
    cfg = LLMConfig.from_env(requests_per_minute=rpm, concurrency=2)
    client = OpenAICompatClient(cfg)
    rows = []
    try:
        for i in range(calls):
            r = await client.complete(SYSTEM, USER, max_tokens=max_tokens,
                                      temperature=0.7, json_mode=True)
            if r.error:
                print(f"  [{i+1}/{calls}] ERRORE {r.error[:80]}")
                continue
            rows.append({"purpose": "action", "prompt_tokens": r.prompt_tokens,
                         "completion_tokens": r.completion_tokens,
                         "max_tokens": max_tokens,
                         "finish_reason": r.finish_reason,
                         "latency_ms": r.latency_ms})
            print(f"  [{i+1}/{calls}] prompt={r.prompt_tokens} "
                  f"out={r.completion_tokens} {r.latency_ms:.0f}ms "
                  f"{r.finish_reason}")
    finally:
        await client.aclose()
    return rows


def from_db(db_path: str) -> list[dict]:
    store = Store(db_path)
    meta = store.get_meta("llm", {}) or {}
    if meta.get("stub"):
        raise SystemExit(
            f"{db_path} e' un run STUB.\n"
            "I token di StubLLM sono len(testo)//4, un'euristica sui "
            "caratteri: non sono una misura e non vanno presentati come tale.\n"
            "Fai un run breve vero (--stub omesso) oppure usa --live."
        )
    rows = store.conn.execute(
        "SELECT purpose, prompt_tokens, completion_tokens, max_tokens, "
        "finish_reason, latency_ms FROM llm_call WHERE error IS NULL "
        "AND completion_tokens > 0"
    ).fetchall()
    if not rows:
        raise SystemExit(f"Nessuna chiamata riuscita in {db_path}.")
    return [dict(r) for r in rows]


def report(rows: list[dict], tpm: int, rpm_now: float, parallel: int,
           latency_hint: float | None) -> None:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["purpose"], []).append(r)

    print("\n" + "=" * 74)
    print(" CONSUMO MISURATO PER CHIAMATA")
    print("=" * 74)
    print(f"  {'tipo':<12}{'n':>5}{'prompt med':>12}{'out med':>10}{'out p95':>9}"
          f"{'budget':>8}{'troncate':>10}")
    print("  " + "-" * 64)

    worst = 0.0
    for purpose, rs in sorted(by.items()):
        pr = [r["prompt_tokens"] for r in rs]
        co = [r["completion_tokens"] for r in rs]
        p95 = sorted(co)[max(0, int(len(co) * 0.95) - 1)]
        budget = max(r["max_tokens"] for r in rs)
        trunc = sum(1 for r in rs if r["finish_reason"] == "length")
        print(f"  {purpose:<12}{len(rs):>5}{statistics.mean(pr):>12.0f}"
              f"{statistics.mean(co):>10.0f}{p95:>9}{budget:>8}{trunc:>10}")
        worst = max(worst, statistics.mean(pr) + budget)

    all_lat = [r["latency_ms"] for r in rows if r["latency_ms"]]
    lat = statistics.mean(all_lat) if all_lat else (latency_hint or 3000)

    print("\n" + "=" * 74)
    print(" QUANTA QUOTA TOKEN RESTA LIBERA")
    print("=" * 74)
    print(f"  Limite token concesso        : {tpm:,} / minuto")
    print(f"  Costo peggiore per chiamata  : ~{worst:.0f} token")
    print(f"    (prompt medio + max_tokens: il gateway sembra pre-scalare sul")
    print(f"     budget richiesto, non sul consumo effettivo — da verificare)")

    rpm_by_tokens = tpm / worst
    print(f"\n  Richieste/min sostenibili col SOLO limite di token: "
          f"{rpm_by_tokens:.0f}")
    print(f"  Richieste/min attualmente concesse                : {rpm_now:.0f}")
    print(f"  Utilizzo attuale della finestra token             : "
          f"{rpm_now * worst / tpm * 100:.1f}%")

    print("\n" + "=" * 74)
    print(" RICHIESTA SOSTENIBILE")
    print("=" * 74)
    # Restare sotto il 60% della finestra token lascia margine agli altri.
    target = min(rpm_by_tokens * 0.6, parallel * 60_000 / max(lat, 1))
    print(f"  Con {parallel} richieste parallele concesse e latenza media "
          f"{lat:.0f} ms,")
    print(f"  il tetto fisico del parallelismo e' gia' "
          f"~{parallel * 60_000 / max(lat, 1):.0f} req/min.")
    print(f"\n  --> chiedere {target:.0f} req/min userebbe il "
          f"{target * worst / tpm * 100:.0f}% della finestra token")
    print(f"      gia' concessa, senza chiedere un token in piu'.")
    print(f"\n  Il vincolo attuale non e' il consumo: e' il tetto RPM, che")
    print(f"  lascia inutilizzato il "
          f"{100 - rpm_now * worst / tpm * 100:.0f}% dei token gia' assegnati.")

    print("\n" + "=" * 74)
    print(" EFFETTO SUI TEMPI DI SIMULAZIONE")
    print("=" * 74)
    for calls in (4_283, 10_000, 35_000):
        h_now = calls / rpm_now / 60
        h_new = calls / target / 60
        print(f"  {calls:>6,} chiamate:  {h_now:>6.1f} h  ->  {h_new:>5.1f} h")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="run.db di un run VERO")
    ap.add_argument("--live", action="store_true", help="misura con chiamate reali")
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_TOKEN_BUDGET["action"])
    ap.add_argument("--tpm", type=int, default=100_000,
                    help="x-ratelimit-team_member-limit-tokens")
    ap.add_argument("--rpm", type=float, default=8.0,
                    help="x-ratelimit-team_member-limit-requests")
    ap.add_argument("--parallel", type=int, default=5,
                    help="x-ratelimit-api_key-limit-max_parallel_requests")
    args = ap.parse_args()

    if args.live:
        if not os.environ.get("LLM_API_KEY"):
            raise SystemExit("Serve LLM_API_KEY nell'ambiente.")
        print(f"Misuro {args.calls} chiamate reali a {args.rpm} req/min "
              f"(~{args.calls / args.rpm:.0f} min)...")
        rows = await measure_live(args.calls, args.max_tokens, args.rpm)
        if not rows:
            raise SystemExit("Nessuna chiamata riuscita: controlla la quota.")
        Path("quota_measurement.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
        print("\nMisure grezze salvate in quota_measurement.json")
    elif args.db:
        rows = from_db(args.db)
    else:
        raise SystemExit("Serve --live oppure --db.")

    report(rows, args.tpm, args.rpm, args.parallel, None)


if __name__ == "__main__":
    asyncio.run(main())
