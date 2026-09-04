#!/usr/bin/env python3
"""
Misura l'effetto di max_tokens, del thinking mode e della concorrenza sul
throughput dell'endpoint. Da lanciare quando la workstation torna su.

    export LLM_API_KEY=...
    python scripts/diagnose_llm.py --concurrency 1 2 4 8 --requests 16

Perche' serve una misura e non un ragionamento
----------------------------------------------
Il fix (imporre max_tokens) e' giusto a prescindere. Il MECCANISMO pero'
merita cautela prima di finire in tesi.

L'idea intuitiva e' che vLLM "prenoti" nella KV cache tutti i 261.678 token.
Con PagedAttention non e' cosi': i blocchi della KV cache vengono allocati
on-demand, un blocco alla volta, man mano che i token vengono generati. Non
c'e' preallocazione proporzionale a max_tokens.

Il costo reale e' un altro, e agisce su tre fronti:
  1. Ammissione: lo scheduler ragiona su budget di sequenza e watermark, e con
     un tetto altissimo diventa piu' conservativo nell'ammettere richieste.
  2. Nessuna garanzia di terminazione: senza tetto, una risposta che "pensa"
     (Qwen3 in thinking mode) puo' generare migliaia di token prima di
     rispondere. Ogni token occupa uno slot di decode e blocchi di KV per
     tutto il tempo. Sotto burst questo ammazza il throughput.
  3. Coda: le richieste lunghe restano in volo, quindi le successive aspettano.

Nota sulla riga "Maximum concurrency for 262,144 tokens per request: 13.43x":
e' un log di AVVIO del server, calcolato come (token totali di KV cache) /
max_model_len. Dipende da --max-model-len e dalla VRAM disponibile, NON dal
max_tokens che manda il client. Quindi e' una diagnostica sulla configurazione
del server, non una prova di cosa stesse facendo il nostro client. Utile per
dimensionare, ma non citabile come evidenza del problema.

Questo script produce l'evidenza che invece e' citabile: latenza e throughput
misurati, a parita' di prompt, variando un parametro alla volta.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirrorfish.config import LLMConfig          # noqa: E402
from mirrorfish.llm import OpenAICompatClient    # noqa: E402

PROMPT_SYSTEM = (
    "Sei un cittadino italiano su un social network. Rispondi solo con JSON: "
    '{"action": "POST"|"IGNORE", "content": "max 280 caratteri"}'
)
PROMPT_USER = (
    "Hai appena letto una notizia sul referendum costituzionale riguardante "
    "la separazione delle carriere dei magistrati. Cosa scrivi? Solo JSON."
)


async def one_batch(
    cfg: LLMConfig, n: int, max_tokens: int, concurrency: int,
    disable_thinking: bool, rpm: float, progress: bool = True,
    no_gate: bool = False,
) -> dict:
    # Il limitatore interno frena a `requests_per_minute` a prescindere dalla
    # concorrenza richiesta: se resta al default (8) il benchmark misura se
    # stesso, non l'endpoint. Va alzato esplicitamente.
    # ATTENZIONE: con il gate attivo il benchmark misura il GATE, non il
    # server. Se l'intervallo (60/rpm) supera la latenza, non c'e' mai piu'
    # di una chiamata in volo e concorrenza e max_tokens non possono avere
    # alcun effetto: e' cio' che e' successo nella misura del 3 settembre,
    # dove tutte e sei le configurazioni davano 33,4 s.
    cfg.requests_per_minute = 10_000 if no_gate else rpm
    cfg.concurrency = concurrency
    cfg.min_interval_s = 0.0          # qui vogliamo misurare il limite, non spalmare
    cfg.disable_thinking = disable_thinking
    cfg.max_retries = 1               # nessun retry: vogliamo vedere i fallimenti
    client = OpenAICompatClient(cfg)
    try:
        t0 = time.perf_counter()
        done = 0

        async def one():
            nonlocal done
            r = await client.complete(PROMPT_SYSTEM, PROMPT_USER,
                                      max_tokens=max_tokens, temperature=0.7,
                                      json_mode=True)
            done += 1
            if progress and done % 5 == 0:
                el = time.perf_counter() - t0
                print(f"      {done}/{n} ({done/max(el,.1)*60:.0f}/min)",
                      flush=True)
            return r

        results = await asyncio.gather(*[one() for _ in range(n)])
        wall = time.perf_counter() - t0
    finally:
        await client.aclose()

    ok = [r for r in results if not r.error]
    lat = [r.latency_ms for r in ok] or [0.0]
    out_tok = sum(r.completion_tokens for r in ok)
    return {
        "max_tokens": max_tokens,
        "concurrency": concurrency,
        "thinking_off": disable_thinking,
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "truncated": sum(1 for r in ok if r.truncated),
        "wall_s": round(wall, 2),
        "req_per_s": round(len(ok) / wall, 2) if wall else 0,
        "out_tok": out_tok,
        "tok_per_s": round(out_tok / wall) if wall else 0,
        "lat_mean_ms": round(statistics.mean(lat)),
        "lat_p95_ms": round(sorted(lat)[int(len(lat) * 0.95) - 1]) if len(lat) > 1 else round(lat[0]),
        "first_error": next((r.error for r in results if r.error), None),
    }


def row(d: dict) -> str:
    return (f"  {d['max_tokens']:>10} {str(d['thinking_off']):>9} {d['concurrency']:>6} "
            f"{d['wall_s']:>8} {d['req_per_s']:>8} {d['tok_per_s']:>8} "
            f"{d['lat_mean_ms']:>9} {d['lat_p95_ms']:>8} {d['truncated']:>6} {d['errors']:>7}")


HEADER = (f"  {'max_tokens':>10} {'no-think':>9} {'conc':>6} {'wall_s':>8} "
          f"{'req/s':>8} {'tok/s':>8} {'lat_med':>9} {'lat_p95':>8} {'tronc':>6} {'err':>7}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--max-tokens", type=int, nargs="+", default=[256, 512, 2048])
    ap.add_argument("--rpm", type=float, default=25.0,
                    help="tetto del limitatore interno durante la misura. "
                         "Se resta basso il benchmark misura se stesso. "
                         "Alzalo fino a dove vuoi sondare, e ricorda che il "
                         "limite di TEAM e' condiviso con i colleghi.")
    ap.add_argument("--no-gate", action="store_true",
                    help="disattiva il limitatore per misurare il tetto REALE "
                         "del server. Prende 429 e consuma quota di squadra: "
                         "usalo solo se serve davvero, con poche richieste.")
    ap.add_argument("--cooldown", type=float, default=70.0,
                    help="pausa fra configurazioni; deve superare la finestra "
                         "del rate limit, altrimenti i risultati sono un artefatto")
    ap.add_argument("--compare-thinking", action="store_true",
                    help="misura anche col thinking mode ATTIVO")
    ap.add_argument("--danger-unbounded", action="store_true",
                    help="include una prova con max_tokens enorme. NON lanciarlo "
                         "mentre altri usano la GPU.")
    args = ap.parse_args()

    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("Serve LLM_API_KEY nell'ambiente.")

    budgets = list(args.max_tokens)
    if args.danger_unbounded:
        budgets.append(261_678)

    print(f"\nEndpoint: {LLMConfig.from_env().base_url}")
    print(f"{args.requests} richieste per configurazione, prompt identico.")
    n_cfg = len(budgets) * len(args.concurrency) * (2 if args.compare_thinking else 1)
    eta = n_cfg * args.cooldown / 60
    print(f"{n_cfg} configurazioni, pausa {args.cooldown}s fra una e l'altra "
          f"(~{eta:.0f} min).\n")
    print(HEADER)
    print("  " + "-" * (len(HEADER) - 2))

    rows = []
    for mt in budgets:
        for c in args.concurrency:
            variants = [True, False] if args.compare_thinking else [True]
            for no_think in variants:
                print(f"    misuro: max_tokens={mt}, concorrenza={c}, "
                      f"rpm={args.rpm}", flush=True)
                d = await one_batch(LLMConfig.from_env(), args.requests, mt, c,
                                    no_think, args.rpm, no_gate=args.no_gate)
                rows.append(d)
                print(row(d))
                if d["errors"]:
                    print(f"      primo errore: {d['first_error']}")
                # ATTENZIONE: 2 secondi NON bastano. Se il limite e' a
                # finestra di 60s, le configurazioni successive partono con
                # il budget gia' esaurito e falliscono tutte, dando
                # l'impressione che il colpevole sia il parametro variato.
                # E' esattamente l'artefatto visto nella prima esecuzione.
                await asyncio.sleep(args.cooldown)

    print("\nCome leggerlo:")
    print("  - se tok/s NON cala alzando max_tokens a parita' di output reale,")
    print("    il tetto di per se' non e' il collo di bottiglia: lo e' la")
    print("    lunghezza effettiva delle risposte (quindi il thinking mode).")
    print("  - se 'tronc' > 0, il budget e' troppo stretto per quel tipo di")
    print("    chiamata: alzalo, altrimenti perdi risposte silenziosamente.")
    print("  - scegli la concorrenza dove req/s smette di crescere: oltre quel")
    print("    punto stai solo allungando la coda, tua e dei colleghi.")

    best = max(rows, key=lambda r: r["req_per_s"])
    print(f"\n  throughput massimo: {best['req_per_s']} req/s "
          f"@ concurrency={best['concurrency']}, max_tokens={best['max_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())
