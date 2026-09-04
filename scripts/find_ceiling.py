#!/usr/bin/env python3
"""
Trova la velocita' massima sostenibile, misurandola.

    export LLM_API_KEY=...
    python scripts/find_ceiling.py

Parte basso e sale finche' non incassa un 429, poi torna indietro. Serve
perche' dopo un aumento di quota il limite che scatta non e' necessariamente
quello che e' stato alzato: il tetto effettivo e'

    min(api_key_parallel, team_member_rpm, team_rpm - consumo_altri)

e l'ultimo termine non e' osservabile localmente.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirrorfish.config import LLMConfig      # noqa: E402
from mirrorfish.llm import OpenAICompatClient  # noqa: E402

SYS = "Rispondi solo con JSON."
USR = 'Rispondi {"ok": true} e basta.'


async def burst(rpm: float, n: int) -> dict:
    cfg = LLMConfig.from_env(requests_per_minute=rpm, concurrency=5)
    cfg.max_retries = 1
    client = OpenAICompatClient(cfg)
    try:
        t0 = time.perf_counter()
        res = await asyncio.gather(*[
            client.complete(SYS, USR, max_tokens=64, temperature=0.2)
            for _ in range(n)])
        wall = time.perf_counter() - t0
        limits = {}
        ok = [r for r in res if not r.error]
        gate = client.gate
        return {
            "rpm_target": rpm, "ok": len(ok), "err": len(res) - len(ok),
            "wall": wall, "rate": len(ok) / wall * 60 if wall else 0,
            "scope": gate.observed_scope, "remaining": gate.observed_remaining,
            "first_err": next((r.error for r in res if r.error), None),
        }
    finally:
        await client.aclose()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, nargs="+",
                    default=[8, 15, 20, 25, 30, 40, 55])
    ap.add_argument("--requests", type=int, default=25)
    ap.add_argument("--cooldown", type=float, default=70.0,
                    help="pausa fra i gradini: deve superare la finestra")
    a = ap.parse_args()

    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("Serve LLM_API_KEY nell'ambiente.")

    print(f"\n{a.requests} richieste per gradino, pausa {a.cooldown}s "
          f"(~{len(a.steps) * a.cooldown / 60:.0f} min in tutto)\n")
    print(f"  {'target':>7}{'ok':>5}{'err':>5}{'misurato':>11}"
          f"{'limite piu stretto':>22}{'residuo':>9}")
    print("  " + "-" * 60)

    best = 0.0
    for rpm in a.steps:
        d = await burst(rpm, a.requests)
        scope = d["scope"] or "-"
        print(f"  {rpm:>7.0f}{d['ok']:>5}{d['err']:>5}{d['rate']:>10.1f}/m"
              f"{scope:>22}{str(d['remaining']):>9}")
        if d["err"]:
            print(f"      {str(d['first_err'])[:110]}")
            break
        best = max(best, d["rate"])
        await asyncio.sleep(a.cooldown)

    print(f"\n  Massimo sostenuto senza rifiuti: {best:.0f} req/min")
    safe = best * 0.8
    print(f"  Da usare in produzione (80%, margine per i colleghi): "
          f"--rpm {safe:.0f}")
    print(f"\n  Se 'limite piu stretto' dice 'team', il tetto e' condiviso:")
    print(f"  quello che misuri ora cambia quando Chiara lancia qualcosa.")
    for calls, label in ((4012, "run da 30 giorni"), (20800, "run da 150 giorni")):
        print(f"  {label:<22} {calls / max(safe,1) / 60:>5.1f} h a --rpm {safe:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
