#!/usr/bin/env python3
"""
Quale concorrenza serve, a partire dalle latenze misurate.

    python scripts/pick_concurrency.py runs/base_s42/run.db --rpm 40

Non serve un benchmark. Con una quota di R richieste al minuto il limitatore
rilascia una partenza ogni 60/R secondi. Se una chiamata dura L secondi,
il numero di chiamate contemporaneamente in volo e' circa L / (60/R).

La concorrenza serve solo a NON rallentare rispetto alla quota. Alzarla oltre
non aumenta il ritmo — quello lo decide la quota — aumenta solo la coda sul
server condiviso.
"""
from __future__ import annotations

import argparse
import math
import sqlite3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--rpm", type=float, default=40.0)
    ap.add_argument("--margin", type=float, default=1.5,
                    help="fattore di sicurezza sulla latenza p95")
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute(
        "SELECT purpose, latency_ms FROM llm_call "
        "WHERE error IS NULL AND latency_ms > 0"
    ).fetchall()
    c.close()
    if not rows:
        raise SystemExit("Nessuna latenza registrata in questo run.")

    per: dict[str, list[float]] = {}
    for r in rows:
        per.setdefault(r["purpose"], []).append(r["latency_ms"])

    interval = 60.0 / a.rpm
    print(f"\nQuota richiesta : {a.rpm:.0f} req/min "
          f"-> una partenza ogni {interval:.2f} s\n")
    print(f"  {'tipo':<12}{'n':>6}{'lat med':>10}{'lat p95':>10}"
          f"{'in volo':>10}")
    print("  " + "-" * 48)

    worst = 0.0
    for purpose, lats in sorted(per.items()):
        lats.sort()
        med = lats[len(lats) // 2] / 1000
        p95 = lats[max(0, int(len(lats) * 0.95) - 1)] / 1000
        inflight = p95 / interval
        worst = max(worst, inflight)
        print(f"  {purpose:<12}{len(lats):>6}{med:>9.2f}s{p95:>9.2f}s"
              f"{inflight:>10.1f}")

    needed = max(1, math.ceil(worst * a.margin))
    print(f"\n  Chiamate in volo nel caso peggiore : {worst:.1f}")
    print(f"  Con margine {a.margin}x                  : {needed}")
    print(f"\n  --> --concurrency {needed}")
    print(f"\n  Oltre questo valore il ritmo non sale: lo fissa la quota.")
    print(f"  Sotto, invece, non riesci a saturarla e il run dura di piu'.")

    # Quanto durerebbe un run tipico
    tot = len(rows)
    print(f"\n  Questo run ({tot:,} chiamate) a {a.rpm:.0f} req/min: "
          f"{tot/a.rpm/60:.1f} h")


if __name__ == "__main__":
    main()
