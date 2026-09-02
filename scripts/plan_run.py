#!/usr/bin/env python3
"""
Pianificatore di run: quante chiamate, quanto tempo, quanta differenziazione.

Non tocca la rete. Serve a scegliere `--hours-per-tick`, il numero di agenti e
la frequenza di riflessione PRIMA di bruciare una notte di quota.

    python scripts/plan_run.py --agents 105 --days 30
    python scripts/plan_run.py --agents 105 --days 30 --rpm 25   # se alzano la quota

Il vincolo vero non e' la GPU: e' `x-ratelimit-team_member-limit-requests`.
Con 8 richieste/minuto ogni chiamata costa 7,5 secondi di orologio, punto.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirrorfish.population import CRONOTIPI, activation_prob  # noqa: E402


def estimate(agents: int, days: int, hours_per_tick: int, reflect_every: int,
             base_activity: float, rpm: float) -> dict:
    ticks = max(1, days * 24 // hours_per_tick)

    # Frazione media di agenti attivi per tick, dai cronotipi reali.
    probs = []
    for tick in range(ticks):
        start_h = (tick * hours_per_tick) % 24
        for hours in CRONOTIPI.values():
            probs.append(activation_prob(
                hours, start_h, hours_per_tick,
                base_activity * 24 / max(1, hours_per_tick)))
    active_frac = statistics.mean(probs)

    actions = ticks * agents * active_frac
    # Alla riflessione va chi ha letto qualcosa dall'ultimo giro: satura
    # rapidamente verso la popolazione intera.
    rounds = ticks // reflect_every if reflect_every else 0
    seen_frac = min(1.0, 1 - (1 - active_frac) ** reflect_every)
    reflections = rounds * agents * seen_frac
    surveys = 2 * agents
    total = actions + reflections + surveys

    minutes = total / rpm
    return {
        "ticks": ticks, "active_frac": active_frac,
        "actions": round(actions), "reflections": round(reflections),
        "surveys": surveys, "total": round(total),
        "hours": minutes / 60,
    }


def differentiation(hours_per_tick: int, base_activity: float) -> float:
    """
    Quanto i cronotipi restano distinguibili a questa lunghezza di tick.

    Coefficiente di variazione delle probabilita' di attivazione fra cronotipi,
    mediato sui tick. Vicino a 0 = tutti uguali: il profilo orario derivato
    dai dati ISTAT non produce piu' alcuna differenza di comportamento, e
    tenerlo nel modello diventa decorativo.
    """
    cvs = []
    for start_h in range(0, 24, max(1, hours_per_tick)):
        ps = [activation_prob(h, start_h, hours_per_tick,
                              base_activity * 24 / max(1, hours_per_tick))
              for h in CRONOTIPI.values()]
        m = statistics.mean(ps)
        if m > 0:
            cvs.append(statistics.pstdev(ps) / m)
    return statistics.mean(cvs) if cvs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=105)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--reflect-every", type=int, default=4)
    ap.add_argument("--base-activity", type=float, default=0.35)
    ap.add_argument("--rpm", type=float, default=8.0,
                    help="richieste al minuto concesse (header team_member)")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--conditions", type=int, default=4,
                    help="condizioni sperimentali (base, controllo, controfattuali)")
    args = ap.parse_args()

    print(f"\n{args.agents} agenti, {args.days} giorni simulati, "
          f"riflessione ogni {args.reflect_every} tick, {args.rpm} req/min\n")
    print(f"  {'h/tick':>7}{'tick':>7}{'attivi%':>9}{'azioni':>9}{'rifles.':>9}"
          f"{'TOTALE':>9}{'ore/run':>9}{'differenz.':>12}")
    print("  " + "-" * 71)

    rows = []
    for hpt in (2, 4, 6, 8, 11, 12, 24):
        e = estimate(args.agents, args.days, hpt, args.reflect_every,
                     args.base_activity, args.rpm)
        d = differentiation(hpt, args.base_activity)
        rows.append((hpt, e, d))
        flag = "  <- appiattito" if d < 0.15 else ""
        print(f"  {hpt:>7}{e['ticks']:>7}{e['active_frac'] * 100:>8.0f}%"
              f"{e['actions']:>9}{e['reflections']:>9}{e['total']:>9}"
              f"{e['hours']:>9.1f}{d:>11.2f}{flag}")

    print("\n  'differenz.' = quanto i cronotipi restano distinguibili.")
    print("  Sotto ~0.15 il profilo orario non produce piu' differenze di")
    print("  comportamento: tenerlo nel modello diventa decorativo.")

    print(f"\n--- COSTO DEL DISEGNO SPERIMENTALE "
          f"({args.conditions} condizioni x {args.replicates} repliche) ---")
    print(f"  {'h/tick':>7}{'ore/run':>10}{'run totali':>13}{'ore totali':>13}"
          f"{'giorni continui':>17}")
    print("  " + "-" * 60)
    n_runs = args.conditions * args.replicates
    for hpt, e, _ in rows:
        tot_h = e["hours"] * n_runs
        print(f"  {hpt:>7}{e['hours']:>10.1f}{n_runs:>13}{tot_h:>13.0f}"
              f"{tot_h / 24:>17.1f}")

    print("\n  Le repliche non sono facoltative: con un LLM non deterministico")
    print("  la differenza fra due condizioni non e' interpretabile senza")
    print("  conoscere la variabilita' fra repliche della stessa condizione.")


if __name__ == "__main__":
    main()
