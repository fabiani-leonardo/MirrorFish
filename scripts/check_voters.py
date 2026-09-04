#!/usr/bin/env python3
"""
Chi sta votando nella survey, e chi non dovrebbe.

    python scripts/check_voters.py runs/base_s42/run.db

Gli account istituzionali (partiti, comitati, testate) partecipano al dibattito
ma NON sono elettori. Se finiscono nel conteggio, il risultato del referendum
e' contaminato da voci che nella realta' non hanno una scheda.
"""
from __future__ import annotations

import argparse
import re
import sqlite3

MARCATORI = [
    r"non è un elettore", r"non e' un elettore",
    r"non è una persona fisica", r"non e' una persona fisica",
    r"account istituzionale", r"account ufficiale",
    r"\bcomitato\b", r"\btestata\b", r"agenzia di stampa",
]
RE_MARK = re.compile("|".join(MARCATORI), re.IGNORECASE)


def classify(bio: str, username: str) -> str | None:
    if RE_MARK.search(bio or ""):
        return "marcatore nella bio"
    if re.search(r"partito|movimento|lega_|forza_|fratelli|comitato|ansa|"
                 r"repubblica|corriere|stampa", username or "", re.IGNORECASE):
        return "username istituzionale"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    a = ap.parse_args()
    conn = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    sospetti = []
    for r in conn.execute("SELECT * FROM agent ORDER BY agent_id"):
        why = classify(r["static_bio"], r["username"])
        if why and not r["is_source"]:
            sospetti.append((r, why))

    tot = conn.execute("SELECT COUNT(*) c FROM agent WHERE is_source=0").fetchone()["c"]
    print(f"\n{tot} agenti conteggiati come elettori.")
    if not sospetti:
        print("Nessun account istituzionale rilevato fra loro.")
        return

    print(f"{len(sospetti)} NON dovrebbero votare:\n")
    print(f"  {'id':>4}  {'username':<28}{'voto base':<11}{'voto fin.':<11}motivo")
    print("  " + "-" * 74)
    for r, why in sospetti:
        v = {x["label"]: x["vote"] for x in conn.execute(
            "SELECT label, vote FROM vote WHERE agent_id=?", (r["agent_id"],))}
        print(f"  {r['agent_id']:>4}  {r['username'][:27]:<28}"
              f"{v.get('baseline','-'):<11}{v.get('final','-'):<11}{why}")

    ids = [r["agent_id"] for r, _ in sospetti]
    marks = ",".join("?" * len(ids))
    for label in ("baseline", "final"):
        rows = conn.execute(
            f"SELECT vote, COUNT(*) n FROM vote WHERE label=? GROUP BY vote",
            (label,)).fetchall()
        tally = {x["vote"]: x["n"] for x in rows}
        if not tally:
            continue
        rows2 = conn.execute(
            f"SELECT vote, COUNT(*) n FROM vote WHERE label=? "
            f"AND agent_id NOT IN ({marks}) GROUP BY vote", [label] + ids).fetchall()
        clean = {x["vote"]: x["n"] for x in rows2}
        tv = sum(v for k, v in tally.items() if k != "ERROR") or 1
        cv = sum(v for k, v in clean.items() if k != "ERROR") or 1
        print(f"\n  [{label}]   con istituzionali -> senza")
        for k in ("SI", "NO", "ASTENUTO"):
            print(f"    {k:<9} {tally.get(k,0)/tv:>6.1%}  ->  {clean.get(k,0)/cv:>6.1%}")
    conn.close()
    print("\nCorrezione: aggiorna il codice e rigenera con --force, oppure")
    print("escludi questi id in fase di analisi documentando la scelta.")


if __name__ == "__main__":
    main()
