#!/usr/bin/env python3
"""
Esporta bio e note nel formato JSON di MiroFish.

    python scripts/export_notes.py runs/base_s42/run.db -o runs/base_s42/personas.json

Nella vecchia versione questo file ERA lo stato: la simulazione ci scriveva
dentro durante l'esecuzione, con lock e scritture atomiche fatte a mano.
Qui non lo e' e non deve esserlo — la fonte di verita' e' SQLite, per tre
motivi concreti:

  - le scritture sono transazionali: un'interruzione a meta' tick non lascia
    un JSON troncato e illeggibile;
  - `--resume` funziona perche' esiste un checkpoint coerente;
  - "quante note ha prodotto ogni fascia d'eta'" e' una query, non uno script.

Questo file resta pero' utile, ed e' quello che chiedevi: stato leggibile a
occhio, diffabile fra run, e allegabile in appendice alla tesi. E' una VISTA
derivata, rigenerabile in qualunque momento.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def export(db: str, out: str | None = None, indent: int = 2) -> Path:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    data: dict[str, dict] = {}
    for a in conn.execute("SELECT * FROM agent ORDER BY agent_id"):
        data[str(a["agent_id"])] = {
            "username": a["username"],
            "static_bio": a["static_bio"],
            "profession": a["profession"],
            "age": a["age"],
            "region": a["region"],
            "is_source": bool(a["is_source"]),
            "notes": [],
        }

    for n in conn.execute("SELECT * FROM note ORDER BY agent_id, note_id"):
        key = str(n["agent_id"])
        if key in data:
            data[key]["notes"].append({
                "round": n["tick"],
                "note": n["note"],
                "reasoning": n["reasoning"],
                "timestamp": n["created_at"],
            })

    # voto prima/dopo accanto alle note: e' il confronto che serve leggendo
    for v in conn.execute("SELECT label, agent_id, vote, motivation FROM vote"):
        key = str(v["agent_id"])
        if key in data and v["label"] in ("baseline", "final"):
            data[key].setdefault("votes", {})[v["label"]] = {
                "vote": v["vote"], "motivation": v["motivation"]}
    conn.close()

    path = Path(out) if out else Path(db).parent / "personas.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=indent),
                    encoding="utf-8")
    n_notes = sum(len(v["notes"]) for v in data.values())
    print(f"{path}  ({len(data)} agenti, {n_notes} note, "
          f"{path.stat().st_size / 1024:.0f} KB)")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--indent", type=int, default=2)
    a = ap.parse_args()
    export(a.db, a.out, a.indent)
