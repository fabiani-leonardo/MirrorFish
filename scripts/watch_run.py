#!/usr/bin/env python3
"""
E' vivo o e' fermo?

    python scripts/watch_run.py runs/test_2/run.db
    python scripts/watch_run.py runs/test_2/run.db --follow

Il terminale puo' restare muto per minuti senza che nulla sia rotto: la riga
di tick si stampa solo a tick concluso, e con 106 agenti attivi a 25 req/min
un tick dura oltre due minuti. `run.db` invece registra ogni chiamata con
l'orario, quindi dice la verita' indipendentemente da cosa si vede a schermo.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import datetime, timedelta


def snapshot(db: str) -> dict:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    now = datetime.now()
    out: dict = {}
    r = c.execute("SELECT COUNT(*) n, MAX(created_at) last FROM llm_call").fetchone()
    out["calls"] = r["n"]
    out["last"] = r["last"]
    out["silence_s"] = None
    if r["last"]:
        try:
            out["silence_s"] = (now - datetime.fromisoformat(r["last"])).total_seconds()
        except ValueError:
            pass
    for label, delta in (("last_1m", 60), ("last_5m", 300)):
        cutoff = (now - timedelta(seconds=delta)).isoformat()
        out[label] = c.execute(
            "SELECT COUNT(*) n FROM llm_call WHERE created_at > ?", (cutoff,)
        ).fetchone()["n"]
    out["errors"] = c.execute(
        "SELECT COUNT(*) n FROM llm_call WHERE error IS NOT NULL").fetchone()["n"]
    recent_err = c.execute(
        "SELECT error, COUNT(*) n FROM llm_call WHERE error IS NOT NULL "
        "GROUP BY substr(error,1,40) ORDER BY n DESC LIMIT 3").fetchall()
    out["top_errors"] = [(e["error"][:70], e["n"]) for e in recent_err]
    try:
        meta = {x["key"]: x["value"] for x in c.execute("SELECT * FROM run_meta")}
        out["tick"] = int(meta.get("last_completed_tick", "-1")) + 1
    except Exception:
        out["tick"] = None
    out["posts"] = c.execute("SELECT COUNT(*) n FROM post").fetchone()["n"]
    out["votes"] = c.execute("SELECT COUNT(*) n FROM vote").fetchone()["n"]
    c.close()
    return out


def verdict(s: dict) -> str:
    if s["calls"] == 0:
        return ("NESSUNA CHIAMATA. Non ha ancora superato la fase di avvio: "
                "controlla che il processo esista (ps aux | grep run.py).")
    sil = s["silence_s"]
    if sil is None:
        return "stato incerto"
    if s["last_1m"] > 0:
        return f"IN LAVORAZIONE: {s['last_1m']} chiamate nell'ultimo minuto."
    if sil < 120:
        return (f"RALLENTATO: nessuna chiamata da {sil:.0f}s. Compatibile con "
                f"una pausa da rate limit (fino a 60s).")
    return (f"FERMO: nessuna chiamata da {sil/60:.1f} minuti. "
            f"Il processo e' bloccato o e' morto.")


def show(db: str) -> None:
    s = snapshot(db)
    print(f"\n  tick completati   : {s['tick']}")
    print(f"  chiamate totali   : {s['calls']:,}")
    print(f"  ultimo minuto     : {s['last_1m']}")
    print(f"  ultimi 5 minuti   : {s['last_5m']}  "
          f"({s['last_5m']/5:.1f}/min effettivi)")
    print(f"  post / voti       : {s['posts']} / {s['votes']}")
    print(f"  errori            : {s['errors']}")
    for e, n in s["top_errors"]:
        print(f"      {n:>4}x {e}")
    print(f"\n  {verdict(s)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--every", type=float, default=30.0)
    a = ap.parse_args()
    if not a.follow:
        show(a.db)
        return
    try:
        while True:
            print("\033[2J\033[H", end="")
            print(f"  {a.db}   {time.strftime('%H:%M:%S')}")
            show(a.db)
            time.sleep(a.every)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
