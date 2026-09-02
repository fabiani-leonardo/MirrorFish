#!/usr/bin/env python3
"""
Server di sola lettura per seguire una simulazione in corso.

    python scripts/serve_feed.py runs/base_s42/run.db --port 8000

Processo SEPARATO, di proposito. Avviarlo dentro `run.py` aggiungerebbe un
thread e una socket a un processo che deve girare otto ore senza sorveglianza:
se il server morisse o bloccasse una porta, si porterebbe dietro la
simulazione. E' lo stesso errore di `ansa_injector.py`, che accoppiava due
cose che non avevano motivo di essere accoppiate.

Il database viene aperto in modalita' `mode=ro` sull'URI: il server non puo'
scrivere nemmeno per un bug. SQLite in WAL — come lo apre `Store` — consente
letture concorrenti mentre la simulazione scrive, quindi non c'e' contesa.

Nota per macOS: la porta 5000 e' occupata da AirPlay Receiver di sistema.
Il default qui e' 8000.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from render_feed import build, load  # noqa: E402

POLL_JS = """
<script>
(function(){
  const seen = %(count)d;
  const every = %(every)d * 1000;
  // Conserva la posizione di scorrimento: un ricaricamento che riporta in
  // cima ogni 15 secondi rende la pagina inutilizzabile mentre si legge.
  const key = 'feedScroll';
  const saved = sessionStorage.getItem(key);
  if (saved) { window.scrollTo(0, parseInt(saved, 10)); }
  window.addEventListener('beforeunload', function(){
    sessionStorage.setItem(key, String(window.scrollY));
  });
  async function tick(){
    try {
      const r = await fetch('/status', {cache:'no-store'});
      const s = await r.json();
      const el = document.getElementById('live');
      if (el) {
        el.textContent = s.done
          ? 'simulazione conclusa · ' + s.posts + ' post'
          : 'in corso · tick ' + (s.tick + 1) + '/' + s.total
            + ' · ' + s.posts + ' post';
      }
      if (s.posts !== seen) { location.reload(); }
    } catch (e) { /* la simulazione potrebbe essere fra due commit */ }
  }
  setInterval(tick, every);
  tick();
})();
</script>
"""

BANNER = ('<p class="sub" id="live" style="margin-top:.5rem">'
          'in attesa di aggiornamenti…</p>')


def read_only(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def status(db: Path) -> dict:
    try:
        conn = read_only(db)
        posts = conn.execute("SELECT COUNT(*) c FROM post").fetchone()["c"]
        row = conn.execute(
            "SELECT value FROM run_meta WHERE key='last_completed_tick'"
        ).fetchone()
        tick = json.loads(row["value"]) if row else -1
        cfg = conn.execute(
            "SELECT value FROM run_meta WHERE key='sim_config'").fetchone()
        total = 0
        if cfg:
            c = json.loads(cfg["value"])
            from datetime import date
            days = (date.fromisoformat(c["end_date"])
                    - date.fromisoformat(c["start_date"])).days + 1
            total = max(1, int(days * 24 / c.get("hours_per_tick", 8)))
        done = conn.execute(
            "SELECT COUNT(*) c FROM vote WHERE label='final'").fetchone()["c"] > 0
        conn.close()
        return {"posts": posts, "tick": tick, "total": total, "done": done}
    except sqlite3.Error as e:
        return {"posts": -1, "tick": -1, "total": 0, "done": False,
                "error": str(e)}


class Handler(BaseHTTPRequestHandler):
    db: Path
    every: int
    _lock = threading.Lock()

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/status"):
            self._send(json.dumps(status(self.db)).encode(), "application/json")
            return
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        with self._lock:
            try:
                d = load(f"file:{self.db}?mode=ro")
                html = build(d, f"Feed simulato · {self.db.parent.name}")
            except sqlite3.Error as e:
                self._send(f"<p>Database non leggibile: {e}</p>".encode(),
                           "text/html; charset=utf-8")
                return
        n = len(d["posts"])
        html = html.replace("</header>", BANNER + "</header>", 1)
        html = html.replace("</body>",
                            POLL_JS % {"count": n, "every": self.every} + "</body>")
        self._send(html.encode("utf-8"), "text/html; charset=utf-8")

    def log_message(self, *a):  # silenzia il log per richiesta
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--port", type=int, default=8000,
                    help="default 8000; su macOS la 5000 e' presa da AirPlay")
    ap.add_argument("--every", type=int, default=20,
                    help="secondi fra un controllo e l'altro")
    args = ap.parse_args()

    db = Path(args.db).resolve()
    if not db.exists():
        raise SystemExit(f"{db} non esiste. Avvia prima la simulazione.")

    handler = partial(Handler)
    Handler.db, Handler.every = db, args.every
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    s = status(db)
    print(f"Feed su http://127.0.0.1:{args.port}")
    print(f"  database : {db}  (sola lettura)")
    print(f"  stato    : tick {s['tick'] + 1}/{s['total']}, {s['posts']} post")
    print("  Ctrl-C per fermare. Non tocca la simulazione.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nfermato")


if __name__ == "__main__":
    main()
