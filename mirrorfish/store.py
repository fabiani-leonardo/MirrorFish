"""
Persistenza: un file SQLite per run.

Perche' SQLite e non Neo4j (vedi note in README):
  - il feed e' una join a due tabelle, non serve un graph DB per questo;
  - un file per run = snapshot, diff e archiviazione banali, che e' cio' che
    serve per confrontare run controfattuali in tesi;
  - zero infrastruttura da tenere viva mentre scrivi.
Il grafo sociale resta comunque ricostruibile in qualunque momento dalle
tabelle `agent` e `follow` (vedi export_graph) per l'analisi di rete a
posteriori con networkx.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS agent (
    agent_id        INTEGER PRIMARY KEY,
    username        TEXT NOT NULL UNIQUE,
    static_bio      TEXT NOT NULL DEFAULT '',
    profession      TEXT,
    age             INTEGER,
    region          TEXT,
    education       TEXT,
    activity        REAL NOT NULL DEFAULT 0.35,
    is_source       INTEGER NOT NULL DEFAULT 0,
    attrs           TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS follow (
    follower_id     INTEGER NOT NULL REFERENCES agent(agent_id),
    followee_id     INTEGER NOT NULL REFERENCES agent(agent_id),
    PRIMARY KEY (follower_id, followee_id)
);

CREATE TABLE IF NOT EXISTS post (
    post_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        INTEGER NOT NULL REFERENCES agent(agent_id),
    parent_id       INTEGER REFERENCES post(post_id),
    content         TEXT NOT NULL,
    kind            TEXT NOT NULL,          -- post | reply | news
    tick            INTEGER NOT NULL,
    sim_date        TEXT NOT NULL,
    news_id         TEXT                    -- provenienza, se iniettato
);
CREATE INDEX IF NOT EXISTS idx_post_tick ON post(tick);
CREATE INDEX IF NOT EXISTS idx_post_agent ON post(agent_id);

CREATE TABLE IF NOT EXISTS reaction (
    agent_id        INTEGER NOT NULL REFERENCES agent(agent_id),
    post_id         INTEGER NOT NULL REFERENCES post(post_id),
    kind            TEXT NOT NULL,          -- like | dislike | share
    tick            INTEGER NOT NULL,
    PRIMARY KEY (agent_id, post_id, kind)
);

CREATE TABLE IF NOT EXISTS note (
    note_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        INTEGER NOT NULL REFERENCES agent(agent_id),
    tick            INTEGER NOT NULL,
    note            TEXT NOT NULL,
    reasoning       TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_note_agent ON note(agent_id);

CREATE TABLE IF NOT EXISTS vote (
    vote_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    label           TEXT NOT NULL,          -- baseline | final | tick_N
    agent_id        INTEGER NOT NULL REFERENCES agent(agent_id),
    vote            TEXT NOT NULL,          -- SI | NO | ASTENUTO | ERROR
    confidence      REAL NOT NULL DEFAULT 0,
    motivation      TEXT NOT NULL DEFAULT '',
    tick            INTEGER,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vote_label ON vote(label);

-- Telemetria: e' cio' che rende misurabile il costo computazionale.
CREATE TABLE IF NOT EXISTS llm_call (
    call_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tick              INTEGER,
    agent_id          INTEGER,
    purpose           TEXT NOT NULL,        -- action | reflection | vote
    max_tokens        INTEGER NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    finish_reason     TEXT NOT NULL DEFAULT '',
    latency_ms        REAL NOT NULL DEFAULT 0,
    error             TEXT,
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_call_purpose ON llm_call(purpose);

CREATE TABLE IF NOT EXISTS run_meta (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL
);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ----------------------------------------------------------------- meta #
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO run_meta (key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False, default=str)),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value FROM run_meta WHERE key = ?", (key,)
        ).fetchone()
        return json.loads(row["value"]) if row else default

    # --------------------------------------------------------------- agenti #
    def add_agents(self, agents: Iterable[dict[str, Any]]) -> int:
        rows = [
            (
                a["agent_id"], a["username"], a.get("static_bio", ""),
                a.get("profession"), a.get("age"), a.get("region"),
                a.get("education"), a.get("activity", 0.35),
                int(a.get("is_source", 0)),
                json.dumps(a.get("attrs", {}), ensure_ascii=False),
            )
            for a in agents
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO agent (agent_id, username, static_bio, "
            "profession, age, region, education, activity, is_source, attrs) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def agents(self, include_sources: bool = False) -> list[sqlite3.Row]:
        q = "SELECT * FROM agent"
        if not include_sources:
            q += " WHERE is_source = 0"
        return self.conn.execute(q + " ORDER BY agent_id").fetchall()

    def add_follows(self, edges: Iterable[tuple[int, int]]) -> int:
        edges = list(edges)
        self.conn.executemany(
            "INSERT OR IGNORE INTO follow (follower_id, followee_id) VALUES (?,?)",
            edges,
        )
        self.conn.commit()
        return len(edges)

    # ------------------------------------------------------------- contenuti #
    def add_post(
        self, agent_id: int, content: str, kind: str, tick: int,
        sim_date: str, parent_id: int | None = None, news_id: str | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO post (agent_id, parent_id, content, kind, tick, "
            "sim_date, news_id) VALUES (?,?,?,?,?,?,?)",
            (agent_id, parent_id, content, kind, tick, sim_date, news_id),
        )
        return int(cur.lastrowid)

    def add_reaction(self, agent_id: int, post_id: int, kind: str, tick: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO reaction (agent_id, post_id, kind, tick) "
            "VALUES (?,?,?,?)",
            (agent_id, post_id, kind, tick),
        )

    _FEED_SELECT = """
        SELECT p.post_id, p.content, p.kind, p.tick, p.sim_date,
               a.username, a.agent_id AS author_id,
               (SELECT COUNT(*) FROM reaction r
                 WHERE r.post_id = p.post_id AND r.kind = 'like') AS likes
        FROM post p
        JOIN agent a ON a.agent_id = p.agent_id
        WHERE p.tick < ? AND p.agent_id != ? AND {clause}
        ORDER BY p.tick DESC, p.post_id DESC LIMIT ?
    """

    def feed_for(
        self, agent_id: int, limit: int, before_tick: int, news_slots: int = 2
    ) -> list[sqlite3.Row]:
        """
        Feed = quota di notizie + quota di post dei seguiti.

        Gli slot sono SEPARATI di proposito. Con una sola query ordinata per
        recency le notizie vincono sempre (sono inserite a inizio tick e
        l'agenzia e' visibile a tutti), quindi la quota di esposizione
        mediatica finiva per dipendere da quante notizie ci sono nell'archivio
        quel giorno: un parametro sperimentale determinato per caso.
        Cosi' invece e' dichiarato, e variarlo e' un asse della sensitivity
        analysis (esposizione mediatica vs esposizione ai pari).
        """
        news_slots = max(0, min(news_slots, limit))
        news = self.conn.execute(
            self._FEED_SELECT.format(clause="a.is_source = 1"),
            (before_tick, agent_id, news_slots),
        ).fetchall() if news_slots else []

        peers = self.conn.execute(
            self._FEED_SELECT.format(
                clause="a.is_source = 0 AND p.agent_id IN "
                       "(SELECT followee_id FROM follow WHERE follower_id = ?)"
            ),
            (before_tick, agent_id, agent_id, limit - len(news)),
        ).fetchall()
        return list(news) + list(peers)

    def posts_by(self, agent_id: int, limit: int = 5) -> list[str]:
        rows = self.conn.execute(
            "SELECT content FROM post WHERE agent_id = ? "
            "ORDER BY post_id DESC LIMIT ?",
            (agent_id, limit),
        ).fetchall()
        return [r["content"] for r in rows]

    # ----------------------------------------------------------------- note #
    def add_note(self, agent_id: int, tick: int, note: str, reasoning: str = "") -> None:
        self.conn.execute(
            "INSERT INTO note (agent_id, tick, note, reasoning, created_at) "
            "VALUES (?,?,?,?,?)",
            (agent_id, tick, note, reasoning, datetime.now().isoformat()),
        )

    def notes_for(self, agent_id: int, limit: int | None = None) -> list[str]:
        q = "SELECT note FROM note WHERE agent_id = ? ORDER BY note_id"
        rows = self.conn.execute(q, (agent_id,)).fetchall()
        notes = [r["note"] for r in rows]
        return notes[-limit:] if limit else notes

    # ---------------------------------------------------------------- voti  #
    def add_vote(
        self, label: str, agent_id: int, vote: str,
        confidence: float, motivation: str, tick: int | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO vote (label, agent_id, vote, confidence, motivation, "
            "tick, created_at) VALUES (?,?,?,?,?,?,?)",
            (label, agent_id, vote, confidence, motivation, tick,
             datetime.now().isoformat()),
        )

    def vote_tally(self, label: str) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT vote, COUNT(*) AS n FROM vote WHERE label = ? GROUP BY vote",
            (label,),
        ).fetchall()
        return {r["vote"]: r["n"] for r in rows}

    def votes(self, label: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM vote WHERE label = ? ORDER BY agent_id", (label,)
        ).fetchall()

    # ----------------------------------------------------------- telemetria #
    def log_call(
        self, purpose: str, max_tokens: int, resp: Any,
        tick: int | None = None, agent_id: int | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO llm_call (tick, agent_id, purpose, max_tokens, "
            "prompt_tokens, completion_tokens, finish_reason, latency_ms, "
            "error, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (tick, agent_id, purpose, max_tokens,
             getattr(resp, "prompt_tokens", 0), getattr(resp, "completion_tokens", 0),
             getattr(resp, "finish_reason", ""), getattr(resp, "latency_ms", 0.0),
             getattr(resp, "error", None), datetime.now().isoformat()),
        )

    def call_stats(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT purpose,
                   COUNT(*)                       AS calls,
                   SUM(prompt_tokens)             AS prompt_tok,
                   SUM(completion_tokens)         AS completion_tok,
                   ROUND(AVG(completion_tokens),1) AS avg_out,
                   MAX(completion_tokens)         AS max_out,
                   MAX(max_tokens)                AS budget,
                   SUM(finish_reason = 'length')  AS truncated,
                   SUM(error IS NOT NULL)         AS errors,
                   ROUND(AVG(latency_ms))         AS avg_ms
            FROM llm_call GROUP BY purpose ORDER BY purpose
            """
        ).fetchall()

    # ---------------------------------------------------------------- utils #
    def export_graph(self) -> dict[str, Any]:
        """Grafo sociale in formato node-link, pronto per networkx."""
        nodes = [
            {"id": r["agent_id"], "username": r["username"],
             "region": r["region"], "age": r["age"], "is_source": r["is_source"]}
            for r in self.conn.execute("SELECT * FROM agent").fetchall()
        ]
        links = [
            {"source": r["follower_id"], "target": r["followee_id"]}
            for r in self.conn.execute("SELECT * FROM follow").fetchall()
        ]
        return {"directed": True, "nodes": nodes, "links": links}

    def truncate_after_tick(self, tick: int) -> dict[str, int]:
        """
        Cancella tutto cio' che appartiene a tick successivi a `tick`.

        Serve alla ripresa. Il checkpoint viene scritto solo a tick COMPLETO,
        quindi un crash a meta' tick lascia in DB le azioni degli agenti che
        avevano gia' risposto. Senza questa pulizia la ripresa rifa' quel tick
        e le somma alle precedenti: post duplicati, conteggi gonfiati, e
        nessun errore visibile.

        L'ordine di cancellazione non e' arbitrario: le reaction e le reply
        referenziano i post, quindi vanno rimosse prima, altrimenti scatta
        FOREIGN KEY constraint failed.
        """
        removed = {}
        cur = self.conn.execute(
            "DELETE FROM reaction WHERE tick > ?", (tick,))
        removed["reaction"] = cur.rowcount
        cur = self.conn.execute(
            "DELETE FROM post WHERE tick > ? AND parent_id IS NOT NULL", (tick,))
        removed["reply"] = cur.rowcount
        cur = self.conn.execute("DELETE FROM post WHERE tick > ?", (tick,))
        removed["post"] = cur.rowcount
        cur = self.conn.execute("DELETE FROM note WHERE tick > ?", (tick,))
        removed["note"] = cur.rowcount
        cur = self.conn.execute("DELETE FROM llm_call WHERE tick > ?", (tick,))
        removed["llm_call"] = cur.rowcount
        cur = self.conn.execute("DELETE FROM vote WHERE label = 'final'")
        removed["vote_final"] = cur.rowcount
        self.conn.commit()
        return {k: v for k, v in removed.items() if v > 0}

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
