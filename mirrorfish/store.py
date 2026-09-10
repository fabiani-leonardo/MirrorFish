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
import random
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
    activity_hours  TEXT NOT NULL DEFAULT '[]',
    is_source       INTEGER NOT NULL DEFAULT 0,
    is_voter        INTEGER NOT NULL DEFAULT 1,
    media_depth     TEXT NOT NULL DEFAULT 'titolo',
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

-- Indici della generazione dei candidati. Senza questi tre, il piano di
-- esecuzione di _CAND_SELECT degrada a
--     CORRELATED SCALAR SUBQUERY -> SCAN r
--     CORRELATED SCALAR SUBQUERY -> SCAN c
-- cioe' una scansione COMPLETA di `reaction` e una di `post` per OGNI post
-- candidato, per ogni agente, per ogni tick. Il costo cresce col quadrato
-- della durata del run: misurato su un DB da 3.708 post, 42,90 ms per
-- chiamata contro 0,36 ms con gli indici, 119 volte piu' lento. E' la causa
-- del rallentamento progressivo osservato nel run da 432 tick (209 agenti/min
-- al tick 288, 129/min al tick 428) con la CPU quasi ferma: il tempo era
-- speso in SQLite, non nel modello.
CREATE INDEX IF NOT EXISTS idx_post_parent   ON post(parent_id);
CREATE INDEX IF NOT EXISTS idx_post_agent_tick ON post(agent_id, tick);
CREATE INDEX IF NOT EXISTS idx_reaction_post ON reaction(post_id);

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
                json.dumps(a.get("activity_hours") or []),
                int(a.get("is_source", 0)), int(a.get("is_voter", 1)),
                a.get("media_depth", "titolo"),
                json.dumps(a.get("attrs", {}), ensure_ascii=False),
            )
            for a in agents
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO agent (agent_id, username, static_bio, "
            "profession, age, region, education, activity, activity_hours, "
            "is_source, is_voter, media_depth, attrs) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()
        return len(rows)

    def agents(self, include_sources: bool = False,
               voters_only: bool = False) -> list[sqlite3.Row]:
        """
        `voters_only=True` esclude gli account istituzionali: partecipano al
        dibattito ma non hanno una scheda elettorale.
        """
        conds = []
        if not include_sources:
            conds.append("is_source = 0")
        if voters_only:
            conds.append("is_voter = 1")
        q = "SELECT * FROM agent"
        if conds:
            q += " WHERE " + " AND ".join(conds)
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

    _CAND_SELECT = """
        SELECT p.post_id, p.content, p.kind, p.tick, p.sim_date, p.news_id,
               a.username, a.agent_id AS author_id,
               (SELECT COUNT(*) FROM reaction r
                 WHERE r.post_id = p.post_id AND r.kind = 'like') AS likes,
               (SELECT COUNT(*) FROM post c
                 WHERE c.parent_id = p.post_id)                   AS replies
        FROM post p JOIN agent a ON a.agent_id = p.agent_id
        WHERE p.tick <= ? AND p.agent_id != ? AND a.is_source = 0 AND {clause}
        ORDER BY p.tick DESC, p.post_id DESC LIMIT ?
    """

    def candidates_in_network(self, agent_id: int, tick: int, limit: int):
        return self.conn.execute(
            self._CAND_SELECT.format(
                clause="p.agent_id IN (SELECT followee_id FROM follow "
                       "WHERE follower_id = ?)"),
            (tick, agent_id, agent_id, limit),
        ).fetchall()

    def candidates_out_network(self, agent_id: int, tick: int, limit: int):
        return self.conn.execute(
            self._CAND_SELECT.format(
                clause="p.agent_id NOT IN (SELECT followee_id FROM follow "
                       "WHERE follower_id = ?)"),
            (tick, agent_id, agent_id, limit),
        ).fetchall()

    def recent_news(self, tick: int, limit: int):
        """
        Notizie recenti, fuori dal grafo di follow: portata editoriale.

        I like sono quelli veri. Prima erano `0 AS likes` fissi, quindi nel
        feed una notizia appariva sempre senza riscontro anche quando mezza
        popolazione l'aveva rilanciata: gli agenti non potevano vedere che
        una notizia stava girando, che e' meta' di come funziona un social.
        """
        return self.conn.execute(
            """
            SELECT p.post_id, p.content, p.kind, p.tick, p.sim_date, p.news_id,
                   a.username, a.agent_id AS author_id,
                   (SELECT COUNT(*) FROM reaction r
                     WHERE r.post_id = p.post_id AND r.kind = 'like') AS likes,
                   (SELECT COUNT(*) FROM post c
                     WHERE c.parent_id = p.post_id)                   AS replies
            FROM post p JOIN agent a ON a.agent_id = p.agent_id
            WHERE p.tick <= ? AND a.is_source = 1
            ORDER BY p.tick DESC, p.post_id DESC LIMIT ?
            """, (tick, limit)).fetchall()

    def follow_candidates(self, tick: int, prob_per_like: float,
                          max_new_per_agent: int, seed: int) -> list[tuple[int, int]]:
        """
        Chi inizia a seguire chi, sulla base dei LIKE ricevuti.

        Solo i like, non le risposte. Una risposta si scrive anche — anzi
        soprattutto — a chi non si condivide: contarla come segnale di
        avvicinamento premierebbe il litigio quanto il consenso, e in una
        simulazione sulla polarizzazione e' proprio l'errore da non fare.
        Il like invece e' approvazione quasi per definizione.

        Forma probabilistica invece che a soglia: ogni like verso la stessa
        persona e' un'occasione indipendente di seguirla, con probabilita'
        `prob_per_like`. Dopo k like la probabilita' cumulata e'
        1 - (1-p)^k, quindi cresce e satura invece di scattare di colpo a una
        soglia arbitraria. Nessuno segue per forza al primo like, e nessuno
        resta immune dopo il decimo.

        L'estrazione e' deterministica su (seme, follower, seguito, tick):
        il run resta riproducibile a parita' di seme.
        """
        rows = self.conn.execute(
            """
            SELECT r.agent_id AS follower, p.agent_id AS followee,
                   COUNT(*) AS like_dati
              FROM reaction r JOIN post p ON p.post_id = r.post_id
             WHERE r.kind = 'like' AND r.tick <= :tick
               AND r.agent_id != p.agent_id
             GROUP BY r.agent_id, p.agent_id
             ORDER BY r.agent_id, like_dati DESC
            """, {"tick": tick}).fetchall()

        gia = {(r["follower_id"], r["followee_id"]) for r in
               self.conn.execute("SELECT follower_id, followee_id FROM follow")}
        fonti = {r["agent_id"] for r in self.conn.execute(
            "SELECT agent_id FROM agent WHERE is_source = 1")}

        nuovi: list[tuple[int, int]] = []
        per_agente: dict[int, int] = {}
        for r in rows:
            a, b = int(r["follower"]), int(r["followee"])
            if a in fonti or (a, b) in gia:
                continue
            if per_agente.get(a, 0) >= max_new_per_agent:
                continue
            k = int(r["like_dati"])
            p_cum = 1.0 - (1.0 - prob_per_like) ** k
            rng = random.Random(f"{seed}|{a}|{b}|{tick}|follow")
            if rng.random() >= p_cum:
                continue
            per_agente[a] = per_agente.get(a, 0) + 1
            nuovi.append((a, b))
        return nuovi

    def affinity_for(self, agent_id: int, tick: int) -> dict[int, float]:
        """
        Quanto ogni altro agente e' relazionalmente vicino a questo, dalle
        interazioni avvenute fino a `tick`. -> {author_id: peso}

        Sostituisce la similarita' semantica del vecchio ramo embedding. Una
        risposta pesa il doppio di un like perche' costa di piu' e segnala
        piu' attenzione; le direzioni contano entrambe, perche' la
        reciprocita' e' proprio cio' che distingue una relazione da un
        semplice arco di follow.

        Deliberatamente NON pesa la concordanza di opinione: se lo facesse,
        la camera d'eco sarebbe imposta dalla metrica invece che emergere
        dall'interazione, e il risultato in tesi sarebbe circolare.
        """
        rows = self.conn.execute(
            """
            SELECT other, SUM(w) AS score FROM (
                -- mi ha messo like
                SELECT r.agent_id AS other, 1.0 AS w
                  FROM reaction r JOIN post p ON p.post_id = r.post_id
                 WHERE p.agent_id = :me AND r.tick <= :tick
                UNION ALL
                -- mi ha risposto
                SELECT c.agent_id AS other, 2.0 AS w
                  FROM post c JOIN post p ON p.post_id = c.parent_id
                 WHERE p.agent_id = :me AND c.tick <= :tick
                UNION ALL
                -- gli ho messo like
                SELECT p.agent_id AS other, 1.0 AS w
                  FROM reaction r JOIN post p ON p.post_id = r.post_id
                 WHERE r.agent_id = :me AND r.tick <= :tick
                UNION ALL
                -- gli ho risposto
                SELECT p.agent_id AS other, 2.0 AS w
                  FROM post c JOIN post p ON p.post_id = c.parent_id
                 WHERE c.agent_id = :me AND c.tick <= :tick
            )
            WHERE other != :me
            GROUP BY other
            """,
            {"me": agent_id, "tick": tick},
        ).fetchall()
        return {int(r["other"]): float(r["score"]) for r in rows}

    def parents_of(self, post_ids: list[int]) -> dict[int, sqlite3.Row]:
        """
        Post padre delle reply presenti nel feed.

        Serve perche' una risposta senza il messaggio a cui risponde e' un non
        sequitur: "Carlo, concordo sulla cautela" letto da solo non dice ne'
        chi sia Carlo ne' su cosa si concordi. Finora il feed le consegnava
        cosi', indistinguibili dai post originali.
        """
        if not post_ids:
            return {}
        marks = ",".join("?" * len(post_ids))
        rows = self.conn.execute(
            f"""SELECT c.post_id AS child_id, p.post_id, p.content, p.kind,
                       a.username, a.is_source
                FROM post c JOIN post p ON p.post_id = c.parent_id
                JOIN agent a ON a.agent_id = p.agent_id
                WHERE c.post_id IN ({marks})""",
            post_ids,
        ).fetchall()
        return {r["child_id"]: r for r in rows}

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
