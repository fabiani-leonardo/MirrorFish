"""
Motore di simulazione: il tick loop.

Determinismo. Ogni sorgente di casualita' passa da un RNG derivato dal seed
del run e dal numero di tick: `Random(f"{seed}|{tick}|{scopo}")`. Due run con
lo stesso seed e la stessa config producono la stessa sequenza di agenti
attivi, gli stessi feed e le stesse notizie.

Attenzione a un limite reale, da dichiarare in tesi: questo rende deterministico
tutto TRANNE l'LLM. Un server vLLM sotto batching dinamico non e' bit-exact
nemmeno a temperature=0, perche' il raggruppamento delle richieste cambia
l'ordine delle riduzioni in virgola mobile. Quindi la riproducibilita' qui e'
"stessa configurazione, stessa struttura del run", non "stesso output token per
token". La conseguenza pratica: ogni condizione sperimentale va eseguita in
piu' repliche e riportata con media e dispersione, non come numero singolo.
"""

from __future__ import annotations

import asyncio
import json
import time
import random
import sqlite3
from datetime import date
from typing import Any

from .agent import decide, render_feed
from .config import LLMConfig, SimConfig
from .llm import LLMClient
from .memory import ReflectionEngine
from .news import NewsStream
from .population import activation_prob
from .recommender import Recommender
from .survey import run_survey
from .store import Store


class Engine:
    def __init__(
        self,
        store: Store,
        client: LLMClient,
        sim: SimConfig,
        llm_cfg: LLMConfig,
        news: NewsStream,
        source_agent_id: int,
        verbose: bool = True,
        recommender: Recommender | None = None,
    ):
        self.store = store
        self.client = client
        self.sim = sim
        self.llm_cfg = llm_cfg
        self.news = news
        self.source_agent_id = source_agent_id
        self.verbose = verbose
        self.reflector = ReflectionEngine(client)
        self.recommender = recommender or Recommender(
            store, policy=sim.recommender,
            out_of_network=sim.out_of_network, seed=sim.seed)
        self.stats: dict[str, int] = {
            "posts": 0, "replies": 0, "likes": 0,
            "ignored": 0, "errors": 0, "notes": 0,
        }
        # Cosa ha letto ogni agente dall'ultima riflessione, nella STESSA
        # resa che ha visto nel prompt (vedi agent.render_feed).
        self._seen: dict[int, list[str]] = {}
        # news_id -> NewsItem, per rendere il testo alla profondita' giusta
        self.news_by_id = {
            it.news_id: it
            for t_ in range(sim.total_ticks()) for it in news.at(t_)}

    # ------------------------------------------------------------------ util #
    def _rng(self, tick: int, purpose: str) -> random.Random:
        return random.Random(f"{self.sim.seed}|{tick}|{purpose}")

    def _active_agents(self, tick: int, agents: list[sqlite3.Row]) -> list[sqlite3.Row]:
        """Chi agisce in questo tick. Deterministico dato (seed, tick)."""
        rng = self._rng(tick, "activity")
        start_h, span = self.sim.tick_hours(tick)
        active = []
        for a in agents:
            try:
                hours = json.loads(a["activity_hours"] or "[]")
            except (TypeError, ValueError):
                hours = []
            scale = a["activity"] or self.sim.base_activity
            p = (activation_prob(hours, start_h, span, scale * 24 / max(1, span))
                 if len(hours) == 24 else scale)
            if rng.random() < p:
                active.append(a)
        rng.shuffle(active)  # ordine stabile ma non per agent_id
        return active

    # ------------------------------------------------------------------ news #
    def _inject_news(self, tick: int, sim_date: date) -> int:
        items = self.news.at(tick)
        for item in items:
            self.store.add_post(
                agent_id=self.source_agent_id,
                content=item.as_post(),
                kind="news",
                tick=tick,
                sim_date=sim_date.isoformat(),
                news_id=item.news_id,
            )
        if items:
            self.store.commit()
        return len(items)

    # ----------------------------------------------------------------- azioni #
    async def _act(self, agent: sqlite3.Row, tick: int, sim_date: date) -> dict[str, Any]:
        agent_id = int(agent["agent_id"])
        # Quante notizie riceve dipende da quanto a fondo legge: chi non segue
        # la politica ('nessuna') non ne riceve affatto e ne sente parlare
        # solo dagli altri. Gli slot liberati vanno ai pari.
        depth = agent["media_depth"] if "media_depth" in agent.keys() else "titolo"
        feed = self.recommender.feed(
            agent_id, tick, limit=self.sim.feed_size,
            news_slots=self.sim.news_slots_for(depth),
        )
        parents = self.store.parents_of(
            [r["post_id"] for r in feed if r["kind"] == "reply"])

        # UNA sola resa del feed, usata sia per il prompt sia per la memoria
        # di cio' che e' stato letto. Erano due, e divergevano sulle notizie.
        feed_lines = render_feed(feed, parents, depth, self.news_by_id,
                                 max_body_chars=self.sim.max_body_chars)

        notes = self.store.notes_for(agent_id, limit=self.sim.max_notes_in_prompt)
        own = self.store.posts_by(agent_id, limit=3)

        actions, resp = await decide(
            self.client, agent, feed_lines, notes, own, sim_date.isoformat(),
            max_tokens=self.llm_cfg.token_budget["action"],
            temperature=self.llm_cfg.temperature,
            max_actions=self.sim.max_actions,
        )
        if feed_lines:
            bucket = self._seen.setdefault(agent_id, [])
            bucket.extend(text for _, text in feed_lines)
            del bucket[:-30]

        return {"agent_id": agent_id, "actions": actions, "resp": resp,
                "tick": tick, "sim_date": sim_date}

    def _apply(self, outcome: dict[str, Any]) -> None:
        """Scritture su DB: serializzate nel thread principale, non nei task."""
        agent_id = outcome["agent_id"]
        tick, sim_date = outcome["tick"], outcome["sim_date"]

        self.store.log_call(
            "action", self.llm_cfg.token_budget["action"],
            outcome["resp"], tick=tick, agent_id=agent_id,
        )

        for action in outcome["actions"]:
            if action.error:
                self.stats["errors"] += 1
                continue
            if action.action == "IGNORE":
                self.stats["ignored"] += 1
            elif action.action == "POST":
                self.store.add_post(agent_id, action.content, "post", tick,
                                    sim_date.isoformat())
                self.stats["posts"] += 1
            elif action.action == "REPLY":
                self.store.add_post(agent_id, action.content, "reply", tick,
                                    sim_date.isoformat(),
                                    parent_id=action.target_post_id)
                self.stats["replies"] += 1
            elif action.action == "LIKE":
                self.store.add_reaction(agent_id, action.target_post_id,
                                        "like", tick)
                self.stats["likes"] += 1

    # ------------------------------------------------------------ riflessione #
    async def _reflect_all(self, agents: list[sqlite3.Row], tick: int) -> None:
        targets = [a for a in agents if self._seen.get(int(a["agent_id"]))]
        if not targets:
            return

        async def one(a: sqlite3.Row):
            aid = int(a["agent_id"])
            return await self.reflector.reflect(
                a, self.store.notes_for(aid), self._seen.get(aid, []),
                max_tokens=self.llm_cfg.token_budget["reflection"],
                temperature=self.llm_cfg.reflection_temperature,
            )

        results = await asyncio.gather(*(one(a) for a in targets))
        for res, resp in results:
            if resp is not None:
                self.store.log_call("reflection",
                                    self.llm_cfg.token_budget["reflection"],
                                    resp, tick=tick, agent_id=res.agent_id)
            if res.note_added:
                self.store.add_note(res.agent_id, tick, res.note, res.reasoning)
                self.stats["notes"] += 1
        self.store.commit()
        self._seen.clear()

    # ---------------------------------------------------------------- il loop #
    async def run(self, start_tick: int = 0) -> dict[str, Any]:
        agents = self.store.agents(include_sources=False)
        total = self.sim.total_ticks()
        if start_tick:
            self.stats.update(self.store.get_meta("stats", {}) or {})
            print(f"[engine] RIPRESA dal tick {start_tick + 1}/{total}")
        if self.verbose:
            print(f"[engine] {len(agents)} agenti, {total} tick, seed={self.sim.seed}")
            print(f"[engine] {self.news.summary()}")

        run_t0 = time.perf_counter()
        for tick in range(start_tick, total):
            sim_date = self.news.tick_date(tick)
            start_h, _ = self.sim.tick_hours(tick)
            n_news = self._inject_news(tick, sim_date)
            active = self._active_agents(tick, agents)

            if active:
                done = 0
                t0 = time.perf_counter()
                last_beat = [0.0]

                async def tracked(a):
                    nonlocal done
                    out = await self._act(a, tick, sim_date)
                    done += 1
                    # Battito a TEMPO, non a conteggio: un tick con 13
                    # agenti attivi non raggiungeva mai la soglia di 15 e
                    # restava muto anche per minuti.
                    el = time.perf_counter() - t0
                    if self.verbose and el - last_beat[0] >= 20.0:
                        last_beat[0] = el
                        print(f"    tick {tick + 1}: {done}/{len(active)} "
                              f"agenti ({done / max(el, .1) * 60:.1f}/min)",
                              flush=True)
                    return out

                outcomes = await asyncio.gather(*(tracked(a) for a in active))
                for o in outcomes:
                    self._apply(o)
                self.store.commit()

            if self.sim.reflection_every and (tick + 1) % self.sim.reflection_every == 0:
                await self._reflect_all(agents, tick)

            # Survey intermedia. E' l'unico modo per avere la TRAIETTORIA
            # dell'opinione: con solo baseline e final si osservano due punti
            # e si e' costretti a chiamare "cambio di idea" qualunque
            # differenza fra i due, senza sapere quando e' avvenuta ne' se e'
            # stabile. Costa N chiamate ogni volta, quindi e' spenta di
            # default e va accesa sapendo cosa si spende.
            if self.sim.survey_every and (tick + 1) % self.sim.survey_every == 0:
                if tick + 1 < total:      # l'ultimo punto e' gia' `final`
                    await run_survey(self.store, self.client, self.llm_cfg,
                                     label=f"tick_{tick}", baseline=False,
                                     tick=tick, verbose=self.verbose)

            # Checkpoint a ogni tick: con un endpoint a rate limit un run
            # lungo viene interrotto spesso, e ricominciare da zero ogni volta
            # rende impossibile finire.
            self.store.set_meta("last_completed_tick", tick)
            self.store.set_meta("stats", self.stats)

            if self.verbose:
                el = time.perf_counter() - run_t0
                per = el / max(1, tick - start_tick + 1)
                eta_h = (total - tick - 1) * per / 3600
                print(f"  tick {tick + 1:3d}/{total} {sim_date} {start_h:02d}h  "
                      f"attivi={len(active):3d} news={n_news:2d}  "
                      f"post={self.stats['posts']} reply={self.stats['replies']} "
                      f"like={self.stats['likes']} note={self.stats['notes']} "
                      f"err={self.stats['errors']}  ETA {eta_h:.1f}h", flush=True)

        self.store.set_meta("stats", self.stats)
        return self.stats
