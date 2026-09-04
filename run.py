#!/usr/bin/env python3
"""
Runner di MirrorFish.

    # smoke test offline, nessuna GPU richiesta
    python run.py --stub --agents 40 --days 5 --out runs/smoke

    # run vero (quando l'endpoint torna su)
    export LLM_API_KEY=...
    python run.py --profiles /path/reddit_profiles.json \
                  --news /path/news_ansa --out runs/base_seed42 --seed 42

    # controfattuale: dal tick 40 in poi, notizie alternative
    python run.py --profiles ... --news ... --cf-news /path/news_alt \
                  --cf-from-tick 40 --out runs/cf_A --seed 42
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import pathlib
from datetime import date
from pathlib import Path

from mirrorfish.config import LLMConfig, SimConfig
from mirrorfish.engine import Engine
from mirrorfish.llm import build_client
from mirrorfish.news import NewsStream, NewsItem, load_news
from mirrorfish.population import (
    build_follow_graph, load_mirofish_profiles, source_agent, synthetic,
)
from mirrorfish.recommender import POLICIES, Recommender, explain_policies
from mirrorfish.store import Store
from mirrorfish.survey import crosstab, run_survey, shift_report


def synth_news(start: date, days: int, per_day: int = 3) -> list[NewsItem]:
    """Notizie finte per lo smoke test."""
    from datetime import timedelta
    items = []
    for d in range(days):
        day = start + timedelta(days=d)
        for i in range(per_day):
            items.append(NewsItem(
                news_id=f"stub-{day.isoformat()}-{i}",
                published=day,
                content=(f"Notizia simulata del {day.isoformat()} n.{i}: "
                         f"dibattito sulla riforma della giustizia."),
            ))
    return items


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(args.start)
    from datetime import timedelta
    end = start + timedelta(days=args.days - 1)

    sim = SimConfig(
        run_id=out_dir.name, seed=args.seed,
        start_date=start, end_date=end,
        hours_per_tick=args.hours_per_tick,
        feed_size=args.feed_size, news_slots=args.news_slots,
        max_actions=args.max_actions,
        max_news_per_tick=args.max_news_per_tick,
        reflection_every=args.reflection_every,
        counterfactual_from_tick=args.cf_from_tick,
        counterfactual_news_dir=args.cf_news,
    )
    llm_cfg = LLMConfig.from_env(concurrency=args.concurrency,
                                 requests_per_minute=args.rpm)
    if args.max_actions > 1:
        # Piu' azioni = output piu' lungo, ma sempre UNA richiesta.
        llm_cfg.token_budget["action"] = 384 + 128 * (args.max_actions - 1)
    if args.max_action_tokens:
        llm_cfg.token_budget["action"] = args.max_action_tokens

    db_path = out_dir / "run.db"
    resuming = False
    if db_path.exists():
        if args.resume:
            resuming = True
        elif not args.force:
            raise SystemExit(
                f"ERRORE: {db_path} esiste gia'.\n"
                "Rilanciare sullo stesso file NON sovrascrive: accumula voti e "
                "post, e il conteggio finale esce sbagliato senza errori.\n"
                "Usa --out con un nome diverso (un run = una cartella), "
                "oppure --force per cancellare e rifare."
            )
        else:
            for suffix in ("", "-wal", "-shm"):
                pathlib.Path(str(db_path) + suffix).unlink(missing_ok=True)

    store = Store(db_path)
    store.set_meta("sim_config", sim.to_dict())
    store.set_meta("fingerprint", sim.fingerprint())
    store.set_meta("actions", {"max_actions": args.max_actions})
    store.set_meta("feed", {"policy": args.recommender,
                           "out_of_network": args.out_of_network})
    store.set_meta("llm", {"model": llm_cfg.model, "budget": llm_cfg.token_budget,
                           "concurrency": llm_cfg.concurrency,
                           "stub": bool(args.stub)})

    # --- popolazione ------------------------------------------------------- #
    if args.profiles:
        agents = load_mirofish_profiles(args.profiles)
        if args.agents:
            agents = agents[: args.agents]
    else:
        agents = synthetic(args.agents or 40, seed=args.seed)

    src_id = max((a["agent_id"] for a in agents), default=0) + 1
    if not any(a.get("is_source") for a in agents):
        agents.append(source_agent(src_id))
    else:
        src_id = next(a["agent_id"] for a in agents if a.get("is_source"))

    if not resuming:
        store.add_agents(agents)
        edges = build_follow_graph(agents, seed=args.seed,
                                   avg_degree=args.avg_degree,
                                   homophily=args.homophily)
        store.add_follows(edges)
        print(f"[setup] {len(agents)} agenti, {len(edges)} archi, "
              f"fonte=agent_id {src_id}")
    else:
        done = store.get_meta("last_completed_tick", -1)
        wiped = store.truncate_after_tick(done)
        print(f"[setup] RIPRESA: {len(store.agents())} agenti in DB, "
              f"ultimo tick completo = {done + 1}")
        if wiped:
            print(f"[setup] ripulito il tick parziale: {wiped}")

    # --- notizie ----------------------------------------------------------- #
    total_ticks = sim.total_ticks()
    items = (load_news(args.news, args.news_full) if args.news
             else synth_news(start, args.days))
    stream = NewsStream(items, start, sim.hours_per_tick, total_ticks,
                        max_per_tick=args.max_news_per_tick)

    if args.cf_from_tick is not None:
        alt = (load_news(args.cf_news) if args.cf_news
               else synth_news(start, args.days))
        stream = stream.fork_at(args.cf_from_tick, alt)
        print(f"[setup] CONTROFATTUALE attivo dal tick {args.cf_from_tick}")

    # --- client ------------------------------------------------------------ #
    client = build_client(llm_cfg, stub=args.stub, seed=args.seed)

    try:
        # baseline PRIMA: solo bio statica, nessuna nota esiste ancora.
        # Se e' gia' stata completata non si rifa': sono N chiamate LLM, e con
        # un rate limit stretto sprecarle a ogni ripresa e' proibitivo.
        done_baseline = store.vote_tally("baseline")
        n_ok = sum(v for k, v in done_baseline.items() if k != "ERROR")
        if n_ok >= len(store.agents(voters_only=True)):
            print(f"[survey] baseline gia' completa ({n_ok} voti), salto.")
        else:
            if n_ok:
                store.conn.execute("DELETE FROM vote WHERE label = 'baseline'")
                store.commit()
            await run_survey(store, client, llm_cfg, label="baseline", baseline=True)

        rec = Recommender(store, policy=args.recommender,
                          out_of_network=args.out_of_network, seed=args.seed)
        if rec.needs_embeddings:
            raise SystemExit(
                f"La politica '{args.recommender}' richiede gli embedding, "
                f"che non sono ancora collegati al motore. Usa random, "
                f"recency o engagement.")
        print(f"[feed] politica: {args.recommender}, "
              f"fuori-rete {args.out_of_network:.0%}")
        engine = Engine(store, client, sim, llm_cfg, stream, src_id,
                        verbose=not args.quiet, recommender=rec)
        start_tick = (store.get_meta("last_completed_tick", -1) + 1) if resuming else 0
        stats = await engine.run(start_tick=start_tick)

        await run_survey(store, client, llm_cfg, label="final", baseline=False)
    finally:
        await client.aclose()

    # --- report ------------------------------------------------------------ #
    report = shift_report(store)
    print(f"\n--- SPOSTAMENTI ---")
    print(f"  cambiato idea : {report['shifted']}/{report['total']} "
          f"({report['shift_rate'] * 100:.1f}%)")
    for k, v in report["transitions"].items():
        print(f"    {k:<18} {v}")

    print("\n--- VOTO FINALE PER FASCIA D'ETA' ---")
    for r in crosstab(store, "final", "age_band"):
        print(f"  {r['bucket'] or '?':<8} {r['vote']:<9} {r['n']}")

    print("\n--- TELEMETRIA LLM ---")
    print(f"  {'scopo':<12}{'chiamate':>9}{'out medio':>11}{'out max':>9}"
          f"{'budget':>8}{'troncate':>10}{'errori':>8}")
    total_out = 0
    for r in store.call_stats():
        total_out += r["completion_tok"] or 0
        print(f"  {r['purpose']:<12}{r['calls']:>9}{r['avg_out']:>11}"
              f"{r['max_out']:>9}{r['budget']:>8}{r['truncated']:>10}"
              f"{r['errors']:>8}")
    print(f"  token di output totali: {total_out:,}")
    gate = getattr(client, "gate", None)
    if gate is not None and gate.trips:
        g = gate.stats()
        print(f"  rate limit: {g['trips']} interventi, "
              f"{g['paused_s']}s di pausa totale, "
              f"ritmo finale {g['final_interval_s']}s/richiesta")

    (out_dir / "shift_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "graph.json").write_text(
        json.dumps(store.export_graph(), ensure_ascii=False), encoding="utf-8")
    store.set_meta("shift_report", {k: v for k, v in report.items() if k != "detail"})
    store.close()
    print(f"\n[done] risultati in {out_dir}/  (run.db, shift_report.json, graph.json)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MirrorFish — simulazione referendaria")
    p.add_argument("--out", default="runs/dev")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", default="2026-03-01")
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--hours-per-tick", type=int, default=8,
                   help="durata del tick in ore: leva principale sul costo")
    p.add_argument("--agents", type=int, default=None)
    p.add_argument("--profiles", default=None, help="reddit_profiles.json")
    p.add_argument("--news", default=None, help="cartella .txt ANSA (sommari)")
    p.add_argument("--news-full", default=None,
                   help="cartella dei testi integrali (notizie_referendum). "
                        "Stessi nomi file dei sommari. Senza, tutti leggono "
                        "il sommario e l'esposizione mediatica e' uniforme.")
    p.add_argument("--cf-news", default=None, help="cartella notizie controfattuali")
    p.add_argument("--cf-from-tick", type=int, default=None)
    p.add_argument("--feed-size", type=int, default=8)
    p.add_argument("--reflection-every", type=int, default=4)
    p.add_argument("--avg-degree", type=int, default=12)
    p.add_argument("--homophily", type=float, default=0.6)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--rpm", type=float, default=8.0,
                   help="richieste/minuto concesse dalla quota")
    p.add_argument("--max-action-tokens", type=int, default=None)
    p.add_argument("--stub", action="store_true", help="LLM finto, offline")
    p.add_argument("--max-actions", type=int, default=1,
                   help="azioni per agente per tick, in una sola chiamata. "
                        "1 = comportamento dei run precedenti; 3 = sessione "
                        "realistica (qualche like + al piu' un contenuto)")
    p.add_argument("--recommender", default="recency", choices=sorted(POLICIES),
                   help="politica del feed. 'recency' (default) riproduce "
                        "esattamente il comportamento dei run precedenti; "
                        "'random' e' il controllo")
    p.add_argument("--out-of-network", type=float, default=0.0,
                   help="quota di post da fuori la rete dei seguiti "
                        "(0.0 = comportamento dei run precedenti)")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="riprende un run interrotto dall'ultimo tick completato")
    p.add_argument("--max-news-per-tick", type=int, default=3)
    p.add_argument("--news-slots", type=int, default=2,
                   help="quanti slot del feed sono riservati alle notizie")
    p.add_argument("--force", action="store_true",
                   help="cancella un run.db preesistente invece di rifiutarsi")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
