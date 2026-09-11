#!/usr/bin/env python3
"""
Runner di MirrorFish.

    # smoke test offline, nessuna GPU richiesta
    python run.py --stub --agents 40 --days 5 --out runs/smoke

    # run vero
    export LLM_API_KEY=...
    python run.py --profiles start/A/reddit_profiles.json \\
                  --news start/notizieansa/notizie_referendum \\
                  --out runs/base_s42 --seed 42

    # controfattuale: dal tick 200 in poi, notizie alternative
    python run.py --profiles ... --news ... --cf-news /path/news_alt \\
                  --cf-from-tick 200 --out runs/cf_A --seed 42

    # sweep sulla politica del feed (il trattamento principale)
    for pol in random recency engagement affinity hybrid; do
      python run.py --profiles ... --news ... --recommender $pol \\
                    --out runs/pol_$pol --seed 42
    done

Tutto cio' che influenza il risultato finisce in SimConfig, il cui hash e'
scritto nel run.db come `fingerprint`. Se aggiungi un parametro che cambia il
risultato, mettilo li' dentro e non qui.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import atexit
import json
import os
import pathlib
from datetime import date, timedelta
from pathlib import Path

from mirrorfish.config import LLMConfig, SimConfig
from mirrorfish.engine import Engine
from mirrorfish.llm import EndpointDown, build_client
from mirrorfish.news import NewsStream, NewsItem, load_news
from mirrorfish.population import (
    build_follow_graph, force_media_depth, load_mirofish_profiles,
    source_agent, synthetic,
)
from mirrorfish.recommender import POLICIES, Recommender, explain_policies
from mirrorfish.store import Store
from mirrorfish.survey import crosstab, run_survey, shift_report


def synth_news(start: date, days: int, per_day: int = 3) -> list[NewsItem]:
    """Notizie finte per lo smoke test. NON usare per risultati."""
    items = []
    for d in range(days):
        day = start + timedelta(days=d)
        for i in range(per_day):
            items.append(NewsItem(
                news_id=f"stub-{day.isoformat()}-{i}",
                published=day,
                title=f"Notizia simulata del {day.isoformat()} n.{i}",
                body=("Testo integrale simulato sul dibattito relativo alla "
                      "riforma della giustizia e alla separazione delle "
                      "carriere. " * 6),
            ))
    return items


def acquire_lock(out_dir: Path) -> Path:
    """
    Impedisce due run sulla stessa cartella.

    E' successo davvero: due processi sulla stessa `--out`, il secondo con
    --force che cancella il run.db mentre il primo lo sta scrivendo. Il
    risultato non e' un errore chiaro ma due processi che sembrano bloccati,
    piu' le due quote sommate contro lo stesso limite di squadra.
    """
    lock_path = out_dir / "run.lock"
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().split()[0])
        except (ValueError, IndexError):
            pid = -1
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)      # segnale 0: verifica soltanto
                alive = True
            except (ProcessLookupError, PermissionError):
                alive = False
        if alive:
            raise SystemExit(
                f"ERRORE: un altro run sta gia' usando {out_dir} (pid {pid}).\n"
                f"Due processi sulla stessa cartella si cancellano i dati a "
                f"vicenda e sommano il consumo sulla stessa quota.\n"
                f"Usa un --out diverso, oppure ferma quel processo "
                f"(kill {pid}) e rilancia."
            )
        print(f"[setup] lock orfano di un processo terminato (pid {pid}): "
              f"lo rimuovo.")
        lock_path.unlink(missing_ok=True)
    from datetime import datetime
    lock_path.write_text(f"{os.getpid()} {datetime.now().isoformat()}\n")
    atexit.register(lambda: lock_path.unlink(missing_ok=True))
    return lock_path


def code_version() -> dict[str, str]:
    """
    Commit git e stato dell'albero, salvati nel run.db.

    Il fingerprint copre la CONFIG, non il CODICE. Ma alcune modifiche al
    codice cambiano il comportamento senza toccare un solo parametro: la
    rotazione delle notizie introdotta l'8 settembre 2026 fa vedere agli
    agenti notizie diverse a parita' di config, e la riscrittura del testo del
    quesito cambia il voto a parita' di `vote_question`. Due run con lo stesso
    fingerprint e commit diversi NON sono repliche, e senza questo campo non
    c'e' modo di accorgersene mesi dopo, quando si scrive la tesi.
    """
    import subprocess
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True,
                                  timeout=5).stdout.strip()
        except Exception:
            return ""
    commit = git("rev-parse", "--short", "HEAD") or "sconosciuto"
    sporco = bool(git("status", "--porcelain"))
    return {"commit": commit, "modifiche_non_committate": sporco}


def build_config(args: argparse.Namespace) -> SimConfig:
    start = date.fromisoformat(args.start)
    return SimConfig(
        run_id=Path(args.out).name,
        seed=args.seed,
        start_date=start,
        end_date=start + timedelta(days=args.days - 1),
        hours_per_tick=args.hours_per_tick,
        feed_size=args.feed_size,
        news_slots=args.news_slots,
        news_slots_mode=args.news_slots_mode,
        max_news_per_tick=args.max_news_per_tick,
        recommender=args.recommender,
        out_of_network=args.out_of_network,
        max_actions=args.max_actions,
        reflection_every=args.reflection_every,
        follow_drift_every=args.follow_drift_every,
        survey_every=args.survey_every,
        counterfactual_from_tick=args.cf_from_tick,
        counterfactual_news_dir=args.cf_news,
        force_media_depth=args.force_media_depth,
        vote_question=args.vote_question,
    )


async def main_async(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = build_config(args)
    llm_cfg = LLMConfig.from_env(concurrency=args.concurrency,
                                 requests_per_minute=args.rpm)
    if sim.max_actions > 1:
        # Piu' azioni = output piu' lungo, ma sempre UNA richiesta. Il
        # supplemento e' largo: il gateway addebita i token usati, non il
        # budget richiesto (misurato con endpoint.py probe), quindi un tetto
        # alto non costa quota e previene i troncamenti.
        llm_cfg.token_budget["action"] = 640 + 192 * (sim.max_actions - 1)
    if args.max_action_tokens:
        llm_cfg.token_budget["action"] = args.max_action_tokens

    acquire_lock(out_dir)

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

    # Su --resume la configurazione DEVE coincidere. Il fingerprint esiste
    # apposta e finora non era collegato a questo controllo, con una
    # conseguenza silenziosa: `--resume` salta `store.add_agents`, quindi un
    # `--force-media-depth` passato in ripresa modificava la lista in memoria
    # e veniva buttato via. Il motore leggeva gli agenti dal DB, con le
    # profondita' vecchie, e il run sembrava l'esperimento richiesto senza
    # esserlo. Vale per qualunque parametro, non solo per quello.
    if resuming:
        atteso = store.get_meta("fingerprint")
        if atteso and atteso != sim.fingerprint():
            vecchia = store.get_meta("sim_config", {}) or {}
            nuova = sim.to_dict()
            diff = [k for k in set(vecchia) | set(nuova)
                    if vecchia.get(k) != nuova.get(k)]
            righe = "\n".join(
                f"    {k}: era {vecchia.get(k)!r}, ora {nuova.get(k)!r}"
                for k in sorted(diff))
            store.close()
            raise SystemExit(
                f"ERRORE: --resume con una configurazione diversa.\n"
                f"  fingerprint nel DB : {atteso}\n"
                f"  fingerprint attuale: {sim.fingerprint()}\n"
                f"Parametri cambiati:\n{righe}\n\n"
                f"Riprendere un run cambiando i parametri produce un ibrido: "
                f"i primi tick con una configurazione e i successivi con "
                f"un'altra, sotto un unico fingerprint. Il risultato non e' "
                f"riconducibile a nessuna delle due.\n"
                f"Usa un --out nuovo, oppure --force per rifare da capo."
            )

    store.set_meta("sim_config", sim.to_dict())
    store.set_meta("fingerprint", sim.fingerprint())
    store.set_meta("code_version", code_version())
    store.set_meta("llm", {"model": llm_cfg.model, "budget": llm_cfg.token_budget,
                           "concurrency": llm_cfg.concurrency,
                           "stub": bool(args.stub)})

    # --- popolazione ------------------------------------------------------- #
    if args.profiles:
        agents = load_mirofish_profiles(args.profiles)
        if args.agents:
            agents = agents[: args.agents]
    else:
        agents = synthetic(args.agents or 40, seed=sim.seed)

    if sim.force_media_depth:
        n = force_media_depth(agents, sim.force_media_depth)
        print(f"[setup] profondita' di lettura FORZATA a "
              f"'{sim.force_media_depth}' per {n} agenti: "
              f"esposizione manipolata, non dedotta dalla biografia")
        if not args.news_slots_pinned:
            print(f"[setup] ATTENZIONE: --news-slots non e' stato fissato. "
                  f"Con profondita' '{sim.force_media_depth}' gli slot "
                  f"notizia sono {sim.news_slots_for(sim.force_media_depth)}, "
                  f"ma un altro braccio ne avrebbe un numero diverso.")
            print("[setup] I due bracci differirebbero per DUE variabili: "
                  "quanto testo si legge E quante notizie si ricevono "
                  "(e quindi quanti post di altri agenti). Per una "
                  "manipolazione a variabile singola passa lo stesso "
                  "--news-slots a entrambi, per esempio --news-slots 1.")

    src_id = max((a["agent_id"] for a in agents), default=0) + 1
    if not any(a.get("is_source") for a in agents):
        agents.append(source_agent(src_id))
    else:
        src_id = next(a["agent_id"] for a in agents if a.get("is_source"))

    if not resuming:
        store.add_agents(agents)
        edges = build_follow_graph(agents, seed=sim.seed,
                                   avg_degree=args.avg_degree,
                                   homophily=args.homophily)
        store.add_follows(edges)
        n_vot = len(store.agents(voters_only=True))
        print(f"[setup] {len(agents)} agenti ({n_vot} elettori), "
              f"{len(edges)} archi, fonte=agent_id {src_id}")
    else:
        done = store.get_meta("last_completed_tick", -1)
        wiped = store.truncate_after_tick(done)
        print(f"[setup] RIPRESA: {len(store.agents())} agenti in DB, "
              f"ultimo tick completo = {done + 1}")
        if wiped:
            print(f"[setup] ripulito il tick parziale: {wiped}")

    # La distribuzione delle profondita' si stampa SEMPRE, letta dal DB, che
    # e' cio' che il motore usera' davvero. Prima si stampava solo su un run
    # nuovo, quindi in ripresa non c'era modo di accorgersi che l'esposizione
    # non era quella richiesta.
    depths: dict[str, int] = {}
    for a in store.agents():
        depths[a["media_depth"]] = depths.get(a["media_depth"], 0) + 1
    print("[setup] profondita' di lettura NEL DB: " + ", ".join(
        f"{k}={v} ({sim.news_slots_for(k)} slot)"
        for k, v in sorted(depths.items())))

    # --- notizie ----------------------------------------------------------- #
    # `--news` e' la cartella degli ARTICOLI INTEGRALI: e' quella che fa da
    # indice. I sommari sono un arricchimento opzionale.
    items = (load_news(args.news) if args.news
             else synth_news(sim.start_date, args.days))
    stream = NewsStream(items, sim.start_date, sim.hours_per_tick,
                        sim.total_ticks(), max_per_tick=sim.max_news_per_tick)

    if sim.counterfactual_from_tick is not None:
        alt = (load_news(sim.counterfactual_news_dir)
               if sim.counterfactual_news_dir
               else synth_news(sim.start_date, args.days))
        stream = stream.fork_at(sim.counterfactual_from_tick, alt)
        print(f"[setup] CONTROFATTUALE attivo dal tick "
              f"{sim.counterfactual_from_tick}")

    # --- client ------------------------------------------------------------ #
    client = build_client(llm_cfg, stub=args.stub, seed=sim.seed)

    try:
        # baseline PRIMA: solo bio statica, nessuna nota esiste ancora.
        # Se e' gia' completata non si rifa': sono N chiamate LLM, e con un
        # rate limit stretto sprecarle a ogni ripresa e' proibitivo.
        done_baseline = store.vote_tally("baseline")
        n_ok = sum(v for k, v in done_baseline.items() if k != "ERROR")
        if n_ok >= len(store.agents(voters_only=True)):
            print(f"[survey] baseline gia' completa ({n_ok} voti), salto.")
        else:
            if n_ok:
                store.conn.execute("DELETE FROM vote WHERE label = 'baseline'")
                store.commit()
            await run_survey(store, client, llm_cfg, label="baseline",
                             baseline=True, question=sim.vote_question)

        rec = Recommender(store, policy=sim.recommender,
                          out_of_network=sim.out_of_network, seed=sim.seed)
        print(f"[feed] politica: {sim.recommender}, "
              f"fuori-rete {sim.out_of_network:.0%}, "
              f"{sim.feed_size} post per tick")
        engine = Engine(store, client, sim, llm_cfg, stream, src_id,
                        verbose=not args.quiet, recommender=rec)
        start_tick = (store.get_meta("last_completed_tick", -1) + 1) if resuming else 0
        await engine.run(start_tick=start_tick)

        await run_survey(store, client, llm_cfg, label="final",
                         baseline=False, question=sim.vote_question)
    except EndpointDown as e:
        done = store.get_meta("last_completed_tick", -1) + 1
        store.close()
        raise SystemExit(
            f"\nENDPOINT NON RAGGIUNGIBILE\n{e}\n\n"
            f"Tick completati e salvati: {done}. Nulla e' andato perso."
        )
    finally:
        await client.aclose()

    # --- report ------------------------------------------------------------ #
    report = shift_report(store)
    print("\n--- SPOSTAMENTI ---")
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
    store.set_meta("shift_report",
                   {k: v for k, v in report.items() if k != "detail"})
    store.close()
    print(f"\n[done] risultati in {out_dir}/  "
          f"(run.db, shift_report.json, graph.json)")
    cv = code_version()
    print(f"[done] fingerprint config: {sim.fingerprint()}  "
          f"codice: {cv['commit']}"
          f"{' (albero sporco)' if cv['modifiche_non_committate'] else ''}")


def parse_args() -> argparse.Namespace:
    d = SimConfig()          # i default vivono in config.py, non qui
    p = argparse.ArgumentParser(
        description="MirrorFish — simulazione referendaria",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default="runs/dev")
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--start", default=d.start_date.isoformat())
    p.add_argument("--days", type=int,
                   default=(d.end_date - d.start_date).days + 1)
    p.add_argument("--hours-per-tick", type=int, default=d.hours_per_tick,
                   help="durata del tick in ore: leva principale sul costo")

    p.add_argument("--profiles", default=None, help="reddit_profiles.json")
    p.add_argument("--agents", type=int, default=None,
                   help="tronca la popolazione ai primi N profili")
    p.add_argument("--avg-degree", type=int, default=12)
    p.add_argument("--homophily", type=float, default=0.6)

    p.add_argument("--news", default=None,
                   help="cartella degli ARTICOLI INTEGRALI (notizie_referendum). "
                        "E' l'indice: definisce quali notizie esistono")
    p.add_argument("--cf-news", default=None,
                   help="cartella notizie controfattuali (integrali)")
    p.add_argument("--cf-from-tick", type=int, default=None)
    p.add_argument("--max-news-per-tick", type=int, default=d.max_news_per_tick)
    p.add_argument("--vote-question", default=d.vote_question,
                   choices=["minimal", "ballot", "informed"],
                   help="'minimal' nomina solo il tema e con un modello che "
                        "non conosce la riforma produce il suo prior; "
                        "'ballot' riproduce la scheda reale, che non riassume "
                        "nulla; 'informed' aggiunge il contenuto della legge "
                        "in termini neutri. Confrontare i tre baseline e' una "
                        "analisi di sensibilita' alla formulazione")
    p.add_argument("--force-media-depth", default=None,
                   choices=["integrale", "titolo", "nessuna"],
                   help="impone la stessa profondita' a tutti: serve a "
                        "separare l'effetto dell'esposizione da quello della "
                        "personalita'. Due run che differiscono SOLO per "
                        "questo sono un esperimento, non un'osservazione")
    p.add_argument("--news-slots-mode", default=d.news_slots_mode,
                   choices=["gradiente", "solo_profondita", "uniforme"],
                   help="'gradiente': l'attivista riceve 2 notizie, il "
                        "moderato 1, il disinteressato 0 (quantita' e "
                        "profondita' variano insieme). 'solo_profondita': "
                        "stessa quantita' per chi le riceve, cambia solo "
                        "quanto testo ne legge. 'uniforme': una notizia a "
                        "tutti, disinteressati compresi")
    p.add_argument("--news-slots", type=int, default=d.news_slots,
                   help="TETTO agli slot notizia. Quelli effettivi dipendono "
                        "dalla profondita' di lettura dell'agente")

    p.add_argument("--feed-size", type=int, default=d.feed_size)
    p.add_argument("--recommender", default=d.recommender,
                   choices=sorted(POLICIES),
                   help="politica del feed: e' il trattamento sperimentale")
    p.add_argument("--out-of-network", type=float, default=d.out_of_network,
                   help="quota di post da fuori la rete dei seguiti")
    p.add_argument("--explain-policies", action="store_true",
                   help="stampa la tabella delle politiche ed esce")

    p.add_argument("--max-actions", type=int, default=d.max_actions,
                   help="azioni per agente per tick, in una sola chiamata")
    p.add_argument("--reflection-every", type=int, default=d.reflection_every)
    p.add_argument("--follow-drift-every", type=int, default=d.follow_drift_every,
                   help="ogni N tick chi interagisce ripetutamente con "
                        "qualcuno inizia a seguirlo. Senza, il grafo resta "
                        "congelato e nessuno puo' accumulare pubblico: la "
                        "figura dell'influencer e' esclusa per costruzione")
    p.add_argument("--survey-every", type=int, default=d.survey_every,
                   help="survey intermedia ogni N tick, per la traiettoria "
                        "dell'opinione. Costa N chiamate ogni volta")

    p.add_argument("--concurrency", type=int, default=4,
                   help="chiamate in volo. Oltre quelle che "
                        "servono a saturare --rpm non accelera "
                        "nulla e toglie slot agli altri")
    p.add_argument("--rpm", type=float, default=30.0,
                   help="richieste/minuto. Il tetto di squadra e' 40 ed e' "
                        "CONDIVISO: a 40 prendi tutta la capacita' e le "
                        "chiamate dei colleghi cadono per timeout, perche' "
                        "LiteLLM non ha code a priorita'")
    p.add_argument("--max-action-tokens", type=int, default=None)
    p.add_argument("--stub", action="store_true", help="LLM finto, offline")

    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="riprende un run interrotto dall'ultimo tick completato")
    p.add_argument("--force", action="store_true",
                   help="cancella un run.db preesistente invece di rifiutarsi")
    ns = p.parse_args()
    # Serve a sapere se --news-slots e' stato scritto davvero o e' il default:
    # con force-media-depth la differenza fra le due cose e' sperimentale.
    ns.news_slots_pinned = any(
        x.startswith("--news-slots") for x in __import__("sys").argv)
    return ns


if __name__ == "__main__":
    a = parse_args()
    if a.explain_policies:
        print(explain_policies())
        raise SystemExit(0)
    asyncio.run(main_async(a))
