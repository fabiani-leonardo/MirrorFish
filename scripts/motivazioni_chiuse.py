#!/usr/bin/env python3
"""
motivazioni_chiuse — sottopone agli agenti l'elenco di motivazioni della
rilevazione reale, nella stessa forma a risposta multipla.

Perche'. La rilevazione sugli elettori reali non poneva una domanda aperta:
offriva un elenco fra cui scegliere, con piu' risposte ammesse. Le motivazioni
simulate sono invece testo libero codificato a posteriori. Confrontarle
significa confrontare due strumenti diversi, oltre che due popolazioni. Questo
script elimina quella differenza: stesso elenco, stessa forma.

Non sostituisce la domanda aperta, la affianca. La domanda aperta misura quali
ragioni la popolazione ha PRODOTTO; questa misura quali ragioni RICONOSCE fra
quelle che le vengono offerte. Il sorteggio per il CSM, che nel corpus compare
in 2 dispacci su 609, puo' benissimo essere scelto qui da chi non lo ha mai
incontrato: proporre un'opzione significa introdurla.

Per questo l'elenco contiene, su ciascun fronte, una motivazione FITTIZIA,
riferita a qualcosa che la riforma non contiene. La quota di agenti che la
seleziona misura quanto lo strumento fabbrichi le risposte che raccoglie, ed
e' cio' che rende interpretabile il resto della tabella.

Lavora in sola lettura sul database: non scrive nulla nella run.

Uso, dalla radice del progetto:
  export LLM_API_KEY=...
  python scripts/motivazioni_chiuse.py runs/campagna_completa1/run.db
  python scripts/motivazioni_chiuse.py runs/campagna_completa1/run.db --label tick_60
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirrorfish.config import LLMConfig                    # noqa: E402
from mirrorfish.llm import build_client, parse_json_response  # noqa: E402
from mirrorfish.store import Store                         # noqa: E402
from mirrorfish.survey import build_context                # noqa: E402

# Elenco della rilevazione reale, con la quota dichiarata da chi ha votato in
# quel modo. FITTIZIA = opzione di controllo, non presente nella riforma.
OPZIONI = {
    "SI": [
        ("S1", "sono favorevole alla separazione delle carriere fra giudici e pubblici ministeri", 59),
        ("S2", "sono favorevole alla divisione del CSM in due organi distinti", 35),
        ("S3", "sono favorevole all'istituzione dell'Alta Corte disciplinare", 34),
        ("S4", "ritengo giusto modificare la Costituzione in questa direzione", 24),
        ("S5", "sostengo il Governo in carica", 18),
        ("SX", "sono favorevole all'abolizione di un grado di giudizio", None),
    ],
    "NO": [
        ("N1", "non ritengo giusto modificare la Costituzione", 61),
        ("N2", "sono contrario al sorteggio dei componenti del CSM", 39),
        ("N3", "sono all'opposizione del Governo in carica", 31),
        ("N4", "sono contrario alla divisione del CSM in due organi distinti", 27),
        ("N5", "sono contrario all'istituzione dell'Alta Corte disciplinare", 17),
        ("N6", "seguo l'indicazione di voto del mio partito", 7),
        ("N7", "sono contrario alla separazione delle carriere", 4),
        ("NX", "sono contrario all'abolizione di un grado di giudizio", None),
    ],
}

DOMANDA = """Hai votato {voto} al referendum costituzionale sulla giustizia.

Quali fra queste ragioni hanno pesato sulla tua scelta? Puoi indicarne piu' di \
una. Indica soltanto quelle che valgono davvero per te: se nessuna ti \
corrisponde, restituisci un elenco vuoto.

{elenco}

Rispondi esclusivamente con un oggetto JSON, senza testo prima o dopo:
{{"codici": ["..."]}}"""

SISTEMA = """Sei {username}, professione: {profession}.

{label}:
{context}

Rispondi come il tuo personaggio, coerentemente con come le tue opinioni si sono \
evolute."""


async def main_async(args) -> None:
    store = Store(Path(args.db))
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    voti = {r["agent_id"]: r["vote"] for r in con.execute(
        "SELECT agent_id, vote FROM vote WHERE label = ? AND vote != 'ERROR'",
        (args.label,))}
    if not voti:
        sys.exit(f"nessun voto con label={args.label} in {args.db}")

    agenti = [a for a in store.agents(include_sources=False, voters_only=True)
              if voti.get(int(a["agent_id"])) in ("SI", "NO")]
    print(f"{len(agenti)} agenti con voto SI o NO alla rilevazione '{args.label}'")
    print(f"(esclusi {len(voti) - len(agenti)} astenuti: la rilevazione reale "
          f"non riporta le loro ragioni)\n")

    cfg = LLMConfig.from_env(requests_per_minute=args.rpm, concurrency=args.concurrency)
    cfg.vote_temperature = args.temperature
    client = build_client(cfg)

    async def chiedi(agent):
        aid = int(agent["agent_id"])
        voto = voti[aid]
        ctx_label, ctx = build_context(store, agent, baseline=False)
        elenco = "\n".join(f"{c}  {t}" for c, t, _ in OPZIONI[voto])
        resp = await client.complete(
            SISTEMA.format(username=agent["username"],
                           profession=agent["profession"] or "non specificata",
                           label=ctx_label, context=ctx),
            DOMANDA.format(voto=voto, elenco=elenco),
            max_tokens=args.max_tokens, temperature=args.temperature, json_mode=True)
        dati = parse_json_response(resp.text)
        validi = {c for c, _, _ in OPZIONI[voto]}
        if resp.error or not isinstance(dati, dict):
            return voto, None
        return voto, [str(c).strip().upper() for c in dati.get("codici", [])
                      if str(c).strip().upper() in validi]

    try:
        esiti = await asyncio.gather(*(chiedi(a) for a in agenti))
    finally:
        await client.aclose()
        con.close()

    for voto in ("SI", "NO"):
        righe = [c for v, c in esiti if v == voto]
        validi = [c for c in righe if c is not None]
        if not validi:
            continue
        n = len(validi)
        conteggio = Counter(c for codici in validi for c in codici)
        vuote = sum(1 for c in validi if not c)
        print(f"{'=' * 72}\n Ha votato {voto}: {n} agenti "
              f"({len(righe) - n} risposte non interpretabili)\n{'=' * 72}")
        print(f"  {'cod':<5}{'simulata':>10}{'reale':>8}{'scarto':>9}  motivazione")
        for cod, testo, reale in OPZIONI[voto]:
            quota = 100 * conteggio[cod] / n
            if reale is None:
                print(f"  {cod:<5}{quota:>9.0f}%{'fittizia':>8}{'—':>9}  {testo[:44]}")
            else:
                print(f"  {cod:<5}{quota:>9.0f}%{reale:>7}%{quota - reale:>+9.0f}  {testo[:44]}")
        print(f"\n  nessuna opzione scelta: {100 * vuote / n:.0f}% degli agenti")
        fittizia = [c for c, _, r in OPZIONI[voto] if r is None][0]
        q = 100 * conteggio[fittizia] / n
        print(f"  OPZIONE FITTIZIA {fittizia}: scelta dal {q:.0f}% degli agenti.")
        if q >= 10:
            print("  Quota non trascurabile: lo strumento a risposta chiusa sta")
            print("  in parte fabbricando le risposte, e le altre quote vanno")
            print("  lette con la stessa cautela.\n")
        else:
            print("  Quota bassa: gli agenti non accolgono indiscriminatamente")
            print("  le opzioni proposte, e il resto della tabella e' leggibile.\n")

    print("Confronta questa tabella con quella della domanda aperta. Una ragione")
    print("assente nel testo libero e frequente qui non e' un'opinione della")
    print("popolazione: e' un'opinione che l'elenco le ha suggerito.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--label", default="final")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--rpm", type=float, default=38)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
