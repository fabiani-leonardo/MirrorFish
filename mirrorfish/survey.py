"""
Survey di voto — port del tuo vote_survey.py.

Il disegno che avevi e' conservato integralmente perche' e' corretto:
la static_bio e' congelata al tick 0 e non viene mai riscritta, quindi il
"PRIMA" si ottiene votando con la sola bio, senza rieseguire la simulazione.
Il "DOPO" vota con bio + intera traiettoria delle note (non troncata a 8:
e' una chiamata sola per agente, non mille).

Due aggiunte:
  - i voti vanno su SQLite insieme al resto, quindi il cross-tab per regione,
    eta' o titolo di studio e' una query invece che uno script a parte. E'
    quello che ti serve per la validazione multi-punto: non un numero solo,
    ma la struttura per sottogruppi confrontabile con i sondaggi reali.
  - survey intermedie (`label="tick_N"`) durante il run, per ricostruire la
    traiettoria dell'opinione nel tempo invece che solo il punto finale.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from typing import Any

from .config import LLMConfig
from .llm import LLMClient, parse_json_response
from .store import Store

# --------------------------------------------------------------------------- #
# La domanda di voto
# --------------------------------------------------------------------------- #
# Questa e' la variabile piu' sottovalutata dell'intero impianto.
#
# La versione MINIMAL e' quella usata fino al 2026-09-08: nomina il tema ma non
# dice cosa la legge preveda, ne' cosa significhino SI e NO. Un modello con
# cutoff 2024 non ha alcun modo di saperlo. Davanti a una parola come "riforma"
# ricade sul proprio prior — riforma uguale efficienza, efficienza uguale bene —
# e risponde SI quasi a tutti. La firma di questo comportamento e' nei dati:
# nei profili B, dove il voto non e' scritto in biografia, il baseline dava
# SI 87% al centrodestra e SI 96% al centrosinistra. Non e' un orientamento
# politico: e' la stessa risposta data a chiunque, cioe' nessuna risposta.
#
# La versione BALLOT riproduce cio' che un elettore vero ha davanti nella
# cabina: la formula del quesito e il contenuto della legge costituzionale su
# cui si vota. Non e' informazione aggiuntiva rispetto alla realta', e' il
# minimo perche' la domanda sia rispondibile. Il contenuto e' tratto dai due
# articoli ANSA di spiegazione del quesito (23 febbraio e 19 marzo 2026).
#
# Sono tenute entrambe perche' il confronto fra le due E' un risultato: dice
# quanta parte del voto simulato dipenda dall'aver posto la domanda in modo
# rispondibile, e non dalla dinamica sociale che si intende misurare. Si
# sceglie con --vote-question, ed e' coperta dal fingerprint.

VOTE_QUESTION_MINIMAL = (
    "Oggi si tiene il Referendum Costituzionale sulla separazione delle "
    "carriere dei magistrati e ogni cittadino italiano sopra i 18 anni e' "
    "chiamato alle urne. TU voterai SI, NO, o ASTENUTO?"
)

# Formula reale della scheda: e' letteralmente cio' che un elettore trova
# davanti, cioe' il titolo della legge e nessuna spiegazione. Nei referendum
# costituzionali italiani la scheda NON riassume il contenuto.
VOTE_QUESTION_BALLOT = """Oggi, 22 marzo 2026, si vota il referendum \
costituzionale sulla giustizia. Sulla scheda c'e' scritto:

  \u00abApprovate il testo della legge costituzionale concernente
  \u00abNorme in materia di ordinamento giurisdizionale e di istituzione
  della Alta Corte disciplinare\u00bb, approvato dal Parlamento e pubblicato
  nella Gazzetta Ufficiale?\u00bb

Votando SI si approva la legge, votando NO la si respinge.

TU cosa voti: SI, NO, o ASTENUTO?"""

# Scheda piu' il contenuto della legge: e' cio' che sa un elettore che si e'
# informato.
#
# LA FORMULAZIONE E' UNA VARIABILE, NON UN DETTAGLIO. La prima stesura di
# questo testo, scritta il 2026-09-08, descriveva le modifiche con verbi di
# sottrazione — "non sono piu' eletti", "toglie ai Csm", "non sono
# ricorribili" — e metteva ESTRATTI A SORTE in maiuscolo. Sono tutte scelte
# che orientano verso il NO, e il baseline che ne e' uscito (NO 53 su 100)
# potrebbe rifletterle. Qui le stesse modifiche sono descritte con verbi
# neutri di sostituzione. Chi legge questo codice deve poter vedere che il
# testo e' stato riscritto e perche': confrontare i baseline ottenuti con le
# tre varianti e' una analisi di sensibilita', ed e' un risultato da
# riportare, non un passaggio da nascondere.
VOTE_QUESTION_INFORMED = VOTE_QUESTION_BALLOT.replace(
    "\nTU cosa voti", """
COSA CAMBIA LA LEGGE, RISPETTO A OGGI:
- i magistrati sono distinti in due carriere, giudicante e requirente, e la
  distinzione viene inserita in Costituzione;
- all'attuale Consiglio superiore della magistratura subentrano due Consigli,
  uno per ciascuna carriera, entrambi presieduti dal Presidente della
  Repubblica;
- i componenti dei due Consigli sono designati per sorteggio anziche' per
  elezione, per un terzo da un elenco di giuristi compilato dal Parlamento e
  per due terzi fra i magistrati;
- la funzione disciplinare passa dai Consigli a una nuova Alta Corte
  disciplinare di quindici membri, in parte nominati e in parte sorteggiati;
- le decisioni dell'Alta Corte si impugnano davanti alla stessa Corte in
  diversa composizione.

TU cosa voti""")

VOTE_QUESTIONS = {
    "minimal": VOTE_QUESTION_MINIMAL,     # solo il tema: misura il prior
    "ballot": VOTE_QUESTION_BALLOT,       # la scheda reale, senza sintesi
    "informed": VOTE_QUESTION_INFORMED,   # scheda + contenuto, in neutro
}

VOTE_SYSTEM = """Sei {username}, professione: {profession}.

{label}:
{context}

Rispondi alla domanda di voto in modo coerente con il tuo personaggio{evolved}.
Rispondi SOLO con un JSON valido:
{{"vote": "SI"|"NO"|"ASTENUTO", "motivation": "spiegazione in prima persona", "confidence": 0.0-1.0}}"""


def normalize_vote(raw: Any) -> str:
    v = str(raw or "").upper().strip()
    if v.startswith("S") and "SI" in v.replace("Ì", "I"):
        return "SI"
    if v == "NO" or v.startswith("NO"):
        return "NO"
    if "ASTEN" in v:
        return "ASTENUTO"
    return "ASTENUTO"


def build_context(store: Store, agent: sqlite3.Row, baseline: bool) -> tuple[str, str]:
    bio = agent["static_bio"] or ""
    if baseline:
        return "CHI SEI", bio[:1500]
    notes = store.notes_for(int(agent["agent_id"]))
    if not notes:
        return "CHI SEI", bio[:1500]
    joined = "\n".join(f"- {n}" for n in notes)
    return ("CHI SEI E COME SI E' EVOLUTO IL TUO PENSIERO",
            f"{bio}\n\n[Evoluzione durante la simulazione]\n{joined}")


async def run_survey(
    store: Store,
    client: LLMClient,
    llm_cfg: LLMConfig,
    *,
    label: str,
    baseline: bool = False,
    question: str = "ballot",
    tick: int | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    agents = store.agents(include_sources=False, voters_only=True)
    budget = llm_cfg.token_budget["vote"]

    async def one(agent: sqlite3.Row) -> tuple[int, str, float, str, Any]:
        ctx_label, ctx = build_context(store, agent, baseline)
        system = VOTE_SYSTEM.format(
            username=agent["username"],
            profession=agent["profession"] or "non specificata",
            label=ctx_label, context=ctx,
            evolved="" if baseline else " e con come le tue opinioni si sono evolute",
        )
        resp = await client.complete(
            system,
            f"DOMANDA DI VOTO: {VOTE_QUESTIONS[question]}\nRispondi solo in JSON.",
            max_tokens=budget, temperature=llm_cfg.vote_temperature, json_mode=True,
        )
        aid = int(agent["agent_id"])
        data = parse_json_response(resp.text)
        if resp.error or not isinstance(data, dict):
            return aid, "ERROR", 0.0, (resp.error or "unparsable"), resp
        try:
            conf = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        return (aid, normalize_vote(data.get("vote")), conf,
                str(data.get("motivation") or "")[:1000], resp)

    # Progresso incrementale. `asyncio.gather` non emette nulla finche' non
    # ha finito: con 105 agenti a 8 req/min sono 13 minuti in cui il processo
    # sembra bloccato. Su un run non sorvegliato e' la differenza fra
    # "sta lavorando" e "l'ho ammazzato per sbaglio".
    done = 0
    t0 = time.perf_counter()

    async def tracked(agent):
        nonlocal done
        out = await one(agent)
        done += 1
        if verbose and (done == 1 or done % 10 == 0 or done == len(agents)):
            el = time.perf_counter() - t0
            rate = done / el * 60 if el else 0
            eta = (len(agents) - done) / max(rate, 0.1)
            print(f"  [{label}] {done}/{len(agents)} voti "
                  f"({rate:.1f}/min, ~{eta:.0f} min alla fine)", flush=True)
        return out

    results = await asyncio.gather(*(tracked(a) for a in agents))
    for aid, vote, conf, motivation, resp in results:
        store.add_vote(label, aid, vote, conf, motivation, tick=tick)
        store.log_call("vote", budget, resp, tick=tick, agent_id=aid)
    store.commit()

    tally = store.vote_tally(label)
    if verbose:
        print_tally(tally, label)
    return tally


def print_tally(tally: dict[str, int], label: str) -> None:
    valid = sum(v for k, v in tally.items() if k != "ERROR")
    print(f"\n--- RISULTATI [{label}] ---")
    for k in ("SI", "NO", "ASTENUTO"):
        n = tally.get(k, 0)
        pct = f"{n / valid * 100:5.1f}%" if valid else "   n/a"
        print(f"  {k:<9} {n:5d}  {pct}")
    print(f"  {'ERRORI':<9} {tally.get('ERROR', 0):5d}")
    print(f"  {'validi':<9} {valid:5d}")


def crosstab(store: Store, label: str, dimension: str) -> list[sqlite3.Row]:
    """
    Distribuzione del voto per una dimensione demografica.

    E' la funzione che regge la validazione multi-punto: confrontare il
    risultato aggregato col reale e' un solo grado di liberta', confrontare
    la struttura per regione/eta'/titolo ne fornisce decine.
    """
    if dimension not in {"region", "education", "profession", "age_band"}:
        raise ValueError(f"dimensione non supportata: {dimension}")
    col = ("CASE WHEN a.age < 35 THEN '18-34' WHEN a.age < 55 THEN '35-54' "
           "ELSE '55+' END" if dimension == "age_band" else f"a.{dimension}")
    return store.conn.execute(
        f"""
        SELECT {col} AS bucket, v.vote, COUNT(*) AS n
        FROM vote v JOIN agent a ON a.agent_id = v.agent_id
        WHERE v.label = ? AND v.vote != 'ERROR'
        GROUP BY bucket, v.vote ORDER BY bucket, v.vote
        """,
        (label,),
    ).fetchall()


def shift_report(store: Store, before: str = "baseline", after: str = "final") -> dict[str, Any]:
    """Chi ha cambiato idea fra le due survey, e come."""
    b = {r["agent_id"]: r for r in store.votes(before)}
    a = {r["agent_id"]: r for r in store.votes(after)}
    shifted, stable = [], []
    for aid in set(b) & set(a):
        vb, va = b[aid]["vote"], a[aid]["vote"]
        if "ERROR" in (vb, va):
            continue
        entry = {"agent_id": aid, "before": vb, "after": va,
                 "notes": store.notes_for(aid)}
        (shifted if vb != va else stable).append(entry)
    total = len(shifted) + len(stable)
    return {
        "total": total,
        "shifted": len(shifted),
        "stable": len(stable),
        "shift_rate": round(len(shifted) / total, 4) if total else 0.0,
        "transitions": _transitions(shifted),
        "detail": shifted,
    }


def _transitions(shifted: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in shifted:
        key = f"{e['before']}->{e['after']}"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
