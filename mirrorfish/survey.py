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

VOTE_QUESTION = (
    "Oggi si tiene il Referendum Costituzionale sulla separazione delle "
    "carriere dei magistrati e ogni cittadino italiano sopra i 18 anni e' "
    "chiamato alle urne. TU voterai SI, NO, o ASTENUTO?"
)

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
    tick: int | None = None,
    verbose: bool = True,
) -> dict[str, int]:
    agents = store.agents(include_sources=False)
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
            system, f"DOMANDA DI VOTO: {VOTE_QUESTION}\nRispondi solo in JSON.",
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
