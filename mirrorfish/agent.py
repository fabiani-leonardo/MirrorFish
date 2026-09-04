"""
Logica dell'agente: costruzione del prompt e parsing dell'azione.

Tre livelli di contesto, esattamente come nel tuo reflective_memory:
  - identita' di lungo periodo -> static_bio (mai riscritta)
  - opinioni di medio periodo  -> ultime N note della memoria riflessiva
  - memoria di breve periodo   -> ultimi post scritti dall'agente

Non c'e' una ChatHistory che cresce all'infinito: il prompt viene ricostruito
a ogni tick da questi tre pezzi, quindi la sua lunghezza e' limitata per
costruzione. Sparisce sia ContextWindowExceededError sia la necessita' di fare
lo sliding window a mano dentro la memoria di CAMEL.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient, LLMResponse, parse_json_response

VALID_ACTIONS = {"POST", "REPLY", "LIKE", "IGNORE"}

SYSTEM_TEMPLATE = """Sei {username}, un cittadino italiano che usa un social network.

CHI SEI:
{bio}
{notes_block}
REGOLE:
- Scrivi in italiano, in prima persona, con il registro linguistico che ti e' proprio.
- Massimo 280 caratteri per ogni contenuto che scrivi.
- Non menzionare mai di essere un'IA o una simulazione.
- Non citare mai identificatori numerici tipo "post 47": tu non li vedi.
- Puoi anche non fare nulla: IGNORE e' una risposta legittima e frequente.

COME SI STA SU UN SOCIAL:
Una sessione non e' un gesto solo. Si scorre, si mettono un paio di "mi
piace" a cio' che convince, e ogni tanto — non sempre — si scrive qualcosa.
Mettere "mi piace" e' molto piu' frequente che scrivere. Puoi elencare fino a
{max_actions} azioni, oppure una sola IGNORE se niente ti ha colpito.

Rispondi SOLO con un oggetto JSON valido, senza testo attorno:
{{"actions": [{{"action": "LIKE", "content": "", "target_post_id": 12}},
             {{"action": "REPLY", "content": "...", "target_post_id": 15}}]}}"""

USER_TEMPLATE = """Oggi e' {sim_date}.

QUELLO CHE VEDI SULLA TUA HOME:
{feed}

{own_block}
Cosa fai adesso? Rispondi in JSON."""


@dataclass
class AgentAction:
    action: str = "IGNORE"
    content: str = ""
    target_post_id: int | None = None
    error: str | None = None

    @property
    def is_noop(self) -> bool:
        return self.action == "IGNORE" or self.error is not None


def format_feed(rows: list[sqlite3.Row], parents: dict | None = None,
                news_depth: str = "sommario",
                news_by_id: dict | None = None) -> str:
    """
    Rende il feed come lo legge l'agente.

    Le risposte vengono mostrate con il messaggio a cui rispondono, citato e
    troncato. Senza, l'agente legge una replica senza sapere a cosa: era il
    caso finora, ed e' il motivo per cui certe risposte sembravano scollegate.
    """
    if not rows:
        return "(la tua home e' vuota, non c'e' ancora niente da leggere)"
    parents = parents or {}
    lines = []
    for r in rows:
        tag = "NOTIZIA" if r["kind"] == "news" else "@" + r["username"]
        likes = f" [{r['likes']} mi piace]" if r["likes"] else ""
        if r["kind"] == "news" and news_by_id:
            nid = r["news_id"] if "news_id" in r.keys() else None
            item = news_by_id.get(nid)
            if item is not None:
                lines.append(f"#{r['post_id']} {tag}{likes}: "
                             f"{item.at_depth(news_depth)}")
                continue
        p = parents.get(r["post_id"])
        if p is not None:
            who = "ANSA" if p["is_source"] else "@" + p["username"]
            quoted = p["content"]
            if len(quoted) > 200:
                quoted = quoted[:200].rsplit(" ", 1)[0] + "..."
            lines.append(f"#{r['post_id']} {tag}{likes} risponde a {who} "
                         f"(\u00ab{quoted}\u00bb): {r['content']}")
        else:
            lines.append(f"#{r['post_id']} {tag}{likes}: {r['content']}")
    return "\n".join(lines)


def build_prompts(
    agent: sqlite3.Row,
    feed_rows: list[sqlite3.Row],
    notes: list[str],
    own_posts: list[str],
    sim_date: str,
    parents: dict | None = None,
    max_actions: int = 1,
    news_depth: str = "sommario",
    news_by_id: dict | None = None,
) -> tuple[str, str]:
    notes_block = ""
    if notes:
        joined = "\n".join(f"- {n}" for n in notes)
        notes_block = f"\nCOME LA PENSI ADESSO (evoluzione recente):\n{joined}\n"

    own_block = ""
    if own_posts:
        joined = "\n".join(f"- {p}" for p in own_posts[:3])
        own_block = f"LE ULTIME COSE CHE HAI SCRITTO TU:\n{joined}\n"

    system = SYSTEM_TEMPLATE.format(
        username=agent["username"],
        bio=(agent["static_bio"] or "")[:1500],
        notes_block=notes_block,
        max_actions=max_actions,
    )
    user = USER_TEMPLATE.format(
        sim_date=sim_date,
        feed=format_feed(feed_rows, parents, news_depth, news_by_id),
        own_block=own_block,
    )
    return system, user


def parse_actions(
    resp: LLMResponse, valid_post_ids: set[int], max_actions: int = 1
) -> list[AgentAction]:
    """
    Estrae la lista di azioni.

    Accetta sia il formato nuovo {"actions": [...]} sia quello vecchio
    {"action": ...}, cosi' i run gia' fatti restano riproducibili e un modello
    che ignora l'istruzione non manda tutto in errore.

    Deduplica i bersagli: due LIKE sullo stesso post nella stessa sessione
    sono un artefatto del modello, non un comportamento.
    """
    if resp.error:
        return [AgentAction(error=f"llm:{resp.error}")]
    data = parse_json_response(resp.text)
    if not isinstance(data, dict):
        reason = "truncated" if resp.truncated else "unparsable"
        return [AgentAction(error=f"parse:{reason}")]

    raw = data.get("actions")
    if raw is None:
        raw = [data]                      # formato a singola azione
    if not isinstance(raw, list):
        return [AgentAction(error="parse:actions_not_list")]

    out: list[AgentAction] = []
    seen_targets: set[tuple[str, int]] = set()
    wrote_content = False
    for item in raw[: max_actions * 2]:   # margine, poi si taglia
        if len(out) >= max_actions:
            break
        if not isinstance(item, dict):
            continue
        act = _one_action(item, valid_post_ids)
        if act.error or act.action == "IGNORE":
            if not out and act.error:
                out.append(act)
            continue
        key = (act.action, act.target_post_id or -1)
        if key in seen_targets:
            continue
        # Al massimo un contenuto scritto per sessione: due post nello stesso
        # momento sono spam, non partecipazione.
        if act.action in ("POST", "REPLY"):
            if wrote_content:
                continue
            wrote_content = True
        seen_targets.add(key)
        out.append(act)

    return out or [AgentAction(action="IGNORE")]


def _one_action(data: dict, valid_post_ids: set[int]) -> AgentAction:
    """
    Converte la risposta grezza in un'azione validata.

    Ogni fallimento diventa IGNORE con un campo `error` popolato: un errore di
    parsing non deve mai far crollare il tick, ma deve restare visibile nella
    telemetria, altrimenti rischi di scrivere in tesi che "il 40% degli agenti
    e' rimasto passivo" quando in realta' il 40% delle risposte non era JSON.
    """
    action = str(data.get("action", "")).upper().strip()
    if action not in VALID_ACTIONS:
        return AgentAction(error=f"parse:bad_action:{action[:20]}")

    content = str(data.get("content") or "").strip()[:280]
    raw_target = data.get("target_post_id")
    target: int | None = None
    if raw_target is not None:
        try:
            target = int(raw_target)
        except (TypeError, ValueError):
            target = None
    if target is not None and target not in valid_post_ids:
        target = None

    if action in ("REPLY", "LIKE") and target is None:
        return AgentAction(error="parse:missing_target")
    if action in ("POST", "REPLY") and not content:
        return AgentAction(error="parse:empty_content")

    return AgentAction(action=action, content=content, target_post_id=target)


async def decide(
    client: LLMClient,
    agent: sqlite3.Row,
    feed_rows: list[sqlite3.Row],
    notes: list[str],
    own_posts: list[str],
    sim_date: str,
    *,
    max_tokens: int,
    temperature: float,
    parents: dict | None = None,
    max_actions: int = 1,
    news_depth: str = "sommario",
    news_by_id: dict | None = None,
) -> tuple[list[AgentAction], LLMResponse]:
    system, user = build_prompts(agent, feed_rows, notes, own_posts, sim_date,
                                 parents, max_actions, news_depth, news_by_id)
    resp = await client.complete(
        system, user, max_tokens=max_tokens, temperature=temperature, json_mode=True
    )
    valid_ids = {r["post_id"] for r in feed_rows}
    return parse_actions(resp, valid_ids, max_actions), resp
