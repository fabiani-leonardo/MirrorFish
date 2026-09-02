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

Rispondi SOLO con un oggetto JSON valido, senza testo attorno:
{{"action": "POST"|"REPLY"|"LIKE"|"IGNORE", "content": "testo o stringa vuota", "target_post_id": numero o null}}"""

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


def format_feed(rows: list[sqlite3.Row]) -> str:
    if not rows:
        return "(la tua home e' vuota, non c'e' ancora niente da leggere)"
    lines = []
    for r in rows:
        tag = "NOTIZIA" if r["kind"] == "news" else "@" + r["username"]
        likes = f" [{r['likes']} mi piace]" if r["likes"] else ""
        lines.append(f"#{r['post_id']} {tag}{likes}: {r['content']}")
    return "\n".join(lines)


def build_prompts(
    agent: sqlite3.Row,
    feed_rows: list[sqlite3.Row],
    notes: list[str],
    own_posts: list[str],
    sim_date: str,
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
    )
    user = USER_TEMPLATE.format(
        sim_date=sim_date,
        feed=format_feed(feed_rows),
        own_block=own_block,
    )
    return system, user


def parse_action(resp: LLMResponse, valid_post_ids: set[int]) -> AgentAction:
    """
    Converte la risposta grezza in un'azione validata.

    Ogni fallimento diventa IGNORE con un campo `error` popolato: un errore di
    parsing non deve mai far crollare il tick, ma deve restare visibile nella
    telemetria, altrimenti rischi di scrivere in tesi che "il 40% degli agenti
    e' rimasto passivo" quando in realta' il 40% delle risposte non era JSON.
    """
    if resp.error:
        return AgentAction(error=f"llm:{resp.error}")
    data = parse_json_response(resp.text)
    if not isinstance(data, dict):
        reason = "truncated" if resp.truncated else "unparsable"
        return AgentAction(error=f"parse:{reason}")

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
) -> tuple[AgentAction, LLMResponse]:
    system, user = build_prompts(agent, feed_rows, notes, own_posts, sim_date)
    resp = await client.complete(
        system, user, max_tokens=max_tokens, temperature=temperature, json_mode=True
    )
    valid_ids = {r["post_id"] for r in feed_rows}
    return parse_action(resp, valid_ids), resp
