"""
Memoria riflessiva — port diretto del tuo reflective_memory.py.

Cosa resta identico (e' la parte tua, quella che vale):
  - append-only scratchpad invece di riscrivere la bio;
  - la bio statica non passa MAI attraverso l'LLM di riflessione;
  - "nessun cambiamento" deve essere il caso comune, non l'eccezione;
  - le due CRITICAL STYLE RULE (niente ID numerici, forma impersonale).

Cosa sparisce: tutta la sezione 3 del file originale, cioe' la glue con
OASIS/CAMEL (_rewrite_agent_persona_and_slide_memory, _extract_last_env_prompt,
il rimescolamento di agent.memory, la riassegnazione di agent._system_message).
Erano ~150 righe che esistevano solo per aggirare il fatto che CAMEL non
espone un setter per il system message e non fa trimming della memoria. Senza
CAMEL non servono: il prompt viene ricostruito da zero a ogni tick.

Cosa cambia: le note stanno su SQLite invece che su dynamic_profiles.json.
Niente piu' lock threading, niente scritture atomiche fatte a mano, e le note
sono interrogabili con SQL insieme a tutto il resto (utile per la tesi:
"quante note ha generato ogni fascia d'eta'" e' una query, non uno script).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .llm import LLMClient, LLMResponse, parse_json_response

REFLECTION_SYSTEM = (
    "Simuli il processo di riflessione interiore di un utente di social media. "
    "Ricevi la biografia statica (solo come contesto), le note accumulate "
    "finora su come e' evoluto il suo pensiero, e i post che ha appena letto. "
    "Nella maggior parte dei casi non cambia nulla: segnala un cambiamento "
    "solo quando i post danno una ragione reale per spostare, rafforzare o "
    "sfumare un'opinione. "
    "REGOLA 1: non citare mai identificatori interni come 'post 47' o "
    "'commento 12' — l'agente non sa che esistono. Riferisciti all'autore "
    "quando e' noto, altrimenti al tema o all'affermazione. "
    "REGOLA 2: nota e motivazione vanno scritte in forma impersonale, "
    "omettendo del tutto il nome dell'agente. Inizia direttamente dall'azione "
    "(es. 'Rafforza la propria opposizione...', 'Sviluppa scetticismo verso...'). "
    "Rispondi solo con un oggetto JSON, in italiano, senza prosa attorno."
)

REFLECTION_USER = """Biografia statica (solo contesto, non riscriverla):
\"\"\"{bio}\"\"\"

Note accumulate finora:
{notes}

Post appena letti:
{posts}

Decidi se questi post sono abbastanza significativi da spostare, rafforzare o
sfumare le opinioni dell'agente. Se sono banali o gia' coerenti con le note
esistenti, NON aggiungere una nota: questo deve essere il caso comune.

Se e solo se qualcosa e' davvero cambiato, scrivi UNA frase breve (max {max_chars}
caratteri) in forma impersonale.

Rispondi esattamente in questa forma:
{{"note_added": true/false, "note": "la frase, o stringa vuota", "reasoning": "una proposizione"}}"""


@dataclass
class ReflectionResult:
    agent_id: int
    note_added: bool = False
    note: str = ""
    reasoning: str = ""
    error: str | None = None


class ReflectionEngine:
    def __init__(self, client: LLMClient, max_note_chars: int = 400):
        self.client = client
        self.max_note_chars = max_note_chars

    async def reflect(
        self,
        agent: sqlite3.Row,
        existing_notes: list[str],
        recent_posts: list[str],
        *,
        max_tokens: int,
        temperature: float,
    ) -> tuple[ReflectionResult, LLMResponse | None]:
        agent_id = int(agent["agent_id"])
        if not recent_posts:
            return ReflectionResult(agent_id, reasoning="niente da leggere"), None

        user = REFLECTION_USER.format(
            bio=(agent["static_bio"] or "")[:1500],
            notes="\n".join(f"- {n}" for n in existing_notes[-10:]) or "(nessuna)",
            posts="\n".join(f"- {p}" for p in recent_posts if p and p.strip()),
            max_chars=self.max_note_chars,
        )
        resp = await self.client.complete(
            REFLECTION_SYSTEM, user,
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
        if resp.error:
            return ReflectionResult(agent_id, error=f"llm:{resp.error}"), resp

        data = parse_json_response(resp.text)
        if not isinstance(data, dict):
            reason = "truncated" if resp.truncated else "unparsable"
            return ReflectionResult(agent_id, error=f"parse:{reason}"), resp

        note = str(data.get("note") or "").strip()
        added = bool(data.get("note_added", False))
        if len(note) > self.max_note_chars:
            note = note[: self.max_note_chars].rsplit(" ", 1)[0] + "..."
        if added and not note:
            added = False

        return ReflectionResult(
            agent_id=agent_id, note_added=added, note=note,
            reasoning=str(data.get("reasoning") or ""),
        ), resp
