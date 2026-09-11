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
    "omettendo del tutto il nome dell'agente. Inizia direttamente dall'azione. "
    # I due esempi precedenti erano 'Rafforza la propria opposizione...' e
    # 'Sviluppa scetticismo verso...': entrambi contrari. E' lo stesso guasto
    # del prompt d'azione che non nominava POST — il modello copia lo schema
    # che gli si da'. Nell'audit del run rep7_titolo il dibattito dei
    # cittadini era al 22% pro-riforma e le note al 6%: la riflessione
    # aggiungeva da sola uno sbilanciamento di quasi quattro volte, e
    # 'Rafforza la propria opposizione' compariva verbatim nelle note.
    # Gli esempi ora sono quattro e bilanciati per direzione.
    "Esempi di apertura, senza preferenza fra loro: 'Si convince che...', "
    "'Rafforza la propria posizione su...', 'Sviluppa una riserva verso...', "
    "'Cambia idea riguardo a...'. "
    # Vincolo nato da due casi reali del run fix7. Una nota apriva con
    # "Rafforza la diffidenza verso il referendum" e proseguiva sostenendo che
    # la separazione delle carriere tutela dall'ingerenza politica, che e' la
    # tesi del SI: verbo di apertura e contenuto dicevano il contrario. Il voto
    # finale ha seguito il contenuto, ed e' passato da NO a SI mentre la nota
    # sembrava confermare il NO. Imporre un'apertura "dall'azione" fa scegliere
    # il verbo prima di sapere cosa si scrivera' dopo.
    "REGOLA 4: il verbo di apertura deve concordare con l'argomento che "
    "riporti. Se l'argomento che ha colpito l'agente e' a FAVORE della "
    "riforma, non aprire con 'rafforza la diffidenza'. "
    "Indica poi in `direzione` da che parte spinge quanto hai scritto: "
    "'verso_si' se avvicina all'approvazione della riforma, 'verso_no' se "
    "l'allontana, 'nessuna' se e' solo una sfumatura senza direzione. "
    "Ricorda che la riforma IN VOTO introduce la separazione delle carriere: "
    "chi la sostiene vota SI, chi la osteggia vota NO. "
    "REGOLA 3: non attribuire all'agente una posizione che non risulti dalla "
    "biografia o dalle note gia' scritte. Tu NON sai come voterebbe. Se i post "
    "lo hanno colpito, descrivi cio' che ha trovato convincente o discutibile, "
    "non una convinzione pregressa che potrebbe non avere. "
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
{{"note_added": true/false, "note": "la frase, o stringa vuota",
  "direzione": "verso_si" | "verso_no" | "nessuna",
  "reasoning": "una proposizione"}}"""


@dataclass
class ReflectionResult:
    agent_id: int
    note_added: bool = False
    note: str = ""
    direzione: str = "nessuna"
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

        direzione = str(data.get("direzione") or "nessuna").strip().lower()
        if direzione not in ("verso_si", "verso_no", "nessuna"):
            direzione = "nessuna"

        return ReflectionResult(
            agent_id=agent_id, note_added=added, note=note,
            reasoning=str(data.get("reasoning") or ""), direzione=direzione,
        ), resp
