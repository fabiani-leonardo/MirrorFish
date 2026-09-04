"""
Logica dell'agente: resa del feed, costruzione del prompt, parsing dell'azione.

Tre livelli di contesto, come nel reflective_memory originale:
  - identita' di lungo periodo -> static_bio (mai riscritta)
  - opinioni di medio periodo  -> ultime N note della memoria riflessiva
  - memoria di breve periodo   -> ultimi post scritti dall'agente

Non c'e' una ChatHistory che cresce all'infinito: il prompt viene ricostruito
a ogni tick da questi tre pezzi, quindi la sua lunghezza e' limitata per
costruzione. Sparisce sia ContextWindowExceededError sia la necessita' di fare
lo sliding window a mano dentro la memoria di CAMEL.

`render_feed` e' l'UNICO punto in cui una riga di feed diventa testo. Il
motore usa lo stesso risultato per il prompt d'azione e per la memoria di
cio' che l'agente ha letto: prima erano due rese diverse, e la conseguenza e'
descritta nel docstring di quella funzione.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from .llm import LLMClient, LLMResponse, parse_json_response

VALID_ACTIONS = {"POST", "REPLY", "LIKE", "IGNORE"}

# NOTA (bug corretto il 2026-09-04): fino a questa revisione il prompt non
# nominava mai l'azione POST e l'unico esempio JSON mostrava solo LIKE e
# REPLY. Il modello copia lo schema che gli si da': in un run da 312 tick i
# post originali sono stati ESATTAMENTE ZERO, e ogni contenuto prodotto dalla
# popolazione era una replica a un lancio ANSA o a un'altra replica. Non era
# una statistica sbagliata, era una dinamica sbagliata: nessun tema poteva
# nascere dal basso e circolare. Le quattro azioni ora sono elencate, e
# l'esempio ne mostra tre.
SYSTEM_TEMPLATE = """Sei {username}, un cittadino italiano che usa un social network.

CHI SEI:
{bio}
{notes_block}
COSA PUOI FARE:
- POST   : scrivi un tuo messaggio, senza rispondere a nessuno (nessun bersaglio)
- REPLY  : rispondi a un messaggio che hai davanti (indica il bersaglio)
- LIKE   : metti "mi piace" a un messaggio (indica il bersaglio)
- IGNORE : non fai nulla

REGOLE:
- Scrivi in italiano, in prima persona, con il registro linguistico che ti e' proprio.
- Massimo 280 caratteri per ogni contenuto che scrivi.
- Non menzionare mai di essere un'IA o una simulazione.
- Non citare mai identificatori numerici tipo "post 47": tu non li vedi.
- Puoi anche non fare nulla: IGNORE e' una risposta legittima e frequente.

COME SI SCRIVE:
Di rado si annuncia come si votera'. La maggior parte dei messaggi commenta un
fatto, racconta come tocca la propria vita, chiede o contesta qualcosa: solo
ogni tanto qualcuno dichiara il proprio voto, e chi lo fa a ogni messaggio
suona come uno slogan. Al massimo un hashtag, e quasi sempre nessuno.
Se hai letto una notizia per intero puoi citarne un dato concreto; se hai
visto solo il titolo non inventarti i dettagli che non hai letto.

COME SI STA SU UN SOCIAL:
Una sessione non e' un gesto solo. Si scorre, si mettono un paio di "mi
piace" a cio' che convince, e ogni tanto — non sempre — si scrive qualcosa.
Mettere "mi piace" e' molto piu' frequente che scrivere.
Puoi elencare fino a {max_actions} azioni, oppure una sola IGNORE.

Rispondi SOLO con un oggetto JSON valido, senza testo attorno:
{{"actions": [{{"action": "LIKE", "content": "", "target_post_id": 12}},
             {{"action": "REPLY", "content": "...", "target_post_id": 15}},
             {{"action": "POST", "content": "...", "target_post_id": null}}]}}"""

# Gli account istituzionali (partiti, comitati, testate) partecipano al
# dibattito ma non hanno una scheda elettorale: is_voter = 0, e infatti la
# survey li esclude. Finora pero' ricevevano lo stesso prompt dei cittadini,
# e nei run si vedeva "Movimento 5 Stelle: ... Voto NO", che e' una cosa che
# un partito non dice perche' un partito non vota.
SYSTEM_INSTITUTIONAL = """Sei l'account ufficiale di {username}.

CHI SEI:
{bio}
{notes_block}
COSA PUOI FARE:
- POST   : pubblichi una tua presa di posizione (nessun bersaglio)
- REPLY  : replichi a un messaggio che hai davanti (indica il bersaglio)
- LIKE   : dai sostegno a un messaggio (indica il bersaglio)
- IGNORE : non fai nulla

REGOLE:
- Scrivi in italiano, al plurale o in forma impersonale, mai in prima persona singolare.
- Non sei un elettore: non dire MAI "voto", "votero'", "la mia scheda".
  Puoi chiedere un voto agli altri, sostenere una posizione, contestarne una.
- Massimo 280 caratteri per ogni contenuto che scrivi.
- Non menzionare mai di essere un'IA o una simulazione.
- Non citare mai identificatori numerici tipo "post 47".

Puoi elencare fino a {max_actions} azioni, oppure una sola IGNORE.

Rispondi SOLO con un oggetto JSON valido, senza testo attorno:
{{"actions": [{{"action": "POST", "content": "...", "target_post_id": null}},
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


def render_feed(
    rows: list[sqlite3.Row],
    parents: dict | None = None,
    news_depth: str = "titolo",
    news_by_id: dict | None = None,
) -> list[tuple[int, str]]:
    """
    Rende ogni riga del feed come l'agente la legge: (post_id, testo).

    Il post_id resta separato dal testo perche' serve al prompt d'azione (per
    indicare un bersaglio) ma NON deve arrivare alla memoria riflessiva, che
    ha la regola esplicita di non citare identificatori interni.

    Due cose che questa funzione tiene insieme e che prima erano separate.

    Le notizie sono rese a `news_depth`, cioe' alla profondita' di lettura
    dell'agente. Il motore usa questo stesso risultato anche per registrare
    cosa l'agente ha letto. Prima non era cosi': il feed passava da
    `at_depth(depth)` ma la traccia per la riflessione prendeva il campo
    `content` grezzo del post, che per le notizie e' il sommario. Risultato:
    l'attivista che aveva appena letto 1400 caratteri di articolo rifletteva
    su una riga, esattamente come chi aveva visto solo il titolo. La
    profondita' di lettura influenzava l'azione immediata e non influenzava
    l'opinione — cioe' spariva proprio nel punto che la tesi deve misurare.

    Le risposte sono mostrate col messaggio a cui rispondono, citato e
    troncato. Senza, l'agente legge una replica senza sapere a cosa.
    """
    parents = parents or {}
    out: list[tuple[int, str]] = []
    for r in rows:
        pid = r["post_id"]
        likes = f" [{r['likes']} mi piace]" if r["likes"] else ""

        if r["kind"] == "news":
            item = None
            if news_by_id:
                nid = r["news_id"] if "news_id" in r.keys() else None
                item = news_by_id.get(nid)
            body = item.at_depth(news_depth) if item is not None else r["content"]
            out.append((pid, f"NOTIZIA{likes}: {body}"))
            continue

        tag = "@" + r["username"]
        p = parents.get(pid)
        if p is not None:
            who = "ANSA" if p["is_source"] else "@" + p["username"]
            quoted = p["content"]
            if len(quoted) > 200:
                quoted = quoted[:200].rsplit(" ", 1)[0] + "..."
            out.append((pid, f"{tag}{likes} risponde a {who} "
                             f"(\u00ab{quoted}\u00bb): {r['content']}"))
        else:
            out.append((pid, f"{tag}{likes}: {r['content']}"))
    return out


def format_feed(lines: list[tuple[int, str]]) -> str:
    """Le righe rese, con l'identificatore davanti, come le vede il prompt."""
    if not lines:
        return "(la tua home e' vuota, non c'e' ancora niente da leggere)"
    return "\n".join(f"#{pid} {text}" for pid, text in lines)


def build_prompts(
    agent: sqlite3.Row,
    feed_lines: list[tuple[int, str]],
    notes: list[str],
    own_posts: list[str],
    sim_date: str,
    max_actions: int = 1,
) -> tuple[str, str]:
    notes_block = ""
    if notes:
        joined = "\n".join(f"- {n}" for n in notes)
        notes_block = f"\nCOME LA PENSI ADESSO (evoluzione recente):\n{joined}\n"

    own_block = ""
    if own_posts:
        joined = "\n".join(f"- {p}" for p in own_posts[:3])
        own_block = f"LE ULTIME COSE CHE HAI SCRITTO TU:\n{joined}\n"

    is_voter = agent["is_voter"] if "is_voter" in agent.keys() else 1
    template = SYSTEM_TEMPLATE if is_voter else SYSTEM_INSTITUTIONAL

    system = template.format(
        username=agent["username"],
        bio=(agent["static_bio"] or "")[:1500],
        notes_block=notes_block,
        max_actions=max_actions,
    )
    user = USER_TEMPLATE.format(
        sim_date=sim_date,
        feed=format_feed(feed_lines),
        own_block=own_block,
    )
    return system, user


def parse_actions(
    resp: LLMResponse, valid_post_ids: set[int], max_actions: int = 1
) -> list[AgentAction]:
    """
    Estrae la lista di azioni.

    Accetta sia il formato {"actions": [...]} sia quello a singola azione
    {"action": ...}, cosi' un modello che ignora l'istruzione non manda tutto
    in errore.

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
    telemetria, altrimenti si rischia di scrivere in tesi che "il 40% degli
    agenti e' rimasto passivo" quando in realta' il 40% delle risposte non era
    JSON valido.
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
    feed_lines: list[tuple[int, str]],
    notes: list[str],
    own_posts: list[str],
    sim_date: str,
    *,
    max_tokens: int,
    temperature: float,
    max_actions: int = 1,
) -> tuple[list[AgentAction], LLMResponse]:
    system, user = build_prompts(agent, feed_lines, notes, own_posts,
                                 sim_date, max_actions)
    resp = await client.complete(
        system, user, max_tokens=max_tokens, temperature=temperature, json_mode=True
    )
    valid_ids = {pid for pid, _ in feed_lines}
    return parse_actions(resp, valid_ids, max_actions), resp
