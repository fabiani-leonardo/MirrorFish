"""
Caricamento della popolazione.

`load_mirofish_profiles` legge il formato che gia' produci
(reddit_profiles.json), cosi' la popolazione ISTAT/YouTrend che hai gia'
generato si riusa tale e quale: non rigeneriamo niente, e la baseline resta
confrontabile con i run vecchi.

`synthetic` serve solo per lo smoke test offline.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

# Cronotipi: propensione relativa all'uso dei social per ora del giorno.
# Derivati per fascia; nel run vero vanno tarati sui dati ISTAT sull'uso del
# tempo, non su queste stime.
CRONOTIPI: dict[str, list[float]] = {
    # 0h                                   12h                          23h
    "studente":  [.4,.3,.2,.1,.1,.1,.1,.2,.4,.5,.6,.7,.8,.7,.6,.6,.7,.8,.9,1.,1.,.9,.8,.6],
    "lavoratore":[.1,.1,.0,.0,.0,.1,.3,.6,.7,.5,.4,.4,.8,.7,.4,.4,.5,.7,.9,1.,.9,.7,.4,.2],
    "pensionato":[.0,.0,.0,.0,.1,.2,.5,.8,.9,1.,.9,.8,.7,.6,.7,.8,.8,.7,.6,.5,.4,.2,.1,.0],
    "notturno":  [.9,.8,.6,.4,.2,.1,.1,.1,.2,.3,.3,.4,.5,.5,.5,.5,.6,.7,.8,.9,1.,1.,1.,1.],
    # Partiti, comitati, testate: pubblicano in orario di redazione, con i
    # picchi della rassegna del mattino e del telegiornale della sera.
    "istituzionale":[0,0,0,0,0,0,.2,.6,1.,1.,.9,.8,.7,.8,.9,.9,.8,.7,.9,1.,.6,.3,.1,0],
}

# Marcatori di account non elettore. Vengono dalle bio che generi tu:
# "non e' una persona fisica", "account istituzionale", eccetera.
_NON_ELETTORE = re.compile(
    r"non (?:è|e') un(?:a)? (?:elettore|persona fisica)|"
    r"account (?:istituzionale|ufficiale)|\bcomitato\b|\btestata\b|"
    r"agenzia di stampa", re.IGNORECASE)
_USERNAME_IST = re.compile(
    r"partito|movimento|lega_|forza_|fratelli|comitato|ansa|repubblica|"
    r"corriere|stampa", re.IGNORECASE)


# Profondita' di lettura delle notizie, dedotta dal testo della biografia.
# I marcatori vengono dalle bio che generi tu ("Segue la politica in modo
# discontinuo", "interviene solo quando un tema lo tocca da vicino").
_ATTENZIONE_BASSA = re.compile(
    r"non segue la politica|disinteressat|si informa poco|"
    r"non si interessa di politica|lontan[oa] dalla politica", re.I)
_ATTENZIONE_ALTA = re.compile(
    r"segue (?:molto |assiduamente |con attenzione )|appassionat[oa] di politica|"
    r"legge i (?:giornali|quotidiani)|si informa (?:molto|quotidianamente)|"
    r"militant|attivist|molto informat", re.I)


def force_media_depth(agents: list[dict], depth: str) -> int:
    """
    Impone la stessa profondita' di lettura a tutti i cittadini.

    Serve a togliere un confondimento che il disegno osservativo non puo'
    togliere. Normalmente `media_depth` si deduce dalla biografia: chi legge
    l'articolo integrale e' un attivista, e un attivista scriverebbe post piu'
    lunghi e piu' documentati QUALUNQUE cosa gli venga mostrato. Trovare che
    chi legge l'integrale argomenta di piu' non distingue quindi l'effetto
    dell'esposizione da quello della personalita'.

    Forzando la stessa profondita' per tutti, e confrontando due run identici
    (stesso seed, stessa popolazione, stesso grafo) che differiscono SOLO per
    quanto testo ricevono, la differenza residua e' attribuibile
    all'esposizione e a nient'altro.

    Le fonti restano escluse: non leggono, pubblicano.
    """
    n = 0
    for a in agents:
        if a.get("is_source"):
            continue
        a["media_depth"] = depth
        n += 1
    return n


def media_depth(bio: str | None, institutional: bool = False) -> str:
    """
    Quanto a fondo un agente legge una notizia: integrale, titolo o nessuna.

    Tre livelli, non quattro. Il livello 'sommario' e' stato tolto: esisteva
    per i rilanci in stile social di una cartella separata, ma la simulazione
    usa solo l'archivio degli articoli integrali, da cui titolo e corpo si
    ricavano nello stesso file. Restava una categoria che nel codice esisteva
    e nei dati coincideva con 'titolo': meta' della popolazione vi finiva
    dentro e leggeva esattamente quanto chi era classificato 'titolo'.

    Le fonti e gli account istituzionali leggono tutto (e' il loro mestiere).
    Per i cittadini la profondita' viene dedotta dalla biografia; in assenza
    di marcatori il default e' 'titolo', il caso piu' comune.

    Chi ha attenzione minima ('nessuna') non riceve notizie nel feed: viene a
    sapere del referendum solo attraverso quello che ne dicono gli altri. E'
    il meccanismo di trasmissione di seconda mano, che con un'esposizione
    uniforme non poteva esistere.
    """
    if institutional:
        return "integrale"
    b = bio or ""
    if _ATTENZIONE_ALTA.search(b):
        return "integrale"
    if _ATTENZIONE_BASSA.search(b):
        return "nessuna"
    return "titolo"


def is_institutional(bio: str | None, username: str | None) -> bool:
    """
    Distingue chi partecipa al dibattito da chi ha una scheda elettorale.

    Un account di partito posta, viene letto e influenza, ma non vota. Finora
    l'unico filtro era `"ansa" in username`, quindi i partiti finivano nel
    conteggio del referendum come cittadini qualunque — contaminando proprio
    il numero che la tesi deve confrontare col risultato reale.
    """
    return bool(_NON_ELETTORE.search(bio or "")
                or _USERNAME_IST.search(username or ""))


def cronotipo_for(age: int | None, profession: str | None,
                  institutional: bool = False) -> str:
    if institutional:
        return "istituzionale"
    prof = (profession or "").lower()
    if "student" in prof or (age is not None and age < 25):
        return "studente"
    if "pension" in prof or (age is not None and age >= 67):
        return "pensionato"
    return "lavoratore"


def activation_prob(
    activity_hours: list[float], start_hour: int, span_hours: int, scale: float
) -> float:
    """
    Probabilita' che l'agente sia attivo in un tick che copre
    [start_hour, start_hour + span_hours).

    Questo rende superfluo il trucco dei tick coprimi con 24: non serve che i
    tick "ruotino" attraverso le ore per dare a tutti la stessa occasione,
    perche' la probabilita' e' gia' calcolata sulla sovrapposizione reale fra
    la finestra del tick e il profilo orario dell'agente. Ogni tick campiona
    ogni agente in modo corretto, sempre.

    Nota: piu' il tick e' lungo, piu' la media si appiattisce e i cronotipi si
    somigliano. A 24h tutti hanno la stessa probabilita' e la differenziazione
    demografica sparisce del tutto.
    """
    if span_hours >= 24:
        hours = range(24)
    else:
        hours = [(start_hour + i) % 24 for i in range(span_hours)]
    mean = sum(activity_hours[h] for h in hours) / max(1, len(list(hours)))
    return max(0.0, min(1.0, mean * scale))


REGIONI = [
    "Lombardia", "Lazio", "Campania", "Sicilia", "Veneto", "Emilia-Romagna",
    "Piemonte", "Puglia", "Toscana", "Calabria",
]
TITOLI = ["licenza media", "diploma", "laurea triennale", "laurea magistrale"]


def load_mirofish_profiles(path: str | Path) -> list[dict[str, Any]]:
    """Legge reddit_profiles.json di MiroFish e lo normalizza."""
    profiles = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for p in profiles:
        username = p.get("username", "")
        bio = p.get("persona") or p.get("bio") or ""
        ist = is_institutional(bio, username)
        is_ansa = "ansa" in username.lower()
        out.append({
            "agent_id": p.get("user_id"),
            "username": username,
            "static_bio": p.get("persona") or p.get("bio") or "",
            "profession": p.get("profession"),
            "age": p.get("age"),
            "region": p.get("region") or p.get("location"),
            "education": p.get("education"),
            "activity": float(p.get("activity_level", 0.35) or 0.35),
            "activity_hours": p.get("activity_hours") or CRONOTIPI[
                cronotipo_for(p.get("age"), p.get("profession"), ist)],
            "is_source": 1 if is_ansa else 0,
            "is_voter": 0 if (ist or is_ansa) else 1,
            "media_depth": media_depth(bio, ist or is_ansa),
            "attrs": {k: v for k, v in p.items()
                      if k not in {"user_id", "username", "persona", "bio"}},
        })
    return out


def synthetic(n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Popolazione finta per i test offline. NON usare per risultati."""
    rng = random.Random(seed)
    agents = []
    for i in range(1, n + 1):
        eta = rng.randint(18, 82)
        reg = rng.choice(REGIONI)
        tit = rng.choice(TITOLI)
        lean = rng.choice(["favorevole", "contrario", "indeciso"])
        agents.append({
            "agent_id": i,
            "username": f"utente_{i:03d}",
            "static_bio": (
                f"Ho {eta} anni, vivo in {reg}, titolo di studio: {tit}. "
                f"Sul referendum sulla separazione delle carriere sono {lean}. "
                f"Uso i social soprattutto la sera."
            ),
            "profession": rng.choice(["impiegato", "insegnante", "artigiano",
                                      "pensionato", "studente", "commerciante"]),
            "age": eta, "region": reg, "education": tit,
            "activity": round(rng.uniform(0.15, 0.6), 2),
            "activity_hours": CRONOTIPI[
                "notturno" if rng.random() < 0.12
                else cronotipo_for(eta, None)],
            "is_source": 0, "is_voter": 1,
            "media_depth": rng.choice(
                ["integrale", "titolo", "titolo", "titolo", "nessuna"]),
            "attrs": {"lean": lean},
        })
    # Due account istituzionali: partecipano al dibattito ma NON votano.
    # Servono anche perche' lo smoke test copra il prompt istituzionale, che
    # altrimenti resterebbe non esercitato in tutti i test offline.
    for j, nome in enumerate(("partito_esempio", "comitato_esempio"), start=n + 1):
        agents.append({
            "agent_id": j,
            "username": nome,
            "static_bio": ("Account istituzionale. Non e' una persona fisica e "
                           "non e' un elettore: prende posizione nel dibattito "
                           "pubblico sul referendum."),
            "profession": "organizzazione", "age": None,
            "region": None, "education": None,
            "activity": 0.5,
            "activity_hours": CRONOTIPI["istituzionale"],
            "is_source": 0, "is_voter": 0, "media_depth": "integrale",
            "attrs": {"lean": "istituzionale"},
        })
    return agents


def source_agent(agent_id: int = 0, username: str = "ANSA") -> dict[str, Any]:
    return {
        "agent_id": agent_id, "username": username,
        "static_bio": "Agenzia di stampa. Pubblica notizie, non commenta.",
        "profession": "agenzia di stampa", "activity": 0.0, "is_source": 1,
        "is_voter": 0, "media_depth": "integrale",
        "attrs": {},
    }


def build_follow_graph(
    agents: list[dict[str, Any]],
    seed: int = 0,
    avg_degree: int = 12,
    homophily: float = 0.6,
) -> list[tuple[int, int]]:
    """
    Grafo dei follow con omofilia su regione + orientamento.

    `homophily` e' la probabilita' che un arco venga scelto dentro il gruppo
    simile invece che a caso. E' un parametro della simulazione, quindi va
    dichiarato e variato nella sensitivity analysis: la struttura della rete
    influenza la diffusione tanto quanto il contenuto delle notizie.
    """
    rng = random.Random(f"follow|{seed}")
    people = [a for a in agents if not a.get("is_source")]
    by_key: dict[tuple, list[int]] = {}
    for a in people:
        key = (a.get("region"), (a.get("attrs") or {}).get("lean"))
        by_key.setdefault(key, []).append(a["agent_id"])

    ids = [a["agent_id"] for a in people]
    edges: set[tuple[int, int]] = set()
    for a in people:
        me = a["agent_id"]
        key = (a.get("region"), (a.get("attrs") or {}).get("lean"))
        similar = [x for x in by_key.get(key, []) if x != me]
        for _ in range(avg_degree):
            if similar and rng.random() < homophily:
                other = rng.choice(similar)
            else:
                other = rng.choice(ids)
            if other != me:
                edges.add((me, other))
    return sorted(edges)
