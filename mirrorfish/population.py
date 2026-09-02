"""
Caricamento della popolazione.

`load_mirofish_profiles` legge il formato che gia' produci
(reddit_profiles.json), cosi' la popolazione ISTAT/YouTrend che hai gia'
generato si riusa tale e quale: non rigeneriamo niente, e la baseline resta
confrontabile con i run vecchi.

`synthetic` serve solo per lo smoke test offline.
"""

from __future__ import annotations

import csv
import json
import random
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
}


def cronotipo_for(age: int | None, profession: str | None) -> str:
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
                cronotipo_for(p.get("age"), p.get("profession"))],
            "is_source": 1 if "ansa" in username.lower() else 0,
            "attrs": {k: v for k, v in p.items()
                      if k not in {"user_id", "username", "persona", "bio"}},
        })
    return out


def load_twitter_csv(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


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
            "is_source": 0,
            "attrs": {"lean": lean},
        })
    return agents


def source_agent(agent_id: int = 0, username: str = "ANSA") -> dict[str, Any]:
    return {
        "agent_id": agent_id, "username": username,
        "static_bio": "Agenzia di stampa. Pubblica notizie, non commenta.",
        "profession": "agenzia di stampa", "activity": 0.0, "is_source": 1,
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