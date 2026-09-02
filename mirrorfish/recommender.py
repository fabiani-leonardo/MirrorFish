"""
Recommender a due stadi: generazione dei candidati, poi ranking.

E' l'architettura delle piattaforme reali, ed e' importante che i due stadi
siano separati: il grafo sociale decide COSA puo' arrivarti, il ranking decide
COSA vedi per primo. Se si fa similarita' semantica su tutti i post
dell'istanza, il grafo diventa decorativo e non stai piu' simulando un social
network ma un motore di ricerca.

--------------------------------------------------------------------------
AVVERTENZA METODOLOGICA — da riportare in tesi
--------------------------------------------------------------------------
La politica di ranking NON e' un dettaglio implementativo: e' un meccanismo
causale dell'esperimento. Un feed ordinato per similarita' semantica PRODUCE
camere d'eco per costruzione. Concludere "la simulazione mostra
polarizzazione" dopo aver scelto quella politica e' circolare.

Per questo le politiche sono intercambiabili e ce ne sono due di controllo:

  random   - ordine casuale. Se i risultati non cambiano rispetto a questo,
             il feed non sta facendo nulla e le conclusioni sul ruolo dei
             media non reggono.
  recency  - cronologico inverso. Null model: nessuna personalizzazione.

Le altre tre sono trattamenti:

  engagement - popolarita' (like + risposte). Nessun embedding richiesto.
  similarity - affinita' semantica col vettore di interesse dell'agente.
  hybrid     - combinazione pesata, la piu' vicina a una piattaforma reale.

Ogni condizione sperimentale andrebbe eseguita sotto almeno `random`,
`recency` e `hybrid`. Se la conclusione regge solo sotto una politica, quello
e' un risultato da dichiarare, non da nascondere.
"""

from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import dataclass

from .embeddings import cosine, from_blob

# (peso_recency, peso_engagement, peso_similarity)
POLICIES: dict[str, tuple[float, float, float]] = {
    "random":     (0.0, 0.0, 0.0),
    "recency":    (1.0, 0.0, 0.0),
    "engagement": (0.3, 1.0, 0.0),
    "similarity": (0.3, 0.0, 1.0),
    "hybrid":     (0.4, 0.4, 0.6),
}

NEEDS_EMBEDDINGS = {"similarity", "hybrid"}


@dataclass
class FeedItem:
    row: sqlite3.Row
    score: float
    recency: float
    engagement: float
    similarity: float


def _minmax(values: list[float]) -> list[float]:
    """Normalizza in [0,1] cosi' che i pesi siano comparabili fra loro."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class Recommender:
    def __init__(
        self,
        store,
        policy: str = "hybrid",
        *,
        out_of_network: float = 0.15,
        candidate_pool: int = 60,
        seed: int = 0,
    ):
        if policy not in POLICIES:
            raise ValueError(f"politica sconosciuta: {policy}. "
                             f"Disponibili: {sorted(POLICIES)}")
        self.store = store
        self.policy = policy
        self.weights = POLICIES[policy]
        self.out_of_network = out_of_network
        self.candidate_pool = candidate_pool
        self.seed = seed

    @property
    def needs_embeddings(self) -> bool:
        return self.policy in NEEDS_EMBEDDINGS

    # ------------------------------------------------- stadio 1: candidati #
    def candidates(self, agent_id: int, tick: int) -> list[sqlite3.Row]:
        """
        Post dei seguiti + una quota fuori-rete.

        La quota fuori-rete non e' un dettaglio: senza, la rete si frammenta in
        componenti che non si parlano mai e la polarizzazione e' garantita
        dalla topologia, non dalla dinamica. Le piattaforme reali iniettano
        contenuti fuori rete proprio per questo.
        """
        in_net = self.store.candidates_in_network(
            agent_id, tick, self.candidate_pool)
        n_out = int(self.candidate_pool * self.out_of_network)
        if n_out <= 0:
            return in_net
        seen = {r["post_id"] for r in in_net}
        rng = random.Random(f"{self.seed}|{agent_id}|{tick}|oon")
        out_net = [r for r in self.store.candidates_out_network(
            agent_id, tick, n_out * 3) if r["post_id"] not in seen]
        rng.shuffle(out_net)
        return in_net + out_net[:n_out]

    # --------------------------------------------------- stadio 2: ranking #
    def rank(
        self,
        agent_id: int,
        rows: list[sqlite3.Row],
        tick: int,
        interest_vec: list[float] | None,
        limit: int,
    ) -> list[FeedItem]:
        if not rows:
            return []

        w_rec, w_eng, w_sim = self.weights

        if self.policy == "random":
            rng = random.Random(f"{self.seed}|{agent_id}|{tick}|rand")
            picked = rows[:]
            rng.shuffle(picked)
            return [FeedItem(r, 0.0, 0.0, 0.0, 0.0) for r in picked[:limit]]

        # Recency: decadimento esponenziale, mezza vita 1 giorno di tick
        raw_rec = [math.exp(-(tick - r["tick"]) / 4.0) for r in rows]
        raw_eng = [math.log1p((r["likes"] or 0) + 2 * (r["replies"] or 0))
                   for r in rows]

        if w_sim > 0 and interest_vec:
            raw_sim = []
            for r in rows:
                blob = r["embedding"] if "embedding" in r.keys() else None
                raw_sim.append(cosine(interest_vec, from_blob(blob)) if blob else 0.0)
        else:
            raw_sim = [0.0] * len(rows)

        rec, eng, sim = _minmax(raw_rec), _minmax(raw_eng), _minmax(raw_sim)
        items = [
            FeedItem(r, w_rec * rec[i] + w_eng * eng[i] + w_sim * sim[i],
                     rec[i], eng[i], raw_sim[i])
            for i, r in enumerate(rows)
        ]
        items.sort(key=lambda it: (-it.score, -it.row["post_id"]))
        return items[:limit]

    # ------------------------------------------------------------- il feed #
    def feed(
        self,
        agent_id: int,
        tick: int,
        limit: int,
        news_slots: int,
        interest_vec: list[float] | None = None,
    ) -> list[sqlite3.Row]:
        """
        Slot garantiti alle notizie + resto dal ranking.

        Le notizie hanno una quota riservata perche' modellano la portata
        editoriale: un'agenzia arriva anche a chi non la segue e non la
        cerca. Lasciarle competere sul ranking le farebbe sparire sotto
        `similarity` (un agente non e' semanticamente "affine" a un lancio
        d'agenzia), il che eliminerebbe proprio la variabile indipendente
        dello studio.
        """
        news_slots = max(0, min(news_slots, limit))
        news = self.store.recent_news(tick, news_slots) if news_slots else []
        seen = {r["post_id"] for r in news}
        cands = [r for r in self.candidates(agent_id, tick)
                 if r["post_id"] not in seen]
        ranked = self.rank(agent_id, cands, tick, interest_vec,
                           limit - len(news))
        return list(news) + [it.row for it in ranked]


def explain_policies() -> str:
    lines = ["politica     recency  engag.  simil.  ruolo"]
    role = {"random": "CONTROLLO", "recency": "null model",
            "engagement": "trattamento", "similarity": "trattamento",
            "hybrid": "trattamento"}
    for name, (a, b, c) in POLICIES.items():
        lines.append(f"{name:<12} {a:>7.1f} {b:>7.1f} {c:>7.1f}  {role[name]}")
    return "\n".join(lines)
