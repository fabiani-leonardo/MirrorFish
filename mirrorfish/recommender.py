"""
Recommender a due stadi: generazione dei candidati, poi ranking.

E' l'architettura delle piattaforme reali, ed e' importante che i due stadi
siano separati: il grafo sociale decide COSA puo' arrivarti, il ranking decide
COSA vedi per primo.

--------------------------------------------------------------------------
AVVERTENZA METODOLOGICA — da riportare in tesi
--------------------------------------------------------------------------
La politica di ranking NON e' un dettaglio implementativo: e' un meccanismo
causale dell'esperimento. Un feed che ordina per affinita' PRODUCE camere
d'eco. Concludere "la simulazione mostra polarizzazione" dopo aver scelto
quella politica e' circolare.

Per questo le politiche sono intercambiabili e ce ne sono due di controllo:

  random     - ordine casuale. Se i risultati non cambiano rispetto a questo,
               il feed non sta facendo nulla e le conclusioni sul ruolo dei
               media non reggono.
  recency    - cronologico inverso. Null model: nessuna personalizzazione.

Le altre tre sono trattamenti:

  engagement - popolarita' del post (like + risposte). Uguale per tutti.
  affinity   - vicinanza *relazionale* fra chi legge e chi ha scritto.
  hybrid     - combinazione pesata, la piu' vicina a una piattaforma reale.

Ogni condizione sperimentale va eseguita sotto almeno `random`, `recency` e
`hybrid`. Se una conclusione regge solo sotto una politica, quello e' un
risultato da dichiarare, non da nascondere.

--------------------------------------------------------------------------
PERCHE' AFFINITA' RELAZIONALE E NON SIMILARITA' SEMANTICA
--------------------------------------------------------------------------
La versione precedente prevedeva `similarity` e `hybrid` calcolate col coseno
fra l'embedding del post e un "vettore di interesse" dell'agente. Erano
irraggiungibili: il ramo embedding non e' mai stato collegato al motore, il
vettore di interesse era un dizionario inizializzato vuoto e mai riempito, e
il runner terminava con un errore se si chiedevano quelle due politiche. Sono
state rimosse insieme a `embeddings.py` e alle quattro funzioni di `store.py`
che le servivano.

Al loro posto c'e' l'affinita' costruita sulle INTERAZIONI: conta chi ti ha
messo like, chi ti ha risposto, a chi hai messo like, a chi hai risposto. E'
SQL puro — nessun modello di embedding, nessun vettore da mantenere, nessuna
chiamata in piu' per ogni post — ma soprattutto e' piu' difendibile in tesi.
Se una camera d'eco si forma perche' continui a vedere chi ti ha risposto,
hai osservato un fenomeno emergente; se si forma perche' ordini per coseno,
hai osservato la tua funzione di ordinamento.
"""

from __future__ import annotations

import math
import random
import sqlite3
from dataclasses import dataclass

# (peso_recency, peso_engagement, peso_affinity)
POLICIES: dict[str, tuple[float, float, float]] = {
    "random":     (0.0, 0.0, 0.0),
    "recency":    (1.0, 0.0, 0.0),
    "engagement": (0.3, 1.0, 0.0),
    "affinity":   (0.3, 0.0, 1.0),
    "hybrid":     (0.4, 0.4, 0.6),
}

ROLE = {"random": "CONTROLLO", "recency": "null model",
        "engagement": "trattamento", "affinity": "trattamento",
        "hybrid": "trattamento"}


@dataclass
class FeedItem:
    row: sqlite3.Row
    score: float
    recency: float
    engagement: float
    affinity: float


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
        policy: str = "recency",
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
        # L'affinita' cambia poco dentro un tick e costa una query per agente:
        # la si calcola una volta per (agente, tick).
        self._aff_cache: dict[tuple[int, int], dict[int, float]] = {}

    @property
    def uses_affinity(self) -> bool:
        return self.weights[2] > 0

    # ------------------------------------------------- stadio 1: candidati #
    def candidates(self, agent_id: int, tick: int) -> list[sqlite3.Row]:
        """
        Post dei seguiti + una quota fuori-rete.

        La quota fuori-rete non e' un dettaglio: senza, la rete si frammenta
        in componenti che non si parlano mai e la polarizzazione e' garantita
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
    def _affinity(self, agent_id: int, tick: int) -> dict[int, float]:
        key = (agent_id, tick)
        if key not in self._aff_cache:
            self._aff_cache[key] = self.store.affinity_for(agent_id, tick)
            # La cache serve dentro un tick, non fra tick: senza questo,
            # su un run da 300 tick e 100 agenti tiene in RAM 30.000 dizionari.
            if len(self._aff_cache) > 4096:
                self._aff_cache = {key: self._aff_cache[key]}
        return self._aff_cache[key]

    def rank(
        self,
        agent_id: int,
        rows: list[sqlite3.Row],
        tick: int,
        limit: int,
    ) -> list[FeedItem]:
        if not rows:
            return []

        w_rec, w_eng, w_aff = self.weights

        if self.policy == "random":
            rng = random.Random(f"{self.seed}|{agent_id}|{tick}|rand")
            picked = rows[:]
            rng.shuffle(picked)
            return [FeedItem(r, 0.0, 0.0, 0.0, 0.0) for r in picked[:limit]]

        # Recency: decadimento esponenziale, mezza vita ~3 tick
        raw_rec = [math.exp(-(tick - r["tick"]) / 4.0) for r in rows]
        raw_eng = [math.log1p((r["likes"] or 0) + 2 * (r["replies"] or 0))
                   for r in rows]

        if w_aff > 0:
            aff = self._affinity(agent_id, tick)
            raw_aff = [math.log1p(aff.get(r["author_id"], 0.0)) for r in rows]
        else:
            raw_aff = [0.0] * len(rows)

        rec, eng, af = _minmax(raw_rec), _minmax(raw_eng), _minmax(raw_aff)
        items = [
            FeedItem(r, w_rec * rec[i] + w_eng * eng[i] + w_aff * af[i],
                     rec[i], eng[i], raw_aff[i])
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
    ) -> list[sqlite3.Row]:
        """
        Slot garantiti alle notizie + resto dal ranking.

        Le notizie hanno una quota riservata perche' modellano la portata
        editoriale: un'agenzia arriva anche a chi non la segue e non la cerca.
        Lasciarle competere sul ranking le farebbe sparire sotto `affinity`
        (con un'agenzia non si hanno interazioni reciproche), il che
        eliminerebbe proprio la variabile indipendente dello studio.

        Quanti siano quegli slot dipende dall'agente: vedi
        SimConfig.news_slots_for e config.NEWS_SLOTS_BY_DEPTH.
        """
        news_slots = max(0, min(news_slots, limit))
        news = self.store.recent_news(tick, news_slots) if news_slots else []
        seen = {r["post_id"] for r in news}
        cands = [r for r in self.candidates(agent_id, tick)
                 if r["post_id"] not in seen]
        ranked = self.rank(agent_id, cands, tick, limit - len(news))
        return list(news) + [it.row for it in ranked]


def explain_policies() -> str:
    """Tabella delle politiche. La stampa `run.py --explain-policies`."""
    lines = ["politica     recency  engag.  affin.  ruolo"]
    for name, (a, b, c) in POLICIES.items():
        lines.append(f"{name:<12} {a:>7.1f} {b:>7.1f} {c:>7.1f}  {ROLE[name]}")
    return "\n".join(lines)
