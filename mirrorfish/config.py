"""
Configurazione della simulazione.

Principio guida: *tutto* cio' che influenza il risultato di un run sta qui
dentro ed e' serializzabile. `fingerprint()` ne e' l'hash e viene scritto nel
run.db: due run con lo stesso fingerprint hanno gli stessi parametri.

Perche' la regola e' rigida. Prima di questa revisione la politica del feed e
la quota fuori-rete stavano FUORI da SimConfig, passate da riga di comando
direttamente al Recommender. Il fingerprint quindi non le copriva, e due run
con politiche di ranking diverse — cioe' con trattamenti sperimentali diversi
— risultavano indistinguibili. Se un parametro cambia il risultato, sta in
questa dataclass. Senza eccezioni.
"""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any


# --------------------------------------------------------------------------- #
# Budget di token
# --------------------------------------------------------------------------- #
# NOTA METODOLOGICA (per la tesi): omettere max_tokens fa si' che vLLM lo
# derivi da max_model_len - prompt_tokens (262.144 - 466 nel nostro caso). Qui
# ogni tipo di chiamata ha un budget esplicito, dimensionato sull'output che
# serve davvero.
#
#   azione agente : un post/reply e' ~280 caratteri -> ~120 token, + JSON
#   riflessione   : una frase -> ~60 token, + JSON
#   voto          : JSON con motivazione breve
#
# Se finish_reason == "length" ricorre spesso il budget e' troppo stretto: la
# telemetria in store.llm_call lo rende misurabile invece che opinabile.
# Massimi OSSERVATI su quattro run reali da 14 giorni (100 agenti ciascuno):
#   action      327, 312, 297, 348
#   reflection  404, 386, 512*, 512*
#   vote        289, 328, 512*, 242
# Gli asterischi sono TRONCAMENTI: il massimo coincide col budget, quindi il
# valore vero e' ignoto e maggiore. In expa-full un voto troncato e' diventato
# un errore di parsing, e il run finale ha 99 voti validi invece di 100.
#
# I budget sono generosi DI PROPOSITO, e la ragione e' misurata. Due probe
# indipendenti (2026-09-05 e 2026-09-08) hanno stabilito come il gateway
# contabilizza: PRENOTA max_tokens all'arrivo della richiesta e RIMBORSA
# (max_tokens - uso reale) alla conclusione della precedente. Il modello
# spiega tutte e sei le osservazioni:
#
#   req  max_tokens  prenota  rimborso prec.  netto  osservato
#     2         128      128              89     39         39
#     3         640      640              89    551        551
#     4         640      640             601     39         39
#
# A regime, quindi, si paga l'USO REALE: alzare max_tokens non costa quota, e
# la prenotazione e' solo un transitorio pari a max_tokens per il numero di
# chiamate in volo (con concorrenza 4 e budget 640, meno di 10.000 token
# trattenuti contro un tetto di 750.000).
#
# Conseguenza pratica: alzare max_tokens non costa nulla in quota, mentre
# abbassarlo costa risposte troncate — e una risposta troncata e' una chiamata
# sprecata piu' un dato perso. Quando il costo e' asimmetrico in questo modo,
# la scelta prudente e' il budget alto.
#
# Da riverificare se cambia il gateway:
#     python scripts/endpoint.py probe --n 2 --max-tokens 128 640 --sleep 20
DEFAULT_TOKEN_BUDGET = {
    "action": 640,
    "reflection": 768,
    "vote": 640,
}


# --------------------------------------------------------------------------- #
# Esposizione mediatica
# --------------------------------------------------------------------------- #
# Quanti slot del feed sono riservati alle notizie, secondo quanto a fondo
# l'agente legge (population.media_depth).
#
# Prima erano 2 su 8 per tutti, a ogni tick, per tutta la simulazione: il 25%
# della dieta informativa di ogni cittadino era filo d'agenzia, il che non
# assomiglia a nessun comportamento reale. Soprattutto `titolo` e `integrale`
# ricevevano la STESSA quantita' di notizie e differivano solo per lunghezza
# del testo: la profondita' di lettura non era una variabile di esposizione,
# era una variabile tipografica.
NEWS_SLOTS_BY_DEPTH = {
    "integrale": 2,   # attivista: apre l'articolo e ne legge il corpo
    "titolo": 1,      # moderato: scorre i titoli
    "nessuna": 0,     # disinteressato: non riceve notizie, ne sente parlare
}


@dataclass
class LLMConfig:
    base_url: str = "https://api.ailabroma3.it/v1"
    model: str = "lab-qwen36"
    api_key: str = ""                      # mai hardcoded: viene da env
    temperature: float = 0.7
    vote_temperature: float = 0.2
    reflection_temperature: float = 0.4

    # Qwen3 ha il thinking mode: disattivarlo e' la leva piu' grossa sulla
    # lunghezza dell'output, piu' di max_tokens. Se il server rifiuta il
    # parametro il client fa fallback automatico (vedi llm.py).
    disable_thinking: bool = True

    token_budget: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_TOKEN_BUDGET)
    )

    # --- controllo del carico ---------------------------------------------- #
    # Header osservati il 2026-09-08, dopo l'intervento sui limiti:
    #   x-ratelimit-api_key-limit-max_parallel_requests : 10
    #   x-ratelimit-team-limit-requests                 : 40   <- vincolante
    #   x-ratelimit-team-limit-tokens                   : 750000
    # I limiti per membro (team_member-*) sono stati RIMOSSI: esiste solo il
    # tetto di squadra, ed e' condiviso con colleghi e ricercatori.
    #
    # Il default e' 30, non 40, e la ragione non e' tecnica. Senza un limite
    # per membro, un run che gira a 40 richieste/minuto prende TUTTA la
    # capacita' della squadra per tutta la sua durata, e LiteLLM non ha code a
    # priorita': le chiamate degli altri non vengono messe in attesa, cadono
    # per timeout. Un run da 14 giorni simulati impiega 113 minuti a 30
    # richieste/minuto contro 84 a 40: mezz'ora in piu' a fronte di lasciare
    # un quarto della capacita' a chi lavora nello stesso momento. Alzare a 40
    # e' ragionevole di notte o dopo essersi accordati.
    #
    # La concorrenza segue dal ritmo, non lo determina (legge di Little): a 40
    # richieste/minuto parte una chiamata ogni 1,5 s, quindi con latenza p95
    # di 6 s ne bastano 4 in volo e con 10 s ne bastano 7. Il tetto di 10
    # chiamate parallele concesso dal professore e' su un totale di circa 16
    # per tutto il laboratorio: usarne 10 quando ne servono 4 non fa andare
    # piu' veloce e toglie slot agli altri. Ritarare sui dati veri con
    #     python scripts/endpoint.py concurrency runs/<run>/run.db --rpm 30
    requests_per_minute: float = 30.0
    concurrency: int = 4
    min_interval_s: float = 0.0   # 0 = derivato da requests_per_minute
    timeout_s: float = 180.0
    # Timeout di CONNESSIONE, separato da quello di lettura. Un endpoint
    # spento deve fallire in secondi, non in minuti: col valore unico a 180 s
    # e 6 tentativi una sola chiamata verso un server irraggiungibile
    # occupava fino a 20 minuti, e il run sembrava bloccato invece che rotto.
    connect_timeout_s: float = 5.0
    # Fallimenti consecutivi dopo i quali si interrompe tutto. Senza, il
    # motore registra gli errori e prosegue: 223 tick di chiamate fallite
    # producono un run vuoto che ha comunque l'aria di essere valido.
    circuit_breaker_failures: int = 25
    max_retries: int = 6
    backoff_base_s: float = 2.0
    # Quanto aspettare su 429 se il server non manda Retry-After. I limiti di
    # gateway hanno finestre da 60s: un backoff da pochi secondi non fa che
    # ritriggerare il limite.
    rate_limit_cooldown_s: float = 60.0

    def pace_interval(self) -> float:
        """Secondi fra due partenze. Lascia un 10% di margine sul limite."""
        if self.min_interval_s > 0:
            return self.min_interval_s
        return 60.0 / max(0.1, self.requests_per_minute) * 1.15

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMConfig":
        cfg = cls(
            base_url=os.environ.get("LLM_BASE_URL", cls.base_url),
            model=os.environ.get("LLM_MODEL_NAME", cls.model),
            api_key=os.environ.get("LLM_API_KEY", ""),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg


@dataclass
class SimConfig:
    """Parametri della simulazione veri e propri."""

    run_id: str = "run_dev"
    seed: int = 42

    # Finestra canonica: apertura della campagna referendaria -> giorno del voto.
    start_date: date = date(2025, 10, 30)
    end_date: date = date(2026, 3, 22)
    # Durata del tick in ORE. E' la leva principale sul costo: dimezzarla
    # raddoppia le chiamate. Vedi mirrorfish/plan_run.py per il compromesso
    # fra costo e differenziazione comportamentale.
    hours_per_tick: int = 8

    # --- feed --------------------------------------------------------------- #
    feed_size: int = 8                     # post mostrati per tick
    # Tetto agli slot notizia. Quelli EFFETTIVI dipendono dalla profondita' di
    # lettura dell'agente: vedi NEWS_SLOTS_BY_DEPTH e news_slots_for().
    news_slots: int = 2
    max_news_per_tick: int = 3
    # Politica di ranking. NON e' un dettaglio implementativo, e' il
    # trattamento sperimentale principale. Vedi recommender.py.
    recommender: str = "recency"
    out_of_network: float = 0.15

    # Azioni per agente per tick, in UNA sola chiamata. Con 1 scrivere e
    # reagire competono per lo stesso slot e scrivere vince, il che rende i
    # "mi piace" innaturalmente rari. Alzarlo non costa richieste in piu',
    # solo qualche token di output.
    max_actions: int = 3

    # Caratteri di articolo mostrati a chi legge l'integrale. Ero sceso a 900
    # temendo un tetto sui token al minuto; il probe ha mostrato che i token
    # non sono il vincolo (al ritmo di 25 richieste/minuto si consuma il 31%
    # del limite). Ripristinato a 1400: accorciare l'articolo indebolisce il
    # trattamento sperimentale, che e' proprio la variabile da misurare.
    max_body_chars: int = 1400

    # Attivita': probabilita' che un agente agisca in un dato tick
    base_activity: float = 0.35

    # Riflessione: ogni quanti tick gira il passo di memoria riflessiva
    reflection_every: int = 4
    max_notes_in_prompt: int = 8

    # Survey intermedie, oltre a baseline e final. None = solo i due estremi.
    # Costa N chiamate ogni volta, ma e' l'unico modo per avere la TRAIETTORIA
    # dell'opinione invece di due soli punti.
    survey_every: int | None = None

    # Se valorizzato, tutti i cittadini leggono a questa profondita' invece
    # di quella dedotta dalla biografia. E' una MANIPOLAZIONE: due run che
    # differiscono solo per questo campo isolano l'effetto dell'esposizione
    # dal confondimento con il tipo di persona.
    force_media_depth: str | None = None

    # Formulazione della domanda di voto: "ballot" riproduce il quesito e il
    # contenuto della legge come li vede un elettore vero, "minimal" nomina
    # solo il tema. Non e' una scelta di stile: e' cio' che determina se il
    # baseline misuri un'opinione o il prior del modello. Vedi survey.py.
    vote_question: str = "informed"

    # Controfattuale: da questo tick in poi si usa lo stream di news alternativo
    counterfactual_from_tick: int | None = None
    counterfactual_news_dir: str | None = None

    def total_ticks(self) -> int:
        days = (self.end_date - self.start_date).days + 1
        return max(1, int(days * 24 / self.hours_per_tick))

    def tick_hours(self, tick: int) -> tuple[int, int]:
        """Ora di inizio e durata del tick, nel giorno simulato."""
        return (tick * self.hours_per_tick) % 24, self.hours_per_tick

    def news_slots_for(self, media_depth: str) -> int:
        """Slot notizia effettivi per un agente, dato quanto a fondo legge."""
        return min(self.news_slots, NEWS_SLOTS_BY_DEPTH.get(media_depth, 1))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat()
        return d

    def fingerprint(self) -> str:
        """Hash stabile della config: identifica il run in modo univoco."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]
