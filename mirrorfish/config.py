"""
Configurazione della simulazione.

Principio guida: *tutto* ciò che influenza il risultato di un run sta qui
dentro ed e' serializzabile. Il file di config viene copiato nel manifest
del run, cosi' ogni risultato in tesi e' riconducibile ai suoi parametri.
"""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Budget di token
# --------------------------------------------------------------------------- #
# NOTA METODOLOGICA (importante per la tesi e per il professore):
# omettere max_tokens fa si' che vLLM lo derivi da max_model_len - prompt_tokens
# (262.144 - 466 = 261.678 nel nostro caso). Qui ogni tipo di chiamata ha un
# budget esplicito, dimensionato sull'output che ci serve davvero.
#
#   azione agente : un post/reply e' ~280 caratteri -> ~120 token, + JSON
#   riflessione   : una frase -> ~60 token, + JSON  (1200 e' il valore che
#                   usavi: generoso perche' Qwen3 puo' "pensare" prima)
#   voto          : JSON con motivazione breve
#
# Se un finish_reason == "length" ricorre spesso, il budget e' troppo stretto:
# la telemetria in store.llm_call lo rende misurabile invece che opinabile.
DEFAULT_TOKEN_BUDGET = {
    "action": 384,
    "reflection": 512,
    "vote": 300,
}


@dataclass
class LLMConfig:
    base_url: str = "https://api.ailabroma3.it/v1"
    model: str = "lab-qwen36"
    embed_model: str = "lab-embed"
    api_key: str = ""                      # mai hardcoded: viene da env
    temperature: float = 0.7
    vote_temperature: float = 0.2
    reflection_temperature: float = 0.4

    # Qwen3 ha il thinking mode: disattivarlo e' la leva piu' grossa sulla
    # lunghezza dell'output, piu' di max_tokens. Se il server rifiuta il
    # parametro, il client fa fallback automatico (vedi llm.py).
    disable_thinking: bool = True

    token_budget: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_TOKEN_BUDGET)
    )

    # --- controllo del carico ---------------------------------------------- #
    # concurrency: quante richieste in volo contemporaneamente.
    # min_interval_s: distanza minima fra due partenze -> spalma il burst.
    # Con 2 GPU e altri utenti sul modello, 6 e 0.15 sono un punto di partenza
    # prudente. Da tarare con scripts/diagnose_llm.py.
    concurrency: int = 6
    min_interval_s: float = 0.15
    timeout_s: float = 180.0
    max_retries: int = 6
    backoff_base_s: float = 2.0
    # Quanto aspettare su 429 se il server non manda Retry-After.
    # I limiti di gateway hanno tipicamente finestre da 60s: un backoff da
    # pochi secondi non fa che ritriggerare il limite.
    rate_limit_cooldown_s: float = 60.0

    @classmethod
    def from_env(cls, **overrides: Any) -> "LLMConfig":
        cfg = cls(
            base_url=os.environ.get("LLM_BASE_URL", cls.base_url),
            model=os.environ.get("LLM_MODEL_NAME", cls.model),
            embed_model=os.environ.get("LLM_EMBED_MODEL", cls.embed_model),
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

    start_date: date = date(2026, 3, 1)
    end_date: date = date(2026, 3, 21)
    ticks_per_day: int = 4                 # 4 tick = mattina/pomeriggio/sera/notte

    # Popolazione
    n_agents: int | None = None            # None = tutti quelli nel file

    # Feed
    feed_size: int = 8                     # post mostrati per tick
    news_slots: int = 2                    # quanti di quegli slot sono notizie
    max_news_per_tick: int = 3
    feed_recency_bias: float = 0.7         # 0 = casuale, 1 = solo i piu' recenti

    # Attivita': probabilita' che un agente agisca in un dato tick
    base_activity: float = 0.35

    # Riflessione: ogni quanti tick gira il passo di memoria riflessiva
    reflection_every: int = 4
    max_notes_in_prompt: int = 8

    # Survey
    survey_every: int | None = None        # None = solo alla fine

    # Controfattuale: da questo tick in poi si usa lo stream di news alternativo
    counterfactual_from_tick: int | None = None
    counterfactual_news_dir: str | None = None

    def total_ticks(self) -> int:
        days = (self.end_date - self.start_date).days + 1
        return days * self.ticks_per_day

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat()
        d["end_date"] = self.end_date.isoformat()
        return d

    def fingerprint(self) -> str:
        """Hash stabile della config: identifica il run in modo univoco."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @classmethod
    def from_json(cls, path: str | Path) -> "SimConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for k in ("start_date", "end_date"):
            if k in raw and isinstance(raw[k], str):
                raw[k] = date.fromisoformat(raw[k])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in raw.items() if k in known})
