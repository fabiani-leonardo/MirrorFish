#!/usr/bin/env python3
"""
Controllo pre-volo. Da eseguire PRIMA di ogni run lungo non sorvegliato.

    python scripts/preflight.py

Verifica che le correzioni critiche siano effettivamente nel codice in
esecuzione. Serve perche' e' gia' successo: una copia con `RateGate` vecchio
ha fatto partire 30 chiamate a raffica e preso tre 429. Su un run da otto ore
lo stesso errore brucia la quota della notte, e quella della squadra.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CHECKS: list[tuple[str, str]] = []


def ok(name: str, detail: str = "") -> None:
    CHECKS.append(("ok", f"{name}{'  ' + detail if detail else ''}"))


def fail(name: str, detail: str) -> None:
    CHECKS.append(("FAIL", f"{name}  {detail}"))


def warn(name: str, detail: str) -> None:
    CHECKS.append(("attenzione", f"{name}  {detail}"))


def main() -> None:
    from mirrorfish.config import DEFAULT_TOKEN_BUDGET, LLMConfig
    from mirrorfish.llm import OpenAICompatClient, RateGate

    # 1. Il gate deve ricevere il ritmo calcolato, non min_interval_s grezzo.
    src = inspect.getsource(OpenAICompatClient.__init__)
    if "pace_interval()" in src:
        ok("gate inizializzato con pace_interval()")
    else:
        fail("gate inizializzato con min_interval_s",
             "con min_interval_s=0 il rallentamento produce 0: raffica e 429")

    # 2. Il floor del rallentamento non deve poter valere 0.
    g = RateGate(0.0, rpm=8)
    asyncio.run(g.trip(1, "preflight"))
    if g.min_interval_s >= 1.0:
        ok("floor del rallentamento", f"{g.min_interval_s:.2f}s con base 0")
    else:
        fail("floor del rallentamento",
             f"{g.min_interval_s}s: il rallentamento azzera la spaziatura")

    # 3. Finestra scorrevole attiva.
    if "self._starts" in inspect.getsource(RateGate.acquire):
        ok("finestra scorrevole attiva")
    else:
        fail("finestra scorrevole assente", "spaziatura fissa: sul bordo del limite")

    # 4. Lettura degli header di quota.
    if hasattr(RateGate, "observe"):
        ok("legge x-ratelimit-*-remaining dalle risposte")
    else:
        warn("non legge gli header di quota",
             "reagisce ai 429 invece di prevenirli")

    # 5. chat_template_kwargs al top level, non dentro extra_body.
    src = inspect.getsource(OpenAICompatClient.complete)
    if 'payload["chat_template_kwargs"]' in src:
        ok("thinking disattivato correttamente")
    elif "extra_body" in src:
        fail("enable_thinking dentro extra_body",
             "convenzione dell'SDK: con httpx grezzo vLLM lo ignora")
    else:
        warn("nessuna disattivazione del thinking", "risposte piu' lunghe e lente")

    # 6. Budget coerenti col p95 misurato.
    for name, p95 in (("action", 136), ("reflection", 222), ("vote", 213)):
        b = DEFAULT_TOKEN_BUDGET.get(name, 0)
        if b >= p95 * 2:
            ok(f"budget {name}", f"{b} (p95 misurato {p95})")
        else:
            warn(f"budget {name} stretto", f"{b} contro p95 {p95}")

    # 7. Chiave presente e non finita nel repo.
    if os.environ.get("LLM_API_KEY"):
        ok("LLM_API_KEY nell'ambiente")
    else:
        fail("LLM_API_KEY assente", "esporta la chiave o riempi .env")

    root = Path(__file__).resolve().parent.parent
    gitignore = (root / ".gitignore")
    if gitignore.exists() and ".env" in gitignore.read_text():
        ok(".env in .gitignore")
    else:
        fail(".env NON in .gitignore", "rischio di committare la chiave")

    # 8. Ritmo effettivo.
    cfg = LLMConfig.from_env()
    ok("ritmo configurato",
       f"{cfg.pace_interval():.2f}s fra richieste "
       f"({cfg.requests_per_minute} req/min, concorrenza {cfg.concurrency})")

    width = max(len(m) for _, m in CHECKS) + 4
    print()
    for level, msg in CHECKS:
        mark = {"ok": "  ok  ", "FAIL": " FAIL ", "attenzione": "  !!  "}[level]
        print(f"[{mark}] {msg:<{width}}")

    bad = sum(1 for l, _ in CHECKS if l == "FAIL")
    print()
    if bad:
        print(f"{bad} controlli falliti. NON lanciare un run lungo cosi': "
              f"aggiorna il codice e ripeti.")
        sys.exit(1)
    print("Tutto a posto per un run non sorvegliato.")


if __name__ == "__main__":
    main()
