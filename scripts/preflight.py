#!/usr/bin/env python3
"""
Controllo pre-volo. Da eseguire PRIMA di ogni run lungo non sorvegliato.

    python scripts/preflight.py
    python scripts/preflight.py --news start/notizieansa/notizie_referendum \\
                               --profiles start/A/reddit_profiles.json

Due famiglie di controlli, ed e' la seconda quella che e' costata di piu'.

INFRASTRUTTURA — che il run arrivi in fondo senza bruciare la quota. Serve
perche' e' gia' successo: una copia con `RateGate` vecchio ha fatto partire 30
chiamate a raffica e preso tre 429.

DISEGNO — che il run misuri cio' per cui e' stato scritto. Un run puo'
concludersi senza un errore e non valere niente: se gli articoli integrali
mancano, tutti gli agenti leggono lo stesso testo e la profondita' di lettura
smette di essere una variabile; se il prompt non nomina un'azione, quell'azione
non accadra' mai. Sono entrambi guasti gia' avvenuti, entrambi silenziosi, ed
entrambi scoperti solo leggendo i risultati a run finito.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
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


# --------------------------------------------------------------------------- #
def check_infrastruttura() -> None:
    from mirrorfish.config import DEFAULT_TOKEN_BUDGET, LLMConfig
    from mirrorfish.llm import OpenAICompatClient, RateGate

    src = inspect.getsource(OpenAICompatClient.__init__)
    if "pace_interval()" in src:
        ok("gate inizializzato con pace_interval()")
    else:
        fail("gate inizializzato con min_interval_s",
             "con min_interval_s=0 il rallentamento produce 0: raffica e 429")

    g = RateGate(0.0, rpm=8)
    asyncio.run(g.trip(1, "preflight"))
    if g.min_interval_s >= 1.0:
        ok("floor del rallentamento", f"{g.min_interval_s:.2f}s con base 0")
    else:
        fail("floor del rallentamento",
             f"{g.min_interval_s}s: il rallentamento azzera la spaziatura")

    if "self._starts" in inspect.getsource(RateGate.acquire):
        ok("finestra scorrevole attiva")
    else:
        fail("finestra scorrevole assente",
             "spaziatura fissa: sul bordo del limite")

    if hasattr(RateGate, "observe"):
        ok("legge x-ratelimit-*-remaining dalle risposte")
    else:
        warn("non legge gli header di quota",
             "reagisce ai 429 invece di prevenirli")

    src = inspect.getsource(OpenAICompatClient.complete)
    if 'payload["chat_template_kwargs"]' in src:
        ok("thinking disattivato correttamente")
    elif "extra_body" in src:
        fail("enable_thinking dentro extra_body",
             "convenzione dell'SDK: con httpx grezzo vLLM lo ignora")
    else:
        warn("nessuna disattivazione del thinking",
             "risposte piu' lunghe e lente")

    for name, p95 in (("action", 136), ("reflection", 222), ("vote", 213)):
        b = DEFAULT_TOKEN_BUDGET.get(name, 0)
        if b >= p95 * 2:
            ok(f"budget {name}", f"{b} (p95 misurato {p95})")
        else:
            warn(f"budget {name} stretto", f"{b} contro p95 {p95}")

    if os.environ.get("LLM_API_KEY"):
        ok("LLM_API_KEY nell'ambiente")
    else:
        fail("LLM_API_KEY assente", "esporta la chiave o riempi .env")

    root = Path(__file__).resolve().parent.parent
    gitignore = root / ".gitignore"
    if gitignore.exists() and ".env" in gitignore.read_text():
        ok(".env in .gitignore")
    else:
        fail(".env NON in .gitignore", "rischio di committare la chiave")

    cfg = LLMConfig.from_env()
    ok("ritmo configurato",
       f"{cfg.pace_interval():.2f}s fra richieste "
       f"({cfg.requests_per_minute} req/min, concorrenza {cfg.concurrency})")


# --------------------------------------------------------------------------- #
def check_prompt() -> None:
    """Ogni azione dichiarata deve essere raggiungibile dal modello."""
    from mirrorfish.agent import (SYSTEM_TEMPLATE, SYSTEM_INSTITUTIONAL,
                                  VALID_ACTIONS)

    testo = SYSTEM_TEMPLATE.format(username="x", bio="y", notes_block="",
                                   max_actions=3)
    assenti = [a for a in sorted(VALID_ACTIONS) if a not in testo]
    if assenti:
        fail("azioni non nominate nel prompt",
             f"{assenti}: il modello non le produrra' MAI "
             f"(e' il bug che ha dato 0 post su 312 tick)")
    else:
        ok("tutte le azioni nominate nel prompt", ", ".join(sorted(VALID_ACTIONS)))

    scrivibili = [a for a in ("POST", "REPLY") if f'"action": "{a}"' in testo]
    if len(scrivibili) == 2:
        ok("esempio JSON mostra sia POST sia REPLY")
    else:
        fail("esempio JSON parziale",
             f"mostra solo {scrivibili}: il modello copia lo schema")

    ist = SYSTEM_INSTITUTIONAL.format(username="x", bio="y", notes_block="",
                                      max_actions=3)
    if "Non sei un elettore" in ist:
        ok("prompt separato per gli account istituzionali")
    else:
        warn("istituzionali col prompt dei cittadini",
             "diranno 'voto NO' pur non avendo una scheda")

    # Lo StubLLM legge il numero massimo di azioni DAL PROMPT: se cambia la
    # formulazione, lo stub torna silenziosamente a una sola azione e tutti
    # gli smoke test smettono di esercitare il percorso multi-azione.
    import re
    if re.search(r"fino a 3 azioni", SYSTEM_TEMPLATE.format(
            username="x", bio="y", notes_block="", max_actions=3)):
        ok("formulazione attesa dallo StubLLM presente")
    else:
        warn("formulazione cambiata", "lo stub non riconoscera' max_actions")


def check_notizie(news_dir: str) -> None:
    """La profondita' di lettura sopravvive a questo archivio?"""
    from mirrorfish.news import load_news, NewsCoverageError
    try:
        items = load_news(news_dir)
    except NewsCoverageError as e:
        fail("copertura articoli integrale insufficiente",
             str(e).splitlines()[0])
        return
    except FileNotFoundError as e:
        fail("cartella notizie", str(e))
        return

    con_corpo = sum(1 for i in items if i.body)
    ok("archivio notizie",
       f"{len(items)} notizie, {con_corpo} con corpo integrale "
       f"({con_corpo / len(items):.0%})")

    # La verifica che conta: profondita' diverse -> testi diversi.
    campione = next((i for i in items if i.body), None)
    if campione is None:
        fail("nessun corpo integrale", "tutte le profondita' collassano in una")
        return
    testi = {d: campione.at_depth(d) for d in ("integrale", "titolo")}
    if len(set(testi.values())) == 1:
        fail("le profondita' di lettura collassano",
             "integrale e titolo danno lo stesso testo")
    else:
        ok("profondita' di lettura distinte",
           " / ".join(f"{d}={len(t)}c" for d, t in testi.items()))

    date_ordinate = [i.published for i in items]
    ok("finestra dell'archivio",
       f"{min(date_ordinate)} -> {max(date_ordinate)}")


def check_popolazione(profiles: str) -> None:
    from mirrorfish.config import SimConfig
    from mirrorfish.population import load_mirofish_profiles

    agents = load_mirofish_profiles(profiles)
    sim = SimConfig()

    elettori = sum(1 for a in agents if a.get("is_voter"))
    fonti = sum(1 for a in agents if a.get("is_source"))
    istituzionali = len(agents) - elettori - fonti
    ok("popolazione",
       f"{len(agents)} profili: {elettori} elettori, "
       f"{istituzionali} istituzionali, {fonti} fonti")
    if istituzionali == 0:
        warn("nessun account istituzionale rilevato",
             "partiti e testate finirebbero nel conteggio del referendum")

    depths: dict[str, int] = {}
    for a in agents:
        depths[a.get("media_depth", "titolo")] = \
            depths.get(a.get("media_depth", "titolo"), 0) + 1
    riga = ", ".join(f"{k}={v} ({sim.news_slots_for(k)} slot)"
                     for k, v in sorted(depths.items()))
    ok("profondita' di lettura nella popolazione", riga)
    if len(depths) == 1:
        fail("profondita' di lettura uniforme",
             "e' una costante, non una variabile: non potrai misurarne l'effetto")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--news", default=None,
                   help="cartella degli articoli integrali")
    p.add_argument("--profiles", default=None)
    a = p.parse_args()

    check_infrastruttura()
    check_prompt()
    if a.news:
        check_notizie(a.news)
    else:
        warn("notizie non verificate", "passa --news per controllare l'archivio")
    if a.profiles:
        check_popolazione(a.profiles)
    else:
        warn("popolazione non verificata", "passa --profiles per controllarla")

    width = max(len(m) for _, m in CHECKS) + 4
    print()
    for level, msg in CHECKS:
        mark = {"ok": "  ok  ", "FAIL": " FAIL ", "attenzione": "  !!  "}[level]
        print(f"[{mark}] {msg:<{width}}")

    bad = sum(1 for l, _ in CHECKS if l == "FAIL")
    print()
    if bad:
        print(f"{bad} controlli falliti. NON lanciare un run lungo cosi': "
              f"correggi e ripeti.")
        sys.exit(1)
    print("Tutto a posto per un run non sorvegliato.")


if __name__ == "__main__":
    main()
