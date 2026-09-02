#!/usr/bin/env python3
"""
Una richiesta sola, e stampa TUTTO quello che il server dice.

Serve a rispondere a una domanda che il benchmark non risolve: quale limite
sta scattando? Richieste al minuto? Token al minuto? Quota della chiave?
Concorrenza massima? Il 429 lo dice quasi sempre, nel body o negli header,
ma il codice di produzione lo scarta.

    export LLM_API_KEY=...
    python scripts/probe.py
    python scripts/probe.py --n 3 --sleep 20   # verifica la finestra temporale

Il risultato di questo script e' l'informazione da girare al professore: non
"ho dei 429", ma "il limite e' X richieste/minuto e la finestra si ricarica
in Y secondi".
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mirrorfish.config import LLMConfig  # noqa: E402

INTERESTING = ("retry-after", "x-ratelimit", "ratelimit", "x-request-id",
               "x-envoy", "x-kong", "server", "date")


async def probe(cfg: LLMConfig, max_tokens: int, no_think: bool, idx: int) -> bool:
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": "Rispondi solo con JSON."},
            {"role": "user", "content": 'Rispondi {"ok": true} e basta.'},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if no_think:
        # Top level, non dentro extra_body: quella e' una convenzione
        # dell'SDK OpenAI, non un campo dell'API.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    async with httpx.AsyncClient(
        base_url=cfg.base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {cfg.api_key}",
                 "Content-Type": "application/json"},
        timeout=httpx.Timeout(120.0),
    ) as c:
        t0 = time.perf_counter()
        r = await c.post("/chat/completions", json=payload)
        dt = (time.perf_counter() - t0) * 1000

    print(f"\n--- richiesta {idx} --------------------------------------------")
    print(f"HTTP {r.status_code}   {dt:.0f} ms")

    shown = {k: v for k, v in r.headers.items()
             if any(k.lower().startswith(p) or p in k.lower() for p in INTERESTING)}
    if shown:
        print("header rilevanti:")
        for k, v in sorted(shown.items()):
            print(f"    {k}: {v}")
    else:
        print("header rilevanti: nessuno (il gateway non espone info di quota)")

    # Quanto budget di token viene scalato? Se il calo dipende da max_tokens
    # e non dai token realmente consumati, il gateway pre-alloca sulla stima:
    # e' il meccanismo con cui una richiesta da 261.678 token esauriva da sola
    # l'intera finestra da 100.000 token/minuto.
    rem = r.headers.get("x-ratelimit-team_member-remaining-tokens")
    if rem is not None:
        try:
            probe.prev_remaining  # type: ignore[attr-defined]
        except AttributeError:
            probe.prev_remaining = None  # type: ignore[attr-defined]
        if probe.prev_remaining is not None:  # type: ignore[attr-defined]
            drop = probe.prev_remaining - float(rem)  # type: ignore[attr-defined]
            print(f"budget token  : -{drop:.0f} scalati (max_tokens={max_tokens})")
        probe.prev_remaining = float(rem)  # type: ignore[attr-defined]

    if r.status_code == 200:
        d = r.json()
        u = d.get("usage") or {}
        ch = d["choices"][0]
        content = (ch.get("message", {}).get("content") or "")
        reasoning = ch.get("message", {}).get("reasoning_content") or ""
        print(f"finish_reason : {ch.get('finish_reason')}")
        print(f"token         : prompt={u.get('prompt_tokens')} "
              f"completion={u.get('completion_tokens')} (budget {max_tokens})")
        print(f"contenuto     : {content[:160]!r}")
        if reasoning:
            print(f"ATTENZIONE: reasoning_content presente ({len(reasoning)} car.) "
                  f"-> il thinking mode e' ANCORA ATTIVO nonostante "
                  f"enable_thinking=False")
        if "<think>" in content:
            print("ATTENZIONE: <think> nel contenuto -> thinking mode ATTIVO")
        return True

    print(f"body:\n{r.text[:800]}")
    return False


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="quante richieste in serie")
    ap.add_argument("--sleep", type=float, default=0.0, help="pausa fra una e l'altra")
    ap.add_argument("--max-tokens", type=int, nargs="+", default=[256],
                    help="uno o piu' valori: vengono provati NELLA STESSA "
                         "invocazione, cosi' il confronto del budget scalato "
                         "e' fatto sulla stessa baseline")
    ap.add_argument("--thinking-on", action="store_true",
                    help="NON disattivare il thinking, per confronto")
    args = ap.parse_args()

    if not os.environ.get("LLM_API_KEY"):
        raise SystemExit("Serve LLM_API_KEY nell'ambiente.")

    cfg = LLMConfig.from_env()
    print(f"endpoint : {cfg.base_url}")
    print(f"modello  : {cfg.model}")
    print(f"chiave   : ...{cfg.api_key[-4:]}  (lunghezza {len(cfg.api_key)})")

    ok = 0
    total = args.n * len(args.max_tokens)
    i = 0
    for mt in args.max_tokens:
        for _ in range(args.n):
            i += 1
            if await probe(cfg, mt, not args.thinking_on, i):
                ok += 1
            if args.sleep and i < total:
                print(f"\n(pausa {args.sleep}s)")
                await asyncio.sleep(args.sleep)

    print(f"\n=== {ok}/{args.n} riuscite ===")
    if ok < args.n:
        print("Se falliscono con 429 anche a una richiesta alla volta, il limite")
        print("non e' la capacita' della GPU ma una quota del gateway o della")
        print("chiave. Manda al professore gli header e il body qui sopra.")


if __name__ == "__main__":
    asyncio.run(main())
