#!/usr/bin/env python3
"""
Tutto cio' che riguarda lo stato e la velocita' dell'endpoint, in un file solo.

    python scripts/endpoint.py check                    # e' su? sono bloccato?
    python scripts/endpoint.py probe --n 3 --sleep 20   # cosa dice il server
    python scripts/endpoint.py ceiling                  # quanto posso spingere
    python scripts/endpoint.py concurrency runs/x/run.db --rpm 40

Sostituisce check_endpoint.py, probe.py, find_ceiling.py e pick_concurrency.py,
che facevano lo stesso mestiere in quattro file con quattro invocazioni diverse
da ricordare.

    check       diagnosi a strati: DNS, TCP, TLS, HTTP. Distingue "e' giu'" da
                "sono bloccato", che e' la domanda che serve poter rispondere
                al professore con precisione:
                  DNS non risolve  -> rete tua o dominio rimosso
                  TCP non connette -> server spento o firewall. NON sei
                                      bloccato: un blocco risponde, e risponde
                                      401 o 403
                  TLS fallisce     -> certificato o proxy
                  401 / 403        -> QUESTO e' un blocco
                  429              -> quota esaurita, non un blocco
    probe       una richiesta e stampa TUTTO quello che il server dice. Serve a
                capire QUALE limite scatta: richieste/minuto, token/minuto,
                concorrenza. Il 429 lo dice quasi sempre negli header, ma il
                codice di produzione li scarta. Con piu' valori di --max-tokens
                mostra anche quanto budget viene scalato per richiesta: se il
                calo dipende da max_tokens e non dai token consumati, il
                gateway pre-alloca sulla stima.
    ceiling     parte basso e sale finche' non incassa un 429. Serve perche' il
                tetto effettivo e'
                    min(api_key_parallel, team_member_rpm, team_rpm - altri)
                e l'ultimo termine non e' osservabile localmente.
    concurrency non misura niente: calcola. Con R richieste/minuto il
                limitatore rilascia una partenza ogni 60/R secondi; se una
                chiamata dura L secondi, le chiamate in volo sono ~L/(60/R).
                La concorrenza serve solo a NON restare sotto la quota.
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import math
import os
import socket
import sqlite3
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirrorfish.config import LLMConfig            # noqa: E402

SYS = "Rispondi solo con JSON."
USR = 'Rispondi {"ok": true} e basta.'
INTERESTING = ("retry-after", "x-ratelimit", "ratelimit", "x-request-id",
               "x-envoy", "x-kong", "server", "date")


# --------------------------------------------------------------- check ----- #
def cmd_check(a: argparse.Namespace) -> int:
    url = a.url or os.environ.get("LLM_BASE_URL", LLMConfig.base_url)
    u = urlparse(url)
    host, port = u.hostname, u.port or (443 if u.scheme == "https" else 80)
    print(f"Endpoint : {url}")
    print(f"Host     : {host}:{port}")
    print(f"Ora      : {datetime.now():%Y-%m-%d %H:%M:%S}")

    print("\n[1] Risoluzione DNS")
    try:
        t0 = time.perf_counter()
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
        ips = sorted({i[4][0] for i in infos})
        print(f"    OK ({(time.perf_counter()-t0)*1000:.0f} ms): {', '.join(ips)}")
    except socket.gaierror as e:
        print(f"    FALLITO: {e}")
        print("\n    Il nome non si risolve: problema di rete o di DNS, non un")
        print("    blocco del laboratorio. Prova da un'altra rete.")
        return 1

    print("\n[2] Connessione TCP")
    reachable = False
    for ip in ips:
        t0 = time.perf_counter()
        try:
            with socket.create_connection((ip, port), timeout=a.timeout):
                print(f"    OK  {ip}:{port} ({(time.perf_counter()-t0)*1000:.0f} ms)")
                reachable = True
        except socket.timeout:
            print(f"    TIMEOUT  {ip}:{port} dopo {a.timeout}s")
        except OSError as e:
            print(f"    RIFIUTATO  {ip}:{port}: {e}")
    if not reachable:
        print("\n    Il DNS risolve ma nessun indirizzo accetta connessioni:")
        print("    server spento, in manutenzione, o firewall che scarta i")
        print("    pacchetti senza rispondere.")
        print("\n    NON e' un blocco della chiave: un blocco risponde, e")
        print("    risponde 401 o 403. Qui non arriva nessuna risposta.")
        return 2

    if u.scheme == "https":
        print("\n[3] Handshake TLS")
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((host, port), timeout=a.timeout) as s:
                with ctx.wrap_socket(s, server_hostname=host) as ss:
                    cert = ss.getpeercert()
                    print(f"    OK, {ss.version()}")
                    if cert and "notAfter" in cert:
                        print(f"    certificato valido fino al {cert['notAfter']}")
        except Exception as e:
            print(f"    FALLITO: {type(e).__name__}: {e}")
            print("    Problema di certificato o proxy TLS.")
            return 3

    print("\n[4] Richiesta HTTP")
    key = os.environ.get("LLM_API_KEY")
    if not key:
        print("    LLM_API_KEY non impostata: salto la prova autenticata.")
        print("    I livelli di rete sono comunque a posto.")
        return 0
    import httpx
    try:
        t0 = time.perf_counter()
        r = httpx.post(
            url.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
            json={"model": os.environ.get("LLM_MODEL_NAME", LLMConfig.model),
                  "messages": [{"role": "user", "content": "Rispondi: ok"}],
                  "max_tokens": 16},
            timeout=a.timeout * 4,
        )
        print(f"    HTTP {r.status_code} ({(time.perf_counter()-t0)*1000:.0f} ms)")
        if r.status_code == 200:
            d = r.json()
            print("    risposta: "
                  f"{d['choices'][0]['message'].get('content','')[:60]!r}")
            for k, v in sorted(r.headers.items()):
                if "ratelimit" in k.lower():
                    print(f"    {k}: {v}")
            print("\n    Tutto funzionante.")
        elif r.status_code in (401, 403):
            print(f"    corpo: {r.text[:300]}")
            print("\n    QUESTO e' un blocco: chiave revocata o non autorizzata.")
        elif r.status_code == 429:
            print(f"    corpo: {r.text[:300]}")
            print("\n    Quota esaurita, non un blocco. Riprova fra un minuto.")
        else:
            print(f"    corpo: {r.text[:300]}")
    except Exception as e:
        print(f"    FALLITO: {type(e).__name__}: {e}")
        print("    Rete a posto ma l'applicazione non risponde: il gateway e'")
        print("    su, il modello dietro probabilmente no.")
        return 4
    return 0


# --------------------------------------------------------------- probe ----- #
async def _one_probe(cfg: LLMConfig, max_tokens: int, no_think: bool,
                     idx: int, state: dict) -> bool:
    import httpx
    payload = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": SYS},
                     {"role": "user", "content": USR}],
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

    print(f"\n--- richiesta {idx} ---------------------------------------")
    print(f"HTTP {r.status_code}   {dt:.0f} ms")

    shown = {k: v for k, v in r.headers.items()
             if any(p in k.lower() for p in INTERESTING)}
    if shown:
        print("header rilevanti:")
        for k, v in sorted(shown.items()):
            print(f"    {k}: {v}")
    else:
        print("header rilevanti: nessuno (il gateway non espone info di quota)")

    rem = r.headers.get("x-ratelimit-team_member-remaining-tokens")
    if rem is not None:
        if state.get("prev") is not None:
            print(f"budget token  : -{state['prev'] - float(rem):.0f} scalati "
                  f"(max_tokens={max_tokens})")
        state["prev"] = float(rem)

    if r.status_code == 200:
        d = r.json()
        u = d.get("usage") or {}
        ch = d["choices"][0]
        content = ch.get("message", {}).get("content") or ""
        reasoning = ch.get("message", {}).get("reasoning_content") or ""
        print(f"finish_reason : {ch.get('finish_reason')}")
        print(f"token         : prompt={u.get('prompt_tokens')} "
              f"completion={u.get('completion_tokens')} (budget {max_tokens})")
        print(f"contenuto     : {content[:160]!r}")
        if reasoning:
            print(f"ATTENZIONE: reasoning_content presente ({len(reasoning)} car.)"
                  f" -> thinking mode ANCORA ATTIVO nonostante enable_thinking=False")
        if "<think>" in content:
            print("ATTENZIONE: <think> nel contenuto -> thinking mode ATTIVO")
        return True

    print(f"body:\n{r.text[:800]}")
    return False


async def cmd_probe(a: argparse.Namespace) -> int:
    if not os.environ.get("LLM_API_KEY"):
        print("Serve LLM_API_KEY nell'ambiente.")
        return 1
    cfg = LLMConfig.from_env()
    print(f"endpoint : {cfg.base_url}")
    print(f"modello  : {cfg.model}")
    print(f"chiave   : ...{cfg.api_key[-4:]}  (lunghezza {len(cfg.api_key)})")

    state: dict = {"prev": None}
    ok = i = 0
    total = a.n * len(a.max_tokens)
    for mt in a.max_tokens:
        for _ in range(a.n):
            i += 1
            if await _one_probe(cfg, mt, not a.thinking_on, i, state):
                ok += 1
            if a.sleep and i < total:
                print(f"\n(pausa {a.sleep}s)")
                await asyncio.sleep(a.sleep)

    print(f"\n=== {ok}/{total} riuscite ===")
    if ok < total:
        print("Se falliscono con 429 anche a una richiesta alla volta, il limite")
        print("non e' la capacita' della GPU ma una quota del gateway o della")
        print("chiave. Manda al professore gli header e il body qui sopra.")
    return 0


# ------------------------------------------------------------- ceiling ----- #
async def _burst(rpm: float, n: int) -> dict:
    from mirrorfish.llm import OpenAICompatClient
    cfg = LLMConfig.from_env(requests_per_minute=rpm, concurrency=5)
    cfg.max_retries = 1
    client = OpenAICompatClient(cfg)
    try:
        t0 = time.perf_counter()
        res = await asyncio.gather(*[
            client.complete(SYS, USR, max_tokens=64, temperature=0.2)
            for _ in range(n)])
        wall = time.perf_counter() - t0
        good = [r for r in res if not r.error]
        return {"rpm_target": rpm, "ok": len(good), "err": len(res) - len(good),
                "rate": len(good) / wall * 60 if wall else 0,
                "scope": client.gate.observed_scope,
                "remaining": client.gate.observed_remaining,
                "first_err": next((r.error for r in res if r.error), None)}
    finally:
        await client.aclose()


async def cmd_ceiling(a: argparse.Namespace) -> int:
    if not os.environ.get("LLM_API_KEY"):
        print("Serve LLM_API_KEY nell'ambiente.")
        return 1
    print(f"\n{a.requests} richieste per gradino, pausa {a.cooldown}s "
          f"(~{len(a.steps) * a.cooldown / 60:.0f} min in tutto)\n")
    print(f"  {'target':>7}{'ok':>5}{'err':>5}{'misurato':>11}"
          f"{'limite piu stretto':>22}{'residuo':>9}")
    print("  " + "-" * 60)

    best = 0.0
    for rpm in a.steps:
        d = await _burst(rpm, a.requests)
        print(f"  {rpm:>7.0f}{d['ok']:>5}{d['err']:>5}{d['rate']:>10.1f}/m"
              f"{(d['scope'] or '-'):>22}{str(d['remaining']):>9}")
        if d["err"]:
            print(f"      {str(d['first_err'])[:110]}")
            break
        best = max(best, d["rate"])
        await asyncio.sleep(a.cooldown)

    safe = best * 0.8
    print(f"\n  Massimo sostenuto senza rifiuti: {best:.0f} req/min")
    print(f"  Da usare in produzione (80%, margine per i colleghi): --rpm {safe:.0f}")
    print("\n  Se 'limite piu stretto' dice 'team', il tetto e' condiviso:")
    print("  quello che misuri ora cambia quando qualcun altro lancia un run.")
    for calls, label in ((4012, "run da 30 giorni"), (20800, "run da 150 giorni")):
        print(f"  {label:<22} {calls / max(safe, 1) / 60:>5.1f} h a --rpm {safe:.0f}")
    return 0


# --------------------------------------------------------- concurrency ----- #
def cmd_concurrency(a: argparse.Namespace) -> int:
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    rows = c.execute("SELECT purpose, latency_ms FROM llm_call "
                     "WHERE error IS NULL AND latency_ms > 0").fetchall()
    c.close()
    if not rows:
        print("Nessuna latenza registrata in questo run.")
        return 1

    per: dict[str, list[float]] = {}
    for r in rows:
        per.setdefault(r["purpose"], []).append(r["latency_ms"])

    interval = 60.0 / a.rpm
    print(f"\nQuota richiesta : {a.rpm:.0f} req/min "
          f"-> una partenza ogni {interval:.2f} s\n")
    print(f"  {'tipo':<12}{'n':>6}{'lat med':>10}{'lat p95':>10}{'in volo':>10}")
    print("  " + "-" * 48)

    worst = 0.0
    for purpose, lats in sorted(per.items()):
        lats.sort()
        med = lats[len(lats) // 2] / 1000
        p95 = lats[max(0, int(len(lats) * 0.95) - 1)] / 1000
        inflight = p95 / interval
        worst = max(worst, inflight)
        print(f"  {purpose:<12}{len(lats):>6}{med:>9.2f}s{p95:>9.2f}s"
              f"{inflight:>10.1f}")

    needed = max(1, math.ceil(worst * a.margin))
    print(f"\n  Chiamate in volo nel caso peggiore : {worst:.1f}")
    print(f"  Con margine {a.margin}x                  : {needed}")
    print(f"\n  --> --concurrency {needed}")
    print("\n  Oltre questo valore il ritmo non sale: lo fissa la quota.")
    print("  Sotto, invece, non riesci a saturarla e il run dura di piu'.")
    print(f"\n  Questo run ({len(rows):,} chiamate) a {a.rpm:.0f} req/min: "
          f"{len(rows)/a.rpm/60:.1f} h")
    return 0


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="e' su? sono bloccato?")
    c.add_argument("--url", default=None)
    c.add_argument("--timeout", type=float, default=8.0)

    pr = sub.add_parser("probe", help="una richiesta, e stampa tutto")
    pr.add_argument("--n", type=int, default=1)
    pr.add_argument("--sleep", type=float, default=0.0,
                    help="pausa fra una e l'altra: verifica la finestra temporale")
    pr.add_argument("--max-tokens", type=int, nargs="+", default=[256],
                    help="piu' valori vengono provati nella stessa invocazione, "
                         "cosi' il budget scalato si confronta sulla stessa base")
    pr.add_argument("--thinking-on", action="store_true",
                    help="NON disattivare il thinking, per confronto")

    ce = sub.add_parser("ceiling", help="quanto posso spingere")
    ce.add_argument("--steps", type=float, nargs="+",
                    default=[8, 15, 20, 25, 30, 40, 55])
    ce.add_argument("--requests", type=int, default=25)
    ce.add_argument("--cooldown", type=float, default=70.0,
                    help="pausa fra i gradini: deve superare la finestra")

    co = sub.add_parser("concurrency", help="quale --concurrency serve")
    co.add_argument("db")
    co.add_argument("--rpm", type=float, default=40.0)
    co.add_argument("--margin", type=float, default=1.5,
                    help="fattore di sicurezza sulla latenza p95")

    a = p.parse_args()
    if a.cmd == "check":
        rc = cmd_check(a)
    elif a.cmd == "probe":
        rc = asyncio.run(cmd_probe(a))
    elif a.cmd == "ceiling":
        rc = asyncio.run(cmd_ceiling(a))
    else:
        rc = cmd_concurrency(a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
