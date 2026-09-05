"""
Client LLM — sostituisce interamente camel-ai.

Tre proprieta' che il layer precedente non garantiva:

1. `max_tokens` e' OBBLIGATORIO. Non e' un default, e' un parametro richiesto:
   una chiamata senza budget solleva un errore invece di lasciare che vLLM
   deduca 261.678 token da max_model_len.
2. La concorrenza e' esplicita e centralizzata (semaforo + intervallo minimo),
   quindi il carico sulla GPU condivisa e' un numero che decidi tu, non un
   effetto collaterale di quanti agenti sono attivi in quel tick.
3. Ogni chiamata restituisce la sua telemetria (token, latenza, finish_reason)
   che finisce su DB. Le affermazioni sul costo computazionale in tesi
   diventano misurate invece che stimate.

Il client reale e lo stub implementano la stessa interfaccia, quindi tutta la
pipeline gira offline (endpoint in manutenzione, test, CI) senza modifiche.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import LLMConfig


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def truncated(self) -> bool:
        """True se il modello ha esaurito il budget prima di finire."""
        return self.finish_reason == "length"


_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_FEED_ID_RE = re.compile(r"^#(\d+) ", re.MULTILINE)


def parse_json_response(text: str) -> dict[str, Any] | None:
    """
    Estrae un oggetto JSON da una risposta che puo' contenere blocchi <think>,
    fence markdown o prosa attorno. Restituisce None se non ci riesce.
    """
    if not text:
        return None
    cleaned = _THINK_RE.sub("", text)
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


class RateGate:
    """
    Cancello CONDIVISO fra tutte le richieste in volo.

    Il problema che risolve: con `asyncio.gather` su 106 agenti, quando il
    server risponde 429 ogni task fa backoff per conto suo e poi riprova piu'
    o meno insieme agli altri. Risultato: thundering herd, il limite si
    ritriggera immediatamente e non si esce piu' dal buco.

    Qui il 429 di UNA richiesta mette in pausa TUTTE le altre fino alla
    scadenza indicata dal server (`Retry-After`) o al backoff calcolato.
    """

    def __init__(self, min_interval_s: float, rpm: float = 0.0,
                 window_s: float = 60.0, reserve: int = 1):
        # Spaziatura fissa (fallback) + finestra scorrevole (vincolo vero).
        self.min_interval_s = max(min_interval_s, 0.05)
        self._base_interval = self.min_interval_s
        self.rpm = rpm
        self.window_s = window_s
        self.reserve = reserve          # margine lasciato libero nella finestra
        self._starts: list[float] = []  # timestamp delle partenze recenti
        self._lock = asyncio.Lock()
        self._last_start = 0.0
        self._cooldown_until = 0.0
        self.trips = 0
        self.paused_s = 0.0
        self.preemptive_pauses = 0
        self.observed_remaining: int | None = None
        self.observed_scope: str | None = None
        # Pausa del freno preventivo. Breve di proposito: serve solo a far
        # scivolare la finestra, non a fermare il run.
        self.probe_pause_s = 12.0
        self._probe_pending = False
        # Risalita dopo un rallentamento. Senza, `trip` era a senso unico: un
        # solo 429 al quinto minuto di un run da quattro ore lo rallentava del
        # 50% per tutte le tre ore e cinquanta rimanenti. Misurato su
        # pilot_14: ritmo nominale 2,76 s/richiesta (--rpm 25), un intervento,
        # ritmo finale 4,14 s per l'intero run, cioe' 14,5 richieste/minuto
        # invece di 21,7. Un terzo del tempo di orologio buttato.
        self.recover_after = 25       # richieste riuscite prima di riprovare
        self.recover_factor = 0.9     # risalita graduale, non a scatto
        self._ok_since_trip = 0
        self.recoveries = 0

    async def acquire(self) -> None:
        """
        Concede il permesso di partire.

        Tre vincoli in AND:
          1. nessuna pausa globale in corso (429 gia' incassato);
          2. distanza minima dalla partenza precedente;
          3. non piu' di (rpm - reserve) partenze nella finestra scorrevole.

        Il terzo e' quello che conta. Una spaziatura fissa di 60/rpm secondi
        sembra sicura ma mette esattamente `rpm` richieste dentro ogni minuto
        solare: si e' sul bordo del limite e basta una latenza irregolare per
        superarlo. La finestra scorrevole con riserva tiene un margine reale.
        """
        while True:
            async with self._lock:
                now = time.monotonic()
                sleep_for = 0.0

                if now < self._cooldown_until:
                    sleep_for = self._cooldown_until - now
                else:
                    gap = self._last_start + self.min_interval_s - now
                    if gap > 0:
                        sleep_for = gap
                    elif self.rpm > 0:
                        cutoff = now - self.window_s
                        self._starts = [t for t in self._starts if t > cutoff]
                        budget = max(1, int(self.rpm) - self.reserve)
                        if len(self._starts) >= budget:
                            # aspetta che la richiesta piu' vecchia esca
                            sleep_for = self._starts[0] + self.window_s - now + 0.1

                if sleep_for <= 0:
                    self._last_start = now
                    self._starts.append(now)
                    return
            await asyncio.sleep(min(sleep_for, 5.0))

    async def trip(self, seconds: float, reason: str = "429") -> None:
        async with self._lock:
            target = time.monotonic() + seconds
            if target > self._cooldown_until:
                first = self._cooldown_until <= time.monotonic()
                self._cooldown_until = target
                self.trips += 1
                self.paused_s += seconds
                # Rallenta anche a regime: se il limite e' scattato una volta,
                # il ritmo precedente era troppo alto.
                # Floor esplicito: con _base_interval = 0 la vecchia formula
                # produceva 0, cioe' "rallenta" azzerava la spaziatura.
                self.min_interval_s = min(
                    max(self._base_interval, 1.0) * 8,
                    max(self.min_interval_s * 1.5, 1.0),
                )
                self._ok_since_trip = 0
                if first:
                    print(f"    [rate-limit] {reason}: pausa globale "
                          f"{seconds:.0f}s, ritmo -> "
                          f"{self.min_interval_s:.2f}s fra le richieste")

    async def _maybe_recover(self) -> None:
        """
        Risale verso il ritmo nominale dopo una serie di richieste riuscite.

        La discesa e' brusca (x1.5 subito) e la risalita lenta (x0.9 ogni 25
        successi) di proposito: si vuole reagire in fretta a un limite e
        tornare su con prudenza, non oscillare attorno alla soglia. Da 4,14 s
        servono circa 100 richieste riuscite per tornare a 2,76 s.
        """
        if self.min_interval_s <= self._base_interval:
            return
        self._ok_since_trip += 1
        if self._ok_since_trip < self.recover_after:
            return
        self._ok_since_trip = 0
        prima = self.min_interval_s
        self.min_interval_s = max(self._base_interval,
                                  self.min_interval_s * self.recover_factor)
        self.recoveries += 1
        if self.recoveries % 4 == 1:
            print(f"    [rate-limit] {self.recover_after} richieste pulite: "
                  f"ritmo {prima:.2f}s -> {self.min_interval_s:.2f}s")

    async def observe(self, headers: Any) -> None:
        """
        Legge il contatore del server dalle risposte riuscite.

        E' piu' affidabile di qualunque stima locale: il gateway conta anche
        le richieste degli altri membri del team sul limite condiviso, che noi
        non possiamo vedere. Quando il residuo scende sotto la riserva ci si
        ferma PRIMA di prendere il 429, invece di reagire dopo.
        """
        async with self._lock:
            await self._maybe_recover()

        # Va guardato il MINIMO fra tutti i limiti, non il primo trovato.
        # Il bug precedente usciva dopo `team_member`, che dopo l'aumento a 60
        # e' sempre abbondante, e non leggeva mai `team` — che e' condiviso
        # con gli altri membri ed e' quello che scatta davvero.
        found: dict[str, int] = {}
        for scope in ("api_key", "team_member", "team"):
            raw = headers.get(f"x-ratelimit-{scope}-remaining-requests")
            if raw is None:
                continue
            try:
                found[scope] = int(float(raw))
            except (TypeError, ValueError):
                pass
        if not found:
            return

        scope, remaining = min(found.items(), key=lambda kv: kv[1])
        self.observed_remaining = remaining
        self.observed_scope = scope
        if remaining <= self.reserve:
            async with self._lock:
                now = time.monotonic()
                # NON riarmare se una pausa e' gia' in corso. La versione
                # precedente confrontava (adesso + 30s) con la scadenza
                # corrente: essendo quasi sempre maggiore, OGNI risposta
                # riuscita rimetteva 30 secondi di pausa. Con il contatore di
                # squadra tenuto basso da un altro run, il ritmo collassava a
                # una richiesta ogni 30 secondi e la simulazione sembrava
                # bloccata.
                if now < self._cooldown_until:
                    return
                # Dopo una pausa lasciamo passare una richiesta di sondaggio
                # prima di poter frenare di nuovo: senza, due run concorrenti
                # si terrebbero a vicenda fermi a zero.
                if self._probe_pending:
                    self._probe_pending = False
                    return
                self._cooldown_until = now + self.probe_pause_s
                self._probe_pending = True
                self.preemptive_pauses += 1
                print(f"    [rate-limit] freno preventivo: {scope} a "
                      f"{remaining} residue, pausa {self.probe_pause_s:.0f}s",
                      flush=True)

    def stats(self) -> dict[str, Any]:
        return {"trips": self.trips, "paused_s": round(self.paused_s, 1),
                "preemptive_pauses": self.preemptive_pauses,
                "final_interval_s": round(self.min_interval_s, 3)}


def _retry_after(resp: httpx.Response, fallback: float) -> float:
    """Legge Retry-After (secondi o data HTTP). Il server sa meglio di noi."""
    raw = resp.headers.get("retry-after")
    if raw:
        try:
            return max(1.0, float(raw))
        except ValueError:
            try:
                from email.utils import parsedate_to_datetime
                from datetime import datetime, timezone
                dt = parsedate_to_datetime(raw)
                return max(1.0, (dt - datetime.now(timezone.utc)).total_seconds())
            except Exception:
                pass
    for h in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens",
              "x-ratelimit-reset"):
        v = resp.headers.get(h)
        if v:
            try:
                return max(1.0, float(v.rstrip("s")))
            except ValueError:
                pass
    return fallback


class LLMClient(ABC):
    """Interfaccia comune. Cambiare provider = scrivere una sottoclasse."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> LLMResponse: ...

    async def aclose(self) -> None:  # pragma: no cover - override se serve
        return None


class EndpointDown(RuntimeError):
    """L'endpoint non risponde da troppe chiamate consecutive."""


class OpenAICompatClient(LLMClient):
    """Client per qualunque endpoint OpenAI-compatible (vLLM, Ollama, ...)."""

    def __init__(self, cfg: LLMConfig):
        if not cfg.api_key:
            raise ValueError(
                "LLM_API_KEY non configurata. Impostala come variabile "
                "d'ambiente (o in .env), mai nel codice o nel repo."
            )
        self.cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrency)
        self.gate = RateGate(cfg.pace_interval(), rpm=cfg.requests_per_minute)
        self._thinking_supported = cfg.disable_thinking
        self._consecutive_failures = 0
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(cfg.timeout_s, connect=cfg.connect_timeout_s),
            limits=httpx.Limits(max_connections=cfg.concurrency + 2),
        )

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not max_tokens or max_tokens <= 0:
            raise ValueError(
                "max_tokens deve essere esplicito e positivo. Ometterlo fa si' "
                "che vLLM usi max_model_len - prompt_tokens."
            )

        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self._thinking_supported:
            # ATTENZIONE: `extra_body` e' una convenzione dell'SDK OpenAI, non
            # un campo dell'API. L'SDK ne spacchetta il contenuto e lo fonde
            # nel body. Con httpx grezzo va messo al TOP LEVEL, altrimenti
            # vLLM lo ignora come campo sconosciuto e il thinking mode resta
            # ATTIVO in silenzio (latenza alta, risposte troncate).
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        last_err = ""
        for attempt in range(self.cfg.max_retries):
            await self.gate.acquire()
            t0 = time.perf_counter()
            try:
                async with self._sem:
                    r = await self._client.post("/chat/completions", json=payload)
                latency = (time.perf_counter() - t0) * 1000

                if r.status_code == 400 and "chat_template_kwargs" in payload:
                    # Il server non accetta enable_thinking: disattiva e riprova
                    # una volta sola, poi ricordatelo per le chiamate successive.
                    payload.pop("chat_template_kwargs", None)
                    self._thinking_supported = False
                    continue

                if r.status_code == 429:
                    # Il body del 429 dice QUALE limite e' scattato (richieste
                    # al minuto, token al minuto, quota della chiave...).
                    # Va conservato: e' l'informazione da girare al professore.
                    body = (r.text or "")[:300].replace("\n", " ")
                    last_err = f"HTTP 429 {body}".strip()
                    wait = _retry_after(r, self.cfg.rate_limit_cooldown_s)
                    await self.gate.trip(wait, reason=f"429 {body[:120]}")
                    continue

                if r.status_code in (500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}"
                    delay = self.cfg.backoff_base_s * (2**attempt)
                    delay *= 0.5 + random.random()  # jitter
                    await asyncio.sleep(delay)
                    continue

                r.raise_for_status()
                await self.gate.observe(r.headers)
                data = r.json()
                choice = data["choices"][0]
                msg = choice.get("message", {})
                text = msg.get("content") or ""
                if not text.strip():
                    # Alcuni build di Qwen mettono tutto in reasoning_content.
                    text = msg.get("reasoning_content") or ""
                usage = data.get("usage") or {}
                self._consecutive_failures = 0
                return LLMResponse(
                    text=text,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason", ""),
                    latency_ms=latency,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(self.cfg.backoff_base_s * (2**attempt))

        # Interruttore. Un 429 non conta: e' una pausa, non un guasto.
        if "429" not in (last_err or ""):
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.cfg.circuit_breaker_failures:
                raise EndpointDown(
                    f"{self._consecutive_failures} chiamate consecutive fallite "
                    f"({last_err}). Interrompo invece di proseguire: un run di "
                    f"sole chiamate fallite ha comunque l'aria di essere valido.\n"
                    f"Diagnostica: python scripts/check_endpoint.py\n"
                    f"Ripresa:     stesso comando piu' --resume"
                )
        return LLMResponse(text="", error=last_err or "unknown", finish_reason="error")

    async def aclose(self) -> None:
        await self._client.aclose()


class StubLLM(LLMClient):
    """
    LLM finto e deterministico per sviluppo offline.

    Serve a due cose:
      - far girare la pipeline completa senza GPU (endpoint in manutenzione,
        test di regressione, debug del loop);
      - misurare l'overhead della simulazione al netto dell'inferenza.

    Le risposte sono generate da un RNG seedato sul contenuto del prompt:
    stesso prompt -> stessa risposta, sempre.
    """

    ACTIONS = ["POST", "REPLY", "LIKE", "IGNORE"]

    def __init__(self, seed: int = 0, latency_ms: float = 0.0):
        self.seed = seed
        self.latency_ms = latency_ms
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> LLMResponse:
        if not max_tokens or max_tokens <= 0:
            raise ValueError("max_tokens deve essere esplicito anche nello stub.")
        self.calls.append({"max_tokens": max_tokens, "temperature": temperature})
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)

        # Il seed deve dipendere dall'INTERO prompt. Usare una fetta (es.
        # user[-400:]) sembra innocuo ma la coda dei prompt e' template fisso:
        # tutte le chiamate finivano con lo stesso seed e quindi con la stessa
        # risposta. E' il motivo per cui il primo smoke test dava 0 note su 90
        # riflessioni.
        digest = hashlib.sha256(f"{self.seed}|{system}|{user}".encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))

        # Gli id disponibili nel feed, letti dal prompt: cosi' REPLY e LIKE
        # puntano a post che esistono davvero, come farebbe un modello vero.
        feed_ids = [int(m) for m in _FEED_ID_RE.findall(user)]

        if "note_added" in user:
            added = rng.random() < 0.25
            body = {
                "note_added": added,
                "note": "Rafforza la propria diffidenza verso la riforma dopo "
                        "un post che ne contesta l'impatto sull'indipendenza."
                        if added else "",
                "reasoning": "post coerente con posizione gia' espressa"
                             if added else "nessun elemento nuovo",
            }
        elif '"vote"' in system or "DOMANDA DI VOTO" in user:
            vote = rng.choices(["SI", "NO", "ASTENUTO"], weights=[42, 45, 13])[0]
            body = {
                "vote": vote,
                "motivation": f"Voto {vote} per ragioni coerenti con la mia storia.",
                "confidence": round(rng.uniform(0.4, 0.95), 2),
            }
        else:
            # Quante azioni consente il prompt? Lo stub deve poter esercitare
            # anche il percorso multi-azione, altrimenti resta non testato.
            m = re.search(r"fino a (\d+) azioni", system)
            max_acts = int(m.group(1)) if m else 1
            weights = [25, 30, 25, 20] if feed_ids else [45, 0, 0, 55]

            def one():
                act = rng.choices(self.ACTIONS, weights=weights)[0]
                return {
                    "action": act,
                    "content": "" if act in ("LIKE", "IGNORE")
                               else f"[stub-{rng.randint(1000,9999)}] Contenuto "
                                    f"simulato, tono {rng.choice(['critico','favorevole','dubbioso'])}.",
                    "target_post_id": (rng.choice(feed_ids)
                                       if act in ("REPLY", "LIKE") and feed_ids
                                       else None),
                }

            if max_acts > 1:
                # Frequenze realistiche: si reagisce spesso, si scrive di rado.
                n = rng.choices(range(1, max_acts + 1),
                                weights=[3, 4, 2][:max_acts])[0]
                body = {"actions": [one() for _ in range(n)]}
            else:
                body = one()
        text = json.dumps(body, ensure_ascii=False)
        return LLMResponse(
            text=text,
            prompt_tokens=len(system + user) // 4,
            completion_tokens=len(text) // 4,
            finish_reason="stop",
            latency_ms=self.latency_ms,
        )


def build_client(cfg: LLMConfig, *, stub: bool = False, seed: int = 0) -> LLMClient:
    return StubLLM(seed=seed) if stub else OpenAICompatClient(cfg)
