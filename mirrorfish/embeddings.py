"""
Embedding: client batch + cache, e uno stub offline che ha semantica vera.

Nota sui costi, perche' e' il motivo per cui questo modulo e' organizzato cosi':
l'embedding di un post non cambia mai, quindi si calcola UNA volta e si mette
in cache. Il vettore di interesse di un agente si ricava per media dai vettori
gia' calcolati dei post che ha scritto e messo "mi piace", quindi non costa
NESSUNA chiamata aggiuntiva. L'unico costo per agente e' l'embedding della bio,
una volta sola.

Ordine di grandezza su un run da 106 agenti e 21 giorni: ~3000 post da
embeddare in batch da 32 = ~95 chiamate, piu' 4 chiamate per le bio. Contro le
~3000 chiamate di generazione, e' rumore.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from abc import ABC, abstractmethod
from typing import Sequence

import httpx

from .config import LLMConfig


def to_blob(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def from_blob(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


def normalize(vec: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n > 1e-12 else vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Assume vettori gia' normalizzati: e' un prodotto scalare."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def mean_vector(vecs: list[Sequence[float]]) -> list[float]:
    if not vecs:
        return []
    dim = len(vecs[0])
    out = [0.0] * dim
    for v in vecs:
        for i, x in enumerate(v):
            out[i] += x
    return normalize([x / len(vecs) for x in out])


class Embedder(ABC):
    dim: int = 768

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None:
        return None


class APIEmbedder(Embedder):
    """nomic-embed-text via endpoint OpenAI-compatible, in batch."""

    def __init__(self, cfg: LLMConfig, batch_size: int = 32, gate=None):
        if not cfg.api_key:
            raise ValueError("LLM_API_KEY non configurata.")
        self.cfg = cfg
        self.batch_size = batch_size
        self.gate = gate           # condiviso col client di generazione
        self.calls = 0
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {cfg.api_key}",
                     "Content-Type": "application/json"},
            timeout=httpx.Timeout(cfg.timeout_s),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = [t[:2000] for t in texts[i : i + self.batch_size]]
            if self.gate is not None:
                # Stesso cancello del client di generazione: gli embedding
                # consumano la stessa quota, ignorarli farebbe scattare il
                # rate limit proprio mentre lo si sta evitando altrove.
                await self.gate.acquire()
            r = await self._client.post(
                "/embeddings", json={"model": self.cfg.embed_model, "input": chunk}
            )
            if r.status_code == 429 and self.gate is not None:
                await self.gate.trip(self.cfg.rate_limit_cooldown_s, "429 embeddings")
                r = await self._client.post(
                    "/embeddings",
                    json={"model": self.cfg.embed_model, "input": chunk},
                )
            r.raise_for_status()
            self.calls += 1
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.extend(normalize(d["embedding"]) for d in data)
        if out:
            self.dim = len(out[0])
        return out

    async def aclose(self) -> None:
        await self._client.aclose()


_TOKEN_RE = re.compile(r"[a-zàèéìòóùA-ZÀÈÉÌÒÓÙ]{3,}")
_STOP = {"che", "non", "per", "con", "una", "del", "della", "dei", "delle",
         "sono", "come", "piu", "anche", "ansa", "questo", "questa", "gli"}


class StubEmbedder(Embedder):
    """
    Embedding deterministico offline con hashing trick su bigrammi di parole.

    NON e' un modello linguistico: non conosce sinonimi. Ma e' semantica vera,
    non rumore: testi che condividono vocabolario hanno cosine alta, testi su
    argomenti diversi hanno cosine bassa. Basta per verificare che il
    recommender faccia quello che dichiara, senza consumare quota.
    """

    dim = 256

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        toks = [w.lower() for w in _TOKEN_RE.findall(text)]
        toks = [w for w in toks if w not in _STOP]
        vec = [0.0] * self.dim
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for f in feats:
            h = hashlib.blake2b(f.encode(), digest_size=8).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0
            vec[idx] += sign
        return normalize(vec)


def build_embedder(cfg: LLMConfig, *, stub: bool = False, gate=None) -> Embedder:
    return StubEmbedder() if stub else APIEmbedder(cfg, gate=gate)
