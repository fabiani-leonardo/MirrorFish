#!/usr/bin/env python3
"""
Test sui segni su piu' coppie di bracci sperimentali.

    python scripts/sign_test.py \\
        runs/fix7_titolo/run.db:runs/fix7_integrale/run.db \\
        runs/sig23_titolo/run.db:runs/sig23_integrale/run.db \\
        ...

Ogni argomento e' una coppia `A:B`, dove A e B sono i due bracci della stessa
replica. L'ordine conta: la differenza riportata e' sempre A meno B.

PERCHE' QUESTO TEST E NON IL PRECEDENTE
---------------------------------------
Il test di permutazione dentro una singola coppia tratta i post come
osservazioni indipendenti. Non lo sono: sono raggruppati per agente, e
l'intero run e' una sola estrazione dal processo generativo. Su una coppia di
run IDENTICI quel test dichiarava significative differenze di 0,119 sugli
hashtag, dove l'effetto vero e' zero per costruzione.

Il test sui segni non assume nulla sulla distribuzione e usa come unita' la
REPLICA, che e' l'unita' giusta. Sotto l'ipotesi nulla ogni coppia ha
probabilita' 1/2 di cadere da un lato, quindi k concordi su n valgono
p = somma_{i>=k} C(n,i) / 2^n. Con 6 coppie su 6 si ottiene p = 0,016 a una
coda, ed e' un risultato che regge anche quando il rumore fra due run
identici e' dello stesso ordine dell'effetto: un singolo confronto non
distingue nulla, sei confronti concordi si'.

Costo: il test e' poco potente, servono almeno cinque o sei repliche per
scendere sotto 0,05. E' il prezzo di non assumere niente.
"""

from __future__ import annotations

import argparse
import re
import json
import sqlite3
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parole(t: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Zàèéìòù]{3,}", t or "")]


def ttr(t: str) -> float:
    w = parole(t)
    return len(set(w)) / len(w) if w else 0.0


METRICHE = {
    "lunghezza": lambda t: float(len(t)),
    "cifre": lambda t: float(len(re.findall(r"\d", t))),
    "hashtag": lambda t: float(t.count("#")),
    "lessico": ttr,
}


def limite_caratteri(c: sqlite3.Connection) -> int:
    """Limite di caratteri con cui quel run e' stato eseguito."""
    try:
        r = c.execute("SELECT value FROM run_meta WHERE key='sim_config'").fetchone()
        return int(json.loads(r["value"]).get("max_post_chars", 280)) if r else 280
    except Exception:
        return 280


def misura(path: str) -> dict[str, float]:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    limite = limite_caratteri(c)
    testi = [r["content"] for r in c.execute(
        "SELECT p.content FROM post p JOIN agent a ON a.agent_id = p.agent_id "
        "WHERE p.kind != 'news' AND a.is_source = 0")]
    c.close()
    if not testi:
        raise SystemExit(f"Nessun post in {path}")
    out = {m: sum(fn(t) for t in testi) / len(testi)
           for m, fn in METRICHE.items()}
    out["n_post"] = float(len(testi))
    # Quota di post che tocca il tetto del run. La soglia e' RELATIVA al
    # limite con cui quel run e' stato eseguito, non fissa: calcolata a 265
    # caratteri, in un run a 500 contava come "al tetto" dei post di lunghezza
    # ordinaria, e produceva un 54,8% privo di significato.
    # Sopra il 10% la media della lunghezza e' compressa dalla censura e la
    # metrica `lunghezza` non e' interpretabile.
    soglia = limite * 0.95
    out["censurati_%"] = 100.0 * sum(1 for t in testi if len(t) >= soglia) / len(testi)
    out["limite"] = float(limite)
    return out


def p_segni(k: int, n: int) -> float:
    """Probabilita' di k o piu' concordi su n, a una coda."""
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("coppie", nargs="+", metavar="A:B",
                    help="coppie di run.db separate da due punti")
    ap.add_argument("--etichette", nargs="*", default=None)
    a = ap.parse_args()

    coppie = []
    for x in a.coppie:
        if ":" not in x:
            raise SystemExit(f"Formato atteso A:B, ricevuto {x!r}")
        coppie.append(tuple(x.split(":", 1)))
    nomi = a.etichette or [Path(x[0]).parent.name.rsplit("_", 1)[0]
                           for x in coppie]

    CHIAVI = list(METRICHE) + ["n_post", "censurati_%"]
    limiti: set[float] = set()
    censura: list[float] = []
    diff: dict[str, list[float]] = {m: [] for m in CHIAVI}
    print(f"\n  {len(coppie)} repliche. Differenza = primo braccio meno secondo.\n")
    intest = "".join(f"{m:>12}" for m in CHIAVI)
    print(f"  {'replica':<12}{intest}")
    for (pa, pb), nome in zip(coppie, nomi):
        ma, mb = misura(pa), misura(pb)
        censura += [ma["censurati_%"], mb["censurati_%"]]
        limiti |= {ma["limite"], mb["limite"]}
        riga = ""
        for m in CHIAVI:
            d = ma[m] - mb[m]
            diff[m].append(d)
            riga += f"{d:>+12.3f}"
        print(f"  {nome:<12}{riga}")

    n = len(coppie)
    print(f"\n  {'metrica':<12}{'concordi':>10}{'p (una coda)':>15}"
          f"{'effetto medio':>16}   esito")
    for m in CHIAVI:
        v = diff[m]
        pos = sum(1 for x in v if x > 0)
        k = max(pos, n - pos)              # la direzione maggioritaria
        p = p_segni(k, n)
        verso = "primo" if pos >= n - pos else "secondo"
        media = sum(v) / n
        esito = "significativo" if p < 0.05 else "non concludente"
        print(f"  {m:<12}{str(k)+'/'+str(n):>10}{p:>15.4f}{media:>+16.3f}"
              f"   {esito} ({verso} braccio piu' alto)")

    cm = sum(censura) / len(censura)
    lim = "/".join(f"{int(x)}" for x in sorted(limiti))
    print(f"\n  Post che toccano il tetto ({lim} caratteri): {cm:.1f}% in media.")
    if len(limiti) > 1:
        print("  I due bracci hanno limiti DIVERSI: la lunghezza fra loro non")
        print("  e' confrontabile per costruzione, perche' il tetto e' parte")
        print("  della condizione sperimentale e non un dettaglio.")
    if cm > 10:
        print("  La media di `lunghezza` e' CENSURATA e non e' interpretabile:")
        print("  con una quota cosi' alta al tetto una differenza vera viene")
        print("  schiacciata verso il limite comune.")
        print("  Usa al suo posto `censurati_%`, che misura la stessa cosa")
        print("  senza subire la censura: quanto spesso un agente vuole")
        print("  scrivere piu' di quanto il mezzo gli consenta. Le altre")
        print("  metriche non sono affette, perche' non hanno un tetto.")

    print(f"\n  Con {n} repliche il minimo p ottenibile e' {p_segni(n, n):.4f}.")
    if p_segni(n, n) >= 0.05:
        serve = 5
        while p_segni(serve, serve) >= 0.05:
            serve += 1
        print(f"  Per scendere sotto 0,05 servono almeno {serve} repliche "
              f"tutte concordi.")


if __name__ == "__main__":
    main()
