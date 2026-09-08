#!/usr/bin/env python3
"""
Confronto fra due o piu' run. E' lo strumento che il disegno sperimentale
richiede e che `analyze_run.py` non puo' dare, perche' guarda un run alla
volta.

    python scripts/compare_runs.py runs/exp_titolo/run.db runs/exp_full/run.db
    python scripts/compare_runs.py runs/*/run.db --etichette titolo integrale

La sezione 6 di analyze_run confronta le profondita' di lettura DENTRO un
run, e va bene solo quando la profondita' e' dedotta dalla biografia. Ma
proprio in quel caso il confronto e' confuso: chi legge l'articolo integrale
e' un attivista, e un attivista scriverebbe post piu' lunghi comunque. Con
`--force-media-depth` la profondita' e' costante dentro il run e variabile
FRA i run, quindi il confronto deve essere fra database. Da qui questo file.

Cosa viene confrontato:
  1. quali parametri differiscono davvero (dal fingerprint)
  2. esito del voto e traiettoria baseline -> final
  3. spostamenti e direzionalita'
  4. produzione di note (il canale del cambio di opinione)
  5. densita' argomentativa, con test di permutazione fra i bracci

Il test di permutazione non assume normalita' e non richiede scipy: rimescola
le etichette dei gruppi 10.000 volte e conta quante volte una differenza
casuale eguaglia quella osservata.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def apri(path: str) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def sezione(titolo: str) -> None:
    print("\n" + "=" * 70)
    print(f" {titolo}")
    print("=" * 70)


def meta(c: sqlite3.Connection, chiave: str, default=None):
    r = c.execute("SELECT value FROM run_meta WHERE key = ?", (chiave,)).fetchone()
    if not r:
        return default
    try:
        return json.loads(r["value"])
    except (json.JSONDecodeError, TypeError):
        return r["value"]


def voti(c: sqlite3.Connection, label: str) -> dict[int, str]:
    return {r["agent_id"]: r["vote"] for r in c.execute(
        "SELECT agent_id, vote FROM vote WHERE label = ? AND vote != 'ERROR'",
        (label,))}


def testi(c: sqlite3.Connection) -> list[str]:
    return [r["content"] for r in c.execute(
        "SELECT p.content FROM post p JOIN agent a ON a.agent_id = p.agent_id "
        "WHERE p.kind != 'news' AND a.is_source = 0")]


def note_per_agente(c: sqlite3.Connection) -> dict[int, int]:
    return {r["agent_id"]: r["n"] for r in c.execute(
        "SELECT agent_id, COUNT(*) n FROM note GROUP BY agent_id")}


def ttr(t: str) -> float:
    w = [x.lower() for x in re.findall(r"\w+", t)]
    return len(set(w)) / len(w) if w else 0.0


METRICHE = {
    "lunghezza": lambda t: float(len(t)),
    "cifre": lambda t: float(len(re.findall(r"\d", t))),
    "hashtag": lambda t: float(t.count("#")),
    "lessico": ttr,
}


def permuta(a: list[float], b: list[float], giri: int = 10000) -> tuple[float, float]:
    """Differenza fra medie e sua probabilita' sotto l'ipotesi nulla."""
    if not a or not b:
        return 0.0, 1.0
    oss = sum(a) / len(a) - sum(b) / len(b)
    tutti = a + b
    na = len(a)
    rng = random.Random(0)
    estremi = 0
    for _ in range(giri):
        rng.shuffle(tutti)
        d = sum(tutti[:na]) / na - sum(tutti[na:]) / (len(tutti) - na)
        if abs(d) >= abs(oss):
            estremi += 1
    return oss, (estremi + 1) / (giri + 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", nargs="+")
    ap.add_argument("--etichette", nargs="*", default=None)
    a = ap.parse_args()

    nomi = a.etichette or [Path(p).parent.name for p in a.db]
    if len(nomi) != len(a.db):
        raise SystemExit("Servono tante etichette quanti database.")
    conn = [apri(p) for p in a.db]

    # --- 1. cosa differisce davvero -------------------------------------- #
    sezione("1. COSA DISTINGUE QUESTI RUN")
    cfg = [meta(c, "sim_config", {}) or {} for c in conn]
    chiavi = sorted({k for d in cfg for k in d})
    diverse = [k for k in chiavi if len({json.dumps(d.get(k)) for d in cfg}) > 1]
    if not diverse:
        print("  Nessuna differenza di configurazione: sono repliche.")
    for k in diverse:
        print(f"  {k:<26}" + "  ".join(
            f"{n}={d.get(k)}" for n, d in zip(nomi, cfg)))
    uguali = [k for k in ("seed", "start_date", "end_date", "hours_per_tick",
                          "recommender", "feed_size") if k not in diverse]
    if uguali:
        print(f"\n  Tenuti costanti: {', '.join(uguali)}")
    if "force_media_depth" in diverse:
        print("\n  Profondita' di lettura MANIPOLATA: la differenza fra i")
        print("  bracci e' attribuibile all'esposizione, non al tipo di")
        print("  persona. Questo e' un esperimento, non un'osservazione.")

    # --- 2. esito e traiettoria ------------------------------------------- #
    sezione("2. ESITO DEL VOTO")
    print(f"  {'run':<16}{'fase':<10}{'SI':>7}{'NO':>7}{'AST':>7}{'validi':>8}")
    for n, c in zip(nomi, conn):
        for fase in ("baseline", "final"):
            v = voti(c, fase)
            t = len(v) or 1
            cnt = {k: sum(1 for x in v.values() if x == k)
                   for k in ("SI", "NO", "ASTENUTO")}
            print(f"  {n if fase == 'baseline' else '':<16}{fase:<10}"
                  f"{cnt['SI']:>7}{cnt['NO']:>7}{cnt['ASTENUTO']:>7}{t:>8}")

    # --- 3. spostamenti ---------------------------------------------------- #
    sezione("3. SPOSTAMENTI E DIREZIONALITA'")
    print(f"  {'run':<16}{'cambiati':>10}{'tasso':>8}{'->NO':>7}{'->SI':>7}"
          f"{'->AST':>7}")
    for n, c in zip(nomi, conn):
        b, f = voti(c, "baseline"), voti(c, "final")
        comuni = set(b) & set(f)
        mossi = [k for k in comuni if b[k] != f[k]]
        verso = {"NO": 0, "SI": 0, "ASTENUTO": 0}
        for k in mossi:
            verso[f[k]] = verso.get(f[k], 0) + 1
        tasso = len(mossi) / max(len(comuni), 1)
        print(f"  {n:<16}{len(mossi):>10}{tasso:>7.0%}"
              f"{verso['NO']:>7}{verso['SI']:>7}{verso['ASTENUTO']:>7}")
    print("\n  Riferimento: due estrazioni indipendenti dalla distribuzione")
    print("  osservata darebbero circa il 60% di cambiamenti. Un tasso molto")
    print("  sotto quella soglia indica ancoraggio, non inerzia del modello.")

    # --- 4. produzione di note --------------------------------------------- #
    sezione("4. PRODUZIONE DI NOTE (il canale del cambio di opinione)")
    print(f"  {'run':<16}{'note':>7}{'agenti con >=1':>16}{'senza note':>12}"
          f"{'cambiati fra':>14}")
    print(f"  {'':<16}{'':>7}{'':>16}{'':>12}{'chi ha note':>14}")
    for n, c in zip(nomi, conn):
        per = note_per_agente(c)
        b, f = voti(c, "baseline"), voti(c, "final")
        comuni = set(b) & set(f)
        con = [k for k in comuni if per.get(k, 0) >= 1]
        senza = [k for k in comuni if per.get(k, 0) == 0]
        mossi_con = sum(1 for k in con if b[k] != f[k])
        mossi_senza = sum(1 for k in senza if b[k] != f[k])
        print(f"  {n:<16}{sum(per.values()):>7}{len(con):>16}{len(senza):>12}"
              f"{mossi_con:>9}/{len(con):<4}")
        if mossi_senza:
            print(f"      ATTENZIONE: {mossi_senza} agenti hanno cambiato voto "
                  f"SENZA alcuna nota. Il voto finale dovrebbe dipendere solo "
                  f"da biografia + note: se cambia comunque, e' rumore di "
                  f"campionamento del modello e va quantificato.")

    # --- 5. densita' argomentativa ----------------------------------------- #
    sezione("5. DENSITA' ARGOMENTATIVA")
    corpora = {n: testi(c) for n, c in zip(nomi, conn)}
    print(f"  {'run':<16}{'post':>7}" + "".join(f"{m:>11}" for m in METRICHE))
    valori: dict[str, dict[str, list[float]]] = {}
    for n in nomi:
        t = corpora[n]
        valori[n] = {m: [fn(x) for x in t] for m, fn in METRICHE.items()}
        riga = "".join(f"{sum(v)/max(len(v),1):>11.3f}"
                       for v in valori[n].values())
        print(f"  {n:<16}{len(t):>7}{riga}")

    if len(nomi) == 2:
        x, y = nomi
        print(f"\n  Test di permutazione: {x} contro {y}")
        print("  (10.000 rimescolamenti, nessuna assunzione di normalita')\n")
        print(f"  {'metrica':<12}{'differenza':>12}{'p':>10}   esito")
        for m in METRICHE:
            d, p = permuta(valori[x][m], valori[y][m])
            esito = "significativo" if p < 0.05 else "non distinguibile"
            print(f"  {m:<12}{d:>+12.3f}{p:>10.4f}   {esito}")
        print(f"\n  Differenza positiva = {x} ha valori piu' alti.")

    for c in conn:
        c.close()


if __name__ == "__main__":
    main()
