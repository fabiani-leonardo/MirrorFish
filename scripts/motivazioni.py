#!/usr/bin/env python3
"""
motivazioni — confronta la STRUTTURA delle ragioni di voto simulate con
quella rilevata sugli elettori reali.

Perche' serve. La ripartizione finale del voto varia di 42 punti al variare
del solo seme, quindi non e' una grandezza su cui basare conclusioni. Le
motivazioni sono un'altra cosa: ce ne sono un centinaio per rilevazione e
undici rilevazioni per esecuzione, e `survey.py` le registra gia' tutte nella
tabella `vote` insieme al grado di fiducia. Sono materiale gia' pagato.

La griglia di codifica riproduce quella della rilevazione reale, comprese le
categorie procedurali (sorteggio, Alta Corte, divisione del CSM) che nel
dibattito pubblico contano poco e nell'urna hanno contato molto.

Ogni motivazione puo' ricevere piu' etichette, come nella rilevazione reale,
dove le quote sommano a piu' di 100.

Uso:
  export KEY=...
  python motivazioni.py runs/campagna_completa1/run.db
  python motivazioni.py runs/campagna_completa1/run.db --label tick_60
  python motivazioni.py a/run.db b/run.db          # confronto fra due run
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

URL = "https://api.ailabroma3.it/v1/chat/completions"
MODEL = "lab-qwen36"

# Griglia e quote della rilevazione reale sugli elettori (fonte da citare in
# tesi). Le quote sono riferite a chi ha votato in quel modo, non al totale.
GRIGLIA = {
    "SI": [
        ("S1", "sostegno alla separazione delle carriere fra giudici e pubblici ministeri", 59),
        ("S2", "favore alla divisione del CSM in due organi", 35),
        ("S3", "sostegno all'istituzione dell'Alta Corte disciplinare", 34),
        ("S4", "volonta' generica di modificare la Costituzione in questa direzione", 24),
        ("S5", "sostegno politico al Governo in carica", 18),
    ],
    "NO": [
        ("N1", "volonta' di non modificare la Costituzione", 61),
        ("N2", "contrarieta' al sorteggio per i componenti del CSM", 39),
        ("N3", "opposizione politica all'esecutivo in carica", 31),
        ("N4", "contrarieta' alla divisione del CSM in due organi", 27),
        ("N5", "contrarieta' all'istituzione dell'Alta Corte disciplinare", 17),
        ("N6", "coerenza con l'indicazione del proprio partito", 7),
        ("N7", "contrarieta' specifica alla separazione delle carriere", 4),
    ],
}

# Categoria fuori griglia: e' l'argomento che domina le note simulate e che
# nella rilevazione reale non compare. Tenerla separata e' il punto dell'analisi.
EXTRA = [
    ("X1", "difesa dell'indipendenza della magistratura da ingerenze politiche"),
    ("X2", "efficienza, durata o lentezza dei processi"),
    ("X3", "sfiducia generica verso la politica o verso i magistrati"),
    ("X0", "nessuna ragione di merito riconoscibile"),
]


def rubrica(voto):
    voci = [f"{c}  {d}" for c, d, _ in GRIGLIA.get(voto, [])]
    voci += [f"{c}  {d}" for c, d in EXTRA]
    return f"""Sei un codificatore. Ti viene data la motivazione con cui un elettore \
italiano spiega di aver votato {voto} al referendum costituzionale sulla \
separazione delle carriere dei magistrati.

Assegna tutti i codici che la motivazione esprime, anche piu' di uno. Assegna \
un codice solo se la ragione corrispondente e' effettivamente presente: non \
dedurre, non completare, non attribuire all'elettore ragioni plausibili che \
non ha scritto. Se nessuna ragione di merito e' riconoscibile, usa X0 da solo.

Codici disponibili:
{chr(10).join(voci)}

Rispondi esclusivamente con un oggetto JSON, senza testo prima o dopo:
{{"codici": ["..."], "prova": "<citazione letterale, max 12 parole>"}}"""


def call(key, voto, testo, timeout=120):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": rubrica(voto)},
            {"role": "user", "content": testo[:2000]},
        ],
        "temperature": 0,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"].get("content") or ""


def parse(raw, validi):
    m = re.search(r"\{.*\}", re.sub(r"```(?:json)?|```", "", raw), re.S)
    if not m:
        return None, ""
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    codici = [str(c).strip().upper() for c in obj.get("codici", [])]
    return [c for c in codici if c in validi], str(obj.get("prova", ""))[:120]


def leggi(db, label):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT vote, motivation, confidence FROM vote "
        "WHERE label = ? AND vote != 'ERROR' AND motivation != ''", (label,)).fetchall()
    con.close()
    return [(r["vote"], r["motivation"], r["confidence"]) for r in rows]


def codifica(key, righe, rpm, concurrency):
    interval = 60.0 / max(rpm, 1)
    out = [None] * len(righe)

    def one(i):
        voto, testo, _ = righe[i]
        validi = {c for c, _, _ in GRIGLIA.get(voto, [])} | {c for c, _ in EXTRA}
        for tentativo in range(3):
            try:
                codici, prova = parse(call(key, voto, testo), validi)
                if codici is not None:
                    return i, codici, prova
            except Exception:
                time.sleep(2 * (tentativo + 1))
        return i, None, ""

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(len(righe)):
            futures.append(pool.submit(one, i))
            time.sleep(interval)
        for f in futures:
            i, codici, prova = f.result()
            out[i] = (codici, prova)
    return out


def riporta(nome, righe, codifiche, confronta_reale=True):
    print(f"\n{'=' * 78}\n {nome}\n{'=' * 78}")
    descr = {c: d for v in GRIGLIA for c, d, _ in GRIGLIA[v]}
    descr.update(dict(EXTRA))
    reale = {c: q for v in GRIGLIA for c, _, q in GRIGLIA[v]}
    quote = {}

    for voto in ("SI", "NO", "ASTENUTO"):
        idx = [i for i, (v, _, _) in enumerate(righe) if v == voto]
        if not idx:
            continue
        n = len(idx)
        cnt = Counter()
        for i in idx:
            for c in (codifiche[i][0] or []):
                cnt[c] += 1
        conf = [righe[i][2] for i in idx if righe[i][2] is not None]
        media_conf = sum(conf) / len(conf) if conf else float("nan")
        print(f"\n  Ha votato {voto}: {n} agenti, fiducia media dichiarata {media_conf:.2f}")
        print(f"  {'cod':<5}{'quota sim.':>11}{'quota reale':>13}{'scarto':>9}  motivazione")
        ordine = [c for c, _, _ in GRIGLIA.get(voto, [])] + [c for c, _ in EXTRA]
        for c in ordine:
            sim = 100 * cnt[c] / n
            quote[(voto, c)] = sim
            if c in reale and confronta_reale:
                print(f"  {c:<5}{sim:>10.0f}%{reale[c]:>12}%{sim - reale[c]:>+9.0f}  {descr[c][:40]}")
            else:
                print(f"  {c:<5}{sim:>10.0f}%{'—':>13}{'—':>9}  {descr[c][:40]}")
        fuori = sum(cnt[c] for c, _ in EXTRA if c != "X0")
        if fuori:
            print(f"\n  Codici fuori griglia assegnati: {fuori} su {n} motivazioni.")
            print("  Sono ragioni che gli elettori reali non hanno indicato fra le")
            print("  principali. Una quota alta significa che la popolazione sintetica")
            print("  argomenta come il dibattito pubblico e non come l'elettorato.")
    return quote


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db", nargs="+", help="uno o due run.db")
    ap.add_argument("--label", default="final")
    ap.add_argument("--rpm", type=int, default=38)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="motivazioni.json")
    args = ap.parse_args()

    key = os.environ.get("KEY") or sys.exit("variabile d'ambiente KEY non impostata")
    dump, quote_per_run = {}, {}

    for db in args.db:
        righe = leggi(db, args.label)
        if not righe:
            print(f"{db}: nessuna motivazione per label={args.label}")
            continue
        print(f"\n{db}: {len(righe)} motivazioni da codificare…")
        cod = codifica(key, righe, args.rpm, args.concurrency)
        falliti = sum(1 for c, _ in cod if c is None)
        if falliti:
            print(f"  {falliti} codifiche fallite, escluse dai conteggi")
        quote_per_run[db] = riporta(f"{db}  [{args.label}]", righe, cod)
        dump[db] = [{"voto": v, "motivazione": t, "fiducia": f,
                     "codici": c, "prova": p}
                    for (v, t, f), (c, p) in zip(righe, cod)]

    if len(quote_per_run) == 2:
        a, b = list(quote_per_run)
        chiavi = sorted(set(quote_per_run[a]) | set(quote_per_run[b]))
        print(f"\n{'=' * 78}\n STABILITA' FRA ESECUZIONI\n{'=' * 78}")
        print(f"  {'voto':<10}{'cod':<6}{'A':>8}{'B':>8}{'scarto':>9}")
        scarti = []
        for voto, c in chiavi:
            va, vb = quote_per_run[a].get((voto, c), 0), quote_per_run[b].get((voto, c), 0)
            scarti.append(abs(va - vb))
            print(f"  {voto:<10}{c:<6}{va:>7.0f}%{vb:>7.0f}%{vb - va:>+9.0f}")
        print(f"\n  Scarto medio assoluto fra le due esecuzioni: {sum(scarti)/len(scarti):.1f} punti.")
        print("  Confrontarlo con i 42,8 punti di variabilita' della ripartizione")
        print("  del voto. Se e' molto piu' basso, il modello ragiona in modo")
        print("  stabile pur votando in modo instabile, ed e' un risultato.")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    print(f"\nCodifiche complete con citazione giustificativa in {args.out}")
    print("Controllarne una trentina a mano prima di portare le cifre in tesi.")


if __name__ == "__main__":
    main()
