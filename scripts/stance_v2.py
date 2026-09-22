#!/usr/bin/env python3
"""
stance_v2 — classificatore di orientamento unico per Mirorendum.

Nasce per sostituire i due classificatori esistenti, che sullo stesso corpus
di 609 dispacci restituiscono composizioni incompatibili (53/56 contro 1/5).
Finche' la misura non e' affidabile, nessuna quota riportata in tesi regge.

Che cosa cambia rispetto a un classificatore lessicale:
  - rubrica esplicita con criterio decisionale, non parole chiave
  - esempi di riferimento nel prompt, inclusi i casi che sbagliavano
  - classifica la POSIZIONE ESPRESSA, non il tono: una nota che dice
    "si convince che la separazione sia necessaria" e' PRO, non neutra
  - temperatura 0 e JSON forzato
  - ogni etichetta accompagnata da una citazione testuale che la giustifica,
    cosi' la codifica e' ispezionabile e finisce in appendice

Tre comandi:

  classify   classifica i testi di una run e stampa la catena
             notizie -> istituzionali -> cittadini -> note
  sample     estrae N testi in un CSV da codificare a mano
  agree      confronta la codifica manuale con quella automatica
             e calcola accordo grezzo e kappa di Cohen

Uso tipico:
  export KEY=...
  python stance_v2.py classify runs/campagna_completa1/run.db
  python stance_v2.py sample runs/campagna_completa1/run.db --what note -n 50
  # compilare a mano la colonna 'manuale' del CSV prodotto
  python stance_v2.py agree stance_sample.csv
"""

import argparse
import csv
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
LABELS = ["PRO", "CONTRO", "AMBIVALENTE", "NESSUNA"]

RUBRICA = """Sei un codificatore che classifica testi italiani sul referendum \
costituzionale per la separazione delle carriere dei magistrati.

Chi e' PRO sostiene la riforma: separare le carriere di giudici e pubblici \
ministeri, in nome dell'efficienza, della terzieta' del giudice o del giusto \
processo.
Chi e' CONTRO avversa la riforma: la considera un rischio per l'indipendenza \
della magistratura, un assoggettamento del pubblico ministero all'esecutivo, \
o una misura inefficace sui problemi reali della giustizia.

Assegna una di queste quattro etichette:

PRO          il testo sostiene la riforma o accoglie come validi gli argomenti
             a suo favore
CONTRO       il testo avversa la riforma o accoglie come validi gli argomenti
             contrari
AMBIVALENTE  il testo espone entrambe le posizioni senza propendere, oppure
             dichiara esplicitamente indecisione
NESSUNA      il testo non contiene alcuna posizione sul merito: cronaca di
             eventi, calendari, procedure, dichiarazioni riportate senza
             che l'autore le faccia proprie, temi diversi dal referendum

CRITERIO DECISIVO: conta la posizione che il testo ESPRIME, non il tono con
cui la esprime. Un testo pacato, in terza persona o dal lessico tecnico, che
afferma una posizione sul merito, NON e' NESSUNA.

Casi limite risolti:
- "Si convince che la separazione delle carriere sia una misura necessaria
  per l'efficienza del sistema" -> PRO. Esprime una posizione, il fatto che
  sia in terza persona non la rende neutra.
- "Rafforza la posizione contraria alla riforma, considerandola un rischio
  di politicizzazione" -> CONTRO. Stessa ragione.
- "Si consolida la diffidenza verso la retorica dell'efficienza" -> CONTRO.
  Respingere gli argomenti di una parte equivale a schierarsi con l'altra.
- "Il Senato ha approvato il testo in quarta lettura" -> NESSUNA. Cronaca.
- "Meloni ha dichiarato che la riforma garantira' processi piu' equi"
  -> NESSUNA se l'autore si limita a riportare, PRO se la fa propria.
- "La riforma promette efficienza ma rischia di indebolire le garanzie"
  -> AMBIVALENTE solo se non propende; se il 'ma' chiude a sfavore, CONTRO.

Rispondi esclusivamente con un oggetto JSON, senza testo prima o dopo:
{"etichetta": "<una delle quattro>", "prova": "<citazione letterale dal testo, \
max 15 parole, che giustifica l'etichetta; stringa vuota se NESSUNA>"}"""


def call(key, text, timeout=120):
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": RUBRICA},
            {"role": "user", "content": text[:4000]},
        ],
        "temperature": 0,
        "max_tokens": 200,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return payload["choices"][0]["message"].get("content") or ""


def parse(raw):
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    m = re.search(r"\{.*\}", cleaned, re.S)
    if not m:
        return None, ""
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, ""
    lab = str(obj.get("etichetta", "")).strip().upper()
    return (lab if lab in LABELS else None), str(obj.get("prova", ""))[:200]


def classify_many(key, texts, rpm=38, concurrency=4):
    interval = 60.0 / max(rpm, 1)
    out = [None] * len(texts)

    def one(i):
        for attempt in range(3):
            try:
                lab, prova = parse(call(key, texts[i]))
                if lab:
                    return i, lab, prova
            except Exception:
                time.sleep(2 * (attempt + 1))
        return i, None, ""

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for i in range(len(texts)):
            futures.append(pool.submit(one, i))
            time.sleep(interval)
        for f in futures:
            i, lab, prova = f.result()
            out[i] = (lab, prova)
    return out

QUERIES = {
    # Le notizie sono post con kind = 'news'
    "notizie": "SELECT content FROM post WHERE kind = 'news'",
    
    # Account istituzionali: is_voter=0, is_source=0, escludendo le news
    "istituzionali": """SELECT p.content FROM post p 
                        JOIN agent a ON a.agent_id = p.agent_id 
                        WHERE a.is_voter = 0 AND a.is_source = 0 AND p.kind != 'news'""",
                        
    # Cittadini elettori: is_voter=1, escludendo le fonti
    "cittadini": """SELECT p.content FROM post p 
                    JOIN agent a ON a.agent_id = p.agent_id 
                    WHERE a.is_voter = 1 AND a.is_source = 0 AND p.kind != 'news'""",
                    
    # Le note di riflessione memorizzate dagli agenti (la tabella è 'note' e la colonna è 'note')
    "note": "SELECT note FROM note",
}
def read(db, what):
    con = sqlite3.connect(db)
    try:
        rows = con.execute(QUERIES[what]).fetchall()
    except sqlite3.Error as e:
        sys.exit(f"query '{what}' fallita: {e}\nVerificare lo schema con: sqlite3 {db} .schema")
    finally:
        con.close()
    return [r[0] for r in rows if r[0] and r[0].strip()]


def quota_pro(counts):
    schierati = counts["PRO"] + counts["CONTRO"]
    return f"{100 * counts['PRO'] / schierati:.0f}%" if schierati else "n/d"


def cmd_classify(args):
    key = os.environ.get("KEY") or sys.exit("variabile d'ambiente KEY non impostata")
    print(f"\n{'livello':<16}{'testi':>7}{'PRO':>7}{'CONTRO':>8}{'AMBIV':>7}"
          f"{'NESSUNA':>9}{'falliti':>9}{'quota PRO':>11}")
    print("-" * 74)
    dump = {}
    for what in ["notizie", "istituzionali", "cittadini", "note"]:
        texts = read(args.db, what)
        if not texts:
            print(f"{what:<16}{0:>7}")
            continue
        res = classify_many(key, texts, args.rpm, args.concurrency)
        counts = Counter(lab for lab, _ in res if lab)
        falliti = sum(1 for lab, _ in res if not lab)
        print(f"{what:<16}{len(texts):>7}{counts['PRO']:>7}{counts['CONTRO']:>8}"
              f"{counts['AMBIVALENTE']:>7}{counts['NESSUNA']:>9}{falliti:>9}"
              f"{quota_pro(counts):>11}")
        dump[what] = [
            {"testo": t, "etichetta": lab, "prova": pr}
            for t, (lab, pr) in zip(texts, res)
        ]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dump, f, ensure_ascii=False, indent=1)
    print(f"\nLa quota PRO e' calcolata sui soli testi schierati (PRO + CONTRO).")
    print(f"Etichette complete con citazione giustificativa in {args.out}")


def cmd_sample(args):
    key = os.environ.get("KEY") or sys.exit("variabile d'ambiente KEY non impostata")
    import random
    texts = read(args.db, args.what)
    random.seed(args.seed)
    picked = random.sample(texts, min(args.n, len(texts)))
    res = classify_many(key, picked, args.rpm, args.concurrency)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "testo", "automatica", "prova", "manuale"])
        for i, (t, (lab, pr)) in enumerate(zip(picked, res), 1):
            w.writerow([i, t.replace("\n", " "), lab or "FALLITA", pr, ""])
    print(f"{len(picked)} testi scritti in {args.out}")
    print("Compilare a mano la colonna 'manuale' con PRO, CONTRO, AMBIVALENTE")
    print("o NESSUNA, SENZA guardare la colonna 'automatica', poi lanciare:")
    print(f"  python stance_v2.py agree {args.out}")


def cmd_agree(args):
    rows = [r for r in csv.DictReader(open(args.csv, encoding="utf-8"))
            if r["manuale"].strip()]
    if not rows:
        sys.exit("nessuna riga codificata a mano: compilare la colonna 'manuale'")
    a = [r["automatica"].strip().upper() for r in rows]
    m = [r["manuale"].strip().upper() for r in rows]
    n = len(rows)
    acc = sum(x == y for x, y in zip(a, m)) / n
    ca, cm = Counter(a), Counter(m)
    atteso = sum(ca[k] * cm[k] for k in set(a) | set(m)) / (n * n)
    kappa = (acc - atteso) / (1 - atteso) if atteso < 1 else 1.0

    print(f"\ncoppie codificate       {n}")
    print(f"accordo grezzo          {acc:.1%}")
    print(f"kappa di Cohen          {kappa:.2f}", end="  ")
    print("(sostanziale)" if kappa >= .61 else
          "(moderato, dichiararlo)" if kappa >= .41 else
          "(INSUFFICIENTE: usare la codifica manuale nel testo)")

    print("\nmatrice di confusione (righe manuale, colonne automatica)")
    etich = sorted(set(a) | set(m))
    print(f"{'':<14}" + "".join(f"{e[:6]:>8}" for e in etich))
    for em in etich:
        riga = [sum(1 for x, y in zip(a, m) if y == em and x == ea) for ea in etich]
        print(f"{em:<14}" + "".join(f"{v:>8}" for v in riga))
    print("\nLe celle fuori diagonale indicano dove il classificatore sbaglia:")
    print("se si concentrano in una riga, la rubrica va corretta per quella")
    print("etichetta prima di rilanciare 'classify'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("classify")
    c.add_argument("db")
    c.add_argument("--out", default="stance_v2.json")
    c.add_argument("--rpm", type=int, default=38)
    c.add_argument("--concurrency", type=int, default=4)
    c.set_defaults(fn=cmd_classify)

    s = sub.add_parser("sample")
    s.add_argument("db")
    s.add_argument("--what", default="note", choices=list(QUERIES))
    s.add_argument("-n", type=int, default=50)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="stance_sample.csv")
    s.add_argument("--rpm", type=int, default=38)
    s.add_argument("--concurrency", type=int, default=4)
    s.set_defaults(fn=cmd_sample)

    g = sub.add_parser("agree")
    g.add_argument("csv")
    g.set_defaults(fn=cmd_agree)

    args = ap.parse_args()
    args.fn(args)
