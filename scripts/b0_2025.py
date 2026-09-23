#!/usr/bin/env python3
"""
B0 fuori campione — i cinque referendum abrogativi dell'8-9 giugno 2025.

Ripete la prova pilota (una sola chiamata, ragionamento attivo, cinque quesiti
in un prompt) con lo STESSO protocollo della sonda P1 sul referendum del 2026:
un quesito per chiamata, trenta ripetizioni, temperatura di voto, ragionamento
disattivato, stima aggregata in JSON, data omessa.

Perche' questo referendum. Il modello non ne conosce l'esito (verificato), e i
risultati reali sono lontani dal 50%: e' il caso che distingue una capacita'
predittiva da una stima strutturalmente centrale.

Differenza di contesto rispetto al 2026, dichiarata e necessaria: questi sono
referendum ABROGATIVI, validi solo se vota la maggioranza degli aventi diritto.
La regola del quorum e' nota a ogni elettore prima del voto e va nel prompt,
come nel 2026 ci andava la sua assenza.

I testi dei quesiti sono descrizioni sintetiche e neutre dell'effetto del Si'
e del No. Le schede ufficiali dei referendum abrogativi elencano gli articoli
da abrogare e non ne descrivono il contenuto; chi volesse usarle puo' metterle
in una cartella come q1.txt ... q5.txt e passarla con --testi.

Uso:
  export KEY=...            # oppure LLM_API_KEY
  python b0_2025.py --temperature 0.2 --max-tokens 640
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://api.ailabroma3.it/v1/chat/completions"
MODEL = "lab-qwen36"
AFFLUENZA_REALE = 29.83

# (codice, descrizione, quota di Si' reale fra i voti validi)
QUESITI = [
    ("Q1_reintegro",
     "Abrogazione della disciplina dei licenziamenti introdotta nel 2015 con il "
     "contratto a tutele crescenti. Votando SI si chiede di abrogarla, ripristinando "
     "per i lavoratori interessati la possibilita' del reintegro nel posto di lavoro "
     "in caso di licenziamento illegittimo. Votando NO la disciplina attuale resta "
     "in vigore.", 87.57),
    ("Q2_piccole_imprese",
     "Abrogazione del limite massimo all'indennita' che le piccole imprese devono "
     "corrispondere in caso di licenziamento illegittimo. Votando SI si elimina il "
     "tetto e la misura dell'indennita' e' lasciata al giudice. Votando NO il tetto "
     "resta in vigore.", 86.02),
    ("Q3_contratti_termine",
     "Abrogazione di norme sui contratti di lavoro a tempo determinato. Votando SI "
     "si reintroduce l'obbligo di indicare una motivazione anche per i contratti a "
     "termine di durata inferiore a dodici mesi. Votando NO resta possibile "
     "stipularli senza motivazione.", 87.53),
    ("Q4_appalti",
     "Abrogazione della norma che esclude la responsabilita' del committente per "
     "gli infortuni sul lavoro dovuti ai rischi specifici dell'attivita' "
     "dell'appaltatore o del subappaltatore. Votando SI la responsabilita' solidale "
     "del committente si estende anche a questi casi. Votando NO l'esclusione resta "
     "in vigore.", 85.78),
    ("Q5_cittadinanza",
     "Riduzione da dieci a cinque anni del periodo di residenza legale in Italia "
     "richiesto ai cittadini stranieri maggiorenni non comunitari per chiedere la "
     "cittadinanza italiana. Votando SI il periodo si dimezza. Votando NO resta di "
     "dieci anni.", 65.34),
]

PROMPT = """In Italia si tiene un referendum abrogativo sul seguente quesito.

{quesito}

Il referendum e' valido solo se vota la maggioranza degli aventi diritto; in caso \
contrario la norma resta in vigore qualunque sia il risultato.

Stima la ripartizione del voto nel corpo elettorale italiano. Le tre quote sono \
riferite al totale degli aventi diritto e devono sommare a 100.

Rispondi esclusivamente con un oggetto JSON, senza alcun testo prima o dopo, \
nella forma esatta:
{{"si": <numero>, "no": <numero>, "astenuti": <numero>}}"""


def call(key, prompt, temperature, max_tokens, timeout=180):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"].get("content") or ""


def parse(raw):
    m = re.search(r"\{.*?\}", re.sub(r"```(?:json)?|```", "", raw), re.S)
    if not m:
        return None
    try:
        o = json.loads(m.group(0))
        si, no, ast = float(o["si"]), float(o["no"]), float(o["astenuti"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    tot = si + no + ast
    return None if tot <= 0 else (si * 100 / tot, no * 100 / tot, ast * 100 / tot)


def fmt(xs):
    if not xs:
        return "n/d"
    sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
    return f"{statistics.mean(xs):5.1f} (sd {sd:4.1f}, es {sd / len(xs) ** .5:3.1f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--temperature", type=float, required=True,
                    help="la stessa della chiamata di voto: 0.2")
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--testi", default=None,
                    help="cartella con q1.txt..q5.txt per usare testi diversi")
    ap.add_argument("--rpm", type=float, default=38)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default="b0_2025_raw.jsonl")
    args = ap.parse_args()

    key = os.environ.get("KEY") or os.environ.get("LLM_API_KEY")
    if not key:
        sys.exit("impostare KEY oppure LLM_API_KEY")

    quesiti = list(QUESITI)
    if args.testi:
        for i, (cod, _, reale) in enumerate(quesiti):
            p = Path(args.testi) / f"q{i + 1}.txt"
            if not p.is_file():
                sys.exit(f"manca {p}")
            quesiti[i] = (cod, p.read_text(encoding="utf-8").strip(), reale)

    jobs = [(cod, PROMPT.format(quesito=testo), r) for cod, testo, _ in quesiti
            for r in range(args.reps)]
    print(f"{len(jobs)} chiamate ({len(quesiti)} quesiti x {args.reps} ripetizioni)")
    interval = 60.0 / max(args.rpm, 1)

    def run(job):
        cod, prompt, rep = job
        for tentativo in range(3):
            try:
                raw = call(key, prompt, args.temperature, args.max_tokens)
                return {"quesito": cod, "rep": rep, "raw": raw, "stima": parse(raw)}
            except (urllib.error.URLError, TimeoutError) as e:
                err = str(e)
                time.sleep(2 * (tentativo + 1))
        return {"quesito": cod, "rep": rep, "raw": "", "stima": None, "errore": err}

    risultati = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for j in jobs:
            futures.append(pool.submit(run, j))
            time.sleep(interval)
        for n, f in enumerate(futures, 1):
            risultati.append(f.result())
            if n % 30 == 0:
                print(f"  {n}/{len(jobs)}")

    with open(args.out, "w", encoding="utf-8") as fh:
        for r in risultati:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'quesito':<22}{'V stimato':>28}{'V reale':>9}{'errore':>8}   affluenza stimata")
    print("-" * 96)
    errori, aff_tutte, sbagliati = [], [], 0
    for cod, _, reale in quesiti:
        ok = [r["stima"] for r in risultati if r["quesito"] == cod and r["stima"]]
        falliti = args.reps - len(ok)
        v = [s[0] * 100 / (s[0] + s[1]) for s in ok if s[0] + s[1] > 0]
        aff = [s[0] + s[1] for s in ok]
        aff_tutte += aff
        if v:
            e = statistics.mean(v) - reale
            errori.append(abs(e))
            if (statistics.mean(v) > 50) != (reale > 50):
                sbagliati += 1
            nota = f"   [{falliti} non interpretabili]" if falliti else ""
            print(f"{cod:<22}{fmt(v):>28}{reale:>8.1f}%{e:>+8.1f}   {fmt(aff)}{nota}")
        else:
            print(f"{cod:<22}{'nessuna risposta utilizzabile':>28}")

    if errori:
        print(f"\nerrore medio assoluto su V    : {statistics.mean(errori):.1f} punti")
        print(f"quesiti con vincitore sbagliato: {sbagliati} su {len(errori)}")
    if aff_tutte:
        m = statistics.mean(aff_tutte)
        print(f"affluenza media stimata       : {m:.1f}%  (reale {AFFLUENZA_REALE}%, "
              f"errore {m - AFFLUENZA_REALE:+.1f})")
        print(f"quorum previsto raggiunto     : {'SI' if m > 50 else 'NO'}  (reale: NO)")
    print(f"\nrisposte grezze in {args.out}")


if __name__ == "__main__":
    main()
