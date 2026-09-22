#!/usr/bin/env python3
"""
B0 — interrogazione diretta del modello come oracolo.

Non simula elettori: chiede al modello una stima aggregata. Serve come
termine di paragone piu' economico per B1 (sondaggio sintetico) e B2
(simulazione completa).

Tre sonde, ciascuna ripetuta N volte:
  P1  stima aggregata con il quesito di scheda      (quesiti/ballot.txt)
  P2  stima aggregata con il quesito informato      (quesiti/informed.txt)
  P3  voto di un "elettore italiano medio"          (quesiti/ballot.txt)

IMPORTANTE: --temperature e --max-tokens devono coincidere con quelli
usati dalla chiamata di voto della simulazione, altrimenti B0 e B1/B2
differiscono per piu' di un fattore e il confronto non e' valido.

Uso:
  export KEY=...
  python b0_baseline.py --reps 30 --temperature 0.7 --max-tokens 640

Produce:
  b0_raw.jsonl   ogni risposta grezza, per l'appendice della tesi
  b0_results.csv una riga per ripetizione
  un riepilogo a schermo con media, deviazione standard e V = si/(si+no)
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_URL = "https://api.ailabroma3.it/v1/chat/completions"
DEFAULT_MODEL = "lab-qwen36"

# Bersaglio reale: voti validi in Italia, referendum 22-23 marzo 2026.
TARGET_V = 46.26          # quota Si fra chi si esprime
TARGET_AFFLUENZA = 58.93  # affluenza

# --- Le tre sonde -----------------------------------------------------------
# La data del voto e' deliberatamente omessa: indicandola il modello
# riconosce di essere interrogato oltre il proprio limite di conoscenza e si
# astiene. Quella versione e' la sonda di contaminazione L3, misura separata.

P_AGGREGATE = """In Italia si tiene un referendum costituzionale confermativo sul seguente quesito.

{quesito}

Non e' previsto alcun quorum di partecipazione: l'esito e' valido qualunque sia l'affluenza.

Stima la ripartizione del voto nel corpo elettorale italiano. Le tre quote sono \
riferite al totale degli aventi diritto e devono sommare a 100.

Rispondi esclusivamente con un oggetto JSON, senza alcun testo prima o dopo, \
nella forma esatta:
{{"si": <numero>, "no": <numero>, "astenuti": <numero>}}"""

P_ELETTORE = """Immagina un elettore italiano medio chiamato a pronunciarsi sul seguente quesito \
referendario.

{quesito}

Non e' previsto alcun quorum di partecipazione.

Come voterebbe? Rispondi esclusivamente con un oggetto JSON, senza alcun testo \
prima o dopo, nella forma esatta:
{{"voto": "SI"}} oppure {{"voto": "NO"}} oppure {{"voto": "ASTENUTO"}}"""


def build_probes(ballot: str, informed: str):
    return [
        ("P1_aggregata_ballot", P_AGGREGATE.format(quesito=ballot), "aggregate"),
        ("P2_aggregata_informed", P_AGGREGATE.format(quesito=informed), "aggregate"),
        ("P3_elettore_medio", P_ELETTORE.format(quesito=ballot), "single"),
    ]


def call_model(url, key, model, prompt, temperature, max_tokens, thinking, timeout=180):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if not thinking:
        # Qwen 3.x: la modalita' di ragionamento e' attiva per default e
        # consuma il budget di output prima di arrivare al JSON.
        body["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    choice = payload["choices"][0]
    return {
        "content": choice["message"].get("content") or "",
        "finish_reason": choice.get("finish_reason"),
        "usage": payload.get("usage", {}),
    }


def extract_json(text):
    """Il modello a volte incornicia il JSON in un blocco di codice o vi
    premette del testo. Si cerca il primo oggetto bilanciato."""
    cleaned = re.sub(r"```(?:json)?|```", "", text).strip()
    depth, start = 0, None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def parse(probe_kind, obj):
    """Ritorna (si, no, astenuti) in percentuale sugli aventi diritto, oppure None."""
    if obj is None:
        return None
    if probe_kind == "aggregate":
        try:
            si, no, ast = float(obj["si"]), float(obj["no"]), float(obj["astenuti"])
        except (KeyError, TypeError, ValueError):
            return None
        total = si + no + ast
        if total <= 0:
            return None
        # normalizzazione difensiva: il modello non sempre somma a 100
        return (si * 100 / total, no * 100 / total, ast * 100 / total)
    voto = str(obj.get("voto", "")).strip().upper()
    if voto == "SI":
        return (100.0, 0.0, 0.0)
    if voto == "NO":
        return (0.0, 100.0, 0.0)
    if voto in ("ASTENUTO", "ASTENUTA", "ASTENSIONE"):
        return (0.0, 0.0, 100.0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--temperature", type=float, required=True,
                    help="deve coincidere con quella della chiamata di voto della simulazione")
    ap.add_argument("--max-tokens", type=int, required=True,
                    help="idem: stesso budget della chiamata di voto")
    ap.add_argument("--thinking", action="store_true",
                    help="lascia attiva la modalita' di ragionamento (default: disattiva)")
    ap.add_argument("--ballot", default="quesiti/ballot.txt")
    ap.add_argument("--informed", default="quesiti/informed.txt")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rpm", type=int, default=38)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out-prefix", default="b0")
    args = ap.parse_args()

    key = os.environ.get("KEY")
    if not key:
        sys.exit("variabile d'ambiente KEY non impostata")

    try:
        ballot = open(args.ballot, encoding="utf-8").read().strip()
        informed = open(args.informed, encoding="utf-8").read().strip()
    except OSError as e:
        sys.exit(f"quesito non leggibile: {e}")

    probes = build_probes(ballot, informed)
    jobs = [(name, prompt, kind, r)
            for name, prompt, kind in probes
            for r in range(args.reps)]

    interval = 60.0 / max(args.rpm, 1)
    rows, raw = [], []

    def run(job):
        name, prompt, kind, rep = job
        try:
            res = call_model(args.url, key, args.model, prompt,
                             args.temperature, args.max_tokens, args.thinking)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            return {"probe": name, "rep": rep, "error": str(e), "content": ""}
        return {"probe": name, "rep": rep, "kind": kind, "error": None, **res}

    print(f"{len(jobs)} chiamate ({len(probes)} sonde x {args.reps} ripetizioni)")
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = []
        for job in jobs:
            futures.append(pool.submit(run, job))
            time.sleep(interval)
        for n, fut in enumerate(as_completed(futures), 1):
            out = fut.result()
            raw.append(out)
            kind = out.get("kind", "aggregate")
            parsed = parse(kind, extract_json(out.get("content", "")))
            rows.append({
                "probe": out["probe"],
                "rep": out["rep"],
                "si": parsed[0] if parsed else "",
                "no": parsed[1] if parsed else "",
                "astenuti": parsed[2] if parsed else "",
                "ok": bool(parsed),
                "finish_reason": out.get("finish_reason", ""),
                "error": out.get("error") or "",
            })
            if n % 10 == 0:
                print(f"  {n}/{len(jobs)}")

    with open(f"{args.out_prefix}_raw.jsonl", "w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{args.out_prefix}_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nbersaglio reale: V = {TARGET_V}%  affluenza = {TARGET_AFFLUENZA}%\n")
    for name, _, _ in probes:
        good = [r for r in rows if r["probe"] == name and r["ok"]]
        failed = sum(1 for r in rows if r["probe"] == name and not r["ok"])
        print(f"--- {name} ---")
        print(f"  valide {len(good)}/{args.reps}, non interpretabili {failed}")
        if not good:
            print("  nessuna risposta utilizzabile\n")
            continue
        si = [r["si"] for r in good]
        aff = [r["si"] + r["no"] for r in good]
        v = [r["si"] * 100 / (r["si"] + r["no"]) for r in good if (r["si"] + r["no"]) > 0]
        def fmt(xs):
            if not xs:
                return "n/d"
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            return f"{statistics.mean(xs):.1f} (sd {sd:.1f})"
        print(f"  Si su aventi diritto : {fmt(si)}")
        print(f"  affluenza            : {fmt(aff)}  [errore {statistics.mean(aff) - TARGET_AFFLUENZA:+.1f}]")
        if v:
            print(f"  V = Si/(Si+No)       : {fmt(v)}  [errore {statistics.mean(v) - TARGET_V:+.1f}]")
        print()

    print(f"scritti {args.out_prefix}_raw.jsonl e {args.out_prefix}_results.csv")


if __name__ == "__main__":
    main()
