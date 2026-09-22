#!/usr/bin/env python3
"""
revote — rifà SOLO le rilevazioni di voto sugli stati gia' registrati di una
run, senza rieseguire la simulazione.

Perche' e' possibile. La biografia statica e' congelata al tick 0 e le note
stanno nel database: il "PRIMA" e il "DOPO" di ogni agente sono gia' salvati.
Rifare il voto con un altro quesito o un'altra temperatura costa cento
chiamate per condizione, contro le ventimila della simulazione.

Riusa `run_survey` di mirrorfish, quindi il prompt di voto e' IDENTICO a
quello della simulazione: l'unica cosa che cambia e' cio' che si sceglie di
cambiare.

Lavora SEMPRE su una copia del database. La run originale non viene toccata.

Due modalita':

  --diagnosi          nessuna chiamata al modello. Stampa la traiettoria V(t)
                      dalle rilevazioni intermedie gia' presenti, e il voto
                      iniziale e finale per collocazione politica dichiarata.
                      E' il test dell'ipotesi di ordinamento partitico.

  (default)           rivota per ogni combinazione di fase x quesito x
                      temperatura, e per ciascuna riporta anche l'accordo
                      con il voto originale agente per agente.

Uso, dalla radice del progetto:
  export LLM_API_KEY=...
  python scripts/revote.py runs/campagna_completa1/run.db --diagnosi
  python scripts/revote.py runs/campagna_completa1/run.db --piano
  python scripts/revote.py runs/campagna_completa1/run.db     # tutti i quesiti in quesiti/
  python scripts/revote.py runs/campagna_completa1/run.db \\
      --quesiti ballot --fasi final --temperature 0.2 1.0
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import re
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirrorfish.config import LLMConfig          # noqa: E402
from mirrorfish.llm import build_client          # noqa: E402
from mirrorfish.store import Store               # noqa: E402
from mirrorfish.survey import carica_quesito, run_survey  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def cartella_predefinita() -> Path:
    """I quesiti stanno in quesiti/ alla radice del progetto; survey.py li
    cerca invece in mirrorfish/quesiti/. Si prova la prima, poi la seconda."""
    for c in (ROOT / "quesiti", ROOT / "mirrorfish" / "quesiti"):
        if c.is_dir() and any(c.glob("*.txt")):
            return c
    return ROOT / "quesiti"


def risolvi_quesiti(scelti: list[str] | None, cartella: Path) -> list[tuple[str, Path]]:
    """Senza --quesiti: tutti i .txt della cartella, in ordine alfabetico.
    Con --quesiti: accetta nomi (senza estensione) o percorsi di file."""
    if not scelti:
        trovati = sorted(cartella.glob("*.txt"))
        if not trovati:
            sys.exit(f"nessun quesito .txt in {cartella}")
        return [(p.stem, p) for p in trovati]
    out = []
    for s in scelti:
        p = Path(s)
        if not p.is_file():
            p = cartella / f"{s}.txt"
        if not p.is_file():
            disponibili = ", ".join(x.stem for x in sorted(cartella.glob("*.txt")))
            sys.exit(f"quesito non trovato: {s!r}. In {cartella}: {disponibili}")
        out.append((p.stem, p))
    return out

# Stesso criterio di analyze_run.py, per coerenza fra gli strumenti.
PART_NO = re.compile(r"Partito Democratico|Movimento 5 Stelle|Alleanza Verdi|AVS", re.I)
PART_SI = re.compile(r"Fratelli d'Italia|Forza Italia|Lega\b", re.I)


def schieramento(bio: str) -> str:
    if PART_NO.search(bio or ""):
        return "centrosinistra"
    if PART_SI.search(bio or ""):
        return "centrodestra"
    return "non dichiarato"


# --------------------------------------------------------------------------- #
# Lettura
# --------------------------------------------------------------------------- #

def voti(con: sqlite3.Connection, label: str) -> dict[int, str]:
    return {r[0]: r[1] for r in con.execute(
        "SELECT agent_id, vote FROM vote WHERE label = ? AND vote != 'ERROR'", (label,))}


def bio(con: sqlite3.Connection) -> dict[int, str]:
    return {r[0]: r[1] or "" for r in con.execute(
        "SELECT agent_id, static_bio FROM agent WHERE is_voter = 1 AND is_source = 0")}


def riga_esito(v: dict[int, str]) -> str:
    si = sum(1 for x in v.values() if x == "SI")
    no = sum(1 for x in v.values() if x == "NO")
    ast = sum(1 for x in v.values() if x == "ASTENUTO")
    n = si + no + ast
    V = f"{100 * si / (si + no):5.1f}%" if si + no else "   n/d"
    aff = f"{100 * (si + no) / n:5.1f}%" if n else "   n/d"
    return f"SI {si:>3}  NO {no:>3}  AST {ast:>3}   V {V}   affluenza {aff}"


def per_schieramento(v: dict[int, str], bios: dict[int, str]) -> None:
    gruppi: dict[str, dict[str, int]] = {}
    for aid, voto in v.items():
        g = schieramento(bios.get(aid, ""))
        gruppi.setdefault(g, {"SI": 0, "NO": 0, "ASTENUTO": 0})[voto] += 1
    for g in ("centrodestra", "centrosinistra", "non dichiarato"):
        d = gruppi.get(g)
        if not d:
            continue
        n = sum(d.values())
        print(f"      {g:<16} n={n:>3}   SI {100*d['SI']/n:4.0f}%   "
              f"NO {100*d['NO']/n:4.0f}%   AST {100*d['ASTENUTO']/n:4.0f}%")


def accordo(a: dict[int, str], b: dict[int, str]) -> str:
    comuni = set(a) & set(b)
    if not comuni:
        return "n/d"
    uguali = sum(1 for k in comuni if a[k] == b[k])
    return f"{100 * uguali / len(comuni):.0f}% ({uguali}/{len(comuni)})"


# --------------------------------------------------------------------------- #
# Diagnosi senza chiamate
# --------------------------------------------------------------------------- #

def diagnosi(db: Path) -> None:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    bios = bio(con)

    print("=" * 78)
    print(f" TRAIETTORIA  {db}")
    print("=" * 78)
    labels = [r[0] for r in con.execute(
        "SELECT label FROM vote GROUP BY label ORDER BY "
        "CASE label WHEN 'baseline' THEN -1 WHEN 'final' THEN 1e9 "
        "ELSE MIN(COALESCE(tick, 0)) END")]
    for lab in labels:
        if lab.startswith("rv_"):
            continue
        print(f"  {lab:<12} {riga_esito(voti(con, lab))}")
    print("\n  Se V scende tutto nelle prime rilevazioni e poi resta piatta, la")
    print("  deriva e' un aggiustamento iniziale che si esaurisce presto. Se scende")
    print("  in modo graduale per tutta la campagna, e' un accumulo.")

    print("\n" + "=" * 78)
    print(" VOTO PER COLLOCAZIONE DICHIARATA NELLA BIOGRAFIA")
    print("=" * 78)
    for lab in ("baseline", "final"):
        v = voti(con, lab)
        if v:
            print(f"\n  [{lab}]")
            per_schieramento(v, bios)
    print("""
  Come leggerlo. Nella campagna reale il centrodestra sosteneva la riforma e il
  centrosinistra la avversava.
  - Se al baseline entrambi votano SI e al final si separano per schieramento,
    la deriva e' ORDINAMENTO: la popolazione scopre da che parte sta la riforma
    e si allinea alla propria collocazione. E' una dinamica sensata.
  - Se al final anche il centrodestra vota NO, la deriva e' COLLASSO: gli
    argomenti contrari prevalgono sulla collocazione del personaggio.""")
    con.close()


# --------------------------------------------------------------------------- #
# Rivotazione
# --------------------------------------------------------------------------- #

def copia(db: Path, dest: Path) -> None:
    """Copia coerente anche con il WAL attivo: il backup SQLite include le
    pagine non ancora riversate nel file principale, una copia di file no."""
    if dest.exists():
        dest.unlink()
    src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    dst = sqlite3.connect(dest)
    src.backup(dst)
    dst.close()
    src.close()


async def rivota(args) -> None:
    src = Path(args.db)
    dest = Path(args.out) if args.out else src.with_name("revote.db")
    cartella = Path(args.cartella) if args.cartella else cartella_predefinita()
    quesiti = risolvi_quesiti(args.quesiti, cartella)
    print(f"quesiti da {cartella}: {', '.join(n for n, _ in quesiti)}")
    combinazioni = [(f, n, p, t) for f in args.fasi for n, p in quesiti for t in args.temperature]
    chiamate = len(combinazioni) * 100
    print(f"{len(combinazioni)} condizioni, circa {chiamate} chiamate "
          f"(~{chiamate / args.rpm:.0f} min a {args.rpm:g} rpm)")
    if args.piano:
        for f, n, _, t in combinazioni:
            print(f"  rv_{f}_{n}_t{t:g}")
        return

    copia(src, dest)
    print(f"lavoro sulla copia {dest}; l'originale non viene modificato\n")

    cfg = LLMConfig.from_env(requests_per_minute=args.rpm, concurrency=args.concurrency)
    client = build_client(cfg)
    store = Store(dest)
    con = sqlite3.connect(dest)
    bios = bio(con)
    originali = {"baseline": voti(con, "baseline"), "final": voti(con, "final")}

    riepilogo = []
    try:
        for fase, nome, percorso, t in combinazioni:
            testo, _, impronta = carica_quesito(str(percorso))
            cfg.vote_temperature = t
            label = f"rv_{fase}_{nome}_t{t:g}"
            print(f"\n--- {label}  (quesito {impronta}) ---")
            await run_survey(store, client, cfg, label=label,
                             baseline=(fase == "baseline"), question=testo,
                             verbose=False)
            v = voti(con, label)
            riepilogo.append((label, v, accordo(v, originali[fase])))
            print(f"  {riga_esito(v)}")
            per_schieramento(v, bios)
    finally:
        await client.aclose()
        con.close()

    print("\n" + "=" * 78)
    print(" RIEPILOGO")
    print("=" * 78)
    for fase in ("baseline", "final"):
        if originali[fase]:
            print(f"  {'originale_' + fase:<34} {riga_esito(originali[fase])}")
    for label, v, acc in riepilogo:
        print(f"  {label:<34} {riga_esito(v)}   accordo con originale {acc}")
    print("""
  L'accordo con l'originale, a parita' di quesito e temperatura, misura la
  ripetibilita' della lettura del voto: e' il pavimento di rumore dello
  strumento. Se rivotando lo stesso agente con lo stesso stato si ottiene la
  stessa risposta solo nell'80% dei casi, ogni spostamento sotto il 20% e'
  indistinguibile dal rumore.""")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--diagnosi", action="store_true",
                    help="nessuna chiamata: traiettoria e voto per schieramento")
    ap.add_argument("--piano", action="store_true",
                    help="mostra le condizioni e il costo senza eseguire")
    ap.add_argument("--quesiti", nargs="+", default=None,
                    help="nomi o percorsi; se omesso, tutti i .txt della cartella")
    ap.add_argument("--cartella", default=None,
                    help="cartella dei quesiti (default: quesiti/ alla radice)")
    ap.add_argument("--fasi", nargs="+", default=["baseline", "final"],
                    choices=["baseline", "final"])
    ap.add_argument("--temperature", nargs="+", type=float, default=[0.2])
    ap.add_argument("--rpm", type=float, default=38)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", default=None, help="percorso della copia (default revote.db)")
    args = ap.parse_args()

    if args.diagnosi:
        diagnosi(Path(args.db))
    else:
        asyncio.run(rivota(args))


if __name__ == "__main__":
    main()
