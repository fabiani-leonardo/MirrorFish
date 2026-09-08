#!/usr/bin/env python3
"""
Audit dell'orientamento: l'ambiente informativo della simulazione e'
bilanciato, oppure produce la deriva da solo?

    python scripts/stance_audit.py runs/expbr/run.db

Perche' serve. Il controllo con `--recommender random` ha escluso che la
deriva verso il NO venga dall'ORDINAMENTO del feed: con un ordine casuale la
deriva c'e' lo stesso, e anzi e' leggermente piu' forte. Ma quel controllo non
esclude la spiegazione piu' semplice, cioe' che siano i CONTENUTI a essere
sbilanciati. Il ranking casuale mostra gli stessi post, solo in ordine
diverso: se il 70% del dibattito e' contrario alla riforma, mescolare le carte
non cambia nulla.

Cosa guarda, in ordine di vicinanza al voto:
  1. le notizie ANSA (lo stimolo esterno)
  2. i post degli account istituzionali, uno per uno
  3. i post dei cittadini
  4. le NOTE, che sono l'unico input che il voto finale legge oltre alla
     biografia, e quindi il punto in cui l'eventuale sbilanciamento diventa
     voto

Se le note sono massicciamente contrarie mentre le notizie sono neutre, lo
sbilanciamento nasce DENTRO la popolazione e non e' un effetto
dell'informazione: e' un effetto della composizione degli agenti, o del prior
del modello.

La classificazione e' lessicale e grossolana, quindi va letta come ordine di
grandezza. Un testo che contiene marcatori di entrambi i segni viene contato
come ambivalente, non assegnato d'ufficio.
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Marcatori piu' larghi di quelli della sezione 1 di analyze_run, perche' qui
# si classificano post e note scritti dagli agenti, non lanci d'agenzia.
RE_NO = re.compile(
    r"\bvoto no\b|\bvotare no\b|#\w*no\b|contro la riforma|"
    r"indipendenza (?:della|dei) magistrat|autonomia della magistratura|"
    r"controllo politic|sottomett|piegare i giudici|"
    r"separazione delle carriere (?:non|no)|difendere la costituzione",
    re.I)
RE_SI = re.compile(
    r"\bvoto s[iì]\b|\bvotare s[iì]\b|#\w*s[iì]\b|a favore della riforma|"
    r"separare le carriere|giustizia pi[uù] (?:rapida|efficiente|giusta)|"
    r"casta|privilegi dei magistrat|riforma necessaria|"
    r"processo (?:pi[uù] )?(?:breve|veloce)",
    re.I)


def stance(t: str) -> str:
    si, no = bool(RE_SI.search(t)), bool(RE_NO.search(t))
    if si and no:
        return "ambivalente"
    if si:
        return "SI"
    if no:
        return "NO"
    return "neutro"


def riga(nome: str, testi: list[str], larghezza: int = 30) -> tuple[int, int]:
    c = {"SI": 0, "NO": 0, "neutro": 0, "ambivalente": 0}
    for t in testi:
        c[stance(t)] += 1
    n = len(testi) or 1
    espliciti = c["SI"] + c["NO"]
    bil = f"{c['SI'] / espliciti:.0%} SI" if espliciti else "-"
    print(f"  {nome:<{larghezza}}{n:>6}{c['SI']:>6}{c['NO']:>6}"
          f"{c['ambivalente']:>7}{c['neutro']:>8}{bil:>10}")
    return c["SI"], c["NO"]


def intestazione(titolo: str, larghezza: int = 30) -> None:
    print(f"\n  {titolo}")
    print(f"  {'':<{larghezza}}{'testi':>6}{'SI':>6}{'NO':>6}"
          f"{'ambiv':>7}{'neutro':>8}{'quota SI':>10}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--esempi", type=int, default=6,
                    help="quante note stampare per campione")
    a = ap.parse_args()

    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row

    print("=" * 72)
    print(" AUDIT DELL'ORIENTAMENTO — l'ambiente e' bilanciato?")
    print("=" * 72)

    # --- 1. stimolo esterno ------------------------------------------------ #
    intestazione("1. NOTIZIE ANSA (stimolo esterno)")
    news = [r["content"] for r in c.execute(
        "SELECT content FROM post WHERE kind = 'news'")]
    riga("notizie iniettate", news)

    # --- 2. account istituzionali, uno per uno ----------------------------- #
    intestazione("2. ACCOUNT ISTITUZIONALI (uno per uno)")
    ist = c.execute(
        "SELECT agent_id, username FROM agent "
        "WHERE is_voter = 0 AND is_source = 0 ORDER BY username").fetchall()
    tot_si = tot_no = 0
    for r in ist:
        testi = [x["content"] for x in c.execute(
            "SELECT content FROM post WHERE agent_id = ? AND kind != 'news'",
            (r["agent_id"],))]
        s, n = riga(r["username"][:29], testi)
        tot_si += s
        tot_no += n
    if not ist:
        print("  (nessun account istituzionale in questa popolazione)")
    else:
        espl = tot_si + tot_no
        print(f"\n  Fra gli istituzionali con orientamento esplicito: "
              f"{tot_si} SI, {tot_no} NO"
              + (f" ({tot_si / espl:.0%} SI)" if espl else ""))
        if espl and not 0.35 <= tot_si / espl <= 0.65:
            print("  SQUILIBRIO. Gli account istituzionali sono i piu'")
            print("  prolifici della simulazione e hanno portata garantita.")
            print("  Se sono sbilanciati, l'ambiente informativo lo e' per")
            print("  costruzione, e la deriva del voto misura la composizione")
            print("  della popolazione, non l'effetto della campagna.")

    # --- 3. cittadini ------------------------------------------------------ #
    intestazione("3. CITTADINI")
    cit = [r["content"] for r in c.execute(
        "SELECT p.content FROM post p JOIN agent a ON a.agent_id = p.agent_id "
        "WHERE p.kind != 'news' AND a.is_source = 0 AND a.is_voter = 1")]
    riga("post e risposte", cit)

    # --- 4. le note, che sono cio' che il voto legge ----------------------- #
    intestazione("4. NOTE (l'unico input del voto oltre alla biografia)")
    note = [r["content"] for r in c.execute("SELECT note AS content FROM note")]
    s_note, n_note = riga("note prodotte", note)

    espl = s_note + n_note
    if espl:
        print(f"\n  Quota SI nelle note: {s_note / espl:.0%}")
        print("  Confrontala con la quota SI delle notizie e dei cittadini")
        print("  qui sopra. Se le note sono molto piu' sbilanciate della")
        print("  fonte da cui derivano, lo squilibrio non viene")
        print("  dall'informazione: lo aggiunge il passo di riflessione.")

    # --- 5. campione di note ------------------------------------------------ #
    if note and a.esempi:
        print(f"\n  Campione di {min(a.esempi, len(note))} note "
              f"(estratte a caso, seme fisso):")
        rng = random.Random(0)
        for t in rng.sample(note, min(a.esempi, len(note))):
            testo = t if len(t) <= 150 else t[:150].rsplit(" ", 1)[0] + "..."
            print(f"    [{stance(t):<11}] {testo}")
        print("\n  Leggerle e' l'unico modo per capire se il modello sta")
        print("  ragionando sul referendum o sta ripetendo una formula.")

    # --- 6. le note cambiano il voto nella direzione che dichiarano? ------- #
    print("\n" + "=" * 72)
    print(" COERENZA FRA NOTA E VOTO")
    print("=" * 72)
    b = {r["agent_id"]: r["vote"] for r in c.execute(
        "SELECT agent_id, vote FROM vote WHERE label='baseline' AND vote!='ERROR'")}
    f = {r["agent_id"]: r["vote"] for r in c.execute(
        "SELECT agent_id, vote FROM vote WHERE label='final' AND vote!='ERROR'")}
    per: dict[int, list[str]] = {}
    for r in c.execute("SELECT agent_id, note AS content FROM note"):
        per.setdefault(r["agent_id"], []).append(r["content"])

    coerenti = incoerenti = senza_segno = 0
    mossi_senza_note = 0
    for aid in set(b) & set(f):
        if b[aid] == f[aid]:
            continue
        testi = per.get(aid, [])
        if not testi:
            mossi_senza_note += 1
            continue
        segni = {stance(t) for t in testi}
        if f[aid] in segni:
            coerenti += 1
        elif segni <= {"neutro", "ambivalente"}:
            senza_segno += 1
        else:
            incoerenti += 1

    print(f"  chi cambia verso una posizione presente nelle sue note : {coerenti}")
    print(f"  chi cambia verso la posizione OPPOSTA alle sue note    : {incoerenti}")
    print(f"  chi cambia con note prive di orientamento esplicito    : {senza_segno}")
    print(f"  chi cambia SENZA avere alcuna nota                     : {mossi_senza_note}")
    if mossi_senza_note:
        print("\n  L'ultimo numero e' il pavimento di rumore del sistema: il")
        print("  voto finale dovrebbe dipendere solo da biografia e note, e")
        print("  senza note l'input e' identico a quello del baseline. Ogni")
        print("  cambiamento li' dentro e' variabilita' di campionamento del")
        print("  modello, e va sottratto dal tasso di spostamento prima di")
        print("  chiamarlo dinamica di opinione.")
    c.close()


if __name__ == "__main__":
    main()
