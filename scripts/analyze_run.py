#!/usr/bin/env python3
"""
Analisi critica di un run concluso: prima di trattarlo come risultato.

    python scripts/analyze_run.py runs/base_s42/run.db

Non produce grafici: produce i controlli che possono invalidare il run.
"""
from __future__ import annotations

import argparse
import re
import sqlite3

# Marcatori grossolani di orientamento nel testo. Servono a rilevare uno
# sbilanciamento macroscopico dello stimolo, non a classificare finemente.
RE_NO = re.compile(r"\bvoto no\b|#referendumno|contro la riforma|"
                   r"indipendenza della magistratura|controllo politico", re.I)
RE_SI = re.compile(r"\bvoto s[iì]\b|#referendums[iì]|a favore della riforma|"
                   r"separare le carriere serve|giustizia più (?:rapida|efficiente)", re.I)


def sezione(t): print(f"\n{'='*70}\n {t}\n{'='*70}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    a = ap.parse_args()
    c = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    # I run fatti prima dell'introduzione di is_voter non hanno la colonna:
    # in quel caso gli account istituzionali VOTAVANO, ed e' proprio uno dei
    # controlli da fare.
    cols = {r[1] for r in c.execute("PRAGMA table_info(agent)")}
    HAS_VOTER = "is_voter" in cols
    VOTER = "a.is_voter" if HAS_VOTER else "1"
    if not HAS_VOTER:
        print("\n  NOTA: questo run e' anteriore a is_voter. Gli account")
        print("  istituzionali hanno votato. Usa check_voters.py per l'impatto.")

    # --- 1. lo stimolo era bilanciato? ------------------------------------
    sezione("1. BILANCIAMENTO DELLO STIMOLO (notizie iniettate)")
    news = [r["content"] for r in c.execute("SELECT content FROM post WHERE kind='news'")]
    n_no = sum(1 for x in news if RE_NO.search(x))
    n_si = sum(1 for x in news if RE_SI.search(x))
    print(f"  {len(news)} notizie: {n_si} pro-SI, {n_no} pro-NO, "
          f"{len(news)-n_si-n_no} neutre/non classificate")
    if n_no + n_si:
        print(f"  sbilanciamento fra le classificate: "
              f"{max(n_si,n_no)/(n_si+n_no):.0%} da un lato")
    print("  NB: classificazione lessicale grossolana, indicativa.")

    # --- 2. chi produce i contenuti che gli altri leggono? ----------------
    sezione("2. CHI OCCUPA IL DIBATTITO")
    for r in c.execute(f"""
        SELECT a.username, a.is_source, {VOTER} AS is_voter, COUNT(*) n
        FROM post p JOIN agent a ON a.agent_id=p.agent_id
        WHERE p.kind!='news' GROUP BY p.agent_id ORDER BY n DESC LIMIT 8"""):
        ruolo = "FONTE" if r["is_source"] else ("cittadino" if r["is_voter"] else "ISTITUZIONALE")
        print(f"  {r['username'][:30]:<32}{r['n']:>5} contenuti   {ruolo}")

    tot = c.execute("SELECT COUNT(*) n FROM post WHERE kind!='news'").fetchone()["n"]
    ist = c.execute(f"""SELECT COUNT(*) n FROM post p JOIN agent a ON a.agent_id=p.agent_id
                        WHERE p.kind!='news' AND {VOTER}=0 AND a.is_source=0"""
                    ).fetchone()["n"] if HAS_VOTER else 0
    print(f"\n  Account istituzionali: {ist}/{tot} contenuti ({ist/max(tot,1):.1%})")

    # --- 3. il baseline e' plausibile? ------------------------------------
    sezione("3. IL BASELINE E' UN ARTEFATTO?")
    print("  Il voto baseline usa SOLO la biografia statica. Se il modello non")
    print("  sa cosa sia il referendum (cutoff 2024), puo' rispondere col")
    print("  proprio prior invece che col personaggio.\n")
    for label in ("baseline", "final"):
        rows = c.execute("SELECT vote, COUNT(*) n FROM vote WHERE label=? "
                         "GROUP BY vote", (label,)).fetchall()
        d = {r["vote"]: r["n"] for r in rows}
        tot_v = sum(d.values()) or 1
        print(f"  {label:<10}" + "  ".join(
            f"{k} {d.get(k,0):>3} ({d.get(k,0)/tot_v:>5.1%})"
            for k in ("SI","NO","ASTENUTO")))

    # coerenza fra orientamento dichiarato in bio e voto baseline
    print("\n  Coerenza bio -> voto baseline (bio che citano un partito):")
    part_no = re.compile(r"Partito Democratico|Movimento 5 Stelle|Alleanza Verdi|AVS", re.I)
    part_si = re.compile(r"Fratelli d'Italia|Forza Italia|Lega\b", re.I)
    tab = {}
    for r in c.execute("""SELECT a.static_bio, v.vote FROM agent a
                          JOIN vote v ON v.agent_id=a.agent_id
                          WHERE v.label='baseline' AND """ + VOTER + "=1"):
        lean = "centrosinistra" if part_no.search(r["static_bio"] or "") else (
               "centrodestra" if part_si.search(r["static_bio"] or "") else None)
        if lean:
            tab.setdefault(lean, {}).setdefault(r["vote"], 0)
            tab[lean][r["vote"]] += 1
    for lean, d in sorted(tab.items()):
        t = sum(d.values())
        print(f"    {lean:<16}" + "  ".join(
            f"{k} {d.get(k,0)/t:>5.0%}" for k in ("SI","NO","ASTENUTO")) + f"   (n={t})")
    if tab:
        print("    Se il centrosinistra vota SI nel baseline, il baseline non")
        print("    sta leggendo la biografia: sta rispondendo col prior del modello.")

    # --- 4. gli spostamenti sono unidirezionali? --------------------------
    sezione("4. DIREZIONALITA' DELLO SPOSTAMENTO")
    b = {r["agent_id"]: r["vote"] for r in c.execute(
        "SELECT agent_id, vote FROM vote WHERE label='baseline'")}
    f = {r["agent_id"]: r["vote"] for r in c.execute(
        "SELECT agent_id, vote FROM vote WHERE label='final'")}
    trans = {}
    for k in set(b) & set(f):
        if b[k] != f[k]:
            trans[f"{b[k]}->{f[k]}"] = trans.get(f"{b[k]}->{f[k]}", 0) + 1
    tot_s = sum(trans.values()) or 1
    for k, v in sorted(trans.items(), key=lambda x: -x[1]):
        print(f"  {k:<18}{v:>4}  ({v/tot_s:>5.1%})")
    verso_no = sum(v for k, v in trans.items() if k.endswith("->NO"))
    print(f"\n  Verso NO: {verso_no}/{tot_s} ({verso_no/tot_s:.0%}) degli spostamenti.")
    if verso_no / tot_s > 0.8:
        print("  ATTENZIONE: spostamento quasi monodirezionale. Puo' essere un")
        print("  effetto reale dello stimolo, oppure deriva sistematica del")
        print("  modello. Distinguerli richiede il braccio di controllo:")
        print("    - run senza iniezione ANSA (--news su cartella vuota)")
        print("    - run con --recommender random")
        print("  Se la deriva verso NO compare anche senza notizie, non e' un")
        print("  risultato sull'informazione: e' un artefatto del prompt.")
    c.close()


if __name__ == "__main__":
    main()
