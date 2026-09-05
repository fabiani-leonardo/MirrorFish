#!/usr/bin/env python3
"""
Analisi critica di un run concluso: prima di trattarlo come risultato.

    python scripts/analyze_run.py runs/base_s42/run.db

Non produce grafici: produce i controlli che possono invalidare il run.
"""
from __future__ import annotations

import argparse
import random
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
        print("  istituzionali hanno votato: il conteggio e' contaminato.")

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

    # --- 5. quante note regge davvero ogni cambio di idea? ----------------
    sezione("5. RISOLUZIONE DELLA MEMORIA RIFLESSIVA")
    print("  Il voto finale legge le note. Se chi cambia idea ha UNA nota,")
    print("  lo spostamento non e' esposizione accumulata: e' una sola")
    print("  chiamata LLM che ha deciso. E' la differenza fra una dinamica")
    print("  di opinione e un lancio di moneta ben scritto.\n")
    per_agente = {r["agent_id"]: r["n"] for r in c.execute(
        "SELECT agent_id, COUNT(*) n FROM note GROUP BY agent_id")}
    cambiati = [k for k in set(b) & set(f) if b[k] != f[k]]
    stabili = [k for k in set(b) & set(f) if b[k] == f[k]]
    for nome, gruppo in (("chi ha cambiato idea", cambiati), ("chi non l'ha cambiata", stabili)):
        if not gruppo:
            continue
        note = [per_agente.get(k, 0) for k in gruppo]
        dist = {}
        for n in note:
            dist[min(n, 4)] = dist.get(min(n, 4), 0) + 1
        media = sum(note) / len(note)
        print(f"  {nome:<24} n={len(gruppo):<4} note/agente medie {media:.2f}")
        print("      " + "  ".join(
            f"{k if k < 4 else '4+'} note: {v}" for k, v in sorted(dist.items())))
    una_sola = sum(1 for k in cambiati if per_agente.get(k, 0) <= 1)
    if cambiati and una_sola / len(cambiati) > 0.6:
        print(f"\n  ATTENZIONE: {una_sola}/{len(cambiati)} di chi cambia idea ha")
        print("  al massimo UNA nota. Prima di interpretare il tasso di")
        print("  spostamento come dinamica sociale, abbassa --reflection-every")
        print("  oppure confrontalo con un run a riflessione spenta.")

    # --- 6. si argomenta o si ripetono slogan? ----------------------------
    sezione("6. DENSITA' ARGOMENTATIVA PER PROFONDITA' DI LETTURA")
    print("  L'ipotesi: per argomentare servono dati, e i dati stanno")
    print("  nell'articolo, non nel titolo. Se regge, chi legge integrale")
    print("  scrive piu' lungo, con piu' numeri e meno hashtag.\n")
    if "media_depth" not in cols:
        print("  (run anteriore a media_depth, salto)")
    else:
        gruppi = {}
        print(f"  {'profondita':<12}{'post':>6}{'caratt.':>9}{'n.cifre':>9}"
              f"{'#tag':>7}{'lessico':>9}")
        for r in c.execute("""
            SELECT a.media_depth AS d, p.content
            FROM post p JOIN agent a ON a.agent_id = p.agent_id
            WHERE p.kind != 'news' AND a.is_source = 0"""):
            gruppi.setdefault(r["d"], []).append(r["content"])
        def ttr(t):
            w = [x.lower() for x in re.findall(r"\w+", t)]
            return len(set(w)) / len(w) if w else 0.0

        misure = {}
        for d, testi in sorted(gruppi.items()):
            # Ricchezza lessicale calcolata PER POST e poi mediata. Sul
            # corpus aggregato era inutilizzabile: il type-token ratio cala
            # meccanicamente al crescere del testo, quindi il gruppo con meno
            # post usciva sempre "piu' ricco". Nel pilota il gruppo 'nessuna'
            # (47 post) segnava 0,302 contro 0,118 di 'integrale' (441 post):
            # non era ricchezza lessicale, era la dimensione del campione.
            misure[d] = {
                "n": len(testi),
                "caratt": [len(t) for t in testi],
                "cifre": [len(re.findall(r"\d", t)) for t in testi],
                "tag": [t.count("#") for t in testi],
                "lessico": [ttr(t) for t in testi],
            }
            m = misure[d]
            print(f"  {d:<12}{m['n']:>6}"
                  f"{sum(m['caratt'])/m['n']:>9.0f}"
                  f"{sum(m['cifre'])/m['n']:>9.2f}"
                  f"{sum(m['tag'])/m['n']:>7.2f}"
                  f"{sum(m['lessico'])/m['n']:>9.3f}")

        if "integrale" in misure and "titolo" in misure:
            print("\n  Test di permutazione, integrale contro titolo")
            print("  (10.000 rimescolamenti, nessuna assunzione di normalita')")
            rng = random.Random(0)
            for metrica in ("caratt", "cifre", "tag", "lessico"):
                a = misure["integrale"][metrica]
                b = misure["titolo"][metrica]
                oss = sum(a) / len(a) - sum(b) / len(b)
                tutti = a + b
                na = len(a)
                estremi = 0
                for _ in range(10000):
                    rng.shuffle(tutti)
                    d = (sum(tutti[:na]) / na
                         - sum(tutti[na:]) / (len(tutti) - na))
                    if abs(d) >= abs(oss):
                        estremi += 1
                p = (estremi + 1) / 10001
                verdetto = "significativo" if p < 0.05 else "non distinguibile"
                print(f"    {metrica:<9} differenza {oss:+8.3f}   "
                      f"p = {p:.4f}   {verdetto}")
            print("\n  ATTENZIONE al confondimento: qui la profondita' e'")
            print("  DEDOTTA dalla biografia, quindi chi legge l'integrale e'")
            print("  un attivista, e un attivista scriverebbe post piu' lunghi")
            print("  comunque. Per attribuire la differenza all'esposizione")
            print("  servono due run con --force-media-depth integrale e")
            print("  --force-media-depth titolo, stesso seed.")

        print("\n  caratt. = lunghezza media | n.cifre = riferimenti numerici")
        print("  #tag = hashtag per post | lessico = ricchezza (type-token ratio)")
        print("  E' la metrica da portare in tesi: non dipende dall'aver")
        print("  riprodotto l'esito del referendum.")
    c.close()



if __name__ == "__main__":
    main()
