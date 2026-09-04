# Cosa la tesi dimostra, e cosa no

Documento di lavoro. Serve a fissare le affermazioni difendibili in sede di
discussione e a distinguerle da quelle che, per come è costruito
l'esperimento, non lo sono.

---

## 1. La domanda di partenza, e perché va riformulata

La formulazione iniziale era:

> Dato un sistema di agenti che rappresenta la popolazione italiana sulla base
> di dati ISTAT e YouTrend, simulando il periodo referendario e somministrando
> le notizie ANSA, è possibile ottenere lo stesso risultato del referendum? Se
> sì, si può sostituire una notizia a partire da un istante di cutoff e vedere
> come sarebbe andata — quindi usare lo strumento per prevedere il futuro.

La catena "riproduco l'esito → il modello è valido → i controfattuali sono
predittivi" si rompe in tre punti, e sono i tre punti su cui una commissione
attaccherebbe per primi.

**Primo. Un esito binario riprodotto non è una validazione.** Indovinare SÌ/NO
ha probabilità 0,5 a caso. Con un solo numero riprodotto, il modello e una
moneta sono indistinguibili.

**Secondo. Rischio di calibrazione mascherata da predizione.** Se si regolano
i parametri finché il risultato torna, si è fatto fitting, non previsione. Con
molti parametri liberi — politica del feed, quota di notizie, omofilia,
frequenza di riflessione — il rischio è concreto e non basta dichiararlo.

**Terzo. I controfattuali non sono falsificabili.** Non esiste un dato reale
su "come sarebbe andata se le notizie fossero state altre". Nessun risultato
controfattuale può essere confermato o smentito: quindi non è una previsione,
e "prevedere il futuro" non è un'affermazione sostenibile.

---

## 2. Le quattro affermazioni difendibili

### A. Validazione strutturale, non aggregata

Non "il modello riproduce il risultato" ma **"il modello riproduce la
struttura del voto per sottogruppi"**: distribuzione per fascia d'età, area
geografica, titolo di studio, confrontata con i sondaggi pubblicati durante la
campagna.

Passa da un grado di libertà a diverse decine. Se il modello azzecca
l'aggregato ma sbaglia sistematicamente i sottogruppi, lo si scopre — ed è un
risultato onesto, non un fallimento.

Implementato: `survey.crosstab()`.

### B. Elasticità informativa, non predizione

Non "prevedo come sarebbe andata" ma **"misuro quanto l'esito simulato è
sensibile a una perturbazione dello stream informativo"**.

Si varia l'intensità della perturbazione su più livelli (sostituzione del
10%, 30%, 50% delle notizie dopo il cutoff) e si ottiene una curva
dose-risposta. È un'affermazione sulla dinamica interna del modello:
verificabile rieseguendo, e non esposta all'obiezione fatale della sezione 1.

Implementato: `NewsStream.fork_at()`, flag `--cf-from-tick`.

### C. La politica del feed come trattamento

**Questa è la parte più originale, e l'unica pienamente falsificabile
all'interno dell'esperimento.**

Lo stesso stream di notizie, la stessa popolazione, lo stesso seed, e si varia
solo il criterio con cui il feed è ordinato: `random` (controllo), `recency`,
`engagement`. Se l'esito e la polarizzazione cambiano, si è misurato l'effetto
dell'algoritmo di raccomandazione a parità di informazione disponibile.

Il controllo `random` è ciò che rende l'affermazione seria: se i risultati non
cambiano fra `random` e `recency`, il feed non sta facendo nulla e ogni
conclusione sul ruolo dell'informazione cade.

Attenzione al ragionamento circolare: un feed ordinato per similarità
semantica **produce camere d'eco per costruzione**. Concludere "la simulazione
mostra polarizzazione" dopo aver scelto quella politica non dimostra niente.
Per questo `similarity` non è nel disegno principale.

Implementato: `recommender.py`, flag `--recommender`.

### D. Contributo metodologico: l'infrastruttura di riproducibilità

Il sistema è stato riscritto da zero anche per rendere gli esperimenti
riproducibili: seed espliciti su ogni sorgente di casualità, telemetria di
ogni chiamata al modello, checkpoint per tick con ripresa, manifesto del run
con impronta della configurazione.

**Limite da dichiarare, non da nascondere:** questo rende deterministica ogni
cosa tranne l'LLM. Un server vLLM sotto batching dinamico non è bit-exact
nemmeno a temperatura 0, perché il raggruppamento delle richieste cambia
l'ordine delle riduzioni in virgola mobile. La riproducibilità è "stessa
configurazione, stessa struttura", non "stesso output token per token".

Conseguenza operativa: **ogni condizione va eseguita in più repliche** e
riportata con media e dispersione. La differenza fra due condizioni non è
interpretabile senza conoscere la variabilità fra repliche della stessa
condizione. Questo non è un dettaglio: è la condizione perché B e C
significhino qualcosa.

---

## 3. Stato reale al 3 settembre

**Fatto**
- Sistema riscritto e funzionante end-to-end contro il modello del laboratorio
- Un run in corso: 105 agenti, 30 giorni simulati, tick da 11h, politica
  `recency`, seed 42 (~9 ore, in chiusura)
- Causa del blocco originale identificata e riprodotta: il gateway riserva
  `max_tokens` contro la finestra da 100.000 token/minuto, e le chiamate senza
  tetto ne riservavano 261.678 — 2,6 volte l'intera finestra
- Consumo misurato su 105 chiamate reali: ~730 token per chiamata

**Difetti noti, non ancora corretti nel run in corso**
- Gli account istituzionali (partiti) votano nel referendum, mentre le loro
  stesse biografie dichiarano che non sono elettori. Da quantificare con
  `check_voters.py` a run concluso.
- Una sola azione per agente per tick: scrivere e reagire competono, e
  scrivere vince nel 95% dei casi. I "mi piace" sono al 4,9%, quindi la
  politica `engagement` ha quasi nessun segnale su cui operare.
- I profili orari dei cronotipi sono stime plausibili, non dati ISTAT
  sull'uso del tempo.
- Nessuna replica, nessun braccio di controllo: **il run in corso non è
  ancora un risultato**, è un pilota a scala reale.

**Non ancora collegato**
- Politiche `similarity` e `hybrid` (embedding non agganciati al motore)
- Confronto con le serie storiche dei sondaggi

---

## 4. Il vincolo che decide tutto

Il limite è **8 richieste al minuto**. Non la GPU: il consumo misurato usa il
10% della finestra token già assegnata, il 90% resta inutilizzato.

| condizione | chiamate | a 8 req/min | a 40 req/min |
|---|---:|---:|---:|
| un run (30 giorni, 105 agenti) | ~4.300 | 8,9 h | 1,8 h |
| un run (150 giorni) | ~20.800 | 43,3 h | 8,7 h |
| disegno minimo (4 condizioni × 3 repliche, 30 gg) | ~51.400 | **107 h** | **21 h** |

Con la scadenza a fine mese, 107 ore di GPU più la scrittura non ci stanno.
**La richiesta di quota non è un'ottimizzazione: è ciò che determina se il
disegno sperimentale esiste.**

---

## 5. Piano per i prossimi giorni

### Se la quota sale a 25-40 req/min

1. Correggere `is_voter` e passare a `--max-actions 3`; rifare la baseline
2. Disegno a 4 condizioni × 3 repliche:
   - `random` (controllo del feed)
   - `recency` (null model)
   - `engagement` (trattamento)
   - `recency` senza iniezione ANSA (controllo dell'informazione)
3. Controfattuali sul braccio migliore: 3 livelli di intensità × 3 repliche
4. Validazione strutturale contro i sondaggi per sottogruppo

### Se la quota resta a 8 req/min

Il disegno va ridotto **prima** di lanciare, non dopo:
1. Finestra a 30 giorni, non 5 mesi
2. Due condizioni sole: `random` e `recency`, 2 repliche ciascuna (~36 h)
3. Controfattuale a un solo livello di intensità, 2 repliche (~18 h)
4. I difetti noti restano, ma vanno **dichiarati e quantificati** con un run
   corto di confronto (25 agenti, 3 giorni, ~15 minuti) invece che ignorati

### In entrambi i casi

- Dichiarare i parametri **prima** di guardare l'esito finale, e scriverlo
  nella tesi: è la difesa contro l'accusa di calibrazione a posteriori
- Tarare i cronotipi sull'indagine ISTAT sull'uso del tempo, o dichiararli
  come parametri assunti
- Sezione qualitativa: traiettorie individuali lette dal feed HTML, accanto
  ai dati aggregati. È l'unico controllo che intercetta il caso in cui i
  numeri tornano ma i testi sono degeneri.

---

## 6. La frase per la discussione

> Non sostengo di prevedere l'esito di un referendum. Sostengo di aver
> costruito un ambiente riproducibile in cui una popolazione sintetica
> calibrata su dati demografici reali reagisce a uno stream informativo
> reale, e di aver misurato tre cose: quanto la struttura del voto simulato
> somiglia a quella rilevata dai sondaggi, quanto l'esito è elastico a
> perturbazioni dell'informazione, e quanto dipende dall'algoritmo che
> decide cosa ciascuno legge — con un braccio di controllo che verifica che
> non sia tutto rumore.
