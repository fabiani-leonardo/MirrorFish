# Capitolo — Architettura del sistema

Bozza per la tesi. Descrive MirrorFish come è effettivamente implementato,
ricostruito leggendo il codice e non a memoria. Ogni modulo è indicato con il
nome del file e la sua dimensione, per dare la misura del sistema.

---

## 1. Premessa: perché una riscrittura

MirrorFish nasce come riscrittura da zero di MiroFish, a sua volta fork di
OASIS/CAMEL-AI. La motivazione non è stilistica ed è documentata da un fatto
misurabile.

I moduli scritti per la tesi (`vote_survey.py`, `reflective_memory.py`)
passavano sempre un limite esplicito di token in uscita. Le uniche chiamate
prive di limite provenivano da `perform_action_by_llm()` di CAMEL-AI. In
assenza di quel parametro, vLLM lo deriva da `max_model_len - prompt_tokens`,
ottenendo 261.678 token; il gateway del laboratorio riserva quel valore contro
una finestra da 100.000 token al minuto, e una richiesta simile non può essere
ammessa. Il meccanismo è stato riprodotto sperimentalmente su sei misure
(§ appendice telemetria).

Il difetto risiedeva quindi nel livello di astrazione, non nel codice della
tesi. Riadottare quel livello nella riscrittura avrebbe reintrodotto la causa
del problema che la motivava. La sostituzione è `llm.py` (480 righe), in cui il
limite di token è un **parametro obbligatorio**: una chiamata che lo omette
solleva un'eccezione.

Effetto collaterale rilevante: sono scomparse circa 150 righe di
`reflective_memory.py` (`_rewrite_agent_persona_and_slide_memory`,
`_extract_last_env_prompt`, la riscrittura di `agent._system_message`). Servivano
unicamente ad aggirare l'assenza, in CAMEL, di un modo per sostituire il
messaggio di sistema e per limitare la crescita della memoria. Senza CAMEL il
prompt viene ricostruito da zero a ogni ciclo e il problema non si pone.

---

## 2. Mappa dei moduli

Sistema: **4.408 righe**, di cui 2.209 nel pacchetto di simulazione e 1.932 in
strumenti diagnostici.

### Pacchetto `mirrorfish/`

| modulo | righe | responsabilità |
|---|---:|---|
| `store.py` | 461 | schema SQLite, feed, note, voti, telemetria |
| `llm.py` | 480 | client HTTP, limitatore di frequenza, LLM simulato |
| `engine.py` | 252 | ciclo dei tick, orchestrazione |
| `agent.py` | 234 | costruzione del prompt, schema e validazione delle azioni |
| `population.py` | 213 | biografie, cronotipi, grafo dei follow |
| `news.py` | 210 | datazione ANSA, calendario, ramo controfattuale |
| `recommender.py` | 200 | politiche di ordinamento del feed |
| `survey.py` | 198 | rilevazione del voto, tabelle incrociate |
| `config.py` | 174 | parametri, budget di token, impronta del run |
| `embeddings.py` | 152 | vettori semantici (non ancora collegato al motore) |
| `memory.py` | 121 | memoria riflessiva |

### Grafo delle dipendenze

Verificato sulle importazioni reali. Nessun ciclo: le dipendenze puntano tutte
verso il basso.

```
run.py
  └─> engine.py
        ├─> agent.py ──> llm.py ──> config.py
        ├─> memory.py ─> llm.py
        ├─> recommender.py ──> embeddings.py
        ├─> news.py
        ├─> population.py
        └─> store.py
  └─> survey.py ──> store.py, llm.py
```

`store.py` non importa nulla del pacchetto: è la base. `config.py` è importato
da tutti e non importa nessuno.

---

## 3. Svolgimento di una simulazione

### 3.1 Preparazione (`run.py`)

1. **Configurazione.** `SimConfig` raccoglie ogni parametro che influenza il
   risultato e produce una `fingerprint()`, hash SHA-256 troncato della
   configurazione serializzata, salvata nel manifesto del run. Due run con la
   stessa impronta hanno gli stessi parametri.

2. **Protezione della cartella.** Se `run.db` esiste, il programma si rifiuta
   di partire. Rilanciare sullo stesso file non sovrascrive: accumula voti e
   post, e il conteggio finale risulta errato senza produrre errori. `--force`
   cancella, `--resume` riprende.

3. **Popolazione** (`population.load_mirofish_profiles`). Le biografie
   generate dai dati ISTAT e YouTrend vengono normalizzate. A ciascun agente
   si assegnano:
   - un **cronotipo**, vettore di 24 valori di propensione oraria, scelto fra
     `studente`, `lavoratore`, `pensionato`, `notturno`, `istituzionale`;
   - un **ruolo**: fonte di notizie (pubblica, non vota), account
     istituzionale (pubblica e influenza, non vota), cittadino (pubblica e
     vota). La distinzione fra `is_source` e `is_voter` è necessaria perché
     gli account di partito partecipano al dibattito ma non hanno scheda
     elettorale.

4. **Grafo dei follow** (`population.build_follow_graph`). Archi generati con
   omofilia su regione e orientamento: il parametro `homophily` è la
   probabilità che un arco sia scelto dentro il gruppo simile anziché a caso.
   È un parametro del modello, quindi dichiarato e variabile in analisi di
   sensibilità.

5. **Calendario delle notizie** (`news.NewsStream`). I file ANSA sono datati
   dal nome (`30ottobre2025-607.txt`). Le notizie precedenti all'inizio della
   finestra vengono **scartate con avviso**, non accumulate sul primo giorno:
   con un archivio che parte da ottobre 2025 e una simulazione che parte a
   marzo 2026 significherebbe centinaia di articoli nel primo tick, feed
   saturi e tick iniziali non interpretabili.

6. **Rilevazione baseline** (`survey.run_survey`). Ogni cittadino esprime un
   voto sulla sola biografia statica, prima che esista qualsiasi nota. È il
   "prima" del confronto, ottenuto senza rieseguire la simulazione.

### 3.2 Il ciclo (`engine.run`)

Per ogni tick, nell'ordine:

**a. Pubblicazione delle notizie.** `_inject_news()` inserisce gli articoli
programmati per il tick corrente, attribuiti all'agente-fonte. L'iniezione è
un passo del ciclo, non un processo esterno: la versione precedente usava un
processo separato che interrogava lo stato ogni 30 secondi e scriveva dal di
fuori, con una corsa critica fra la scrittura del post e la costruzione del
feed. Due esecuzioni identiche potevano vedere le notizie in ordine diverso.

**b. Selezione degli attivi.** `_active_agents()` calcola, per ciascun agente,
la probabilità di attivarsi come sovrapposizione fra la finestra oraria del
tick e il proprio cronotipo. Questo rende superfluo l'espediente dei tick
coprimi con 24: non serve che i tick ruotino attraverso le ore per dare a
tutti la stessa occasione, perché la probabilità è già calcolata sulla
sovrapposizione effettiva.

Il numero di attivi varia di conseguenza: una finestra che inizia alle 11 ne
attiva 50-56 su 105, una che inizia alle 22 circa 20.

**c. Azione degli agenti attivi** (`_act`, in parallelo). Per ciascuno:
- `recommender.feed()` compone il feed: slot riservati alle notizie più il
  resto ordinato secondo la politica scelta;
- `store.notes_for()` recupera le note recenti della memoria riflessiva;
- `store.posts_by()` recupera gli ultimi contenuti scritti dall'agente;
- `store.parents_of()` recupera i post a cui le risposte in feed rispondono,
  perché una replica senza il messaggio cui risponde è un non sequitur;
- `agent.decide()` costruisce i due prompt ed effettua **una sola** chiamata
  al modello;
- `agent.parse_actions()` valida la risposta.

Il prompt è ricostruito da zero a ogni tick da tre livelli: identità di lungo
periodo (biografia statica, mai riscritta), opinioni di medio periodo (ultime
N note), memoria di breve periodo (ultimi contenuti propri). Non esiste una
cronologia che cresce indefinitamente, quindi la lunghezza del prompt è
limitata per costruzione.

**d. Applicazione.** `_apply()` scrive su database. Le scritture sono
serializzate nel processo principale, non nei task concorrenti. Ogni azione
non valida diventa un errore registrato nella telemetria anziché
un'interruzione: senza questa distinzione si rischia di scrivere che «il 40%
degli agenti è rimasto passivo» quando in realtà il 40% delle risposte non era
JSON valido.

**e. Riflessione** (ogni `reflection_every` tick). `memory.ReflectionEngine`
riceve la biografia come solo contesto, le note accumulate e i post letti, e
decide se qualcosa è cambiato. La biografia statica non attraversa mai il
modello di riflessione: le note sono un blocco in sola aggiunta, così
l'identità dell'agente non si degrada a forza di riscritture successive. Il
prompt impone che «nessun cambiamento» sia il caso comune.

**f. Checkpoint.** Il tick completato viene registrato nel manifesto. Con un
endpoint a quota limitata le interruzioni sono frequenti, e ricominciare da
capo ogni volta renderebbe impossibile concludere. Alla ripresa,
`truncate_after_tick()` rimuove le scritture di un tick incompleto in ordine
compatibile con i vincoli di integrità referenziale (reazioni, poi risposte,
poi post): il checkpoint si scrive solo a tick concluso, quindi
un'interruzione a metà lascia in memoria le azioni degli agenti che avevano
già risposto, che altrimenti verrebbero duplicate.

### 3.3 Chiusura

Rilevazione finale del voto, con biografia **e** intera traiettoria delle note.
Confronto con la baseline (`survey.shift_report`), tabelle incrociate per
sottogruppo demografico (`survey.crosstab`), esportazione del grafo sociale.

---

## 4. Controllo del carico

Il vincolo operativo non è la capacità di calcolo ma la quota del gateway: 8
richieste al minuto per utente, su una finestra di un minuto solare, con un
limite di 5 richieste parallele e 100.000 token al minuto.

`RateGate` in `llm.py` impone tre vincoli in congiunzione: assenza di pausa
globale in corso, distanza minima fra partenze successive, e non più di
(quota − riserva) partenze in una finestra scorrevole. Il terzo è quello
determinante: una spaziatura fissa di 60/quota secondi colloca esattamente
`quota` richieste in ogni minuto solare, ponendosi sul limite.

Il limitatore legge inoltre le intestazioni `x-ratelimit-*-remaining` dalle
risposte riuscite e si arresta **prima** di ricevere un rifiuto, anziché
reagire dopo. Questo consente anche di tenere conto del consumo degli altri
membri del gruppo sul limite condiviso, non osservabile localmente.

Quando un rifiuto arriva comunque, la pausa è **globale**: senza, ciascuna
delle richieste in volo ritenterebbe per conto proprio e più o meno insieme
alle altre, ritriggerando immediatamente il limite.

---

## 5. Determinismo e suo limite

Ogni sorgente di casualità deriva da `Random(f"{seed}|{tick}|{scopo}")`:
agenti attivi, ordine di azione, costruzione del grafo, ordinamento casuale
del feed. Configurazione e seed identici producono la stessa struttura di
esecuzione.

**L'unica eccezione è il modello linguistico.** Un server vLLM sotto batching
dinamico non è riproducibile bit per bit nemmeno a temperatura 0, perché il
raggruppamento delle richieste modifica l'ordine delle riduzioni in virgola
mobile. La riproducibilità garantita è quindi «stessa configurazione, stessa
struttura», non «stesso output token per token».

Ne discende che ogni condizione sperimentale va eseguita in più repliche e
riportata con media e dispersione: la differenza fra due condizioni non è
interpretabile senza conoscere la variabilità fra repliche della stessa
condizione.

---

## 6. Telemetria

Ogni chiamata al modello è registrata nella tabella `llm_call` con scopo,
budget richiesto, token consumati, motivo di terminazione e latenza. Serve a
tre scopi: quantificare il costo computazionale con misure anziché stime;
rilevare se un budget è troppo stretto (`finish_reason = 'length'` ricorrente
indica risposte perse silenziosamente); e documentare che i limiti sono tarati
e non stimati.

Nella prima esecuzione di prova la telemetria ha individuato immediatamente
due difetti — 77 azioni su 156 scartate per bersaglio inesistente e 0 note su
90 riflessioni per un seme degenere — che sarebbero altrimenti passati per
«gli agenti sono poco reattivi».

---

## 7. Strumenti diagnostici (`scripts/`)

| strumento | funzione |
|---|---|
| `preflight.py` | verifica che le correzioni critiche siano nel codice in esecuzione |
| `probe.py` | una richiesta, con intestazioni e corpo completi: identifica il limite attivo |
| `quota_report.py` | consumo misurato per tipo di chiamata e margine sulla quota |
| `plan_run.py` | costo e durata previsti al variare dei parametri |
| `diagnose_llm.py` | throughput al variare di limite di token e concorrenza |
| `render_feed.py` | ispettore HTML autonomo del feed simulato |
| `serve_feed.py` | server di sola lettura per seguire un run in corso |
| `check_voters.py` | individua account istituzionali inclusi nel conteggio |
| `export_notes.py` | esporta biografie e note in JSON |

`render_feed.py` non è un accessorio: i dati aggregati non distinguono fra
agenti che cambiano posizione per ragioni argomentate e agenti il cui testo è
degenerato. La lettura qualitativa è l'unico controllo che intercetta il
secondo caso.

---

## 8. Scelte architetturali e loro motivazione

**SQLite anziché un database a grafo.** Il feed è una giunzione fra due
tabelle. Ciò che serve davvero è archiviare, confrontare e differenziare
decine di esecuzioni controfattuali: con un file per esecuzione è immediato,
con un'istanza condivisa l'isolamento diventa una questione di disciplina.
`Store.export_graph()` produce comunque il grafo in formato node-link per
l'analisi di rete a posteriori.

**Notizie con quota riservata nel feed.** Modellano la portata editoriale:
un'agenzia raggiunge anche chi non la segue. Lasciandole competere sul
ranking sparirebbero sotto la politica `similarity` — un cittadino non è
semanticamente affine a un lancio d'agenzia — eliminando proprio la variabile
indipendente dello studio.

**Politica del feed come parametro esplicito.** In precedenza l'ordinamento
era per sola recenza, e la quota di esposizione mediatica dipendeva da quanti
articoli l'archivio contenesse quel giorno: un parametro sperimentale
determinato per caso. Renderlo esplicito lo trasforma in un asse
dell'esperimento.

---

## Appendice — Il meccanismo del limite di token

Sei misure sulla stessa base, ottenute con `probe.py --n 3 --max-tokens 256
4096`, si spiegano con un solo modello: il gateway riserva `max_tokens` più
circa 13 token di intestazione contro la finestra, e riconcilia contro la
riserva della richiesta precedente.

| richiesta | max_tokens | riserva | rimborso prec. | netto atteso | osservato |
|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 269 | 0 | 269 | 269 |
| 2 | 256 | 269 | 230 | 39 | 39 |
| 3 | 256 | 269 | 230 | 39 | 39 |
| 4 | 4.096 | 4.109 | 230 | 3.879 | 3.879 |
| 5 | 4.096 | 4.109 | 4.070 | 39 | 39 |
| 6 | 4.096 | 4.109 | 4.070 | 39 | 39 |

A regime il consumo netto coincide con quello effettivo (39 token per 33 di
prompt e 6 di completamento), ma la riserva resta impegnata. Con il limite
omesso, la riserva sarebbe di 261.678 token, pari a 2,6 volte l'intera
finestra: nessuna richiesta poteva essere ammessa.
