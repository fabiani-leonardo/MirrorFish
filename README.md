# MirrorFish

Simulazione di una popolazione sintetica italiana su un social network, per
studiare la dinamica dell'opinione durante il periodo referendario.

```bash
pip install -r requirements.txt
cp .env.example .env          # metti qui la chiave; .env e' gia' in .gitignore

# smoke test offline: nessuna GPU, nessun endpoint
python run.py --stub --agents 40 --days 5 --out runs/smoke

# run vero
python run.py --profiles /path/reddit_profiles.json --news /path/ansa \
              --out runs/base_s42 --seed 42 --concurrency 6

# controfattuale: identico fino al tick 40, poi notizie diverse
python run.py --profiles ... --news ... --cf-news /path/ansa_alt \
              --cf-from-tick 40 --out runs/cf_A --seed 42

# taratura dell'endpoint (quando la workstation torna su)
python scripts/diagnose_llm.py --concurrency 1 2 4 8 --requests 16
```

Un run = una cartella = un file `run.db`. Rilanciare sulla stessa cartella
viene **rifiutato** (`--force` per sovrascrivere): l'accumulo silenzioso di
voti in un DB riusato e' esattamente il tipo di errore che falsa una tabella
senza dare segnali.

---

## Decisioni architetturali

Tre scelte divergono dalla specifica generata da Gemini. Sono le uniche che
contano davvero, quindi vale la pena poterle difendere.

### 1. Niente CamelAI

Era il punto di partenza del rewrite, ed e' documentato da un fatto misurabile:
`vote_survey.py` passa `max_tokens=300`, `reflective_memory.py` passa 1200. Le
uniche chiamate senza tetto — quelle che vLLM espandeva a
`max_model_len - prompt_tokens` = 261.678 token — venivano da
`perform_action_by_llm()` di camel-ai. Il codice proprio del progetto era gia'
corretto; il bug stava nel layer di astrazione.

Riadottare CamelAI nella riscrittura significherebbe reintrodurre la causa del
problema che ha motivato la riscrittura. Il sostituto e' `mirrorfish/llm.py`:
~180 righe, `max_tokens` come parametro **obbligatorio** (una chiamata senza
budget solleva un'eccezione), semaforo di concorrenza, backoff con jitter,
fallback automatico se il server rifiuta `enable_thinking`.

Effetto collaterale importante: spariscono anche le ~150 righe della sezione 3
di `reflective_memory.py` (`_rewrite_agent_persona_and_slide_memory`,
`_extract_last_env_prompt`, la chirurgia su `agent._system_message`). Esistevano
solo per aggirare il fatto che CAMEL non espone un setter per il system message
e non fa trimming della memoria. Senza CAMEL il prompt viene ricostruito da zero
a ogni tick e il problema non si pone.

### 2. SQLite, non Neo4j (per il runtime)

Il feed e' `post JOIN follow`, non serve un graph DB. Quello che serve invece e'
poter archiviare, confrontare e diffare decine di run controfattuali: con un
file per run e' banale, con un'istanza Docker condivisa l'isolamento fra run
diventa un problema di disciplina.

Non e' una porta chiusa. `Store.export_graph()` emette node-link JSON, quindi
l'analisi di rete a posteriori (omofilia, camere d'eco, cammini di diffusione)
si fa con networkx su dati esportati, senza tenere Neo4j vivo durante i run. Se
in futuro serve davvero, si aggiunge un backend dietro la stessa interfaccia.

Aggiungere Neo4j *adesso* significherebbe introdurre un'infrastruttura nuova, un
linguaggio di query nuovo e un nuovo modo di fallire, in un progetto con una
scadenza — che e' la definizione del *second-system effect*.

### 3. Le notizie sono dentro il tick loop

`ansa_injector.py` girava come processo separato, faceva polling su
`run_state.json` ogni 30 secondi e scriveva nel DB dall'esterno. C'e' una race
condition fra la scrittura del post e la costruzione del feed: due run identici
possono vedere le notizie in ordine diverso.

Qui l'iniezione e' un passo del tick. Il parsing dei nomi file
(`30ottobre2025-607.txt`) e' preso dall'originale, che funzionava; la sola
modifica e' una regex ancorata invece di `name[:name.index(mese)]`, che su un
nome anomalo darebbe un risultato sbagliato in silenzio.

Il guadagno vero e' il controfattuale: `stream.fork_at(tick, alt_items)` produce
uno stream identico fino a `tick` e diverso dopo. Il ramo controfattuale non
richiede di toccare il motore.

---

## Determinismo, e il suo limite

Ogni sorgente di casualita' passa da `Random(f"{seed}|{tick}|{scopo}")`: agenti
attivi, ordine di azione, costruzione del grafo. Stesso seed e stessa config
producono la stessa struttura di run (verificato).

**Ma l'LLM no.** Un server vLLM sotto batching dinamico non e' bit-exact nemmeno
a `temperature=0`: il raggruppamento delle richieste cambia l'ordine delle
riduzioni in virgola mobile. La riproducibilita' qui e' "stessa configurazione,
stessa struttura", non "stesso output token per token".

Conseguenza da dichiarare in tesi: **ogni condizione sperimentale va eseguita in
piu' repliche** e riportata con media e dispersione. Un numero singolo per
condizione non e' un risultato, e la differenza fra due run controfattuali non
e' interpretabile se non si conosce la variabilita' fra repliche della stessa
condizione.

## Telemetria

Ogni chiamata finisce in `llm_call` con budget, token consumati,
`finish_reason` e latenza. Serve a tre cose:

- rispondere al professore con numeri misurati invece che con una stima;
- accorgersi se `finish_reason == 'length'` ricorre, cioe' se un budget e'
  troppo stretto e stai perdendo risposte in silenzio;
- quantificare il costo computazionale nella tesi.

Nel primo smoke test la telemetria ha trovato subito due bug (77 azioni su 156
scartate per target inesistente, 0 note su 90 riflessioni per un seed degenere).
Senza, sarebbero passati per "gli agenti sono poco reattivi".

## Struttura

```
mirrorfish/
  config.py      budget di token, concorrenza, parametri; fingerprint del run
  llm.py         client OpenAI-compatible + StubLLM deterministico offline
  store.py       schema SQLite, feed, note, voti, telemetria
  population.py  caricamento profili MiroFish, grafo di follow con omofilia
  news.py        parsing date ANSA, scheduling, fork controfattuale
  agent.py       costruzione prompt, schema azione, parsing validato
  memory.py      memoria riflessiva (port, senza la glue CAMEL)
  engine.py      tick loop deterministico
  survey.py      voto baseline/finale, crosstab, shift report
run.py           CLI
scripts/diagnose_llm.py   benchmark max_tokens / thinking / concorrenza
```
