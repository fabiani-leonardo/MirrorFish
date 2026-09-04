# Ripartire dopo il blocco

## Cosa e' successo, quasi certamente

I due run puntavano alla **stessa cartella** `runs/test_2`, entrambi con
`--force`. Se il primo era ancora vivo, il secondo gli ha cancellato il
`run.db` da sotto i piedi. In piu' 40 + 25 = 65 richieste/minuto contro un
limite di squadra da 25: entrambi in attesa permanente.

Non produce un errore chiaro: produce due processi che sembrano bloccati.

Ora `run.py` scrive un `run.lock` con il proprio pid e si rifiuta di partire
se un altro processo vivo sta usando quella cartella. Un lock rimasto da un
processo morto viene riconosciuto e rimosso.

## Passi, in ordine

### 1. Chiudi tutto quello che e' rimasto in giro

```bash
ps aux | grep "run.py" | grep -v grep
```

Se compare qualcosa, `kill <pid>`. Verifica anche che non ci siano
`serve_feed.py` appesi a database cancellati.

### 2. Leggi i limiti veri

```bash
python scripts/probe.py --n 2
```

Serve `x-ratelimit-team-limit-requests`. **E' quello il tetto**, non il tuo
limite personale da 60. Se e' ancora 25, il massimo prudente e' `--rpm 18`
lasciando margine a Chiara. Se e' salito, regolati su quello.

### 3. Scegli la concorrenza dai dati

```bash
python scripts/pick_concurrency.py runs/base_s42/run.db --rpm 18
```

### 4. Un run alla volta, cartelle distinte

```bash
python -u run.py \
  --profiles ./simulations/sim_14/reddit_profiles.json \
  --news ./start/notizieansa/notizie_social \
  --news-full ./start/notizieansa/notizie_referendum \
  --start 2026-02-20 --days 30 --hours-per-tick 11 \
  --out runs/ctrl_nonews --seed 42 --rpm 18 --concurrency 2 \
  2>&1 | tee runs/ctrl_nonews.log
```

Il nome della cartella dice cosa contiene. `runs/test_2` usato per due
esperimenti diversi e' come si perdono i dati.

### 5. Controlla che sia vivo senza fissare il terminale

```bash
python scripts/watch_run.py runs/ctrl_nonews/run.db --follow
```

La riga di tick si stampa solo a tick concluso, e con 106 agenti attivi a 18
req/min un tick dura diversi minuti: il terminale muto non significa bloccato.
`run.db` registra ogni chiamata con l'orario e dice la verita'.

## Sulla finestra da 144 giorni

Il secondo tentativo copriva 223 tick su 106 agenti: circa 14.600 chiamate,
cioe' 13 ore a 18 req/min. Fattibile, ma non e' il primo run da fare.

Prima serve il **controllo senza notizie** sui 30 giorni: se la deriva verso
NO compare anche senza stimolo informativo, non e' un risultato
sull'informazione ed e' inutile impegnare tredici ore su una finestra piu'
lunga.
