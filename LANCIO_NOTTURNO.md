# Run notturno — procedura

## 0. Controllo pre-volo (obbligatorio)

```bash
export LLM_API_KEY=...
python scripts/preflight.py
```

Deve uscire "Tutto a posto". Se `gate inizializzato con min_interval_s`
risulta FAIL, stai eseguendo la copia vecchia: quella che ha fatto partire
30 chiamate a raffica prendendo tre 429. Su otto ore brucerebbe la quota
della notte e quella della squadra.

## 1. Scegli la finestra

Il tuo archivio arriva al 21 marzo 2026. Trenta giorni indietro:

```bash
--start 2026-02-20 --days 30
```

**Verifica la data del referendum** e fai finire la finestra li'. Se il voto
e' il 22 marzo, questa e' giusta. Se fosse un'altra, cambia `--start` di
conseguenza: un run che finisce tre giorni dopo il voto non e' confrontabile
col risultato reale.

## 2. Lancia

```bash
caffeinate -i nohup python run.py \
  --profiles ./simulations/sim_14/reddit_profiles.json \
  --news ./start/notizieansa/notizie_social \
  --start 2026-02-20 --days 30 --hours-per-tick 11 \
  --out runs/base_s42 --seed 42 --rpm 8 \
  > runs/base_s42.log 2>&1 &
```

- `caffeinate -i` impedisce a macOS di sospendersi: senza, il coperchio
  chiuso ferma tutto e al risveglio trovi un run a meta'.
- `nohup ... &` lo stacca dal terminale, che puoi chiudere.
- Circa 4.300 chiamate, ~9 ore a 8 req/min.

## 3. Guarda il feed mentre gira

```bash
python scripts/serve_feed.py runs/base_s42/run.db --port 8000
```

Poi apri http://127.0.0.1:8000. La pagina si aggiorna da sola e conserva la
posizione di scorrimento. Il database e' aperto in `mode=ro`: il server non
puo' scrivere nemmeno per un bug, e non tocca la simulazione.

Su macOS la 5000 e' occupata da AirPlay Receiver: il default e' 8000.

## 4. Controlla dopo venti minuti, prima di andare a dormire

```bash
tail -20 runs/base_s42.log
```

Cosa deve vedersi:
- i tick avanzano (uno ogni ~4-6 minuti con 105 agenti);
- `err=0` o quasi: qualche errore isolato e' tollerabile, una colonna di
  errori significa che qualcosa non va e conviene fermarsi;
- nessun `[rate-limit]` ripetuto. Uno ogni tanto e' il gate che fa il suo
  lavoro; uno ogni tick significa che il ritmo e' ancora troppo alto.

Se qualcosa e' storto, fermalo (`pkill -f "python run.py"`) e riparti domani:
meglio perdere una notte che presentarsi con dati sbagliati.

## 5. Se si interrompe

```bash
caffeinate -i nohup python run.py [stessi identici argomenti] --resume \
  >> runs/base_s42.log 2>&1 &
```

Riprende dall'ultimo tick completo, ripulisce quello parziale, non rifa' la
survey baseline. Gli argomenti devono essere gli stessi: cambiarli a meta'
run produce un risultato non interpretabile.

## Una sola simulazione per volta

Il limite team e' 25 richieste/minuto condivise. Due run in parallelo
raddoppiano il consumo e la mattina Chiara trova la quota esaurita — il
giorno in cui chiedi di alzarla.

## Cosa avrai domattina

Un run completo su 30 giorni con la popolazione intera: voto baseline,
traiettoria, voto finale, spostamenti, crosstab per fascia d'eta', telemetria
di ~4.300 chiamate reali e il feed navigabile.

E' un risultato, non ancora una prova: **una replica sola**. Con l'LLM non
deterministico la differenza fra due condizioni non e' interpretabile senza
conoscere la variabilita' fra repliche della stessa condizione. Serve per
mostrare che la pipeline produce dati veri, e per motivare la richiesta di
quota — non per concludere niente sul referendum.
