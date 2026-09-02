# Run di stanotte

## Configurazione scelta

105 agenti, 30 giorni simulati, tick da 11 ore, seed 42.
= 65 tick, ~4.283 chiamate, **~8,9 ore a 8 req/min**. Una notte esatta.

I 5 mesi sono 43 ore: non entrano, e un run incompleto non e' confrontabile
con niente. Se domani la quota sale, i 5 mesi si rifanno in 9 ore.

Finestra: dal 20 febbraio al 21 marzo 2026, cioe' l'ultimo mese di campagna
fino alla fine dell'archivio ANSA. Le notizie precedenti vengono scartate con
avviso: e' una scelta dichiarata, non una perdita silenziosa.

Campionamento notizie: 65 tick su 30 giorni = 2,2 tick/giorno, max 3 notizie
per tick = ~6,5 notizie/giorno su ~29 disponibili. Circa il 22% dell'archivio.
Da dichiarare in tesi, e da variare nella sensitivity analysis.

## Prima di lanciare

```bash
export LLM_API_KEY='...'
python scripts/preflight.py
```

Deve uscire con "Tutto a posto". Verifica che le correzioni critiche siano
davvero nel codice in esecuzione: il gate col ritmo giusto, il floor del
rallentamento, la finestra scorrevole, `chat_template_kwargs` al top level.
Se anche solo uno fallisce, un run da otto ore brucia la quota della notte
e quella della squadra.

## Lancio

```bash
tmux new -s mirrorfish

python run.py \
  --profiles ./simulations/sim_14/reddit_profiles.json \
  --news ./start/notizieansa/notizie_social \
  --start 2026-02-20 --days 30 --hours-per-tick 11 \
  --out runs/base_s42 --seed 42 --rpm 8 \
  2>&1 | tee runs/base_s42.log
```

`tmux` serve perche' il processo deve sopravvivere alla chiusura del
terminale. `Ctrl-b d` per staccarsi, `tmux attach -t mirrorfish` per tornare.

In un secondo terminale:

```bash
python scripts/serve_feed.py runs/base_s42/run.db --port 8000
```

Su http://127.0.0.1:8000 — processo separato, database in sola lettura, non
puo' toccare la simulazione. Su macOS la porta 5000 e' presa da AirPlay
Receiver: usa la 8000.

## Cosa guardare nei primi 15 minuti

Non lasciarlo solo subito. Tre cose:

1. **`err=0` nella riga di tick.** Se gli errori salgono, e' parsing o quota.
2. **Nessun `[rate-limit]`.** Il limitatore nuovo dovrebbe fermarsi PRIMA del
   429 leggendo `x-ratelimit-*-remaining`. Se compaiono lo stesso, il gate
   funziona ma il ritmo va abbassato: fermati e rilancia con `--rpm 6`.
3. **Il primo passo di riflessione** (al tick 4). E' il tipo di chiamata mai
   provato in volume: se produce errori di parsing si vede subito.

Poi puoi andare a dormire.

## Se si interrompe

```bash
python run.py ... --resume
```

Riprende dall'ultimo tick completo, ripulisce il tick parziale, non rifa la
survey baseline. Stessi identici argomenti piu' `--resume`.

## Domattina

```bash
python scripts/quota_report.py --db runs/base_s42/run.db
python scripts/render_feed.py runs/base_s42/run.db
```

Il primo da' il consumo misurato su un run completo — molto piu' solido delle
105 chiamate di stasera. Il secondo da' l'HTML da leggere: e' il controllo che
i numeri aggregati non intercettano.
