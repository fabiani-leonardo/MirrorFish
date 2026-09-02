# La simulazione sembra ferma: cosa controllare

## Tempi attesi (105 agenti, 8 req/min)

| output                          | quando          |
|---------------------------------|-----------------|
| `[setup] N agenti, M archi...`  | subito          |
| `[news] finestra simulata...`   | subito          |
| survey baseline (105 chiamate)  | ~13 min         |
| `--- RISULTATI [baseline] ---`  | dopo ~13 min    |
| `tick 1/65 ...`                 | dopo ~18 min    |

**Il primo tick compariva dopo circa 18 minuti.** Non era un blocco: era la
survey baseline, 105 chiamate sequenziali a 8 al minuto, senza alcun output
intermedio perche' `asyncio.gather` non emette nulla finche' non ha finito.

Corretto: ora survey e tick stampano il progresso ogni 10 e ogni 15 chiamate,
con ritmo effettivo e stima. La riga di tick ha anche l'ETA complessiva.

## Ma `[setup]` deve comparire subito

Se non vedi nemmeno quella riga, il problema e' prima. Controlla il comando:

```
python run.py \ 
              ^-- barra seguita da UNO SPAZIO
```

`\` seguito da spazio **non e' una continuazione di riga**: e' uno spazio
letterale, e il comando finisce li'. Le righe successive vengono eseguite come
comandi separati. La barra deve essere l'ultimo carattere della riga, senza
spazi dopo.

Nel comando che hai incollato ogni riga finisce con `\` + spazio. Se e' cosi'
anche nel terminale, `python run.py` e' partito senza argomenti.

## Verifica che stia lavorando davvero

Da un secondo terminale, mentre gira:

```bash
sqlite3 runs/test1/run.db "SELECT COUNT(*), MAX(created_at) FROM llm_call;"
```

Se il numero cresce ogni minuto, sta lavorando. Se resta a 0 dopo due minuti,
e' fermo prima delle chiamate.

```bash
ps aux | grep run.py          # il processo esiste?
```

## Comando corretto, in una riga sola

Cosi' non ci sono barre di continuazione da sbagliare:

```bash
python -u run.py --profiles ./simulations/sim_14/reddit_profiles.json --news ./start/notizieansa/notizie_social --start 2026-02-20 --days 30 --hours-per-tick 11 --out runs/base_s42 --seed 42 --rpm 8 2>&1 | tee runs/base_s42.log
```

`-u` non era necessario nei miei test (l'output attraversava la pipe
comunque), ma non costa nulla e toglie di mezzo la variabile.
