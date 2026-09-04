# Taratura dopo l'aumento a 60 RPM

## Cosa dice il 429

```
Rate limit exceeded for team: b2e816ee...  Limit type: requests
```

**team**, non `team_member`. Il professore ha alzato il TUO limite a 60, ma
quello di squadra — condiviso con Chiara e l'altro collega — e' un'altra
quota, e a ~47 richieste/minuto e' quella che ha ceduto.

Il freno preventivo non l'ha intercettata: leggeva `team_member` e usciva
subito, e dopo l'aumento quel contatore e' sempre abbondante. Ora prende il
**minimo** fra `api_key`, `team_member` e `team`. Aggiorna il codice prima di
ritarare, altrimenti misuri il bug.

## Sequenza (10 minuti)

### 1. Leggi i limiti attuali

```bash
python scripts/probe.py --n 2
```

Guarda le tre righe `x-ratelimit-*-limit-requests`. Serve sapere se il limite
di team e' rimasto a 25 o e' salito anche lui: se e' fermo a 25 e Chiara
lavora, il tuo tetto reale non e' 60.

### 2. Trova il ritmo sostenibile

```bash
python scripts/diagnose_llm.py --concurrency 2 5 --max-tokens 384 \
       --requests 30 --cooldown 70
```

La pausa da 70s fra configurazioni non e' facoltativa: con finestre da un
minuto, pause piu' corte fanno misurare il rate limit invece del parametro.

### 3. Verifica sul campo con margine

Parti **sotto** il limite letto, non sopra. Con `team` a 25 condiviso, `--rpm
18` e' prudente; se il team e' salito a 60, prova `--rpm 40`.

```bash
python -u run.py ... --rpm 18 --out runs/prova --days 2 --agents 25
```

Se in 5 minuti non compaiono `[rate-limit]`, sali di 10 e ripeti.

## Tempi con i vari ritmi

| req/min | un run | 12 run (4 condizioni x 3 repliche) |
|---:|---:|---:|
| 8 | 8,4 h | 100 h |
| 25 | 2,7 h | **32 h** |
| 40 | 1,7 h | 20 h |
| 55 | 1,2 h | 15 h |

Anche restando a 25 effettivi, il disegno completo entra in un fine settimana.
Il vincolo che bloccava la tesi non c'e' piu'.

## Cortesia verso la squadra

Il limite di team e' condiviso. Prima di lanciare 12 run di fila, vale la pena
avvisare Chiara: con `--rpm 40` su una quota di team da 60 le lasci 20, che
per il suo carico basta, ma e' meglio detto che scoperto.
