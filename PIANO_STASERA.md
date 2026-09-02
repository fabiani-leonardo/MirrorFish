# Dati per il colloquio — versione con misure reali

## Cosa e' confermato

**Finestra del rate limit: minuto solare fisso.**
Richieste alle 21:15:01 / :23 / :43 hanno decrementato il contatore 7 -> 6 -> 5
senza ricaricarsi; alle 21:16:05 era di nuovo a 7. Il limite di 8 e' per
minuto di orologio, non scorrevole.

**Consumo reale misurato** (105 chiamate vere contro `lab-qwen36`):

| tipo       |  n | prompt medio | out medio | out p95 | tetto | troncate |
|------------|---:|-------------:|----------:|--------:|------:|---------:|
| action     | 66 |          760 |        73 |     136 |   384 |        0 |
| reflection | 19 |        1.008 |       117 |     222 |   512 |        0 |
| vote       | 48 |          319 |       141 |     213 |   512 |        0 |

**Contabilita' del gateway a regime = consumo reale.** Verificato: 39 token
scalati per 33 di prompt + 6 di output. Nessuna penalizzazione basata su
`max_tokens`.

**Latenza media: ~940 ms.** Con 5 richieste parallele gia' concesse, il tetto
fisico del parallelismo e' ~300 req/min.

## La causa del problema originale, riprodotta

Sei misure sulla stessa baseline (`probe.py --n 3 --max-tokens 256 4096`)
si spiegano con un modello solo: **il gateway riserva `max_tokens` + ~13
token di overhead contro la finestra da 100.000/minuto, e riconcilia contro
la riserva della richiesta precedente.**

| req | max_tokens | riserva | rimborso prec. | netto atteso | osservato |
|----:|-----------:|--------:|---------------:|-------------:|----------:|
|   1 |        256 |     269 |              0 |          269 |       269 |
|   2 |        256 |     269 |            230 |           39 |        39 |
|   3 |        256 |     269 |            230 |           39 |        39 |
|   4 |      4.096 |   4.109 |            230 |        3.879 |     3.879 |
|   5 |      4.096 |   4.109 |          4.070 |           39 |        39 |
|   6 |      4.096 |   4.109 |          4.070 |           39 |        39 |

Sei su sei. A regime il netto e' il consumo reale (39 token per 33+6 usati),
ma la riserva resta appesa alla finestra.

Conseguenza diretta sul problema originale: con `max_tokens` omesso, il valore
diventava 261.678, cioe' una riserva **2,6 volte l'intera finestra da 100.000
token al minuto**. Una singola richiesta non poteva essere ammessa, mai.

Non era la KV cache di vLLM: era la contabilita' del gateway. Spiegazione piu'
semplice, riprodotta, e falsificabile — basta rifare la tabella.

## L'argomento

> Ho misurato 105 chiamate reali. Consumo in media ~730 token per chiamata
> fra prompt e risposta, e il contatore del gateway conferma che scala il
> consumo reale, non il tetto richiesto. Con 100.000 token al minuto ne
> sosterrei oltre 100 al minuto; ne ho 8, quindi sto usando circa il 10% dei
> token che mi sono gia' stati assegnati e il 90% resta inutilizzato.
> Il tetto di 5 richieste parallele, con una latenza media di 940 ms, ne
> sostiene fisicamente ~300 al minuto: non chiedo di alzarlo.
> Chiedo di passare da 8 a 40 richieste al minuto. Resterei sotto il 60%
> della mia finestra token, senza chiedere un token in piu' e senza toccare
> il parallelismo.

Se chiede un numero piu' basso, 25 req/min risolve comunque il problema:
un run sui 5 mesi passa da 43 ore a 14.

## Cosa comporta (105 agenti, tick da 11h)

| finestra simulata | chiamate | a 8 req/min | a 40 req/min |
|-------------------|---------:|------------:|-------------:|
| 30 giorni         |    4.283 |       8,9 h |        1,8 h |
| 150 giorni (5 mesi)|  20.783 |      43,3 h |        8,7 h |

Con 12 run (4 condizioni x 3 repliche) sui 5 mesi: **520 ore contro 104**.
A quota attuale il disegno sperimentale non entra nel mese che ho.

## Da dire, perche' rafforza la richiesta

Il problema originale — chiamate senza `max_tokens`, che vLLM espandeva a
261.678 — e' gia' risolto: i tetti sono espliciti e obbligatori nel codice
(una chiamata senza budget solleva un'eccezione), e la telemetria registra
ogni chiamata con budget, token consumati e `finish_reason`. Zero troncamenti
su 105 chiamate: i tetti sono tarati, non tirati a indovinare.

## Correzioni applicate stasera

- **`vote` portato da 300 a 512 token.** Il p95 misurato era 213 e il massimo
  242: 300 lasciava un margine di due risposte lunghe prima del troncamento.
- **Limitatore riscritto.** I tre 429 durante `quota_report --live` venivano
  da un bug: `RateGate` riceveva `cfg.min_interval_s` (= 0) invece di
  `cfg.pace_interval()`, e la formula di rallentamento con base 0 produceva
  0. Le 30 chiamate sono partite a raffica. Ora c'e' una finestra scorrevole
  con riserva, un floor esplicito, e soprattutto la lettura di
  `x-ratelimit-*-remaining-requests` dalle risposte riuscite: ci si ferma
  PRIMA del 429 invece di reagire dopo. Conta anche il consumo degli altri
  membri sul limite di team condiviso, che localmente non e' osservabile.
