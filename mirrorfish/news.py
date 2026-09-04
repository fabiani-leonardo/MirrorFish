"""
Stream di notizie ANSA.

Differenza sostanziale rispetto ad ansa_injector.py: nessun processo separato
che fa polling su run_state.json e scrive nel DB dall'esterno ogni 30 secondi.
Qui le notizie sono parte del tick loop.

Perche' conta, e non e' solo eleganza:
  - l'injector esterno introduce una race condition fra il momento in cui
    scrive il post e il momento in cui l'agente costruisce il feed. Due run
    identici possono vedere notizie in ordine diverso. Con la validazione su
    piu' punti che ti serve, questo e' esattamente cio' che non puoi
    permetterti;
  - il controfattuale diventa banale: `stream.fork_at(tick, altra_cartella)`
    e da quel tick in poi la popolazione legge notizie diverse. Non serve
    toccare la simulazione.

La logica di parsing dei nomi file e' presa dal tuo ansa_injector, che gia'
funzionava: `30ottobre2025-607.txt` -> 2025-10-30.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

MESI = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4,
    "maggio": 5, "giugno": 6, "luglio": 7, "agosto": 8,
    "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}

_NAME_RE = re.compile(
    r"^(\d{1,2})(" + "|".join(MESI) + r")(\d{4})", re.IGNORECASE
)


def parse_date_from_filename(filename: str) -> date | None:
    """
    `30ottobre2025-607.txt` -> date(2025, 10, 30).

    Rispetto alla versione originale usa una regex ancorata: `name[:index]`
    su un nome tipo `1marzo2026` funziona, ma su un nome che contiene due
    volte una sottostringa-mese darebbe il risultato sbagliato senza segnalarlo.
    """
    stem = Path(filename).stem.split("-")[0].strip().lower()
    m = _NAME_RE.match(stem)
    if not m:
        return None
    day, month_name, year = m.group(1), m.group(2).lower(), m.group(3)
    try:
        return date(int(year), MESI[month_name], int(day))
    except ValueError:
        return None


@dataclass(frozen=True)
class NewsItem:
    """
    Una notizia, alle tre profondita' con cui puo' essere letta.

    Titolo e corpo vengono dallo STESSO file dell'archivio integrale
    (notizie_referendum): la divisione avviene in lettura, non in due
    cartelle diverse.
    """

    news_id: str
    published: date
    title: str                   # titolo originale, dall'articolo integrale
    body: str = ""               # testo integrale

    def at_depth(self, depth: str, max_body_chars: int = 1400) -> str:
        """
        Il testo che l'agente legge, secondo la sua profondita' di lettura.

        Modella un fatto ovvio del consumo di notizie: davanti allo stesso
        lancio d'agenzia, chi segue la politica legge l'articolo, chi la segue
        di sfuggita legge il titolo, chi non la segue non lo apre affatto e
        semmai ne sente parlare da altri.

        Titolo e corpo escono dallo stesso file scrapato: e' `parse_full_article`
        a separarli, e questa funzione decide quanto darne all'agente. Non
        servono due archivi.

        `nessuna` non compare qui perche' non e' un modo di leggere: e' non
        ricevere la notizia. Lo gestisce il feed azzerando gli slot.
        """
        if depth == "integrale":
            body = self.body
            if len(body) > max_body_chars:
                body = body[:max_body_chars].rsplit(" ", 1)[0] + " [...]"
            return f"[ANSA] {self.title}\n{body}"
        return f"[ANSA] {self.title}"          # titolo, e default

    def as_post(self) -> str:
        """
        Testo salvato nella tabella `post`.

        E' il titolo, non l'articolo: la riga in DB e' l'identita' della
        notizia, il testo che ogni agente legge davvero viene ricostruito a
        ogni feed da at_depth(). Salvare qui il corpo integrale significava
        che chiunque ricevesse la notizia di seconda mano — citata in una
        reply, o mostrata a un agente `titolo` — se la ritrovava per intero.
        """
        return f"[ANSA] {self.title}"


_TITOLO = re.compile(r"^TITOLO:\s*(.+)$", re.MULTILINE)
_SEP = re.compile(r"^-{10,}\s*$", re.MULTILINE)


def parse_full_article(text: str) -> tuple[str, str]:
    """
    Estrae titolo e corpo dal formato di `notizie_referendum`:

        TITOLO: ...
        DATA ESTRATTA: 18febbraio2026
        LINK: https://...
        --------------------------------------------------

        <corpo>
    """
    m = _TITOLO.search(text)
    title = m.group(1).strip() if m else ""
    parts = _SEP.split(text, maxsplit=1)
    body = parts[1].strip() if len(parts) > 1 else ""
    body = " ".join(body.split())      # ANSA manda a capo a meta' frase
    return title, body


class NewsCoverageError(RuntimeError):
    """Gli articoli integrali mancano o sono troppo pochi per il disegno."""


def load_news(news_dir: str | Path, *, min_body_ratio: float = 0.9) -> list[NewsItem]:
    """
    Carica le notizie da un'unica cartella di articoli integrali.

    Un solo archivio, non due. La versione precedente iterava sui rilanci in
    stile social e ci attaccava il testo integrale cercando lo stesso nome
    file, con due conseguenze silenziose: un articolo integrale senza rilancio
    omonimo non entrava mai in simulazione, e se il testo integrale mancava
    `title` e `body` restavano vuoti, facendo collassare tutte le profondita'
    di lettura sulla stessa stringa.

    Ora c'e' un archivio solo. Titolo e corpo si ricavano dal medesimo file
    (`parse_full_article`), e la divisione fra chi legge il titolo e chi legge
    l'articolo avviene in lettura, dentro `NewsItem.at_depth`. Se la copertura
    dei corpi scende sotto `min_body_ratio` si solleva NewsCoverageError
    invece di partire con un esperimento che non puo' misurare cio' per cui e'
    stato scritto.
    """
    news_dir = Path(news_dir)
    if not news_dir.exists():
        raise FileNotFoundError(f"Cartella notizie inesistente: {news_dir}")

    items: list[NewsItem] = []
    scartati: list[str] = []
    senza_corpo: list[str] = []
    for f in sorted(news_dir.glob("*.txt")):
        d = parse_date_from_filename(f.name)
        if d is None:
            scartati.append(f.name)
            continue
        title, body = parse_full_article(
            f.read_text(encoding="utf-8", errors="replace"))
        if not title:
            # Senza titolo la notizia non e' leggibile a nessuna profondita'.
            scartati.append(f.name)
            continue
        if not body:
            senza_corpo.append(f.name)
        items.append(NewsItem(news_id=f.stem, published=d, title=title, body=body))

    items.sort(key=lambda n: (n.published, n.news_id))

    if scartati:
        print(f"[news] {len(scartati)} file ignorati (data o titolo non "
              f"leggibili): {scartati[:5]}"
              f"{'...' if len(scartati) > 5 else ''}")
    if not items:
        raise NewsCoverageError(f"Nessuna notizia leggibile in {news_dir}.")

    ratio = 1.0 - len(senza_corpo) / len(items)
    if ratio < min_body_ratio:
        raise NewsCoverageError(
            f"Solo il {ratio:.0%} delle notizie in {news_dir} ha un corpo "
            f"integrale ({len(senza_corpo)} su {len(items)} senza).\n"
            f"Sotto questa soglia gli agenti 'integrale' leggono lo stesso "
            f"testo di quelli 'titolo': la profondita' di lettura smette di "
            f"essere una variabile e il run non puo' misurarla.\n"
            f"Controlla il formato atteso (TITOLO: / riga di trattini / corpo).\n"
            f"Primi file senza corpo: {senza_corpo[:5]}")
    if senza_corpo:
        print(f"[news] {len(senza_corpo)} notizie senza corpo integrale "
              f"({1 - ratio:.1%}): a quelle gli agenti 'integrale' vedono "
              f"solo il titolo.")
    return items


class NewsStream:
    """
    Mappa notizie -> tick. Le notizie del giorno D vengono pubblicate a partire
    dal primo tick del giorno D, distribuite sui tick di quel giorno per non
    scaricarle tutte insieme.
    """

    def __init__(
        self,
        items: list[NewsItem],
        start_date: date,
        hours_per_tick: int,
        total_ticks: int,
        max_per_tick: int = 3,
        verbose: bool = True,
    ):
        self.start_date = start_date
        self.hours_per_tick = hours_per_tick
        self.ticks_per_day = 24 / hours_per_tick
        self.total_ticks = total_ticks
        self.max_per_tick = max_per_tick
        self.dropped_before = 0
        self.dropped_after = 0
        self.dropped_overflow = 0
        self._schedule: dict[int, list[NewsItem]] = {}
        self._build(items)
        if verbose:
            self.report()

    def tick_date(self, tick: int) -> date:
        return self.start_date + timedelta(days=(tick * self.hours_per_tick) // 24)

    def _first_tick_of_day(self, day_offset: int) -> int:
        import math
        return math.ceil(day_offset * 24 / self.hours_per_tick)

    def _ticks_in_day(self, day_offset: int) -> list[int]:
        a = self._first_tick_of_day(day_offset)
        b = self._first_tick_of_day(day_offset + 1)
        return [t for t in range(a, min(b, self.total_ticks))]

    def _build(self, items: list[NewsItem]) -> None:
        by_day: dict[int, list[NewsItem]] = {}
        for n in items:
            day_offset = (n.published - self.start_date).days
            if day_offset < 0:
                # Le notizie precedenti all'inizio finestra vengono SCARTATE.
                # Prima venivano schiacciate sul giorno 0: con un archivio che
                # parte da ottobre 2025 e una simulazione che parte a marzo
                # 2026 significa scaricare centinaia di articoli sul primo
                # giorno, saturare i feed e rendere il tick 0 non
                # interpretabile. Se servono come contesto pregresso vanno
                # messi nella bio, non nel feed.
                self.dropped_before += 1
                continue
            if self._first_tick_of_day(day_offset) >= self.total_ticks:
                self.dropped_after += 1
                continue
            by_day.setdefault(day_offset, []).append(n)

        for day_offset, day_items in by_day.items():
            slots = self._ticks_in_day(day_offset)
            if not slots:
                self.dropped_after += len(day_items)
                continue
            capacity = len(slots) * self.max_per_tick
            if len(day_items) > capacity:
                # Tenere solo le prime N e' una scelta di campionamento, non
                # una perdita silenziosa: va dichiarata nel run.
                self.dropped_overflow += len(day_items) - capacity
                day_items = day_items[:capacity]
            for i, item in enumerate(day_items):
                self._schedule.setdefault(slots[i % len(slots)], []).append(item)

    def report(self) -> None:
        kept = sum(len(v) for v in self._schedule.values())
        end = self.tick_date(self.total_ticks - 1)
        print(f"[news] finestra simulata: {self.start_date} -> {end}")
        print(f"[news] {kept} notizie schedulate su {len(self._schedule)} tick "
              f"(max {self.max_per_tick}/tick)")
        if self.dropped_before:
            print(f"[news] ATTENZIONE: {self.dropped_before} notizie PRIMA "
                  f"di {self.start_date} scartate. Se le vuoi, sposta "
                  f"--start indietro.")
        if self.dropped_after:
            print(f"[news] {self.dropped_after} notizie dopo {end} scartate.")
        if self.dropped_overflow:
            print(f"[news] {self.dropped_overflow} notizie scartate per "
                  f"superamento di max_per_tick.")

    def at(self, tick: int) -> list[NewsItem]:
        return self._schedule.get(tick, [])

    def fork_at(self, tick: int, alt_items: list[NewsItem]) -> "NewsStream":
        """
        Ramo controfattuale: identico fino a `tick` escluso, poi sostituisce
        completamente le notizie con `alt_items`. Questo e' il meccanismo che
        implementa l'analisi di sensitivita' descritta nella metodologia.
        """
        forked = NewsStream.__new__(NewsStream)
        forked.start_date = self.start_date
        forked.hours_per_tick = self.hours_per_tick
        forked.ticks_per_day = self.ticks_per_day
        forked.total_ticks = self.total_ticks
        forked._schedule = {t: v for t, v in self._schedule.items() if t < tick}
        forked.max_per_tick = self.max_per_tick
        forked.dropped_before = forked.dropped_after = forked.dropped_overflow = 0
        alt = NewsStream(alt_items, self.start_date, self.hours_per_tick,
                         self.total_ticks, self.max_per_tick, verbose=False)
        for t, v in alt._schedule.items():
            if t >= tick:
                forked._schedule[t] = v
        return forked

    def summary(self) -> str:
        n = sum(len(v) for v in self._schedule.values())
        return (f"{n} notizie su {len(self._schedule)} tick "
                f"(da {self.tick_date(0)} a {self.tick_date(self.total_ticks - 1)})")
