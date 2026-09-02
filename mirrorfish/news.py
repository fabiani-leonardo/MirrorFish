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
    news_id: str
    published: date
    content: str

    def as_post(self) -> str:
        return f"[ANSA] {self.content}"


def load_news(news_dir: str | Path) -> list[NewsItem]:
    """Carica i .txt datati, ordinati per data poi per nome (deterministico)."""
    news_dir = Path(news_dir)
    if not news_dir.exists():
        raise FileNotFoundError(f"Cartella notizie inesistente: {news_dir}")

    items: list[NewsItem] = []
    skipped: list[str] = []
    for f in sorted(news_dir.glob("*.txt")):
        d = parse_date_from_filename(f.name)
        if d is None:
            skipped.append(f.name)
            continue
        content = f.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            items.append(NewsItem(news_id=f.stem, published=d, content=content))

    items.sort(key=lambda n: (n.published, n.news_id))
    if skipped:
        print(f"[news] {len(skipped)} file ignorati (data non parsabile): "
              f"{skipped[:5]}{'...' if len(skipped) > 5 else ''}")
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
        ticks_per_day: int,
        total_ticks: int,
        max_per_tick: int = 3,
        verbose: bool = True,
    ):
        self.start_date = start_date
        self.ticks_per_day = ticks_per_day
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
        return self.start_date + timedelta(days=tick // self.ticks_per_day)

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
            if day_offset * self.ticks_per_day >= self.total_ticks:
                self.dropped_after += 1
                continue
            by_day.setdefault(day_offset, []).append(n)

        for day_offset, day_items in by_day.items():
            base = day_offset * self.ticks_per_day
            capacity = self.ticks_per_day * self.max_per_tick
            if len(day_items) > capacity:
                # Tenere solo le prime N e' una scelta di campionamento, non
                # una perdita silenziosa: va dichiarata nel run.
                self.dropped_overflow += len(day_items) - capacity
                day_items = day_items[:capacity]
            for i, item in enumerate(day_items):
                t = base + (i % self.ticks_per_day)
                if t < self.total_ticks:
                    self._schedule.setdefault(t, []).append(item)

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
        forked.ticks_per_day = self.ticks_per_day
        forked.total_ticks = self.total_ticks
        forked._schedule = {t: v for t, v in self._schedule.items() if t < tick}
        forked.max_per_tick = self.max_per_tick
        forked.dropped_before = forked.dropped_after = forked.dropped_overflow = 0
        alt = NewsStream(alt_items, self.start_date, self.ticks_per_day,
                         self.total_ticks, self.max_per_tick, verbose=False)
        for t, v in alt._schedule.items():
            if t >= tick:
                forked._schedule[t] = v
        return forked

    def summary(self) -> str:
        n = sum(len(v) for v in self._schedule.values())
        return (f"{n} notizie su {len(self._schedule)} tick "
                f"(da {self.tick_date(0)} a {self.tick_date(self.total_ticks - 1)})")
