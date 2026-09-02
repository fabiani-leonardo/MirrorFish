#!/usr/bin/env python3
"""
Ispettore del feed: da run.db a un file HTML autonomo.

    python scripts/render_feed.py runs/pilot/run.db -o runs/pilot/feed.html

Serve a leggere cosa hanno effettivamente scritto gli agenti. I numeri
aggregati dicono che il 20,8% ha cambiato voto; non dicono se l'hanno fatto
per ragioni sensate o perche' il modello stava producendo testo degenere. La
lettura qualitativa e' l'unico controllo che intercetta il secondo caso, e in
tesi vale come sezione a se': esempi di traiettorie individuali accanto ai
dati aggregati.

Nessuna dipendenza esterna, nessuna CDN: si apre offline con doppio clic.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
from datetime import datetime
from pathlib import Path

CSS = """
:root {
  --ink:#1c1a17; --ink-soft:#5f5952; --line:#ded7cc; --page:#faf8f4;
  --card:#fff; --news:#7a3b1d; --news-bg:#fdf3ec;
  --si:#2f6b4f; --no:#8f2f2f; --ast:#6b6357;
}
* { box-sizing:border-box; }
body {
  margin:0; background:var(--page); color:var(--ink);
  font:16px/1.55 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}
header {
  border-bottom:1px solid var(--line); padding:2rem 1.5rem 1.5rem;
  background:var(--card);
}
.wrap { max-width:52rem; margin:0 auto; }
h1 { margin:0 0 .3rem; font-size:1.55rem; font-weight:600; letter-spacing:-.01em; }
.sub { color:var(--ink-soft); font-size:.9rem; margin:0; }
.meters { display:flex; flex-wrap:wrap; gap:1.8rem; margin-top:1.4rem; }
.meter b { display:block; font-size:1.5rem; font-weight:600; font-variant-numeric:tabular-nums; }
.meter span { font-size:.78rem; color:var(--ink-soft); }
main { padding:1.5rem; }
.controls {
  display:flex; gap:.6rem; flex-wrap:wrap; align-items:center;
  margin-bottom:1.6rem; font-family:system-ui,sans-serif; font-size:.85rem;
}
.controls input, .controls select {
  font:inherit; padding:.4rem .6rem; border:1px solid var(--line);
  border-radius:3px; background:var(--card); color:var(--ink);
}
.controls input[type=search] { flex:1; min-width:12rem; }
.tick-sep {
  display:flex; align-items:center; gap:.8rem; margin:2.2rem 0 1rem;
  color:var(--ink-soft); font-family:system-ui,sans-serif; font-size:.78rem;
}
.tick-sep::after { content:""; flex:1; height:1px; background:var(--line); }
article {
  background:var(--card); border:1px solid var(--line); border-radius:4px;
  padding:1rem 1.15rem; margin-bottom:.9rem;
}
article.news { border-left:3px solid var(--news); background:var(--news-bg); }
.byline {
  display:flex; align-items:baseline; gap:.5rem; margin-bottom:.45rem;
  font-family:system-ui,sans-serif; font-size:.82rem;
}
.who { font-weight:600; color:var(--ink); }
.meta { color:var(--ink-soft); }
.body { margin:0; white-space:pre-wrap; }
.foot {
  margin-top:.7rem; padding-top:.55rem; border-top:1px solid var(--line);
  font-family:system-ui,sans-serif; font-size:.78rem; color:var(--ink-soft);
  display:flex; gap:1.1rem; flex-wrap:wrap; align-items:center;
}
.likers { cursor:help; border-bottom:1px dotted var(--line); }
.replies { margin:.85rem 0 0 1.1rem; padding-left:1rem; border-left:1px solid var(--line); }
.replies article { margin-bottom:.6rem; padding:.75rem .9rem; }
.vote { font-weight:600; font-family:system-ui,sans-serif; font-size:.76rem; }
.v-SI { color:var(--si); } .v-NO { color:var(--no); } .v-ASTENUTO { color:var(--ast); }
.shift { font-family:system-ui,sans-serif; font-size:.76rem; }
.note { color:var(--ink-soft); font-style:italic; }
.notes-box {
  margin-top:.6rem; padding:.6rem .8rem; background:var(--page);
  border-radius:3px; font-size:.86rem;
}
.notes-box p { margin:.25rem 0; }
details summary { cursor:pointer; font-family:system-ui,sans-serif; font-size:.78rem; color:var(--ink-soft); }
.empty { color:var(--ink-soft); padding:3rem 0; text-align:center; }
.hidden { display:none !important; }
@media (max-width:600px){ .replies{margin-left:.3rem;padding-left:.7rem;} }
"""

JS = """
const q = document.getElementById('q');
const who = document.getElementById('who');
const kind = document.getElementById('kind');
function apply(){
  const t = q.value.toLowerCase(), w = who.value, k = kind.value;
  let shown = 0;
  document.querySelectorAll('article.top').forEach(a=>{
    const txt = a.dataset.search;
    const okT = !t || txt.includes(t);
    const okW = !w || a.dataset.authors.split(',').includes(w);
    const okK = !k || (k==='news' ? a.classList.contains('news')
                                  : !a.classList.contains('news'));
    const ok = okT && okW && okK;
    a.classList.toggle('hidden', !ok);
    if(ok) shown++;
  });
  document.querySelectorAll('.tick-sep').forEach(s=>{
    let n=0, el=s.nextElementSibling;
    while(el && !el.classList.contains('tick-sep')){
      if(el.classList.contains('top') && !el.classList.contains('hidden')) n++;
      el = el.nextElementSibling;
    }
    s.classList.toggle('hidden', n===0);
  });
  document.getElementById('empty').classList.toggle('hidden', shown>0);
}
[q,who,kind].forEach(e=>e.addEventListener('input',apply));
"""


def esc(s: str) -> str:
    return html.escape(s or "")


def load(db: str) -> dict:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    d: dict = {}
    d["agents"] = {r["agent_id"]: dict(r)
                   for r in conn.execute("SELECT * FROM agent")}
    d["posts"] = [dict(r) for r in conn.execute(
        "SELECT * FROM post ORDER BY tick, post_id")]
    d["likes"] = {}
    for r in conn.execute("SELECT post_id, agent_id FROM reaction "
                          "WHERE kind='like'"):
        d["likes"].setdefault(r["post_id"], []).append(r["agent_id"])
    d["notes"] = {}
    for r in conn.execute("SELECT agent_id, tick, note FROM note ORDER BY note_id"):
        d["notes"].setdefault(r["agent_id"], []).append((r["tick"], r["note"]))
    d["votes"] = {}
    for r in conn.execute("SELECT label, agent_id, vote, motivation FROM vote"):
        d["votes"].setdefault(r["agent_id"], {})[r["label"]] = (
            r["vote"], r["motivation"])
    d["meta"] = {r["key"]: json.loads(r["value"])
                 for r in conn.execute("SELECT * FROM run_meta")}
    conn.close()
    return d


def render_post(p: dict, d: dict, replies: list[dict], top: bool) -> str:
    a = d["agents"].get(p["agent_id"], {})
    name = a.get("username", f"agent_{p['agent_id']}")
    is_news = p["kind"] == "news"
    likers = d["likes"].get(p["post_id"], [])
    liker_names = ", ".join(
        d["agents"].get(x, {}).get("username", str(x)) for x in likers[:25])

    bits = []
    if likers:
        bits.append(f'<span class="likers" title="{esc(liker_names)}">'
                    f'{len(likers)} mi piace</span>')
    if replies:
        bits.append(f"{len(replies)} rispost{'a' if len(replies)==1 else 'e'}")

    vote = d["votes"].get(p["agent_id"], {})
    if not is_news and "baseline" in vote and "final" in vote:
        b, f = vote["baseline"][0], vote["final"][0]
        if b != f:
            bits.append(f'<span class="shift">voto <span class="vote v-{b}">'
                        f'{b}</span> &rarr; <span class="vote v-{f}">{f}</span></span>')
        else:
            bits.append(f'<span class="shift">voto '
                        f'<span class="vote v-{f}">{f}</span></span>')

    demo = []
    if a.get("age"):
        demo.append(f"{a['age']} anni")
    if a.get("region"):
        demo.append(a["region"])
    if a.get("profession"):
        demo.append(a["profession"])

    inner = "".join(render_post(r, d, [], False) for r in replies)
    reply_block = f'<div class="replies">{inner}</div>' if inner else ""

    search = f"{name} {p['content']}".lower()
    authors = ",".join({str(p["agent_id"]),
                        *(str(r["agent_id"]) for r in replies)})

    return (
        f'<article class="{"top " if top else ""}{"news" if is_news else ""}" '
        f'data-search="{esc(search)}" data-authors="{authors}">'
        f'<div class="byline"><span class="who">'
        f'{"ANSA" if is_news else esc(name)}</span>'
        f'<span class="meta">{esc(" · ".join(demo)) if demo and not is_news else ""}'
        f'{"" if is_news else " · "}{esc(p["sim_date"])}</span></div>'
        f'<p class="body">{esc(p["content"])}</p>'
        f'{f"<div class=foot>{' '.join(bits)}</div>" if bits else ""}'
        f'{reply_block}</article>'
    )


def build(d: dict, title: str) -> str:
    by_parent: dict[int, list[dict]] = {}
    tops: list[dict] = []
    for p in d["posts"]:
        if p["parent_id"]:
            by_parent.setdefault(p["parent_id"], []).append(p)
        else:
            tops.append(p)

    chunks, last_tick = [], None
    for p in tops:
        if p["tick"] != last_tick:
            chunks.append(f'<div class="tick-sep">tick {p["tick"] + 1} · '
                          f'{esc(p["sim_date"])}</div>')
            last_tick = p["tick"]
        chunks.append(render_post(p, d, by_parent.get(p["post_id"], []), True))

    people = sorted(
        ((i, a["username"]) for i, a in d["agents"].items() if not a["is_source"]),
        key=lambda x: x[1])
    opts = "".join(f'<option value="{i}">{esc(n)}</option>' for i, n in people)

    tally = {}
    for v in d["votes"].values():
        if "final" in v:
            tally[v["final"][0]] = tally.get(v["final"][0], 0) + 1
    shifted = sum(1 for v in d["votes"].values()
                  if "baseline" in v and "final" in v
                  and v["baseline"][0] != v["final"][0])

    n_posts = sum(1 for p in d["posts"] if p["kind"] != "news")
    n_news = sum(1 for p in d["posts"] if p["kind"] == "news")
    meters = [
        (n_posts, "post e risposte"), (n_news, "notizie iniettate"),
        (sum(len(v) for v in d["likes"].values()), "mi piace"),
        (sum(len(v) for v in d["notes"].values()), "note di riflessione"),
        (shifted, "hanno cambiato voto"),
    ]
    for k in ("SI", "NO", "ASTENUTO"):
        if tally.get(k):
            meters.append((tally[k], f"voto {k}"))

    meter_html = "".join(
        f'<div class="meter"><b>{n}</b><span>{esc(l)}</span></div>'
        for n, l in meters)

    stub = d["meta"].get("llm", {}).get("stub")
    warn = ('<p class="sub" style="color:#8f2f2f">Run stub: i contenuti sono '
            'generati da un LLM finto, non dal modello.</p>') if stub else ""

    return f"""<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>{CSS}</style></head><body>
<header><div class="wrap">
<h1>{esc(title)}</h1>
<p class="sub">Generato il {datetime.now():%d/%m/%Y %H:%M} · {len(d["agents"])} agenti</p>
{warn}
<div class="meters">{meter_html}</div>
</div></header>
<main><div class="wrap">
<div class="controls">
  <input type="search" id="q" placeholder="Cerca nel testo dei post">
  <select id="who"><option value="">Tutti gli autori</option>{opts}</select>
  <select id="kind"><option value="">Post e notizie</option>
    <option value="news">Solo notizie</option>
    <option value="post">Solo agenti</option></select>
</div>
{"".join(chunks) or '<p class="empty">Nessun post in questo run.</p>'}
<p class="empty hidden" id="empty">Nessun risultato per questo filtro.</p>
</div></main>
<script>{JS}</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    db = Path(args.db)
    out = Path(args.out) if args.out else db.parent / "feed.html"
    d = load(str(db))
    out.write_text(build(d, args.title or f"Feed simulato · {db.parent.name}"),
                   encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1024:.0f} KB, "
          f"{len(d['posts'])} post)")


if __name__ == "__main__":
    main()
