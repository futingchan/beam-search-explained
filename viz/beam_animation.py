"""Generate viz/beam_search.svg — an animated walkthrough of beam search.

Replays the same toy example as src/beam_search_scratch.py: beam_width=2
explores a rigged LM where greedy ("Visit"…) dead-ends and the beam finds
"Book flights early".

Pure SVG + SMIL animation — no deps, plays inline on GitHub via <img>.

Run:  python viz/beam_animation.py
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent / "beam_search.svg"

W, H = 980, 470
CYCLE = 16.0  # seconds per loop

# Node: id -> (cx, cy, label, sub, width)
NODES = {
    "root":     (80,  230, "⟨prompt⟩", "", 120),
    "visit":    (245, 105, "Visit",  "-0.40", 120),
    "book":     (245, 230, "Book",   "-0.60", 120),
    "ask":      (245, 355, "Ask_About", "-1.20", 120),
    "v_the":    (420, 75,  "…museum",  "-3.40", 120),
    "v_a":      (420, 135, "…beach",   "-3.90", 120),
    "b_tix":    (420, 200, "…flights", "-0.90", 120),
    "b_a":      (420, 260, "…hotel",   "-2.60", 120),
    "bt_early": (625, 175, "…2026-06-12/2026-06-19", "-1.10", 185),
    "bt_on":    (625, 225, "…2026-10-01", "-1.40", 185),
    "ba_tour":  (625, 275, "…2026-06-12/2026-06-18", "-4.10", 185),
}
NODE_H = 46

# Edge: id -> (parent, child, step_logprob)
EDGES = {
    "e_visit":  ("root", "visit", "-0.4"),
    "e_book":   ("root", "book", "-0.6"),
    "e_ask":    ("root", "ask", "-1.2"),
    "e_vthe":   ("visit", "v_the", "-3.0"),
    "e_va":     ("visit", "v_a", "-3.5"),
    "e_btix":   ("book", "b_tix", "-0.3"),
    "e_ba":     ("book", "b_a", "-2.0"),
    "e_bte":    ("b_tix", "bt_early", "-0.2"),
    "e_bto":    ("b_tix", "bt_on", "-0.5"),
    "e_bat":    ("b_a", "ba_tour", "-1.5"),
}

# Timeline captions: (text, t_start, t_end)
CAPTIONS = [
    ("Step 1 · Expand — every live beam proposes next-token logprobs", 0.0, 3.0),
    ("Step 2 · Rank candidates by cumulative logprob, keep top k=2 — 'Ask_About' pruned", 3.0, 5.6),
    ("Step 3 · Expand the survivors — a pruned branch never grows back", 5.6, 8.4),
    ("Step 4 · Rank again — 'Visit' won step 1 locally, loses cumulatively", 8.4, 10.8),
    ("Step 5 · Beams emit <eos> → finished pile, ranked by total score", 10.8, 12.6),
    ("Step 6 · Winner: 'Book flights 2026-06-12/19' (-1.10) — greedy's 'Visit museum 2026-10-03' ended at -3.90", 12.6, 16.0),
]


def t(sec: float) -> float:
    return sec / CYCLE


# If FRAME is set, emit a static SVG with opacity resolved at t=FRAME seconds
# (for thumbnails / viewers without SMIL). Otherwise emit SMIL animation.
FRAME: float | None = None


def appear(begin: float, fade_in: float = 0.5) -> str:
    """SMIL opacity: hidden until `begin`, then fade in, stay."""
    if FRAME is not None:
        return ""
    return (
        f'<animate attributeName="opacity" dur="{CYCLE}s" '
        f'repeatCount="indefinite" fill="freeze" '
        f'values="0;0;1;1" keyTimes="0;{t(begin):.4f};{t(begin+fade_in):.4f};1"/>'
    )


def appear_opacity(begin: float) -> float:
    return 1.0 if (FRAME or 0) >= begin else 0.0


def appear_dim(begin: float, dim_at: float, dim_to: float = 0.12) -> str:
    """Appear at `begin`, dim at `dim_at`, stay dimmed until loop restarts."""
    if FRAME is not None:
        return ""
    return (
        f'<animate attributeName="opacity" dur="{CYCLE}s" '
        f'repeatCount="indefinite" fill="freeze" '
        f'values="0;0;1;1;{dim_to};{dim_to}" '
        f'keyTimes="0;{t(begin):.4f};{t(begin+0.5):.4f};'
        f'{t(dim_at):.4f};{t(dim_at+0.4):.4f};1"/>'
    )


def appear_dim_opacity(begin: float, dim_at: float, dim_to: float = 0.12) -> float:
    f = FRAME or 0
    if f < begin:
        return 0.0
    return dim_to if f >= dim_at else 1.0


def pulse(begin: float, end: float) -> str:
    """Brief highlight pulse between two times (used on kept beams)."""
    if FRAME is not None:
        return ""
    return (
        f'<animate attributeName="opacity" dur="{CYCLE}s" '
        f'repeatCount="indefinite" fill="freeze" '
        f'values="0;0;1;1;0;0" '
        f'keyTimes="0;{t(begin):.4f};{t(begin+0.3):.4f};'
        f'{t(end):.4f};{t(end+0.3):.4f};1"/>'
    )


def pulse_opacity(begin: float, end: float) -> float:
    f = FRAME or 0
    return 1.0 if begin <= f <= end else 0.0


def edge_svg(eid: str, anim: str, opacity: float = 0.0) -> str:
    p, c, lp = EDGES[eid]
    x1, y1 = NODES[p][0] + NODES[p][4] / 2, NODES[p][1]
    x2, y2 = NODES[c][0] - NODES[c][4] / 2, NODES[c][1]
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    return f"""<g opacity="{opacity}">
      <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"
            stroke="#8b949e" stroke-width="1.5"/>
      <text x="{mx}" y="{my - 5}" text-anchor="middle" font-size="11"
            fill="#8b949e" font-family="monospace">{lp}</text>
      {anim}
    </g>"""


def node_svg(nid: str, anim: str, fill: str = "#161b22",
             stroke: str = "#58a6ff", opacity: float = 0.0) -> str:
    x, y, label, sub, w = NODES[nid]
    font = 15 if len(label) <= 14 else 11
    sub_svg = (
        f'<text x="{x}" y="{y + 15}" text-anchor="middle" font-size="11" '
        f'fill="#8b949e" font-family="monospace">Σ {sub}</text>'
        if sub else ""
    )
    return f"""<g opacity="{opacity}">
      <rect x="{x - w/2}" y="{y - NODE_H/2}" width="{w}"
            height="{NODE_H}" rx="9" fill="{fill}" stroke="{stroke}"
            stroke-width="1.5"/>
      <text x="{x}" y="{y - 2}" text-anchor="middle" font-size="{font}"
            fill="#e6edf3" font-family="monospace">{escape(label)}</text>
      {sub_svg}
      {anim}
    </g>"""


def main() -> None:
    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="ui-monospace, monospace">'
    )
    parts.append(f'<rect width="{W}" height="{H}" fill="#0d1117" rx="12"/>')
    parts.append(
        f'<text x="24" y="34" font-size="15" fill="#58a6ff" '
        f'font-family="monospace">beam search · beam_width = 2 · '
        f'score = cumulative logprob</text>'
    )

    # --- captions (bottom, one visible at a time) ---
    for text, b, e in CAPTIONS:
        if FRAME is not None:
            op = 1.0 if b <= FRAME < e else 0.0
            parts.append(
                f'<text x="24" y="{H - 18}" font-size="14" fill="#e6edf3" '
                f'font-family="monospace" opacity="{op}">{escape(text)}</text>'
            )
            continue
        # hold 0 until b, quick fade-in, hold 1 until e, quick fade-out
        eps = 0.15 / CYCLE
        parts.append(
            f'<text x="24" y="{H - 18}" font-size="14" fill="#e6edf3" '
            f'font-family="monospace" opacity="0">{escape(text)}'
            f'<animate attributeName="opacity" dur="{CYCLE}s" '
            f'repeatCount="indefinite" fill="freeze" '
            f'values="0;0;1;1;0;0" '
            f'keyTimes="0;{t(b):.4f};{t(b)+eps:.4f};{t(e):.4f};'
            f'{t(e)+eps:.4f};1"/></text>'
        )

    def E(eid: str, begin: float, dim_at: float | None = None) -> None:
        if dim_at is None:
            parts.append(edge_svg(eid, appear(begin), appear_opacity(begin)))
        else:
            parts.append(edge_svg(
                eid, appear_dim(begin, dim_at),
                appear_dim_opacity(begin, dim_at)))

    def N(nid: str, begin: float, dim_at: float | None = None,
          **kw) -> None:
        if dim_at is None:
            parts.append(node_svg(nid, appear(begin),
                                  opacity=appear_opacity(begin), **kw))
        else:
            parts.append(node_svg(
                nid, appear_dim(begin, dim_at),
                opacity=appear_dim_opacity(begin, dim_at), **kw))

    # --- frame 1: root + depth 1 (appear 0.4–1.6), Ask_About dims at 3.4 ---
    N("root", 0.2, fill="#21262d", stroke="#8b949e")
    E("e_visit", 0.8); E("e_book", 1.0); E("e_ask", 1.2, dim_at=3.4)
    N("visit", 0.9); N("book", 1.1); N("ask", 1.3, dim_at=3.4)

    # --- frame 2: depth 2 (appear 6.0–7.6), Visit branch dims at 9.0 ---
    E("e_vthe", 6.0, dim_at=9.0); E("e_va", 6.2, dim_at=9.0)
    E("e_btix", 6.4); E("e_ba", 6.6)
    N("v_the", 6.1, dim_at=9.0); N("v_a", 6.3, dim_at=9.0)
    N("b_tix", 6.5); N("b_a", 6.7)

    # --- frame 3: depth 3 (appear 9.6–11.2); losers dim at 12.8 ---
    E("e_bte", 9.6); E("e_bto", 9.8, dim_at=12.8); E("e_bat", 10.0, dim_at=12.8)
    N("bt_early", 9.7, stroke="#3fb950")
    N("bt_on", 9.9, dim_at=12.8); N("ba_tour", 10.1, dim_at=12.8)

    # --- winner burst: green halo pulses 13.0–15.6 ---
    x, y, w = NODES["bt_early"][0], NODES["bt_early"][1], NODES["bt_early"][4]
    parts.append(f"""<g opacity="{pulse_opacity(13.0, 15.6)}">
      <rect x="{x - w/2 - 6}" y="{y - NODE_H/2 - 6}" width="{w + 12}"
            height="{NODE_H + 12}" rx="12" fill="none" stroke="#3fb950"
            stroke-width="3"/>
      <text x="{x}" y="{y - NODE_H/2 - 14}" text-anchor="middle"
            font-size="13" fill="#3fb950" font-family="monospace">
        ✓ best sequence</text>
      {pulse(13.0, 15.6)}
    </g>""")

    # --- greedy overlay 13.2–15.8: dashed red line root→visit→v_the ---
    greedy_path = (
        f"M {NODES['root'][0] + NODES['root'][4]/2} {NODES['root'][1]} "
        f"L {NODES['visit'][0] - NODES['visit'][4]/2} {NODES['visit'][1]} "
        f"M {NODES['visit'][0] + NODES['visit'][4]/2} {NODES['visit'][1]} "
        f"L {NODES['v_the'][0] - NODES['v_the'][4]/2} {NODES['v_the'][1]}"
    )
    parts.append(f"""<g opacity="{pulse_opacity(13.2, 15.8)}">
      <path d="{greedy_path}" stroke="#f85149" stroke-width="3"
            stroke-dasharray="6 4" fill="none"/>
      <text x="{NODES['v_the'][0]}" y="{NODES['v_the'][1] - NODE_H/2 - 10}"
            text-anchor="middle" font-size="12" fill="#f85149"
            font-family="monospace">greedy path → -3.90</text>
      {pulse(13.2, 15.8)}
    </g>""")

    parts.append("</svg>")
    out = OUT if FRAME is None else OUT.with_name(f"beam_search_t{FRAME:g}.svg")
    out.write_text("\n".join(parts))
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=float, default=None,
                    help="emit a static SVG with state resolved at t=seconds")
    args = ap.parse_args()
    if args.frame is not None:
        FRAME = args.frame
    main()
