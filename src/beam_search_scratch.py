"""Beam search from scratch — a pedagogical implementation.

No ML libraries required. We define a tiny hand-built "language model" whose
next-token log-probabilities are chosen so that GREEDY decoding picks a
locally-best first token that leads to a bad ending, while BEAM SEARCH finds
the globally better sequence.

The scenario is deliberately recommendation-flavoured: the model is
"deciding" how to phrase a travel recommendation.

Run:  python src/beam_search_scratch.py
"""

from __future__ import annotations

import math
from typing import Callable

EOS = "<eos>"


# ---------------------------------------------------------------------------
# A toy language model
# ---------------------------------------------------------------------------
# logprobs(prefix) -> {next_token: logprob}
#
# Tree (numbers are logprobs). Read each root->leaf path as one
# recommendation item: verb -> noun -> when (ISO 8601 date or interval,
# what the downstream API schema expects).
#
#   () ──"Visit"(-0.4)──"museum"(-3.0)──"2026-10-03"(-0.5)──<eos>
#     │                                       total = -3.9   <- greedy lands here
#     ├─"Book"(-0.6)───"flights"(-0.3)──"2026-06-12/2026-06-19"(-0.2)──<eos>
#     │                                       total = -1.1   <- beam=2 finds this
#     └─"Ask_About"(-1.2)───"local_customs"(-0.8)───"2026-04-01/2026-05-31"(-0.6)──<eos>
#                                                total = -2.6
#
# Greedy takes "Visit" because -0.4 is the best *single-step* score — then gets
# stuck on a low-probability branch. Beam width 2 keeps "Book" alive and wins.

_TABLE: dict[tuple[str, ...], dict[str, float]] = {
    (): {"Visit": -0.4, "Book": -0.6, "Ask_About": -1.2},
    ("Visit",): {"museum": -3.0, "beach": -3.5},
    ("Visit", "museum"): {"2026-10-03": -0.5, "2026-10-04": -1.0},
    ("Book",): {"flights": -0.3, "hotel": -2.0},
    ("Book", "flights"): {"2026-06-12/2026-06-19": -0.2, "2026-10-01": -0.5},
    ("Book", "hotel"): {"2026-06-12/2026-06-18": -1.5},
    ("Ask_About",): {"local_customs": -0.8, "visa_rules": -2.0},
    ("Ask_About", "local_customs"): {"2026-04-01/2026-05-31": -0.6},
}


def toy_lm(prefix: tuple[str, ...]) -> dict[str, float]:
    """Next-token logprobs for a completed prefix.

    Every 3-token sequence ends with EOS at logprob 0 (i.e. certainty), so
    sequences all have length 3 + <eos>. Unknown prefixes get a flat tiny
    distribution to keep the demo total.
    """
    if len(prefix) >= 3:
        return {EOS: 0.0}
    return _TABLE.get(prefix, {EOS: -5.0})


# ---------------------------------------------------------------------------
# Beam search — the part that matters
# ---------------------------------------------------------------------------

LogprobFn = Callable[[tuple[str, ...]], dict[str, float]]


def beam_search(
    score_fn: LogprobFn,
    beam_width: int = 2,
    max_len: int = 6,
    length_alpha: float = 0.0,
    verbose: bool = True,
) -> list[tuple[list[str], float]]:
    """Classic beam search.

    At each step we expand every live beam by every candidate next token,
    score each extension by *cumulative* logprob, and keep the top
    `beam_width` partial sequences. Beams that emit EOS are moved to the
    finished pile.

    length_alpha > 0 applies GNMT-style length normalisation when ranking
    finished hypotheses: score / ((5 + len) / 6) ** alpha. This counters the
    bias toward short sequences (logprobs only shrink as you add tokens).

    Returns finished hypotheses sorted by (possibly normalised) score, best
    first.
    """
    # live beams: (tokens, cumulative_logprob)
    beams: list[tuple[tuple[str, ...], float]] = [((), 0.0)]
    finished: list[tuple[tuple[str, ...], float]] = []

    for step in range(max_len):
        candidates: list[tuple[tuple[str, ...], float]] = []
        for tokens, score in beams:
            for tok, lp in score_fn(tokens).items():
                candidates.append((tokens + (tok,), score + lp))

        if verbose:
            print(f"  step {step + 1}: {len(beams)} live beams -> "
                  f"{len(candidates)} candidates, keep top {beam_width}")

        # rank all extensions, keep the best `beam_width`
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for tokens, score in candidates:
            if tokens[-1] == EOS:
                finished.append((tokens[:-1], score))
            else:
                beams.append((tokens, score))
            if len(beams) >= beam_width:
                break

        if verbose:
            for tokens, score in beams:
                print(f"    live:  {' '.join(tokens):<40} {score:+.2f}")
            for tokens, score in finished:
                print(f"    done:  {' '.join(tokens):<40} {score:+.2f}")

        if not beams:
            break

    # any still-live beams count as unfinished hypotheses too
    finished.extend(beams)

    def norm(item: tuple[tuple[str, ...], float]) -> float:
        tokens, score = item
        if length_alpha == 0.0:
            return score
        lp = ((5.0 + len(tokens)) / 6.0) ** length_alpha  # GNMT length penalty
        return score / lp

    finished.sort(key=norm, reverse=True)
    return [(list(t), s) for t, s in finished]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("TOY MODEL — full search tree")
    print("=" * 72)

    for width in (1, 2):
        print(f"\n--- beam_width = {width} {'(== greedy)' if width == 1 else ''} ---")
        results = beam_search(toy_lm, beam_width=width)
        best_seq, best_score = results[0]
        print(f"  BEST: '{' '.join(best_seq)}'  score={best_score:+.2f}")

    print("\n" + "=" * 72)
    print("TAKEAWAY")
    print("=" * 72)
    print(
        "Greedy (beam_width=1) chose 'Visit' — the best first token — and\n"
        "ended in a low-probability dead end. Beam width 2 kept the runner-up\n"
        "'Book' alive and found the sequence with the best *total* likelihood.\n"
        "\n"
        "For recommendations this maps directly:\n"
        "  * tokens        -> pieces of the JSON / list the model emits\n"
        "  * a beam        -> one candidate recommendation document\n"
        "  * beam_width    -> how many candidate docs you explore in parallel\n"
        "  * best beam     -> the highest-likelihood, most 'canonical' answer\n"
        "\n"
        "Same function, real model: swap `toy_lm` for a wrapper that runs a\n"
        "transformer and returns top-k next-token logprobs. That is literally\n"
        "what HF `generate(num_beams=...)` and vLLM `llm.beam_search()` do —\n"
        "plus batching, KV-caching and GPU kernels.\n"
    )


if __name__ == "__main__":
    main()
