"""Greedy vs sampling vs beam search for travel recommendations.

Runs a small instruct model (default Qwen2.5-0.5B-Instruct — CPU-friendly)
on the persona paragraphs in data/personas.json and compares four decoding
strategies:

    greedy        do_sample=False                    1 candidate
    sample        temp=0.9 top_p=0.95, 3 runs        3 candidates (3 passes)
    beam          num_beams=4                        1 best-of-4 candidate
    beam_top4     num_beams=4, return all 4          4 candidates (1 pass)

For *diverse* beam groups (num_beam_groups + diversity_penalty) on
transformers v5 you need
    custom_generate="transformers-community/group-beam-search",
    trust_remote_code=True
in the generate() call — plain multi-beam is built in.

The prompt mirrors a travel-planner wizard style: stable system
prefix + per-request suffix carrying the user's trip paragraph, strict JSON
output. See README.md for why beam search helps and where it doesn't.

Usage:
    python src/demo_hf.py --persona couple_foodie_anniversary
    python src/demo_hf.py --all --model Qwen/Qwen2.5-1.5B-Instruct
    python src/demo_hf.py --sweep          # all personas, aggregated stats
    python src/demo_hf.py --sweep --samples 8 --out results/sweep.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

PERSONAS_PATH = Path(__file__).resolve().parent.parent / "data" / "personas.json"

# Mirrors the travel-planner convention: long byte-stable system prefix
# (cacheable), short variable suffix (the user's paragraph).
SYSTEM = """You are a travel recommendation engine. The traveler describes a trip
in a free-text paragraph. You produce a JSON list of the top 8 next actions
they can take to build their itinerary.

Output STRICT JSON only — no prose, no markdown fences. Shape:

{"recommendations": [
  {"action": "Book", "target": "Hotel in Alfama",
   "when": "2026-06-09/2026-06-13",
   "rationale": "<one sentence why it fits THIS traveler>"},
  {"action": "Visit", "target": "Gothic Quarter", "when": "2026-06-10",
   "rationale": "..."}
]}

Valid actions and example targets:
- "Book": bookable things. Targets like "Hotel in Alfama", "Flight to
  Lisbon", "Uber to airport", "Sagrada Familia tickets", "concert tickets",
  "food tour experience".
- "Visit": places to plan a visit to. Targets like "Gothic Quarter",
  "Boqueria market", "Sintra day trip".
- "Ask_About": topics to research before committing. Targets like
  "stroller access at Sintra", "cider house reservations".

The `when` field is ISO 8601: a single date "2026-06-10" or an interval
"2026-06-09/2026-06-13" for ranges (stays, booking windows). Use null only
when no date applies.

Rules:
- Exactly 8 recommendations.
- Match the stated pace: packed trips skew to Book/Visit anchors, slow trips
  include Ask_About research items.
- Respect hard constraints in the paragraph (diet, mobility, budget, kids).
- Targets must be real, well-known things or generic categories
  (e.g. "family-friendly tapas bar"), never invented proper nouns.
"""

USER_TEMPLATE = """<trip_paragraph>
{paragraph}
</trip_paragraph>

Return the JSON recommendations object now."""


def build_prompt(tokenizer, paragraph: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(paragraph=paragraph)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


VALID_ACTIONS = {"Book", "Visit", "Ask_About"}


def try_parse(text: str) -> list[dict] | None:
    """Parse the recommendations JSON; return None if malformed or
    any item uses an off-vocabulary action (the app would reject it)."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        recs = obj.get("recommendations")
        if not isinstance(recs, list):
            return None
        if any(r.get("action") not in VALID_ACTIONS for r in recs):
            return None
        return recs
    except (ValueError, AttributeError):
        return None


def timed_generate(model, tokenizer, prompt: str, **gen_kwargs) -> tuple[str, float, int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.monotonic()
    out = model.generate(**inputs, **gen_kwargs)
    dt = time.monotonic() - t0
    n_out = out.shape[1] - n_in
    text = tokenizer.decode(out[0][n_in:], skip_special_tokens=True)
    return text, dt, n_out


def timed_generate_multi(model, tokenizer, prompt: str, n: int, **gen_kwargs):
    """Generate n sequences in ONE call (num_return_sequences)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.monotonic()
    out = model.generate(**inputs, num_return_sequences=n, **gen_kwargs)
    dt = time.monotonic() - t0
    texts = [
        tokenizer.decode(o[n_in:], skip_special_tokens=True)
        for o in out
    ]
    ns = [int(o.shape[0] - n_in) for o in out]
    return texts, dt, ns


def show(text: str, dt: float, n_out: int | None = None, tag: str = "") -> bool:
    recs = try_parse(text)
    n_tok = f", {n_out} tok" if n_out is not None else ""
    if recs is None:
        print(f"    [{tag}] {dt:5.1f}s{n_tok}  ✗ malformed JSON")
        print("    raw:", text[:200].replace("\n", " "))
        return False
    print(f"    [{tag}] {dt:5.1f}s{n_tok}  ✓ {len(recs)} recs")
    for r in recs[:4]:
        print(f"        • {r.get('action','?')} → {r.get('target','?')}")
    if len(recs) > 4:
        print(f"        … +{len(recs)-4} more")
    return True


def sweep(model, tokenizer, personas, args) -> None:
    """Run every persona through every strategy; aggregate validity + wall
    time into a JSON summary (the numbers cited in README §4)."""
    # beam-search knobs; run 2 sets --length-penalty / --min-new-tokens
    beam_kwargs = {}
    if args.length_penalty != 1.0:
        beam_kwargs["length_penalty"] = args.length_penalty
    if args.min_new_tokens:
        beam_kwargs["min_new_tokens"] = args.min_new_tokens
    if args.repetition_penalty != 1.0:
        beam_kwargs["repetition_penalty"] = args.repetition_penalty

    results: list[dict] = []
    for persona in personas:
        prompt = build_prompt(tokenizer, persona["paragraph"])
        gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
        rec = {"persona": persona["id"]}
        print(f"\n── {persona['id']} ──", flush=True)

        if not args.beam_only:
            text, dt, n = timed_generate(model, tokenizer, prompt, **gen)
            rec["greedy"] = {"wall": round(dt, 2), "tokens": n,
                             "valid": try_parse(text) is not None}

            samples = []
            for i in range(args.samples):
                text, dt, n = timed_generate(
                    model, tokenizer, prompt, do_sample=True,
                    temperature=0.9, top_p=0.95,
                    **{k: v for k, v in gen.items() if k != "do_sample"},
                )
                recs = try_parse(text)
                samples.append({"wall": round(dt, 2), "tokens": n,
                                "valid": recs is not None,
                                "n_recs": len(recs) if recs else 0})
                print(f"  sample {i + 1}/{args.samples} "
                      f"{'✓' if recs else '✗'} {dt:.1f}s", flush=True)
            rec["sampling"] = samples

        text, dt, n = timed_generate(
            model, tokenizer, prompt, num_beams=args.beams,
            early_stopping=True, **beam_kwargs, **gen,
        )
        recs = try_parse(text)
        rec["beam_best"] = {"wall": round(dt, 2), "tokens": n,
                            "valid": recs is not None,
                            "n_recs": len(recs) if recs else 0,
                            "snippet": "" if recs else text[:160]}

        texts, dt, ns = timed_generate_multi(
            model, tokenizer, prompt, n=args.beams, num_beams=args.beams,
            early_stopping=True, **beam_kwargs, **gen,
        )
        rec["beam_all"] = {
            "wall": round(dt, 2),
            "candidates": [
                {"valid": (r := try_parse(t)) is not None,
                 "n_recs": len(r or []), "tokens": n_t,
                 "snippet": "" if r else t[:160]}
                for t, n_t in zip(texts, ns)
            ],
        }
        print(f"  beam k={args.beams}: {dt:.1f}s, "
              f"{sum(c['valid'] for c in rec['beam_all']['candidates'])}"
              f"/{args.beams} valid", flush=True)
        results.append(rec)

    # ---- aggregate ----
    n_p = len(results)
    k = args.beams
    b_wall = sum(r["beam_all"]["wall"] for r in results) / n_p
    b_valid = sum(c["valid"] for r in results
                  for c in r["beam_all"]["candidates"])
    b_tokens = sum(c["tokens"] for r in results
                   for c in r["beam_all"]["candidates"])
    bb_valid = sum(r["beam_best"]["valid"] for r in results)

    summary = {
        "model": args.model,
        "n_personas": n_p,
        "beam_width": k,
        "length_penalty": args.length_penalty,
        "min_new_tokens": args.min_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "beam_all": {"valid": b_valid, "of": n_p * k,
                     "mean_wall": round(b_wall, 1),
                     "total_tokens": b_tokens},
        "beam_best": {"valid": bb_valid, "of": n_p},
    }
    if not args.beam_only:
        n_s = args.samples
        g_valid = sum(r["greedy"]["valid"] for r in results)
        g_wall = sum(r["greedy"]["wall"] for r in results) / n_p
        g_tokens = sum(r["greedy"]["tokens"] for r in results)
        s_valid = sum(s["valid"] for r in results for s in r["sampling"])
        s_wall = sum(s["wall"] for r in results
                     for s in r["sampling"]) / (n_p * n_s)
        s_tokens = sum(s["tokens"] for r in results for s in r["sampling"])
        p = s_valid / (n_p * n_s) or 1e-9
        summary["samples_per_persona"] = n_s
        summary["greedy"] = {"valid": g_valid, "of": n_p,
                             "mean_wall": round(g_wall, 1),
                             "total_tokens": g_tokens}
        summary["sampling"] = {
            "valid": s_valid, "of": n_p * n_s,
            "mean_wall_per_pass": round(s_wall, 1),
            "total_tokens": s_tokens,
            "expected_passes_for_k_valid": round(k / p, 1)}

    out = {"summary": summary, "per_persona": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 68)
    print("SWEEP SUMMARY")
    print("=" * 68)
    if not args.beam_only:
        print(f"  greedy      {g_valid}/{n_p} valid   "
              f"mean wall {g_wall:.1f}s  (1 candidate)")
        print(f"  sampling    {s_valid}/{n_p * n_s} valid  "
              f"mean wall {s_wall:.1f}s/pass → "
              f"~{k / p:.1f} passes for {k} valid candidates")
    print(f"  beam k={k}    {b_valid}/{n_p * k} valid  "
          f"mean wall {b_wall:.1f}s/pass (all {k} candidates, 1 pass)")
    print(f"  beam best   {bb_valid}/{n_p} valid")
    print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--persona", default="couple_foodie_anniversary")
    ap.add_argument("--all", action="store_true", help="run all personas")
    ap.add_argument("--sweep", action="store_true",
                    help="all personas × all strategies, aggregated stats")
    ap.add_argument("--samples", type=int, default=5,
                    help="sampling passes per persona in --sweep mode")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--beam-only", action="store_true",
                    help="sweep only the beam strategies (run-2 reruns)")
    ap.add_argument("--length-penalty", type=float, default=1.0,
                    help="beam length penalty; >1 favors longer sequences")
    ap.add_argument("--min-new-tokens", type=int, default=0,
                    help="beam min_new_tokens; blocks early-EOS degenerates")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="penalizes tokens already in context; discourages "
                         "beams from parroting the prompt's few-shot example")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "results" / "sweep.json")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    personas = json.loads(PERSONAS_PATH.read_text())
    selected = (personas if args.all or args.sweep
                else [p for p in personas if p["id"] == args.persona])
    if not selected:
        raise SystemExit(f"unknown persona {args.persona!r}; "
                         f"choices: {[p['id'] for p in personas]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {args.model} on {device} …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    if args.sweep:
        sweep(model, tokenizer, selected, args)
        return

    for persona in selected:
        print("\n" + "=" * 72)
        print(f"PERSONA: {persona['label']}")
        print("=" * 72)
        print(f"  {persona['paragraph'][:120]}…")
        prompt = build_prompt(tokenizer, persona["paragraph"])
        gen = dict(max_new_tokens=args.max_new_tokens, do_sample=False)

        print("\n  ── greedy (1 candidate) ──")
        text, dt, n = timed_generate(model, tokenizer, prompt, **gen)
        show(text, dt, n, "greedy")

        print("\n  ── sampling ×3, temp=0.9 (3 separate passes) ──")
        ok = 0
        t0 = time.monotonic()
        for i in range(3):
            text, dt, n = timed_generate(
                model, tokenizer, prompt, do_sample=True,
                temperature=0.9, top_p=0.95, **{k: v for k, v in gen.items() if k != "do_sample"},
            )
            ok += show(text, dt, n, f"sample {i+1}")
        print(f"    total wall time: {time.monotonic()-t0:.1f}s, "
              f"valid JSON: {ok}/3")

        print("\n  ── beam search, num_beams=4 (1 best candidate) ──")
        text, dt, n = timed_generate(
            model, tokenizer, prompt, num_beams=4,
            early_stopping=True, **gen,
        )
        show(text, dt, n, "beam best")

        print("\n  ── beam top-4: all beams returned, ranked by score (1 pass) ──")
        texts, dt, _ = timed_generate_multi(
            model, tokenizer, prompt, n=4, num_beams=4,
            early_stopping=True, **gen,
        )
        ok = 0
        for i, t in enumerate(texts):
            ok += show(t, dt / len(texts), None, f"beam {i+1}")
        print(f"    total wall time (all 4): {dt:.1f}s, valid JSON: {ok}/4")

    print("\nDone. See README.md §'How beam search maps to this app'.")


if __name__ == "__main__":
    main()
