"""Beam-search-for-recommendations benchmark — vLLM edition.

Self-contained harness. Replicates the local HF experiment on a serving
GPU so results carry real vLLM numbers (compiled guided decoding,
continuous batching, prefix caching).

    greedy      n=1, temperature=0
    samp_t07    n=4 in one generate() call, temp=0.7 top_p=0.95
    samp_t09    n=4 in one generate() call, temp=0.9 top_p=0.95
    beam_lp10   llm.beam_search beam_width=4, length_penalty=1.0,
                min_tokens=150 (re-implemented; see install_beam_min_tokens)
    beam_lp12   same, length_penalty=1.2
    *_g         same + structured_outputs=schema (vLLM guided decoding;
                beam arms skip guided to stay comparable to the local run,
                though vLLM >= 0.29 BeamSearchParams does accept a schema)

Phases:
    python src/vllm_experiment.py matrix      # strategy × persona
    python src/vllm_experiment.py width       # beam_width in {1,2,4,8}
    python src/vllm_experiment.py determinism # 5 reps seeded vs unseeded
    python src/vllm_experiment.py judge       # bigger-model judge
    python src/vllm_experiment.py analyze     # aggregate tables

Flags: --model, --tp N, --no-example, --strategies a b c,
--judge-model. Outputs → results/exp_*_<modeltag>.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERSONAS_PATH = ROOT / "data" / "personas.json"
RESULTS = ROOT / "results"
MAX_NEW = 1200  # 35B pretty-prints JSON (~3.1 char/tok): 8 recs need
                # ~600-900 tok. The original 400 truncated nearly every arm.
BEAM_MIN_TOKENS = 150  # reinstated via install_beam_min_tokens()
SEED = 1234

ARGS_MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
ARGS_JUDGE = "Qwen/Qwen3.5-122B-A10B-FP8"
ARGS_TP = 4
ARGS_JUDGE_TP = 8
ARGS_NO_EXAMPLE = False
ARGS_STRATEGIES: list[str] | None = None
ARGS_LIMIT_PERSONAS = 0  # 0 = all; debug aid for cheap smoke runs
ARGS_OUT_SUFFIX = ""     # keeps opt-in arms out of the main result files
ARGS_MAX_NUM_SEQS = 0    # 0 = vLLM default (256). The 122B judge is a hybrid
                         # Mamba/GDN model: each decode seq needs a Mamba cache
                         # block and tp=8 only affords ~136, so 256 aborts
                         # CUDA graph capture.
ARGS_SO_BACKEND = ""     # "" = leave vLLM default ("auto"); beam+guided
                         # needs an explicit backend (see load_llm)

SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string",
                               "enum": ["Book", "Visit", "Ask_About"]},
                    "target": {"type": "string"},
                    "when": {"type": ["string", "null"]},
                    "rationale": {"type": "string"},
                },
                "required": ["action", "target", "when", "rationale"],
            },
        }
    },
    "required": ["recommendations"],
}

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

# Ablation: same prompt minus the concrete JSON exemplar. Schema is carried
# by field descriptions only — tests whether beam's parroting was caused by
# the in-prompt example.
SYSTEM_NO_EXAMPLE = """You are a travel recommendation engine. The traveler describes a trip
in a free-text paragraph. You produce a JSON list of the top 8 next actions
they can take to build their itinerary.

Output STRICT JSON only — no prose, no markdown fences. The response is a
single object with one key "recommendations", an array of objects each with
exactly these keys:
- "action": one of "Book", "Visit", "Ask_About"
- "target": a string naming what to act on
- "when": ISO 8601 — a date "2026-06-10" or an interval
  "2026-06-09/2026-06-13" for ranges; null only when no date applies
- "rationale": one sentence on why it fits THIS traveler

Valid actions and example targets:
- "Book": bookable things, e.g. hotels, flights, rideshares, event tickets,
  tour or experience bookings.
- "Visit": places to plan a visit to.
- "Ask_About": topics to research before committing, e.g. accessibility
  questions, reservation policies.

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

VALID_ACTIONS = {"Book", "Visit", "Ask_About"}

STRATEGIES = {
    "greedy":      {"n": 1, "temp": 0.0},
    "greedy_g":    {"n": 1, "temp": 0.0, "guided": True},
    "samp_t07":    {"n": 4, "temp": 0.7},
    "samp_t07_g":  {"n": 4, "temp": 0.7, "guided": True},
    "samp_t09":    {"n": 4, "temp": 0.9},
    "samp_t09_g":  {"n": 4, "temp": 0.9, "guided": True},
    "beam_lp08":   {"n": 4, "beam": True, "length_penalty": 0.8},
    "beam_lp10":   {"n": 4, "beam": True, "length_penalty": 1.0},
    "beam_lp12":   {"n": 4, "beam": True, "length_penalty": 1.2},
    # vLLM >= 0.29 DOES accept structured_outputs on BeamSearchParams, so
    # these two are now runnable. Opt-in via --strategies (plus --out-suffix
    # so they land in their own file); deliberately excluded from the default
    # matrix to keep the headline numbers comparable to the local HF run.
    "beam_lp10_g": {"n": 4, "beam": True, "length_penalty": 1.0,
                    "guided": True},
    "beam_lp12_g": {"n": 4, "beam": True, "length_penalty": 1.2,
                    "guided": True},
}


# ------------------------------------------------------------------ helpers

def render_chat(tok, msgs) -> str:
    """Render a chat prompt with reasoning mode OFF.

    Qwen3.5/3.6 chat templates default to reasoning mode: they append a
    dangling "<think>\n", so the token budget goes to a reasoning trace
    instead of the answer. Generators emit no JSON at all; the judge emits
    "Thinking Process:\n\n1. **" and a naive [1-5] regex scrapes the "1" off
    its numbered list, silently scoring every candidate 1/5.

    Every prompt in this harness must go through here.
    """
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)


def build_prompt(llm, paragraph: str) -> str:
    tok = llm.get_tokenizer()
    system = SYSTEM_NO_EXAMPLE if ARGS_NO_EXAMPLE else SYSTEM
    return render_chat(tok, [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(paragraph=paragraph)}])


def load_personas(limit: int | None = None):
    ps = json.loads(PERSONAS_PATH.read_text())
    if limit:
        ps = ps[:limit]
    if ARGS_LIMIT_PERSONAS:
        ps = ps[:ARGS_LIMIT_PERSONAS]
    return ps


def encode_prompt(tok, prompt: str) -> list[int]:
    """Token ids for an already-templated prompt (no extra specials)."""
    try:
        return list(tok.encode(prompt, add_special_tokens=False))
    except TypeError:
        return list(tok.encode(prompt))


def _prompt_len(beam) -> int:
    p = beam.orig_prompt
    ids = p.get("prompt_token_ids") if isinstance(p, dict) else None
    return len(ids) if ids else 0


def install_beam_min_tokens(llm, min_tokens: int):
    """Reinstate BeamSearchParams.min_tokens, dropped in vLLM >= 0.29.

    `SamplingParams.min_tokens` works by masking EOS out of the logits. In
    vLLM's beam step each live beam is expanded into 2*beam_width candidates,
    one per top-logprob token, and the candidate whose token is EOS is moved
    to `instance.completed` while the rest stay live. Discarding those
    premature EOS candidates is therefore equivalent to masking EOS: the
    surviving non-EOS candidates, and their relative cum_logprob ranking, are
    untouched.

    Without this, length_penalty=1.0 makes a 9-token stub
    (`{"recommendations": [`) the highest-scoring beam and every beam arm
    collapses — which is exactly why the local HF run set min_tokens=150.
    """
    if getattr(llm, "_min_tokens_patched", False):
        return
    orig = llm._beam_search_step

    def step(**kw):
        stop = orig(**kw)
        for inst in kw["instances_batch"]:
            inst.completed = [
                b for b in inst.completed
                if len(b.tokens) - _prompt_len(b) >= min_tokens]
        if stop and any(inst.beams for inst in kw["instances_batch"]):
            stop = False
        return stop

    llm._beam_search_step = step
    llm._min_tokens_patched = True


def beam_call(llm, prompt: str, beam_width: int, length_penalty: float = 1.0,
              guided: bool = False):
    """Run llm.beam_search and return generated-only results.

    vLLM's BeamSearchSequence seeds `.tokens` with the prompt token ids and
    sets `.text = decode(.tokens)`, so both the text and the token count
    include the entire prompt. Feeding a TokensPrompt lets us slice the
    prompt back off exactly.
    """
    from vllm.sampling_params import BeamSearchParams
    if BEAM_MIN_TOKENS:
        install_beam_min_tokens(llm, BEAM_MIN_TOKENS)
    tok = llm.get_tokenizer()
    prompt_ids = encode_prompt(tok, prompt)
    n_prompt = len(prompt_ids)
    t0 = time.monotonic()
    bs_kw = {}
    if guided:
        from vllm.sampling_params import StructuredOutputsParams
        bs_kw["structured_outputs"] = StructuredOutputsParams(json=SCHEMA)
    out = llm.beam_search(
        [{"prompt_token_ids": prompt_ids}],
        BeamSearchParams(beam_width=beam_width, max_tokens=MAX_NEW,
                         temperature=0.0, length_penalty=length_penalty,
                         **bs_kw),
    )
    dt = time.monotonic() - t0
    seqs = out[0].sequences
    gen_ids = [list(s.tokens)[n_prompt:] for s in seqs]
    texts = [tok.decode(g, skip_special_tokens=True) for g in gen_ids]
    ns = [len(g) for g in gen_ids]
    lps = [s.cum_logprob for s in seqs]
    fins = [getattr(s, "finish_reason", None)
            or ("length" if n >= MAX_NEW else "stop")
            for s, n in zip(seqs, ns)]
    first = [g[0] if g else None for g in gen_ids]
    return texts, dt, ns, lps, fins, first


def _first_json_object(text: str) -> tuple[str | None, int]:
    """(substring, end_index) of the first balanced {...} in text.

    String-aware so braces inside string literals don't affect depth. Taking
    the FIRST balanced object rather than a first-'{' .. last-'}' span matters
    for the beam arms: beams routinely close a valid document and then loop the
    same items until the token cap, and a last-'}' span swallows that
    repetition and reports plain "invalid JSON" instead.
    """
    start = text.find("{")
    if start < 0:
        return None, -1
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], i + 1
    return None, -1


def try_parse_ex(text: str) -> tuple[list[dict] | None, int]:
    """(recs, trailing_chars). trailing_chars = generated text left over after
    the first complete JSON object — the beam degeneration signal."""
    blob, end = _first_json_object(text)
    if blob is None:
        try:
            s, e = text.index("{"), text.rindex("}") + 1
            blob, end = text[s:e], e
        except (ValueError, AttributeError):
            return None, 0
    trailing = max(len(text) - end, 0)
    try:
        obj = json.loads(blob)
    except ValueError:
        return None, trailing
    if not isinstance(obj, dict):
        return None, trailing
    recs = obj.get("recommendations")
    if not isinstance(recs, list):
        return None, trailing
    if any(not isinstance(r, dict) or r.get("action") not in VALID_ACTIONS
           for r in recs):
        return None, trailing
    return recs, trailing


def try_parse(text: str) -> list[dict] | None:
    return try_parse_ex(text)[0]


def load_llm(model: str, tp: int | None = None):
    from vllm import LLM
    kw = {}
    if ARGS_MAX_NUM_SEQS:
        kw["max_num_seqs"] = ARGS_MAX_NUM_SEQS
    if ARGS_SO_BACKEND:
        # beam_search's structured-output path resolves the backend from engine
        # config and only handles "xgrammar"/"guidance" explicitly -- the
        # default "auto" raises ValueError. Sampling arms are fine with "auto",
        # so this stays opt-in to avoid perturbing the specified runs.
        kw["structured_outputs_config"] = {"backend": ARGS_SO_BACKEND}
    return LLM(model=model, dtype="auto",
               tensor_parallel_size=tp if tp is not None else ARGS_TP,
               enable_prefix_caching=True, gpu_memory_utilization=0.85,
               max_model_len=8192, **kw)


def guided_kwargs():
    """vLLM version-tolerant structured-output param."""
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {"structured_outputs": StructuredOutputsParams(json=SCHEMA)}
    except ImportError:
        pass
    try:
        from vllm import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(json=SCHEMA)}
    except ImportError:
        raise RuntimeError(
            "no structured-output API found; need vllm>=0.6.3")


def gen_candidates(llm, prompt: str, cfg: dict, seed=None):
    """One call → (texts, wall, token counts, seq logprobs, finish reasons,
    first token ids)."""
    if cfg.get("beam"):
        return beam_call(llm, prompt, cfg["n"], cfg["length_penalty"],
                         guided=bool(cfg.get("guided")))

    from vllm import SamplingParams
    kw = dict(n=cfg["n"], temperature=cfg["temp"], max_tokens=MAX_NEW,
              top_p=0.95, logprobs=0)
    if cfg.get("guided"):
        kw.update(guided_kwargs())
    if seed is not None:
        kw["seed"] = seed
    t0 = time.monotonic()
    out = llm.generate([prompt], SamplingParams(**kw))
    dt = time.monotonic() - t0
    cands = out[0].outputs
    texts = [c.text for c in cands]
    ns = [len(c.token_ids) for c in cands]
    # cumulative_logprob present when logprobs requested
    lps = [c.cumulative_logprob or 0.0 for c in cands]
    fins = [c.finish_reason for c in cands]
    first = [c.token_ids[0] if c.token_ids else None for c in cands]
    return texts, dt, ns, lps, fins, first


# ---------------------------------------------------- deterministic checks

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
INTERVAL_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})/(\d{4}-\d{2}-\d{2})$")
GENERIC_TARGETS = {
    "hotel", "hostel", "flight", "uber", "taxi", "restaurant", "bar",
    "tickets", "ticket", "tour", "museum", "market", "train", "transfer",
    "rental", "reservation", "class", "workshop", "experience", "show",
    "concert", "spa", "ferry", "bus", "pass", "card", "day trip", "car",
    "guesthouse", "dinner", "lunch", "crawl", "tasting", "booking",
}


def norm_target(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w ]", " ", str(t).lower())).strip()


def check_candidate(persona: dict, recs: list[dict]) -> dict:
    v = {"when_out_of_window": 0, "when_not_iso": 0, "dup_items": 0,
         "missing_verbs": [], "foreign_targets": 0}
    seen = set()
    verbs = {r.get("action") for r in recs}
    w0, w1 = persona["window"]
    for r in recs:
        key = (r.get("action"), norm_target(r.get("target", "")))
        if key in seen:
            v["dup_items"] += 1
        seen.add(key)
        w = r.get("when")
        if w is not None:
            if DATE_RE.match(str(w)):
                if not (w0 <= w <= w1):
                    v["when_out_of_window"] += 1
            elif m := INTERVAL_RE.match(str(w)):
                if m.group(1) > w1 or m.group(2) < w0:
                    v["when_out_of_window"] += 1
            else:
                v["when_not_iso"] += 1
        if r.get("action") in ("Visit", "Ask_About"):
            t = norm_target(r.get("target", ""))
            ok = (any(a in t for a in persona["allowlist"])
                  or any(g in t for g in GENERIC_TARGETS))
            if not ok:
                v["foreign_targets"] += 1
    v["missing_verbs"] = [x for x in persona["implied_verbs"]
                          if x not in verbs]
    hard_pass = (v["when_out_of_window"] == 0 and v["when_not_iso"] == 0
                 and v["dup_items"] == 0 and not v["missing_verbs"])
    return {"violations": v, "hard_pass": hard_pass, "n_recs": len(recs)}


def coverage(persona: dict, recs: list[dict]) -> float:
    blob = " ".join(f"{r.get('target','')} {r.get('rationale','')}"
                    for r in recs).lower()
    hit = sum(any(m in blob for m in it["match"])
              for it in persona["interests"])
    return hit / len(persona["interests"])


def jaccard_diversity(cands: list[list[dict]]) -> float:
    sets = [{(r.get("action"), norm_target(r.get("target", "")))
             for r in c} for c in cands]
    pairs = [len(a & b) / len(a | b) if a | b else 1.0
             for a, b in itertools.combinations(sets, 2)]
    return sum(pairs) / len(pairs) if pairs else 0.0


def score_candidate(persona: dict, text: str) -> dict:
    recs, trailing = try_parse_ex(text)
    # `trailing` counts generated chars after the first complete JSON object;
    # a large value means the decoder kept going (beams loop their own items
    # instead of emitting EOS).
    extra = {"trailing_chars": trailing, "degenerate": trailing > 50}
    if recs is None:
        return {"valid": False, **extra}
    return {"valid": True, **extra, **check_candidate(persona, recs),
            "coverage": round(coverage(persona, recs), 3)}


def out_name(base: str) -> Path:
    tag = f"{base}_{ARGS_MODEL.split('/')[-1]}"
    if ARGS_NO_EXAMPLE:
        tag += "_noex"
    if ARGS_OUT_SUFFIX:
        tag += f"_{ARGS_OUT_SUFFIX}"
    return RESULTS / f"{tag}.json"


def save_meta(phase: str):
    """Append a run record to results/exp_meta.json — vLLM/torch versions,
    model, tp, flags — so results are self-describing for the writeup."""
    def v(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    rec = {"phase": phase, "ts": datetime.now(timezone.utc).isoformat(),
           "model": ARGS_MODEL, "judge_model": ARGS_JUDGE,
           "tp": ARGS_TP, "judge_tp": ARGS_JUDGE_TP,
           "no_example": ARGS_NO_EXAMPLE, "strategies": ARGS_STRATEGIES,
           "limit_personas": ARGS_LIMIT_PERSONAS,
           "out_suffix": ARGS_OUT_SUFFIX,
           "so_backend": ARGS_SO_BACKEND,
           "max_num_seqs": ARGS_MAX_NUM_SEQS,
           "max_new_tokens": MAX_NEW, "beam_min_tokens": BEAM_MIN_TOKENS,
           "seed": SEED,
           "vllm": v("vllm"), "torch": v("torch"),
           "transformers": v("transformers")}
    path = RESULTS / "exp_meta.json"
    data = json.loads(path.read_text()) if path.exists() else []
    data.append(rec)
    path.write_text(json.dumps(data, indent=1))


# ------------------------------------------------------------------ phases

def cmd_matrix(args):
    llm = load_llm(ARGS_MODEL)
    personas = load_personas()
    strategies = ARGS_STRATEGIES or list(STRATEGIES)
    out = []
    for persona in personas:
        prompt = build_prompt(llm, persona["paragraph"])
        prec = {"persona": persona["id"], "prompt": prompt,
                "strategies": {}}
        for name in strategies:
            cfg = STRATEGIES[name]
            texts, dt, ns, lps, fins, _ = gen_candidates(llm, prompt, cfg)
            cands = [{"text": t, "tokens": n, "seq_lp": round(lp, 4),
                      "finish": f, "hit_cap": n >= MAX_NEW,
                      **score_candidate(persona, t)}
                     for t, n, lp, f in zip(texts, ns, lps, fins)]
            prec["strategies"][name] = {
                "wall": round(dt, 2), "total_tokens": sum(ns),
                "jaccard": round(jaccard_diversity(
                    [try_parse(t) or [] for t in texts]), 3),
                "candidates": cands}
            nv = sum(c["valid"] for c in cands)
            print(f"  {persona['id']:>22} {name:>10}: {dt:5.1f}s "
                  f"{nv}/{cfg['n']} valid", flush=True)
        out.append(prec)
    path = out_name("exp_matrix")
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


def cmd_width(args):
    llm = load_llm(ARGS_MODEL)
    personas = load_personas()
    out = []
    for persona in personas:
        prompt = build_prompt(llm, persona["paragraph"])
        for k in (1, 2, 4, 8):
            texts, dt, ns, lps, fins, _ = beam_call(llm, prompt, k)
            cands = [{"text": t, "tokens": n, "seq_lp": round(lp, 4),
                      "finish": f, "hit_cap": n >= MAX_NEW,
                      **score_candidate(persona, t)}
                     for t, n, lp, f in zip(texts, ns, lps, fins)]
            out.append({"persona": persona["id"], "k": k,
                        "wall": round(dt, 2),
                        "total_tokens": sum(ns),
                        "jaccard": round(jaccard_diversity(
                            [try_parse(t) or [] for t in texts]), 3),
                        "top1": cands[0], "candidates": cands})
            print(f"  {persona['id']:>22} k={k}: {dt:5.1f}s "
                  f"top1_valid={cands[0]['valid']}", flush=True)
    path = out_name("exp_width")
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


def cmd_determinism(args):
    llm = load_llm(ARGS_MODEL)
    personas = load_personas(3)
    out = []
    for persona in personas:
        prompt = build_prompt(llm, persona["paragraph"])
        for name in ("samp_t09", "beam_lp10"):
            cfg = STRATEGIES[name]
            for mode in ("seeded", "unseeded"):
                hashes = []
                for _ in range(5):
                    texts, *_ = gen_candidates(
                        llm, prompt, cfg,
                        seed=SEED if mode == "seeded" else None)
                    hashes.append(hashlib.sha256(
                        "|".join(texts).encode()).hexdigest()[:12])
                out.append({"persona": persona["id"], "strategy": name,
                            "mode": mode, "unique_outputs": len(set(hashes)),
                            "of": 5})
                print(f"  {persona['id']:>22} {name:>9} {mode:>8}: "
                      f"{len(set(hashes))}/5 unique", flush=True)
    path = out_name("exp_determinism")
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


def cmd_mechanism(args):
    """First-token divergence test: does beam top-1 commit to a different
    opening token than greedy, and does the beam set recover an action
    verb (e.g. Book) that greedy's single candidate missed?"""
    llm = load_llm(ARGS_MODEL)
    personas = load_personas()
    tok = llm.get_tokenizer()
    out = []
    for persona in personas:
        prompt = build_prompt(llm, persona["paragraph"])
        g_texts, g_dt, _, _, _, g_first = gen_candidates(
            llm, prompt, STRATEGIES["greedy"])
        b_texts, b_dt, b_ns, b_lps, _, b_first = gen_candidates(
            llm, prompt, STRATEGIES["beam_lp10"])
        g_recs = try_parse(g_texts[0]) or []
        g_verbs = {r.get("action") for r in g_recs}
        beam_verbs = set()
        for t in b_texts:
            beam_verbs |= {r.get("action") for r in (try_parse(t) or [])}
        out.append({
            "persona": persona["id"],
            "greedy_first_tok": (
                tok.decode([g_first[0]])
                if g_first and g_first[0] is not None else None),
            "beam_top1_first_tok": (
                tok.decode([b_first[0]])
                if b_first and b_first[0] is not None else None),
            "beam_top1_differs": b_texts[0] != g_texts[0],
            "greedy_verbs": sorted(v for v in g_verbs if v),
            "beam_verbs_union": sorted(v for v in beam_verbs if v),
            "beam_recovers_verbs": sorted(
                v for v in beam_verbs - g_verbs if v),
            "greedy_text": g_texts[0],
            "beam_top1_text": b_texts[0],
            "greedy_wall": round(g_dt, 2), "beam_wall": round(b_dt, 2)})
        print(f"  {persona['id']:>22}: differs="
              f"{b_texts[0] != g_texts[0]} recovered="
              f"{sorted(v for v in beam_verbs - g_verbs if v)}", flush=True)
    path = out_name("exp_mechanism")
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


# ------------------------------------------------------------------ judge

JUDGE_SCORE = """A traveler wrote this trip description:

<trip>{paragraph}</trip>

A recommendation engine proposed this list of next actions (JSON):

<candidate>{candidate}</candidate>

Score the list 1-5 for how well it serves THIS traveler's stated interests,
constraints, and pace. Judge content quality, not JSON syntax.
5 = excellent fit, 1 = poor/generic/mismatched. Reply with the integer only."""


def parse_judge_score(raw: str) -> int | None:
    """Pull the 1-5 verdict out of a judge reply.

    Prefers a standalone digit (the reply should be the integer alone) and
    takes the LAST one, so any preamble cannot be mistaken for the verdict.
    Returns None rather than guessing -- a None is visible in the output,
    a wrong digit is not.
    """
    t = (raw or "").strip()
    bare = re.fullmatch(r"([1-5])\W*", t)   # expected shape: the integer alone
    if bare:
        return int(bare.group(1))
    hits = re.findall(r"(?<![\w.])([1-5])(?![\w.])", t)
    return int(hits[-1]) if hits else None


def cmd_judge(args):
    llm = load_llm(ARGS_JUDGE, tp=ARGS_JUDGE_TP)
    from vllm import SamplingParams
    personas = {p["id"]: p for p in load_personas()}
    matrix = json.loads(out_name("exp_matrix").read_text())
    sp = SamplingParams(temperature=0.0, max_tokens=16)
    scores_out = []
    for prec in matrix:
        p = personas[prec["persona"]]
        prompts, keys = [], []
        tok = llm.get_tokenizer()
        for name in prec["strategies"]:
            for i, cand in enumerate(prec["strategies"][name]["candidates"]):
                recs = try_parse(cand["text"])
                blob = json.dumps(recs) if recs else cand["text"][:1500]
                prompts.append(render_chat(tok, [
                    {"role": "user", "content": JUDGE_SCORE.format(
                        paragraph=p["paragraph"], candidate=blob)}]))
                keys.append((p["id"], name, i))
        outs = llm.generate(prompts, sp)
        for (pid, name, i), o in zip(keys, outs):
            raw = o.outputs[0].text
            scores_out.append({"persona": pid, "strategy": name,
                               "cand_idx": i, "judge_raw": raw.strip(),
                               "score": parse_judge_score(raw)})
        print(f"  judged {p['id']}", flush=True)
    path = out_name("exp_judge_scores")
    path.write_text(json.dumps(scores_out, indent=1))
    print("wrote", path)


# ------------------------------------------------------------------ analyze

def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


def cmd_analyze(args):
    matrix = json.loads(out_name("exp_matrix").read_text())
    names = list(matrix[0]["strategies"])
    print(f"{'strategy':>12} {'valid':>10} {'hard_pass':>10} "
          f"{'wall':>7} {'tok/s':>7} {'cov':>6} {'jaccard':>8} "
          f"{'cost/valid':>10} {'degen':>7} {'cap':>6}")
    for name in names:
        tot = val = hard = degen = cap = 0
        wall = toks = cov = jac = 0.0
        n_p = 0
        for prec in matrix:
            s = prec["strategies"][name]
            n_p += 1
            val += sum(c["valid"] for c in s["candidates"])
            hard += sum(c.get("hard_pass", False) for c in s["candidates"])
            degen += sum(c.get("degenerate", False) for c in s["candidates"])
            cap += sum(c.get("hit_cap", False) for c in s["candidates"])
            tot += len(s["candidates"])
            wall += s["wall"]
            toks += s["total_tokens"]
            cov += sum(c.get("coverage", 0) for c in s["candidates"])
            jac += s["jaccard"]
        print(f"{name:>12} {val:>4}/{tot:<5} {hard:>4}/{tot:<5} "
              f"{wall / n_p:>6.1f}s {toks / wall:>6.1f} "
              f"{cov / tot:>6.2f} {jac / n_p:>8.2f} "
              f"{wall / max(val, 1):>9.1f}s "
              f"{degen / tot:>7.2f} {cap / tot:>6.2f}")

    # finish_reason breakdown per strategy
    print("\nfinish_reason / trailing-chars breakdown:")
    for name in names:
        fins, trail, n = {}, 0, 0
        for prec in matrix:
            for c in prec["strategies"][name]["candidates"]:
                fins[c.get("finish")] = fins.get(c.get("finish"), 0) + 1
                trail += c.get("trailing_chars", 0)
                n += 1
        order = ", ".join(f"{k}={v}" for k, v in sorted(
            fins.items(), key=lambda kv: -kv[1]))
        print(f"  {name:>12} {order}   mean_trailing={trail / max(n, 1):.0f}")

    # per-strategy violation rates (per valid candidate)
    print("\nviolations per valid candidate:")
    for name in names:
        agg = {"when_out_of_window": 0, "when_not_iso": 0,
               "dup_items": 0, "missing_verbs": 0, "foreign_targets": 0}
        nv = 0
        for prec in matrix:
            for c in prec["strategies"][name]["candidates"]:
                if not c.get("valid"):
                    continue
                nv += 1
                for k, v in c["violations"].items():
                    agg[k] += len(v) if isinstance(v, list) else v
        short = {"when_out_of_window": "date_out", "when_not_iso": "not_iso",
                 "dup_items": "dup", "missing_verbs": "miss_verb",
                 "foreign_targets": "foreign"}
        print(f"  {name:>12} " + "  ".join(
            f"{short.get(k, k)}={agg[k] / max(nv, 1):.2f}"
            for k in agg) + f"   (n={nv})")

    try:
        scores = json.loads(out_name("exp_judge_scores").read_text())
        sc = {(s["persona"], s["strategy"], s["cand_idx"]): s["score"]
              for s in scores}
        print("\njudge means (1-5):")
        for name in names:
            vals = [sc[(p["persona"], name, i)]
                    for p in matrix
                    for i in range(len(p["strategies"][name]["candidates"]))
                    if sc.get((p["persona"], name, i))]
            if vals:
                print(f"  {name:>12} mean={sum(vals) / len(vals):.2f} "
                      f"(n={len(vals)})")
        print("\ncalibration Spearman(seq_lp, judge):")
        for name in names:
            if name == "greedy" or name == "greedy_g":
                continue
            a, b = [], []
            for prec in matrix:
                for i, c in enumerate(
                        prec["strategies"][name]["candidates"]):
                    s = sc.get((prec["persona"], name, i))
                    if s:
                        a.append(c["seq_lp"])
                        b.append(s)
            if len(a) > 2:
                print(f"  {name:>12} pooled rho={spearman(a, b):+.2f}")
    except FileNotFoundError:
        print("\n(no judge scores yet — run judge phase)")


def main():
    global ARGS_MODEL, ARGS_JUDGE, ARGS_TP, ARGS_JUDGE_TP
    global ARGS_NO_EXAMPLE, ARGS_STRATEGIES, ARGS_LIMIT_PERSONAS
    global MAX_NEW, BEAM_MIN_TOKENS, ARGS_OUT_SUFFIX, ARGS_SO_BACKEND
    global ARGS_MAX_NUM_SEQS
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["matrix", "width", "determinism",
                                      "mechanism", "judge", "analyze"])
    ap.add_argument("--model", default=ARGS_MODEL)
    ap.add_argument("--judge-model", default=ARGS_JUDGE)
    ap.add_argument("--tp", type=int, default=ARGS_TP)
    ap.add_argument("--judge-tp", type=int, default=ARGS_JUDGE_TP)
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--no-example", action="store_true")
    ap.add_argument("--limit-personas", type=int, default=0,
                    help="debug: only run the first N personas (0 = all)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--max-num-seqs", type=int, default=0,
                    help="cap concurrent seqs (needed for hybrid Mamba judge)")
    ap.add_argument("--so-backend", default="",
                    help="explicit structured-output backend (xgrammar|guidance); "
                         "required for guided beam arms")
    ap.add_argument("--out-suffix", default="",
                    help="suffix result filenames (keeps extra arms separate)")
    ap.add_argument("--beam-min-tokens", type=int, default=BEAM_MIN_TOKENS)
    args = ap.parse_args()
    ARGS_MODEL, ARGS_JUDGE, ARGS_TP = args.model, args.judge_model, args.tp
    ARGS_JUDGE_TP = args.judge_tp
    ARGS_NO_EXAMPLE, ARGS_STRATEGIES = args.no_example, args.strategies
    ARGS_LIMIT_PERSONAS = args.limit_personas
    MAX_NEW, BEAM_MIN_TOKENS = args.max_new, args.beam_min_tokens
    ARGS_OUT_SUFFIX = args.out_suffix
    ARGS_SO_BACKEND = args.so_backend
    ARGS_MAX_NUM_SEQS = args.max_num_seqs
    RESULTS.mkdir(exist_ok=True)
    save_meta(args.phase)
    {"matrix": cmd_matrix, "width": cmd_width, "determinism": cmd_determinism,
     "mechanism": cmd_mechanism, "judge": cmd_judge,
     "analyze": cmd_analyze}[args.phase](args)


if __name__ == "__main__":
    main()
