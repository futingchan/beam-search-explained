"""Fair-comparison experiment: does beam search add value to a structured
recommendation endpoint once the baselines are honest?

Every strategy returns k=4 candidates in ONE call (except greedy, the
single-answer reference). Each runs unguided and with JSON-schema
constrained decoding (lm-format-enforcer prefix fn — the HF-side
equivalent of vLLM's guided_json).

    greedy        n=1, do_sample=False
    samp_t07      n=4 sampled in one call, temp=0.7
    samp_t09      n=4 sampled in one call, temp=0.9
    beam_lp10     num_beams=4, length_penalty=1.0, min_new_tokens=150
    beam_lp12     num_beams=4, length_penalty=1.2, min_new_tokens=150
    *_g           same + LMFE schema constraint

Phases (run in order):
    python src/experiment.py matrix        # strategy × persona → raw cands
    python src/experiment.py width         # beam k in {1,2,4,8}
    python src/experiment.py determinism   # 5 reps seeded vs unseeded
    python src/experiment.py judge         # 3B judge: relevance + pairwise
    python src/experiment.py analyze       # aggregate → report tables

Outputs land in results/exp_*.json.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PERSONAS_PATH = ROOT / "data" / "personas.json"
RESULTS = ROOT / "results"
GEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
JUDGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW = 400
SEED = 1234

# Ablation flags (set from CLI in main()): run a subset of strategies, swap
# the generator model, or strip the few-shot JSON example from the prompt.
ARGS_MODEL = GEN_MODEL
ARGS_STRATEGIES: list[str] | None = None
ARGS_NO_EXAMPLE = False
ARGS_LOAD_4BIT = False

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

STRATEGIES = {
    "greedy":     {"n": 1, "gen": {}},
    "greedy_g":   {"n": 1, "gen": {}, "guided": True},
    "samp_t07":   {"n": 4, "gen": {"do_sample": True, "temperature": 0.7,
                                   "top_p": 0.95}},
    "samp_t07_g": {"n": 4, "gen": {"do_sample": True, "temperature": 0.7,
                                   "top_p": 0.95}, "guided": True},
    "samp_t09":   {"n": 4, "gen": {"do_sample": True, "temperature": 0.9,
                                   "top_p": 0.95}},
    "samp_t09_g": {"n": 4, "gen": {"do_sample": True, "temperature": 0.9,
                                   "top_p": 0.95}, "guided": True},
    "beam_lp10":  {"n": 4, "gen": {"num_beams": 4, "early_stopping": True,
                                   "length_penalty": 1.0,
                                   "min_new_tokens": 150}},
    "beam_lp10_g": {"n": 4, "gen": {"num_beams": 4, "early_stopping": True,
                                    "length_penalty": 1.0,
                                    "min_new_tokens": 150}, "guided": True},
    "beam_lp12":  {"n": 4, "gen": {"num_beams": 4, "early_stopping": True,
                                   "length_penalty": 1.2,
                                   "min_new_tokens": 150}},
    "beam_lp12_g": {"n": 4, "gen": {"num_beams": 4, "early_stopping": True,
                                    "length_penalty": 1.2,
                                    "min_new_tokens": 150}, "guided": True},
}

# ---------------------------------------------------------------- prompts

from demo_hf import SYSTEM, USER_TEMPLATE, VALID_ACTIONS, try_parse  # noqa: E402

# Same SYSTEM prompt minus the concrete JSON example — schema conveyed by
# field description only. Ablation for the "mode is the prompt" finding:
# does beam still parrot when there is nothing to parrot?
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
- "Book": bookable things. Targets like "Hotel in Alfama", "Flight to
  Lisbon", "Uber to airport", "concert tickets", "food tour experience".
- "Visit": places to plan a visit to. Targets like "Gothic Quarter",
  "Boqueria market".
- "Ask_About": topics to research before committing. Targets like
  "stroller access at Sintra", "cider house reservations".

Rules:
- Exactly 8 recommendations.
- Match the stated pace: packed trips skew to Book/Visit anchors, slow trips
  include Ask_About research items.
- Respect hard constraints in the paragraph (diet, mobility, budget, kids).
- Targets must be real, well-known things or generic categories
  (e.g. "family-friendly tapas bar"), never invented proper nouns.
"""


def build_prompt(tokenizer, paragraph: str) -> str:
    messages = [
        {"role": "system",
         "content": SYSTEM_NO_EXAMPLE if ARGS_NO_EXAMPLE else SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(paragraph=paragraph)},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def make_prefix_fn(tokenizer):
    """LMFE schema constraint; shims transformers v5 import breakage."""
    import transformers.tokenization_utils as tu
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(tu, "PreTrainedTokenizerBase"):
        tu.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )
    return build_transformers_prefix_allowed_tokens_fn(
        tokenizer, JsonSchemaParser(SCHEMA))


def load_gen_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ARGS_MODEL)
    kw = {"dtype": torch.float16}
    if ARGS_LOAD_4BIT:
        # big model: 4-bit so it fits the 11 GB Pascal card
        from transformers import BitsAndBytesConfig
        kw = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16)}
    model = AutoModelForCausalLM.from_pretrained(ARGS_MODEL, **kw).to("cuda").eval()
    return model, tok


def gen_candidates(model, tok, prompt, cfg, seed=None):
    """One generate() call → (texts, wall_s, per-seq token counts)."""
    import torch
    kw = dict(max_new_tokens=MAX_NEW, **cfg["gen"])
    kw.setdefault("do_sample", False)
    if cfg.get("guided"):
        kw["prefix_allowed_tokens_fn"] = make_prefix_fn(tok)
    if seed is not None:
        torch.manual_seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    n_in = inputs["input_ids"].shape[1]
    t0 = time.monotonic()
    out = model.generate(**inputs, num_return_sequences=cfg["n"], **kw)
    dt = time.monotonic() - t0
    texts = [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]
    ns = [int(o.shape[0] - n_in) for o in out]
    return texts, dt, ns


def seq_logprobs(model, tok, prompt, texts):
    """Teacher-forced sum logprob of each candidate. Uniform across
    strategies — lets us re-rank ANY candidate set (beam's cum_logprob is
    the same quantity, modulo length normalization)."""
    import torch
    n_in = tok(prompt, return_tensors="pt")["input_ids"].shape[1]
    lps = []
    for t in texts:
        ids = tok(prompt + t, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**ids).logits[0]
        comp = logits[n_in - 1:-1]            # positions predicting completion
        tgt = ids["input_ids"][0, n_in:]
        lp = torch.log_softmax(comp.float(), -1).gather(
            1, tgt.unsqueeze(1)).sum().item()
        lps.append(lp / max(len(tgt), 1))     # per-token mean: length-free
    return lps


# ------------------------------------------------------- deterministic checks

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
    """Deterministic hard checks. Returns violations dict + pass bool."""
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

    v["missing_verbs"] = [v_ for v_ in persona["implied_verbs"]
                          if v_ not in verbs]
    hard_pass = (v["when_out_of_window"] == 0 and v["when_not_iso"] == 0
                 and v["dup_items"] == 0 and not v["missing_verbs"])
    return {"violations": v, "hard_pass": hard_pass,
            "n_recs": len(recs)}


def coverage(persona: dict, recs: list[dict]) -> float:
    blob = " ".join(f"{r.get('target','')} {r.get('rationale','')}"
                    for r in recs).lower()
    hit = sum(any(m in blob for m in it["match"])
              for it in persona["interests"])
    return hit / len(persona["interests"])


def jaccard_diversity(cands: list[list[dict]]) -> float:
    """Mean pairwise Jaccard over (action, target) sets — lower = more
    diverse. 0 = disjoint, 1 = identical."""
    sets = [{(r.get("action"), norm_target(r.get("target", "")))
             for r in c} for c in cands]
    pairs = [len(a & b) / len(a | b) if a | b else 1.0
             for a, b in itertools.combinations(sets, 2)]
    return sum(pairs) / len(pairs) if pairs else 0.0


def first_verbs(recs: list[dict]) -> list[str]:
    return [r.get("action", "?") for r in recs]


def score_candidate(persona: dict, text: str) -> dict:
    recs = try_parse(text)
    if recs is None:
        return {"valid": False}
    out = {"valid": True, **check_candidate(persona, recs),
           "coverage": round(coverage(persona, recs), 3),
           "verbs": sorted({r.get("action") for r in recs
                            if r.get("action")})}
    return out


# ------------------------------------------------------------------ phases

def out_name(base: str) -> Path:
    suffix = Path(ARGS_MODEL.split("/")[-1].replace("Qwen", "qwen"))
    tag = f"{base}_{suffix}"
    if ARGS_NO_EXAMPLE:
        tag += "_noex"
    return RESULTS / f"{tag}.json"


def cmd_matrix(args):
    model, tok = load_gen_model()
    personas = json.loads(PERSONAS_PATH.read_text())
    strategies = (ARGS_STRATEGIES if ARGS_STRATEGIES
                  else list(STRATEGIES))
    out = []
    for persona in personas:
        prompt = build_prompt(tok, persona["paragraph"])
        prec = {"persona": persona["id"], "strategies": {}}
        for name in strategies:
            cfg = STRATEGIES[name]
            texts, dt, ns = gen_candidates(model, tok, prompt, cfg)
            lps = seq_logprobs(model, tok, prompt, texts)
            cands = []
            for t, n, lp in zip(texts, ns, lps):
                sc = score_candidate(persona, t)
                cands.append({"text": t, "tokens": n, "seq_lp": round(lp, 4),
                              **sc})
            prec["strategies"][name] = {
                "wall": round(dt, 2), "total_tokens": sum(ns),
                "jaccard": round(jaccard_diversity(
                    [try_parse(t) or [] for t in texts]), 3),
                "candidates": cands,
            }
            nv = sum(c["valid"] for c in cands)
            print(f"  {persona['id']:>22} {name:>10}: {dt:5.1f}s "
                  f"{nv}/{cfg['n']} valid", flush=True)
        out.append(prec)
    path = out_name("exp_matrix")
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


def cmd_width(args):
    model, tok = load_gen_model()
    personas = json.loads(PERSONAS_PATH.read_text())
    out = []
    for persona in personas:
        prompt = build_prompt(tok, persona["paragraph"])
        for k in (1, 2, 4, 8):
            cfg = {"n": k, "gen": {"num_beams": k, "early_stopping": True,
                                    "min_new_tokens": 150}}
            texts, dt, ns = gen_candidates(model, tok, prompt, cfg)
            lps = seq_logprobs(model, tok, prompt, texts)
            cands = [{"tokens": n, "seq_lp": round(lp, 4),
                      **score_candidate(persona, t)}
                     for t, n, lp in zip(texts, ns, lps)]
            out.append({"persona": persona["id"], "k": k,
                        "wall": round(dt, 2), "total_tokens": sum(ns),
                        "jaccard": round(jaccard_diversity(
                            [try_parse(t) or [] for t in texts]), 3),
                        "top1": cands[0],
                        "candidates": cands})
            print(f"  {persona['id']:>22} k={k}: {dt:5.1f}s "
                  f"top1_valid={cands[0]['valid']}", flush=True)
    path = out_name("exp_width")
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)


def cmd_determinism(args):
    model, tok = load_gen_model()
    personas = json.loads(PERSONAS_PATH.read_text())[:3]
    out = []
    for persona in personas:
        prompt = build_prompt(tok, persona["paragraph"])
        for name in ("samp_t09", "beam_lp10"):
            cfg = STRATEGIES[name]
            for mode in ("seeded", "unseeded"):
                hashes = []
                for rep in range(5):
                    texts, dt, _ = gen_candidates(
                        model, tok, prompt, cfg,
                        seed=SEED if mode == "seeded" else None)
                    h = hashlib.sha256(
                        "|".join(texts).encode()).hexdigest()[:12]
                    hashes.append(h)
                uniq = len(set(hashes))
                out.append({"persona": persona["id"], "strategy": name,
                            "mode": mode, "unique_outputs": uniq,
                            "of": 5})
                print(f"  {persona['id']:>22} {name:>9} {mode:>8}: "
                      f"{uniq}/5 unique", flush=True)
    path = out_name("exp_determinism")
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

JUDGE_PAIR = """A traveler wrote this trip description:

<trip>{paragraph}</trip>

Two recommendation lists were generated (JSON):

<list_a>{a}</list_a>

<list_b>{b}</list_b>

Which list better serves THIS traveler — relevance to stated interests,
respect for constraints, appropriate pace? Reply with exactly A, B, or TIE."""


def judge_chat(model, tok, text, temp=0.0):
    import torch
    msgs = [{"role": "user", "content": text}]
    prompt = tok.apply_chat_template(msgs, tokenize=False,
                                     add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    n_in = ids["input_ids"].shape[1]
    kw = ({"do_sample": True, "temperature": temp, "top_p": 0.95}
          if temp > 0 else {"do_sample": False})
    out = model.generate(**ids, max_new_tokens=16, **kw)
    return tok.decode(out[0][n_in:], skip_special_tokens=True).strip()


def parse_score(s: str):
    m = re.search(r"[1-5]", s)
    return int(m.group()) if m else None


def cmd_judge(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(JUDGE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL, dtype=torch.float16).to("cuda").eval()

    personas = {p["id"]: p for p in
                json.loads(PERSONAS_PATH.read_text())}
    matrix = json.loads(out_name("exp_matrix").read_text())

    scores_out, pairs_out = [], []
    for prec in matrix:
        p = personas[prec["persona"]]
        para = p["paragraph"]
        st = prec["strategies"]

        # relevance scores: every candidate of greedy, samp_t09, beam_lp10,
        # plus the guided top-1s, when those strategies ran
        to_score = []
        for name, take in (("greedy", 4), ("samp_t09", 4), ("beam_lp10", 4),
                           ("samp_t09_g", 1), ("beam_lp10_g", 1)):
            if name in st:
                to_score += [(name, i, c) for i, c in
                             enumerate(st[name]["candidates"][:take])]
        for name, idx, cand in to_score:
            recs = try_parse(cand["text"])
            blob = json.dumps(recs) if recs else cand["text"][:1500]
            s = parse_score(judge_chat(
                model, tok, JUDGE_SCORE.format(paragraph=para,
                                               candidate=blob)))
            scores_out.append({"persona": p["id"], "strategy": name,
                               "cand_idx": idx, "score": s})
            print(f"  score {p['id']:>22} {name:>10}#{idx}: {s}",
                  flush=True)

        # pairwise: beam-top1 vs greedy, beam-top1 vs best samp_t09.
        # best-sample = highest teacher-forced seq logprob among valid ones.
        if not all(k in st for k in ("beam_lp10", "greedy", "samp_t09")):
            continue
        beam1 = st["beam_lp10"]["candidates"][0]
        greedy = st["greedy"]["candidates"][0]
        samps = ([c for c in st["samp_t09"]["candidates"] if c["valid"]]
                 or st["samp_t09"]["candidates"])
        best_samp = max(samps, key=lambda c: c["seq_lp"])
        for opp_name, opp in (("greedy", greedy), ("best_sample", best_samp)):
            wins = {"beam": 0, "other": 0, "tie": 0}
            for rep in range(3):
                # blind: randomize A/B assignment each rep
                order = [("beam", beam1), ("other", opp)]
                random.shuffle(order)
                a_text = json.dumps(try_parse(order[0][1]["text"]) or
                                    order[0][1]["text"][:1500])
                b_text = json.dumps(try_parse(order[1][1]["text"]) or
                                    order[1][1]["text"][:1500])
                ans = judge_chat(
                    model, tok,
                    JUDGE_PAIR.format(paragraph=para, a=a_text, b=b_text),
                    temp=0.6)
                verdict = (order[0][0] if "A" in ans.upper()[:1]
                           else order[1][0] if "B" in ans.upper()[:1]
                           else "tie")
                wins[verdict if verdict in wins else "tie"] += 1
            pairs_out.append({"persona": p["id"],
                              "pair": f"beam_vs_{opp_name}",
                              **wins})
            print(f"  pair {p['id']:>22} beam_vs_{opp_name}: {wins}",
                  flush=True)

    out_name("exp_judge_scores").write_text(
        json.dumps(scores_out, indent=1))
    out_name("exp_judge_pairs").write_text(
        json.dumps(pairs_out, indent=1))
    print("wrote", out_name("exp_judge_scores"), "and",
          out_name("exp_judge_pairs"))


# ------------------------------------------------------------------ analyze

def ranks(xs):
    """Average ranks (1=lowest) with tie handling — for Spearman."""
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
          f"{'wall':>7} {'tok/call':>9} {'tok/s':>7} {'cov':>6} "
          f"{'jaccard':>8} {'cost/valid':>10}")
    for name in names:
        tot = val = hard = 0
        wall = toks = cov = jac = 0.0
        n_p = 0
        for prec in matrix:
            s = prec["strategies"][name]
            n_p += 1
            val += sum(c["valid"] for c in s["candidates"])
            hard += sum(c.get("hard_pass", False) for c in s["candidates"])
            tot += len(s["candidates"])
            wall += s["wall"]
            toks += s["total_tokens"]
            cov += sum(c.get("coverage", 0) for c in s["candidates"])
            jac += s["jaccard"]
        mw = wall / n_p
        print(f"{name:>12} {val:>4}/{tot:<5} {hard:>4}/{tot:<5} "
              f"{mw:>6.1f}s {toks / n_p:>8.0f} {toks / wall:>6.1f} "
              f"{cov / tot:>6.2f} {jac / n_p:>8.2f} "
              f"{wall / max(val,1):>9.1f}s")

    # first-verb divergence: beam top1 vs greedy
    print("\nfirst-verb / verb-set divergence, beam top-1 vs greedy:")
    for prec in matrix:
        if "beam_lp10" not in prec["strategies"] or \
                "greedy" not in prec["strategies"]:
            continue
        b = try_parse(prec["strategies"]["beam_lp10"]["candidates"][0]["text"]) or []
        g = try_parse(prec["strategies"]["greedy"]["candidates"][0]["text"]) or []
        bv, gv = first_verbs(b), first_verbs(g)
        diff = (bv[:1] != gv[:1], set(bv) != set(gv),
                "Book" in set(bv) and "Book" not in set(gv))
        print(f"  {prec['persona']:>22} beam1={bv[:1]} greedy={gv[:1]} "
              f"first_diff={diff[0]} set_diff={diff[1]} "
              f"beam_recovered_Book={diff[2]}")

    # calibration: Spearman(seq_lp rank, judge score) within candidate sets
    try:
        scores = json.loads(out_name("exp_judge_scores").read_text())
        sc = {(s["persona"], s["strategy"], s["cand_idx"]): s["score"]
              for s in scores}
        print("\ncalibration Spearman(seq_lp, judge):")
        for name in ("beam_lp10", "samp_t09"):
            rho_p, pooled_a, pooled_b = [], [], []
            for prec in matrix:
                cands = prec["strategies"][name]["candidates"]
                lps = [c["seq_lp"] for c in cands]
                js = [sc.get((prec["persona"], name, i)) or 0
                      for i in range(len(cands))]
                rho_p.append(spearman(lps, js))
                pooled_a += lps
                pooled_b += js
            per_p = " ".join(f"{r:+.2f}" for r in rho_p)
            print(f"  {name:>10} per-persona: {per_p}  "
                  f"pooled: {spearman(pooled_a, pooled_b):+.2f}")
    except FileNotFoundError:
        print("\njudge results not found yet")


def main():
    global ARGS_MODEL, ARGS_STRATEGIES, ARGS_NO_EXAMPLE, ARGS_LOAD_4BIT
    ap = argparse.ArgumentParser()
    ap.add_argument("phase",
                    choices=["matrix", "width", "determinism",
                             "judge", "analyze"])
    ap.add_argument("--model", default=GEN_MODEL,
                    help="generator model")
    ap.add_argument("--load-4bit", action="store_true",
                    help="load generator with bitsandbytes nf4 "
                         "(for >3B models on the 11 GB card)")
    ap.add_argument("--strategies", nargs="*", default=None,
                    help="subset of STRATEGIES to run")
    ap.add_argument("--no-example", action="store_true",
                    help="ablation: strip the JSON exemplar from SYSTEM")
    ap.add_argument("--max-new", type=int, default=None,
                    help="override MAX_NEW (bigger models pretty-print "
                         "JSON and need a larger budget)")
    args = ap.parse_args()
    ARGS_MODEL = args.model
    ARGS_STRATEGIES = args.strategies
    ARGS_NO_EXAMPLE = args.no_example
    ARGS_LOAD_4BIT = args.load_4bit
    if args.max_new:
        global MAX_NEW
        MAX_NEW = args.max_new
    RESULTS.mkdir(exist_ok=True)
    {"matrix": cmd_matrix, "width": cmd_width,
     "determinism": cmd_determinism, "judge": cmd_judge,
     "analyze": cmd_analyze}[args.phase](args)


if __name__ == "__main__":
    main()
