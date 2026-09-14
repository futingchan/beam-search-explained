"""Canonical-answer benchmark: tasks where beam search SHOULD win.

The recs workload tested beam where the deliverable is a diverse set.
These tasks test it where the deliverable is the single most-likely
correct answer — translation, slot extraction, NL→SQL — against gold
references with objective metrics (chrF, field exact-match, execution
match). No judge needed.

    greedy      n=1 argmax
    beam_lp10   num_beams=4, length_penalty=1.0
    samp_t09    n=4 sampled in one call, temp=0.9

Metrics per strategy:
    top1    score of candidate 1 (beam's own ranking for beam)
    oracle  best score among the k candidates — does the right answer
            exist in the set even if ranking misses it
    wall    seconds per call

    python src/experiment_canonical.py translation --model Qwen/Qwen2.5-3B-Instruct
    python src/experiment_canonical.py translation --model Helsinki-NLP/opus-mt-en-de
    python src/experiment_canonical.py extraction  --model Qwen/Qwen2.5-3B-Instruct
    python src/experiment_canonical.py nl2sql      --model Qwen/Qwen2.5-3B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

STRATEGIES = {
    "greedy":    {"n": 1, "gen": {"do_sample": False}},
    "samp_t09":  {"n": 4, "gen": {"do_sample": True, "temperature": 0.9,
                                  "top_p": 0.95}},
    "beam_lp10": {"n": 4, "gen": {"num_beams": 4, "early_stopping": True,
                                  "length_penalty": 1.0}},
}

MAX_NEW = {"translation": 128, "extraction": 220, "nl2sql": 160}


# ---------------- scoring ----------------

def clean_jsonish(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith(("json", "sql")):
            text = text[4:]
    return text.strip()


def score_translation(cand: str, item) -> float:
    import sacrebleu
    hyp = cand.strip().split("\n")[0].strip()
    return sacrebleu.sentence_chrf(hyp, [item["ref"]]).score


def score_extraction(cand: str, item) -> float:
    try:
        pred = json.loads(clean_jsonish(cand))
        if isinstance(pred, list):
            pred = pred[0] if pred else {}
    except Exception:
        return 0.0
    gold = item["gold"]
    ok = 0
    for k, gv in gold.items():
        pv = str(pred.get(k, "")).strip().lower()
        gv_ = str(gv).strip().lower()
        if k == "amount":
            try:
                ok += abs(float(pv) - float(gv_)) < 0.01
                continue
            except ValueError:
                pass
        ok += pv == gv_
    return ok / len(gold)


_SQL_DB = None

def _sql_db():
    global _SQL_DB
    if _SQL_DB is None:
        _SQL_DB = sqlite3.connect(":memory:")
        spec = json.load(open(ROOT / "data" / "nl2sql.json"))
        _SQL_DB.executescript(spec["schema"])
        for s in spec["seed"]:
            _SQL_DB.execute(s)
        _SQL_DB.commit()
    return _SQL_DB


def _exec_sql(sql: str):
    try:
        rows = _sql_db().execute(sql).fetchall()
        return sorted(tuple(r) for r in rows)
    except Exception:
        return None


def score_nl2sql(cand: str, item) -> float:
    sql = clean_jsonish(cand)
    if ";" in sql:
        sql = sql[: sql.index(";") + 1]
    got = _exec_sql(sql)
    want = _exec_sql(item["sql"])
    if got is None or want is None:
        return 0.0
    return float(got == want)


SCORERS = {"translation": score_translation,
           "extraction": score_extraction,
           "nl2sql": score_nl2sql}


# ---------------- prompts ----------------

def build_prompt(task, item, tok, seq2seq):
    if task == "translation":
        if seq2seq:
            return item["src"]
        msgs = [
            {"role": "system", "content":
             "You are a professional English-to-German translator. "
             "Output only the German translation."},
            {"role": "user", "content": item["src"]},
        ]
    elif task == "extraction":
        msgs = [
            {"role": "system", "content":
             "You extract structured data from emails. Output only valid "
             "JSON — no prose, no code fences."},
            {"role": "user", "content":
             "Extract these fields as JSON: vendor, date (ISO "
             "YYYY-MM-DD), amount (decimal string), currency (ISO code "
             "USD/EUR/GBP), confirmation_code.\n\nEmail:\n"
             + item["input"]},
        ]
    elif task == "nl2sql":
        spec = json.load(open(ROOT / "data" / "nl2sql.json"))
        msgs = [
            {"role": "system", "content":
             "You write SQLite queries. Output only the SQL query — no "
             "prose, no code fences."},
            {"role": "user", "content":
             "Schema:\n" + spec["schema"] +
             "\n\nQuestion: " + item["question"]},
        ]
    return tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)


# ---------------- runner ----------------

def load_model(model_name, seq2seq):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if "opus-mt" in model_name:
        # Marian is numerically unstable in fp16 — keep it fp32 (it's tiny)
        from transformers import MarianMTModel, MarianTokenizer
        tok = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(dev).eval()
        return model, tok
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16).to(dev).eval()
    return model, tok


def run(task, model_name, out_path):
    spec = json.load(open(ROOT / "data" / (
        "translation_en_de.json" if task == "translation" else
        task + ".json")))
    items = spec if isinstance(spec, list) else spec["items"]
    seq2seq = "opus-mt" in model_name or "t5" in model_name.lower()
    model, tok = load_model(model_name, seq2seq)
    scorer = SCORERS[task]
    max_new = MAX_NEW[task]

    rows = []
    for i, item in enumerate(items):
        prompt = build_prompt(task, item, tok, seq2seq)
        for name, cfg in STRATEGIES.items():
            kw = dict(max_new_tokens=max_new, **cfg["gen"])
            kw.setdefault("do_sample", False)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            n_in = inputs["input_ids"].shape[1]
            t0 = time.monotonic()
            out = model.generate(**inputs,
                                 num_return_sequences=cfg["n"], **kw)
            dt = time.monotonic() - t0
            # seq2seq generate() returns decoder tokens only — no prompt
            # prefix to slice off
            texts = [tok.decode(o if seq2seq else o[n_in:],
                                skip_special_tokens=True) for o in out]
            scores = [scorer(t, item) for t in texts]
            rows.append({"item": i, "strategy": name,
                         "wall": round(dt, 3), "texts": texts,
                         "scores": [round(s, 3) for s in scores],
                         "top1": round(scores[0], 3),
                         "oracle": round(max(scores), 3)})
        print(f"  item {i + 1}/{len(items)}", end="\r", flush=True)

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"task": task, "model": model_name,
                                    "rows": rows}, ensure_ascii=False))
    print("\nwrote", out_path)


def analyze(path):
    d = json.load(open(path))
    from collections import defaultdict
    agg = defaultdict(lambda: {"top1": [], "oracle": [], "wall": []})
    for r in d["rows"]:
        a = agg[r["strategy"]]
        a["top1"].append(r["top1"])
        a["oracle"].append(r["oracle"])
        a["wall"].append(r["wall"])
    print(f"\n{d['task']} — {d['model']}")
    print(f"{'strategy':<12} {'top1':>6} {'oracle':>6} {'wall/call':>9}")
    for name, a in agg.items():
        m = lambda xs: sum(xs) / len(xs)
        print(f"{name:<12} {m(a['top1']):>6.2f} {m(a['oracle']):>6.2f} "
              f"{m(a['wall']):>8.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run",
                    choices=["run", "analyze"])
    ap.add_argument("task", nargs="?",
                    choices=["translation", "extraction", "nl2sql"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--file", help="analyze: path to results json")
    args = ap.parse_args()
    if args.cmd == "analyze":
        analyze(Path(args.file))
    else:
        tag = args.model.split("/")[-1].lower()
        run(args.task, args.model,
            RESULTS / f"canonical_{args.task}_{tag}.json")
