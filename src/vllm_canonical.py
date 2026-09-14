"""Canonical-answer control benchmark on vLLM — the §5 tasks at 35B.

Same tasks/metrics as the local HF harness (experiment_canonical.py):
translation (chrF vs Tatoeba refs), extraction (field exact-match),
NL→SQL (execution match on sqlite). The deliverable here is ONE correct
answer — the task family beam was built for.

    python src/vllm_canonical.py translation --model Qwen/Qwen3.6-35B-A3B-FP8
    python src/vllm_canonical.py extraction --model ...
    python src/vllm_canonical.py nl2sql      --model ...

Needs `sacrebleu` in the venv for translation scoring (other tasks are
stdlib-only). Output: results/canonical_<task>_<model>.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm_experiment import beam_call, encode_prompt, render_chat  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

MAX_NEW = {"translation": 128, "extraction": 220, "nl2sql": 160}
TASK_FILE = {"translation": "translation_en_de.json",
             "extraction": "extraction.json", "nl2sql": "nl2sql.json"}


def clean_jsonish(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith(("json", "sql")):
            text = text[4:]
    return text.strip()


def score_translation(cand, item):
    import sacrebleu
    hyp = cand.strip().split("\n")[0].strip()
    return sacrebleu.sentence_chrf(hyp, [item["ref"]]).score


def score_extraction(cand, item):
    try:
        pred = json.loads(clean_jsonish(cand))
        if isinstance(pred, list):
            pred = pred[0] if pred else {}
    except Exception:
        return 0.0
    gold = item["gold"]
    ok = 0
    for k, gv in gold.items():
        pv, gv_ = str(pred.get(k, "")).strip().lower(), str(gv).strip().lower()
        if k == "amount":
            try:
                ok += abs(float(pv) - float(gv_)) < 0.01
                continue
            except ValueError:
                pass
        ok += pv == gv_
    return ok / len(gold)


_DB = None

def _db():
    global _DB
    if _DB is None:
        spec = json.load(open(ROOT / "data" / "nl2sql.json"))
        _DB = sqlite3.connect(":memory:")
        _DB.executescript(spec["schema"])
        for s in spec["seed"]:
            _DB.execute(s)
        _DB.commit()
    return _DB


def _exec(sql):
    try:
        return sorted(tuple(r) for r in _db().execute(sql).fetchall())
    except Exception:
        return None


def score_nl2sql(cand, item):
    sql = clean_jsonish(cand)
    if ";" in sql:
        sql = sql[: sql.index(";") + 1]
    got, want = _exec(sql), _exec(item["sql"])
    return float(got is not None and got == want)


SCORERS = {"translation": score_translation,
           "extraction": score_extraction,
           "nl2sql": score_nl2sql}


def messages(task, item, spec):
    if task == "translation":
        return [{"role": "system", "content":
                 "You are a professional English-to-German translator. "
                 "Output only the German translation."},
                {"role": "user", "content": item["src"]}]
    if task == "extraction":
        return [{"role": "system", "content":
                 "You extract structured data from emails. Output only "
                 "valid JSON — no prose, no code fences."},
                {"role": "user", "content":
                 "Extract these fields as JSON: vendor, date (ISO "
                 "YYYY-MM-DD), amount (decimal string), currency (ISO "
                 "code USD/EUR/GBP), confirmation_code.\n\nEmail:\n"
                 + item["input"]}]
    return [{"role": "system", "content":
             "You write SQLite queries. Output only the SQL query — no "
             "prose, no code fences."},
            {"role": "user", "content":
             "Schema:\n" + spec["schema"] +
             "\n\nQuestion: " + item["question"]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=list(TASK_FILE))
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=4)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    spec = json.load(open(ROOT / "data" / TASK_FILE[args.task]))
    items = spec if isinstance(spec, list) else spec["items"]
    scorer = SCORERS[args.task]
    max_new = MAX_NEW[args.task]

    llm = LLM(model=args.model, dtype="auto",
              tensor_parallel_size=args.tp,
              gpu_memory_utilization=0.90,
              enable_prefix_caching=True)
    tok = llm.get_tokenizer()

    rows = []
    for i, item in enumerate(items):
        prompt = render_chat(tok, messages(args.task, item, spec))

        # greedy
        t0 = time.monotonic()
        out = llm.generate([prompt], SamplingParams(
            temperature=0.0, max_tokens=max_new))
        texts = [out[0].outputs[0].text]
        rows.append({"item": i, "strategy": "greedy",
                     "wall": round(time.monotonic() - t0, 3),
                     "texts": texts,
                     "scores": [scorer(t, item) for t in texts]})

        # sampling n=4
        t0 = time.monotonic()
        out = llm.generate([prompt], SamplingParams(
            temperature=0.9, top_p=0.95, n=4, max_tokens=max_new))
        texts = [o.text for o in out[0].outputs]
        rows.append({"item": i, "strategy": "samp_t09",
                     "wall": round(time.monotonic() - t0, 3),
                     "texts": texts,
                     "scores": [scorer(t, item) for t in texts]})

        # beam k=4 — no min_tokens floor here: canonical outputs are short
        texts, dt, _ns, _lps, _fins, _first = beam_call(
            llm, prompt, 4, length_penalty=1.0)
        rows.append({"item": i, "strategy": "beam_lp10",
                     "wall": round(dt, 3), "texts": texts,
                     "scores": [scorer(t, item) for t in texts]})

        print(f"  item {i + 1}/{len(items)}", flush=True)

    tag = args.model.split("/")[-1].lower()
    out_path = RESULTS / f"canonical_{args.task}_{tag}.json"
    out_path.write_text(json.dumps(
        {"task": args.task, "model": args.model, "rows": rows},
        ensure_ascii=False))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
