"""Beam search with vLLM — the production path.

Two integration shapes for a recommendations backend:

A) OFFLINE ENGINE (recommended for a dedicated recommendations worker)
     llm.beam_search(prompts, BeamSearchParams(beam_width=4, max_tokens=800))
   Available since vLLM ~0.6. All beams are one batched call; with
   enable_prefix_caching=True the shared system+prefix KV is computed once
   and reused across beams AND across requests.

B) OPENAI-COMPAT SERVER (drop-in for their existing VLLMBackend, which
   already calls chat.completions). Since vLLM v0.6.3 the server accepts:

     client.chat.completions.create(
         model=..., messages=..., max_tokens=800,
         extra_body={"use_beam_search": True,
                     "n": 4,                 # n becomes beam_width
                     "length_penalty": 1.0},
     )

   Verified in vllm/entrypoints/openai/protocol.py (v0.7.0–v0.10.0):
   `use_beam_search: bool`, `to_beam_search_params()` maps `n` →
   `beam_width`. The offline path (A) is still the most explicit API.

NOTE: vLLM BeamSearchParams honours `temperature` (scales the logprobs the
search maximises) and `length_penalty`, but NOT top_p/top_k — beam search is
a max-likelihood search, not sampling. If you want *diverse* candidates, use
n>1 sampling or post-dedup beams — vLLM's beam_search returns beams ranked
by score, which are often near-duplicates.

Usage (GPU with compute capability >= 7.0 required):
    python src/vllm_beam.py --model Qwen/Qwen2.5-1.5B-Instruct
    python src/vllm_beam.py --server http://localhost:8000 --persona family_young_kids
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from demo_hf import PERSONAS_PATH, SYSTEM, USER_TEMPLATE, try_parse


def build_messages(paragraph: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TEMPLATE.format(paragraph=paragraph)},
    ]


def run_offline(model_id: str, personas: list[dict], beam_width: int,
                max_tokens: int) -> None:
    from vllm import LLM
    from vllm.sampling_params import BeamSearchParams, SamplingParams

    llm = LLM(model=model_id, enable_prefix_caching=True)
    tok = llm.get_tokenizer()

    prompts = [
        tok.apply_chat_template(
            build_messages(p["paragraph"]), tokenize=False,
            add_generation_prompt=True,
        )
        for p in personas
    ]

    print(f"\n=== vLLM beam_search, beam_width={beam_width}, "
          f"{len(prompts)} personas ===")
    t0 = time.monotonic()
    results = llm.beam_search(
        prompts,
        BeamSearchParams(beam_width=beam_width, max_tokens=max_tokens),
    )
    dt = time.monotonic() - t0
    for p, res in zip(personas, results):
        best = res.sequences[0]
        recs = try_parse(best.text)
        n = len(recs) if recs else "✗"
        print(f"  {p['id']:<28} score={best.cum_logprob:+.1f}  recs={n}")
    print(f"  total: {dt:.1f}s for {len(prompts)} requests "
          f"({dt/len(prompts):.1f}s each, shared-prefix cached)")

    # Contrast: n>1 sampling for the same candidate count
    print(f"\n=== contrast: sampling n={beam_width}, temp=0.9 ===")
    t0 = time.monotonic()
    outs = llm.generate(
        prompts,
        SamplingParams(n=beam_width, temperature=0.9, top_p=0.95,
                       max_tokens=max_tokens),
    )
    dt = time.monotonic() - t0
    for p, out in zip(personas, outs):
        ok = sum(try_parse(o.text) is not None for o in out.outputs)
        print(f"  {p['id']:<28} valid JSON {ok}/{beam_width}")
    print(f"  total: {dt:.1f}s")


def run_server(base_url: str, personas: list[dict], beam_width: int,
               max_tokens: int, model_id: str) -> None:
    """Drop-in shape for a VLLMBackend-style run() call."""
    import openai

    client = openai.OpenAI(base_url=base_url, api_key="EMPTY")
    for p in personas:
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=build_messages(p["paragraph"]),
                max_tokens=max_tokens,
                # vLLM maps request field `n` -> beam_width.
                extra_body={"use_beam_search": True, "n": beam_width},
            )
        except openai.BadRequestError as e:
            print(f"  {p['id']}: server rejected beam params — {e}")
            print("  → check server version (beam search needs vllm>=0.6.3);"
                  " or fall back to offline llm.beam_search()")
            return
        dt = time.monotonic() - t0
        recs = try_parse(resp.choices[0].message.content or "")
        print(f"  {p['id']:<28} {dt:.1f}s  recs={len(recs) if recs else '✗'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--server", default=None,
                    help="vLLM OpenAI base URL; omit for offline engine")
    ap.add_argument("--beam-width", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args()

    personas = json.loads(PERSONAS_PATH.read_text())
    if args.server:
        run_server(args.server, personas, args.beam_width,
                   args.max_tokens, args.model)
    else:
        run_offline(args.model, personas, args.beam_width, args.max_tokens)


if __name__ == "__main__":
    main()
