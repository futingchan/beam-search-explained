# Harness Changes — vLLM 0.29.0 run on 8×L4

Every change below was required to make the 8-strategy matrix actually
produce data on vLLM 0.29.0 with `Qwen/Qwen3.6-35B-A3B-FP8`. Five of the
six were **blocking**: without them the run either crashed or reported 0
valid candidates in every arm.

Environment used:

| | |
|---|---|
| host | `cdm`, 8×NVIDIA L4 24 GB, 192 vCPU, 728 GB RAM, AL2023 |
| vllm | **0.29.0** (not 0.11 — see change 1) |
| torch / transformers | 2.13.0+cu130 / 5.17.0 |
| python | 3.12.13 (uv venv; system python is 3.9, too old) |
| nvcc | 13.4 |
| generator | `Qwen/Qwen3.6-35B-A3B-FP8`, tp=4, 9.3 GB/GPU |
| judge | `Qwen/Qwen3.5-122B-A10B-FP8`, tp=8 |

All four model ids used in the run exist on HF and resolve to
`Qwen3_5MoeForConditionalGeneration`, which vLLM 0.29.0 supports.

---

## 1. vllm 0.11 → 0.29.0, plus `ninja`

`requirements.txt` says `vllm>=0.11`; 0.11 predates these model
architectures. Installed 0.29.0 (current release).

Also needed `ninja` on `PATH`: flashinfer JIT-compiles its top-k/top-p
sampling kernel during engine warmup and dies with
`FileNotFoundError: 'ninja'`. The package alone is not enough — the binary
lands in `.venv/bin`, which is not on `PATH` when you invoke
`.venv/bin/python` directly. `env.sh` now prepends `.venv/bin` and
`/usr/local/cuda/bin`.

## 2. Reasoning mode had to be switched off — *blocking*

`src/vllm_experiment.py:169` `build_prompt`

Qwen3.5/3.6 chat templates default to reasoning mode. The rendered prompt
ended in a dangling `<think>\n`, so the model spent its entire token budget on
a reasoning trace and never emitted JSON. Every arm scored 0 valid.

Now passes `enable_thinking=False` (falls back gracefully if a template
rejects the kwarg), which renders a closed empty `<think>\n\n</think>\n\n`
block. This also keeps the arms comparable to the local Qwen2.5 run, which
was a non-reasoning model.

## 3. `BeamSearchParams.min_tokens` no longer exists — *blocking*

`src/vllm_experiment.py:212` `install_beam_min_tokens`

vLLM 0.29.0 `BeamSearchParams` is
`(beam_width, max_tokens, ignore_eos, temperature, length_penalty,
include_stop_str_in_output, structured_outputs)`. Passing `min_tokens=150`
raises `TypeError: Unexpected keyword argument 'min_tokens'`.

The brief calls `min_tokens=150` load-bearing and it is: without it the
highest-scoring beam is a **9-token stub**, `{\n  "recommendations": [`, then
EOS. Dropping the parameter silently guts every beam arm.

Rather than deviate, `min_tokens` is re-implemented. `SamplingParams.min_tokens`
works by masking EOS out of the logits. In vLLM's beam step each live beam is
expanded into `2*beam_width` candidates, one per top-logprob token; the
candidate whose token is EOS goes to `instance.completed` and the rest stay
live. So **discarding premature EOS candidates is equivalent to masking EOS** —
the surviving non-EOS candidates and their relative `cum_logprob` ranking are
untouched. The wrapper drops any completed beam shorter than `min_tokens`
generated tokens. Controlled by `--beam-min-tokens` (default 150).

`ignore_eos=True` was rejected as the substitute: it forces every beam to the
token cap, so the JSON is always followed by a rambling tail and no beam arm
can ever be valid.

## 4. Beam sequences include the prompt — *blocking, and silently wrong*

`src/vllm_experiment.py:245` `beam_call`

`BeamSearchInstance.__init__` seeds `.tokens` with the prompt token ids, and
`beam_search` sets `beam.text = tokenizer.decode(beam.tokens)`. So in 0.29.0
both the text and the token count of every beam candidate **include the entire
rendered prompt**. Consequences in the original code:

- `try_parse` scanned from the first `{`, which is in the system prompt's JSON
  exemplar → garbage, not the model's output.
- `tokens` counted prompt+generated, so `hit_cap` was always true and `tok/s`
  was inflated.
- `cmd_mechanism`'s "first generated token" was the prompt's first token,
  identical for greedy and beam, making the divergence test vacuous.
- `cmd_width` additionally read `s.token_ids`, which does not exist on
  `BeamSearchSequence` (the field is `.tokens`) → `AttributeError`.

`beam_call` now feeds a `TokensPrompt` so the prompt length is known exactly,
slices it back off, and decodes generated tokens only. `gen_candidates` and
`cmd_width` both route through it, so the two phases can no longer drift.

## 5. `max_tokens` 400 → 1200 — *blocking*

`src/vllm_experiment.py:44`, flag `--max-new`

400 was far too small. The 35B pretty-prints its JSON (~3.1 chars/token), so
8 recommendations with rationales need ~600–900 tokens. At 400, greedy landed
on exactly the cap and **every guided-sampling candidate truncated mid-document**
(`finish=length`, 0/4 valid). At 1200 the same arms are 24/24 valid.

This is a genuine deviation from the brief and it matters for interpretation:
the "guided-sampling failures are token-cap truncations" question from the
local run was reproducible here only because the cap was too low. With an
adequate cap, guided sampling has **zero** truncations.

## 6. `try_parse` took the last `}`, hiding beam degeneration

`src/vllm_experiment.py:278` `_first_json_object`, `:313` `try_parse_ex`

Beams routinely emit a complete, valid 8-item document and then **keep going,
looping their own recommendations** until the token cap. The old
first-`{`-to-last-`}` span swallowed that repetition and reported plain
"invalid JSON", conflating two very different failures.

`try_parse` now extracts the first *balanced* top-level object (string-aware,
so braces inside string literals don't count) and records the leftover as
`trailing_chars` / `degenerate` per candidate. Verified against saved outputs:
greedy and sampling candidates are **bit-identical** under old and new parsers
(`trailing=0`), so no arm is flattered — only beam changes, and it changes from
"invalid" to "valid, plus N chars of measured repetition". Genuinely
unterminated beams still read invalid.

`cmd_analyze` gained `degen` and `cap` columns plus a
finish_reason/mean-trailing breakdown.

## 7. The judge was also in reasoning mode — *silently produced fake results*

`src/vllm_experiment.py` `render_chat`, `parse_judge_score`

`cmd_judge` built its prompt with its own `apply_chat_template` call instead of
going through `build_prompt`, so it never got change 2. The 122B judge is also a
reasoning model: with `max_tokens=8` every one of the 156 replies was the string

```
Thinking Process:\n\n1.  **
```

and `re.search(r"[1-5]", raw)` scraped the `1` off its numbered list. Result: a
clean-looking table reporting **judge mean = 1.00 for all eight strategies** and
`Spearman = nan` (zero variance). No error, no warning.

Fixed three ways: a shared `render_chat()` that both the generator and judge
paths must use (so this cannot drift again), `max_tokens` 8 → 16, and
`parse_judge_score()` which expects a bare integer and returns **None** rather
than guessing when the reply is prose. After the fix the raw replies are
`'5'`×119, `'4'`×23, `'2'`×13, `'3'`×1.

Anything consuming judge output should assert non-degenerate score variance.

## 8. Two engine-config flags the models needed

- `--max-num-seqs` — the 122B judge is a hybrid Mamba/GDN model. Each decode
  sequence needs a Mamba cache block; at tp=8 with
  `gpu_memory_utilization=0.85` only 136 are available, and the default
  `max_num_seqs=256` aborts CUDA graph capture with
  `ValueError: max_num_seqs (256) exceeds available Mamba cache blocks (136)`.
  Judge runs use 128. Default 0 leaves vLLM's default untouched.
- `--so-backend` — beam + structured outputs resolves its backend from engine
  config and only handles `xgrammar`/`guidance`; the default `auto` raises
  `ValueError: Unsupported structured output backend: auto`. Opt-in so the
  sampling arms (fine with `auto`) are unperturbed.

## 9. Non-blocking additions

- `--limit-personas N` (`:189` `load_personas`) — debug aid for cheap smoke
  runs. Defaults to 0 = all, so the specified runs are unaffected.
- `exp_meta.json` now also records `beam_min_tokens` and `limit_personas`.
- Violation table column labels: `when_out_of_window` and `when_not_iso` both
  rendered as `when=`. Now `date_out=` / `not_iso=`.
- Orchestration ran phases GPU-pinned (`CUDA_VISIBLE_DEVICES`): two 35B
  instances concurrently on disjoint GPU halves; the 127 GB judge needs
  all 8 GPUs so it ran alone. Every phase wrote its own return-code file
  and never aborted the queue.

## Correction: guided beam is now possible

> guided beam is intentionally absent — vLLM's BeamSearchParams has no guided params

No longer true. vLLM 0.29.0 `BeamSearchParams` **accepts
`structured_outputs`**, and `beam_search` has a real structured-output path
(`_init_beam_search_structured_output`, xgrammar/guidance backends, with
bitmask filtering of the candidate expansion). A guided beam arm is now
possible.

The 8-strategy matrix was left exactly as specified so the headline numbers
stay comparable to the local run — but `beam_lp10_g` / `beam_lp12_g` are
available work, and worth doing given how much of beam's measured failure here
is malformed/degenerate output that a grammar would prevent.

## Things deliberately *not* changed

`llm.beam_search` accepts a list of prompts and batches them, which would cut
beam wall time severalfold by putting `n_prompts × beam_width` sequences in
flight per step instead of `beam_width`. Not done: per-strategy `wall` is a
headline deliverable ("does batched n=4 sampling match beam's throughput"), and
batching personas into one beam call would make beam's timing incomparable to
the sampling arms. Parallelism was added *across* runs (disjoint GPU halves)
instead, which leaves per-request timing semantics untouched.

Same reasoning blocked batching the `determinism` reps — and there the batch
composition would change kernel reduction order, i.e. it would perturb the very
thing that phase measures.
