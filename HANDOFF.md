# Session Handoff — Track A (BioReasoning Challenge)

Context for continuing work in a new session/CLI. Read this first, then `git pull`.

## Goal
Maximize the Track A score = **avg of DE-AUROC and DIR-AUROC** (see `kaggle_metric_track_a.py`).
- DE-AUROC: effect vs none, scored by `prediction_up + prediction_down`.
- DIR-AUROC: up vs down among effect rows, scored by `prediction_up/(prediction_up+prediction_down)`.
- Constraint: prompt-only, fixed model `gpt-oss-120b`, max 4096 prompt tokens, 3 seeds (42/43/44) averaged.

## Key facts learned
- Train label distribution: **4260 none / 2359 up / 1086 down** (~55/31/14%). None dominates; among effects up ~2x down.
- Test split is disjoint on the **gene axis** — no test gene appears in train. Memorizing genes is useless; must reason from function.
- The submission pipeline (`examples/track_a_prompt_only.py`) auto-appends the `<answer>A/B/C</answer>` instruction via `append_answer_tag()`, so the template must NOT include its own answer-format block.
- Template placeholders are single-brace: `{pert}`, `{gene}`, `{cell_desc}` (filled by `str.format`).

## What was changed this session
- `baml_src/track_a.baml` and `examples/prompt_template.txt`: rewrote the prompt with a reasoning scaffold, real base-rate calibration (55/31/14), and directional-commitment guidance (lean up when direction ambiguous).
- `examples/quick_eval_train.py`: NEW eval harness. Runs N train rows x 3 seeds, builds continuous seed-averaged predictions, scores with the real `kaggle_metric_track_a.py`. Flags: `-n`, `--random SEED`, `--concurrency`, `--save`, `--api-base`, `--model`.
- Installed `scikit-learn` into the venv (metric dependency).

## Results so far
- Local `gpt-oss:20b` (weak, sanity only): score ~0.51.
- `gpt-oss-120b` on cloud RTX PRO 6000 Blackwell (96GB):
  - 100 rows: **0.5901**
  - 500 rows: **0.5618**
- Problem: model over-predicts "none" (~80% of rows), and hard A/B/C parsing yields mostly (0,0) predictions -> flat DE score -> weak AUROC.

## NEXT STEP (highest leverage) — not yet done
Switch extraction from hard-parse to **logprobs** via `examples/track_a_logprobs.py`. It reads the
A/B/C answer-token logprobs and softmaxes them into continuous P(up)/P(down)/P(none) -> much better AUROC.
Requires serving with the patch wrapper:

```bash
# server
uv run --extra serve python serve_with_logprobs_fix.py openai/gpt-oss-120b \
    --port 8000 --enforce-eager --no-enable-prefix-caching \
    --gpu-memory-utilization 0.95 --max-num-seqs 64 --max-model-len 8192
# run
uv run python examples/track_a_logprobs.py \
    --prompt-template examples/prompt_template.txt \
    --api-base http://localhost:8000/v1 --model openai/gpt-oss-120b --concurrency 32
```

Also TODO: add a `--logprobs` mode to `quick_eval_train.py` to A/B hard-parse vs logprobs AUROC on
the same train rows; and consider softening the "default to none" conservatism.

## GPU serve command (RTX PRO 6000 Blackwell, 96GB, CUDA 13)
```bash
export HF_HOME=/workspace/hf
uv sync --extra serve
uv run --extra serve vllm serve openai/gpt-oss-120b \
    --port 8000 --enforce-eager --no-enable-prefix-caching \
    --gpu-memory-utilization 0.95 --max-num-seqs 64 --max-model-len 8192
```

## Repo
Pushed to public repo: https://github.com/MikeyRupert/bio_a_gang (remote `newrepo`).
Original origin: MikeyRupert/BioReasoningChallenge_1 (untouched).
