"""
Track A -- Logprob-based continuous predictions using OpenAI SDK.

Uses the same GPT client config as the BAML setup (baml_src/clients.baml)
but calls the OpenAI SDK directly with logprobs=True so that raw top_logprobs
are accessible. BAML's generated client abstracts away the raw response, so
the logprobs path bypasses it and talks to the same endpoint directly.

The zero-shot prompt (matching track_a.baml) is appended with an
<answer>A/B/C</answer> tag instruction. Logprobs are read at the answer
token and softmax'd over A/B/C to produce continuous (pred_up, pred_down).

Mirrors examples/track_a_logprobs.py but targets gpt-oss:20b and reads
config from the same env vars as the BAML client.

Usage:
    python baml/track_a_logprobs.py

    # Parallel requests:
    python baml/track_a_logprobs.py --concurrency 4

    # Disable reasoning for calibrated logprobs:
    python baml/track_a_logprobs.py --no-reasoning
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from mlgenx import parse_answer
from mlgenx.prompts import CELL_DESC

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "test.csv"
SEEDS = [42, 43, 44]

_DEFAULT_UP, _DEFAULT_DOWN = parse_answer("")

_PROMPT_ZERO = (
    "You are an expert molecular biologist who studies how genes are related using Perturb-seq.\n\n"
    "Context: {cell_desc}\n\n"
    "Question: If you knockdown {pert} using CRISPRi in mouse BMDMs, what is the effect on {gene}?\n\n"
    "Your answer must be one of:\n"
    "A) Knockdown of {pert} results in up-regulation of {gene}.\n"
    "B) Knockdown of {pert} results in down-regulation of {gene}.\n"
    "C) Knockdown of {pert} does not significantly affect {gene}.\n\n"
    "Return ONLY the final choice in this exact format:\n"
    "<answer>A</answer>, <answer>B</answer>, or <answer>C</answer>\n"
    "Do not include any other text."
)


def build_prompt(pert: str, gene: str) -> str:
    return _PROMPT_ZERO.format(pert=pert, gene=gene, cell_desc=CELL_DESC)


# ---------------------------------------------------------------------------
# Logprob extraction (identical to examples/track_a_logprobs.py)
# ---------------------------------------------------------------------------

def prediction_from_logprobs(
    logprobs_content: List[dict],
    debug: bool = False,
) -> Optional[Tuple[float, float]]:
    if not logprobs_content:
        return None

    tokens = [t.get("token", "") for t in logprobs_content]
    reconstructed = "".join(tokens)

    m = re.search(r"<answer>\s*([ABCabc])\s*</answer>", reconstructed)
    if not m:
        if debug:
            tail = reconstructed[-200:]
            print(f"    [debug] no <answer> tag (last 200 chars): {tail!r}")
        return None

    answer_char_start = m.start(1)
    char_pos = 0
    answer_token_idx: int | None = None
    for i, tok_text in enumerate(tokens):
        tok_end = char_pos + len(tok_text)
        if char_pos <= answer_char_start < tok_end:
            answer_token_idx = i
            break
        char_pos = tok_end

    if answer_token_idx is None:
        return None

    top_lps = logprobs_content[answer_token_idx].get("top_logprobs") or []

    if debug:
        chosen = logprobs_content[answer_token_idx]
        print(f"    [debug] answer token idx={answer_token_idx} "
              f"tok={chosen.get('token')!r} lp={chosen.get('logprob')} "
              f"top_lps={[(e.get('token'), e.get('logprob')) for e in top_lps[:5]]}")

    logprob_a: float | None = None
    logprob_b: float | None = None
    logprob_c: float | None = None

    for entry in top_lps:
        tok = entry.get("token", "").strip().upper()
        lp = entry.get("logprob")
        if lp is None:
            continue
        ends_a = tok == "A" or tok.endswith(">A")
        ends_b = tok == "B" or tok.endswith(">B")
        ends_c = tok == "C" or tok.endswith(">C")
        if ends_a and logprob_a is None:
            logprob_a = float(lp)
        elif ends_b and logprob_b is None:
            logprob_b = float(lp)
        elif ends_c and logprob_c is None:
            logprob_c = float(lp)

    if logprob_a is None and logprob_b is None and logprob_c is None:
        if debug:
            print("    [debug] neither A, B, nor C found in top_logprobs")
        return None

    lps = [lp for lp in (logprob_a, logprob_b, logprob_c) if lp is not None]
    floor = min(lps) - 20.0
    if logprob_a is None:
        logprob_a = floor
    if logprob_b is None:
        logprob_b = floor
    if logprob_c is None:
        logprob_c = floor

    max_lp = max(logprob_a, logprob_b, logprob_c)
    exp_a = math.exp(logprob_a - max_lp)
    exp_b = math.exp(logprob_b - max_lp)
    exp_c = math.exp(logprob_c - max_lp)
    total = exp_a + exp_b + exp_c
    if total <= 0:
        return None
    return exp_a / total, exp_b / total


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}


def save_cache(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Track A: BAML-configured logprob predictions (3 seeds)"
    )
    parser.add_argument("--api-base", default=os.environ.get("VLLM_API_BASE", "http://localhost:11434/v1"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "ollama"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL", "gpt-oss:20b"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument(
        "--no-reasoning", action="store_true",
        help="Disable model reasoning via chat_template_kwargs.enable_thinking=false.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "track_a_logprobs" / "baml")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (useful for quick testing).",
    )
    args = parser.parse_args()

    client = OpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
        timeout=args.timeout_s,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "responses_cache.json"
    cache = load_cache(cache_path)
    if "rows" not in cache:
        cache["rows"] = {}

    test_df = pd.read_csv(args.test_csv)
    if args.limit is not None:
        test_df = test_df.head(args.limit)
    total = len(test_df)
    cache_lock = threading.Lock()
    new_count = 0
    logprob_hits = 0
    logprob_misses = 0

    def call_api(prompt: str, seed: int) -> Tuple[str, List[dict]]:
        extra: dict = {}
        if args.no_reasoning:
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            top_p=1.0,
            seed=seed,
            max_tokens=args.max_tokens,
            logprobs=True,
            top_logprobs=args.top_logprobs,
            **extra,
        )
        content = resp.choices[0].message.content or ""
        lp_data = resp.choices[0].logprobs
        if lp_data and lp_data.content:
            raw_lps = [
                {
                    "token": t.token,
                    "logprob": t.logprob,
                    "top_logprobs": [
                        {"token": tl.token, "logprob": tl.logprob}
                        for tl in (t.top_logprobs or [])
                    ],
                }
                for t in lp_data.content
            ]
        else:
            raw_lps = []
        return content, raw_lps

    def process_row(idx: int, row: pd.Series) -> None:
        nonlocal new_count, logprob_hits, logprob_misses
        rid = str(row["id"])
        prompt = build_prompt(row["pert"], row["gene"])

        with cache_lock:
            cached = cache["rows"].get(rid, {})
            if all(
                f"prediction_up_seed{s}" in cached
                and f"prediction_down_seed{s}" in cached
                for s in SEEDS
            ):
                print(f"[{idx+1}/{total}] {rid} cache_hit")
                return

        for seed in SEEDS:
            key_up = f"prediction_up_seed{seed}"
            key_down = f"prediction_down_seed{seed}"
            with cache_lock:
                if key_up in cached and key_down in cached:
                    continue

            content = ""
            raw_lps: List[dict] = []
            for attempt in range(args.max_retries + 1):
                try:
                    content, raw_lps = call_api(prompt, seed)
                    break
                except Exception as e:
                    print(f"  [{rid}] seed={seed} attempt={attempt+1} error={e}")
                    if attempt < args.max_retries:
                        time.sleep(2 ** attempt)

            pair = prediction_from_logprobs(raw_lps, debug=args.debug)
            used_logprobs = pair is not None

            if pair is None:
                tag_m = re.search(r"<answer>\s*([ABCabc])\s*</answer>", content)
                source = tag_m.group(1).upper() if tag_m else content
                pair = parse_answer(source)

            pred_up, pred_down = pair

            with cache_lock:
                if used_logprobs:
                    logprob_hits += 1
                else:
                    logprob_misses += 1

            cached[key_up] = pred_up
            cached[key_down] = pred_down
            cached[f"reasoning_trace_seed{seed}"] = content
            cached[f"used_logprobs_seed{seed}"] = used_logprobs

        cached["prediction_up"] = sum(
            cached.get(f"prediction_up_seed{s}", _DEFAULT_UP) for s in SEEDS
        ) / len(SEEDS)
        cached["prediction_down"] = sum(
            cached.get(f"prediction_down_seed{s}", _DEFAULT_DOWN) for s in SEEDS
        ) / len(SEEDS)
        cached["model_name"] = args.model

        with cache_lock:
            cache["rows"][rid] = cached
            new_count += 1
            print(
                f"[{idx+1}/{total}] {rid} "
                f"pred_up={cached['prediction_up']:.3f} "
                f"pred_down={cached['prediction_down']:.3f}"
            )
            if new_count % args.save_every == 0:
                save_cache(cache_path, cache)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(process_row, idx, row)
            for idx, (_, row) in enumerate(test_df.iterrows())
        ]
        for future in as_completed(futures):
            future.result()

    save_cache(cache_path, cache)
    print(
        f"Collected {total} rows ({new_count} new calls). "
        f"Logprob extraction: {logprob_hits} hit, {logprob_misses} fallback."
    )

    rows_out = []
    for _, row in test_df.iterrows():
        rid = str(row["id"])
        c = cache["rows"].get(rid, {})
        rows_out.append({
            "id": rid,
            "prediction_up": c.get("prediction_up", _DEFAULT_UP),
            "prediction_down": c.get("prediction_down", _DEFAULT_DOWN),
            "prediction_up_seed42": c.get("prediction_up_seed42", _DEFAULT_UP),
            "prediction_down_seed42": c.get("prediction_down_seed42", _DEFAULT_DOWN),
            "prediction_up_seed43": c.get("prediction_up_seed43", _DEFAULT_UP),
            "prediction_down_seed43": c.get("prediction_down_seed43", _DEFAULT_DOWN),
            "prediction_up_seed44": c.get("prediction_up_seed44", _DEFAULT_UP),
            "prediction_down_seed44": c.get("prediction_down_seed44", _DEFAULT_DOWN),
            "reasoning_trace_seed42": c.get("reasoning_trace_seed42") or "none",
            "reasoning_trace_seed43": c.get("reasoning_trace_seed43") or "none",
            "reasoning_trace_seed44": c.get("reasoning_trace_seed44") or "none",
            "model_name": c.get("model_name", args.model),
        })

    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(
        "# Track A (BAML logprobs) -- zero-shot prompt matching baml_src/track_a.baml\n"
        "# Prediction: softmax over A/B/C logprobs at <answer> token\n"
        f"# Model: {args.model}\n\n"
        + _PROMPT_ZERO.format(pert="{pert}", gene="{gene}", cell_desc=CELL_DESC)
    )

    zip_path = args.output_dir / "submission_track_a.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        zf.write(prompt_path, "prompt.txt")

    print(f"Wrote {sub_path}")
    print(f"Wrote {zip_path}  <-- upload this to Kaggle")


if __name__ == "__main__":
    main()
