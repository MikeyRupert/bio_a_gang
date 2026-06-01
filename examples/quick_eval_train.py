"""Quick eval harness: run the prompt template on the first N train rows over
3 seeds, build continuous seed-averaged predictions, and score them with the
real competition metric (kaggle_metric_track_a.py: avg of DE + DIR AUROC).

Usage:
    python examples/quick_eval_train.py \
        --api-base http://localhost:11434/v1 --model gpt-oss:20b \
        -n 100 --concurrency 8
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from mlgenx import parse_answer
from mlgenx.prompts import CELL_DESC
from track_a_prompt_only import (
    append_answer_tag,
    extract_answer_tag,
    load_prompt_template,
    post_chat_completion,
)
from track_a_logprobs import (
    post_chat_completion as post_chat_completion_lp,
    prediction_from_logprobs,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
_DEFAULT_UP, _DEFAULT_DOWN = parse_answer("")


def _load_metric():
    """Import score() from kaggle_metric_track_a.py at the repo root."""
    spec = importlib.util.spec_from_file_location(
        "kaggle_metric_track_a", ROOT / "kaggle_metric_track_a.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score


def run_row(row, template, args):
    """Run all seeds for one row; return a submission record."""
    rec = {"id": str(row["id"])}
    prompt = append_answer_tag(template.format(
        pert=row["pert"], gene=row["gene"], cell_desc=CELL_DESC))
    prompt_tokens = 0.0
    for seed in SEEDS:
        if args.logprobs:
            try:
                text, _reasoning, stats, lp_content = post_chat_completion_lp(
                    api_base=args.api_base, api_key=args.api_key, model=args.model,
                    prompt=prompt, seed=seed, max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s, top_logprobs=args.top_logprobs,
                    no_reasoning=args.no_reasoning,
                )
            except Exception as e:
                text, stats, lp_content = "", {}, []
                print(f"  {row['id']} seed={seed} error={e}", file=sys.stderr)
            lp_pred = prediction_from_logprobs(lp_content)
            if lp_pred is not None:
                pred_up, pred_down = lp_pred
            else:
                # Fall back to hard-parse when the answer token can't be located.
                tag = extract_answer_tag(text)
                pred_up, pred_down = parse_answer(tag if tag else text)
        else:
            try:
                text, stats = post_chat_completion(
                    api_base=args.api_base, api_key=args.api_key, model=args.model,
                    prompt=prompt, seed=seed, max_tokens=args.max_tokens,
                    timeout_s=args.timeout_s,
                )
            except Exception as e:
                text, stats = "", {}
                print(f"  {row['id']} seed={seed} error={e}", file=sys.stderr)
            tag = extract_answer_tag(text)
            pred_up, pred_down = parse_answer(tag if tag else text)
        rec[f"prediction_up_seed{seed}"] = pred_up
        rec[f"prediction_down_seed{seed}"] = pred_down
        rec[f"reasoning_trace_seed{seed}"] = text or "none"
        prompt_tokens = max(prompt_tokens, float(stats.get("prompt_tokens", 0)))
    rec["prediction_up"] = sum(rec[f"prediction_up_seed{s}"] for s in SEEDS) / len(SEEDS)
    rec["prediction_down"] = sum(rec[f"prediction_down_seed{s}"] for s in SEEDS) / len(SEEDS)
    rec["tokens_used"] = 0
    rec["prompt_tokens"] = prompt_tokens
    rec["model_name"] = args.model
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://localhost:11434/v1")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--template", type=Path,
                    default=ROOT / "examples" / "prompt_template.txt")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--random", type=int, default=None,
                    metavar="SEED", help="Sample n random rows with this seed.")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--logprobs", action="store_true",
                    help="Extract continuous P(up)/P(down) from answer-token "
                         "logprobs instead of hard-parsing A/B/C. Requires the "
                         "server started via serve_with_logprobs_fix.py.")
    ap.add_argument("--top-logprobs", type=int, default=20,
                    help="How many top logprobs to request per token (logprobs mode).")
    ap.add_argument("--no-reasoning", action="store_true",
                    help="Disable hidden chain-of-thought (enable_thinking=false) "
                         "for genuinely calibrated answer-token logprobs.")
    ap.add_argument("--save", type=Path, default=None,
                    help="Optional path to write the predictions CSV.")
    args = ap.parse_args()

    template = load_prompt_template(args.template)
    df = pd.read_csv(ROOT / "data" / "train.csv")
    if args.random is not None:
        df = df.sample(n=args.n, random_state=args.random).reset_index(drop=True)
    else:
        df = df.head(args.n)

    records = [None] * len(df)
    rows = list(df.iterrows())
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_row, row, template, args): i
                for i, (_, row) in enumerate(rows)}
        done = 0
        for fut in as_completed(futs):
            records[futs[fut]] = fut.result()
            done += 1
            print(f"  [{done}/{len(df)}]", end="\r", file=sys.stderr)
    print(file=sys.stderr)

    sub = pd.DataFrame(records)
    sol = df[["id", "label"]].copy()

    if args.save:
        sub.to_csv(args.save, index=False)
        print(f"Wrote predictions -> {args.save}")

    # Report the confusion-style breakdown (argmax of seed-avg).
    def argmax_label(r):
        u, d = r["prediction_up"], r["prediction_down"]
        n = 1 - u - d
        return max(("up", u), ("down", d), ("none", n), key=lambda x: x[1])[0]
    sub2 = sub.merge(sol, on="id")
    sub2["pred"] = sub2.apply(argmax_label, axis=1)
    acc = (sub2["pred"] == sub2["label"]).mean()
    print("\nlabel distribution (truth):",
          dict(sub2["label"].value_counts()))
    print("pred  distribution       :",
          dict(sub2["pred"].value_counts()))
    print(f"Argmax accuracy: {acc:.1%}")

    score = _load_metric()
    try:
        s = score(sol, sub, row_id_column_name="id")
        print(f"\n*** Competition metric (avg DE+DIR AUROC): {s:.4f} ***")
    except Exception as e:
        print(f"\nMetric could not be computed: {e}")
        print("(Need both DE classes present and both directions among effects "
              "— increase -n if the sample is too small/skewed.)")


if __name__ == "__main__":
    main()
