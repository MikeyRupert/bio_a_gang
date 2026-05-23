"""
Track A -- Prompt-only baseline using BAML.

Mirrors examples/track_a_prompt_only.py but uses the BAML-generated
PredictPerturbEffect function (baml_src/track_a.baml) with the GPT client
(gpt-oss:20b via Ollama at localhost:11434).

Seeds 42/43/44 are injected per-call via ClientRegistry. The typed
PerturbEffect enum (Up/Down/NoEffect) replaces regex-based text parsing.

Usage:
    python baml/track_a_prompt_only.py

    # Parallel requests:
    python baml/track_a_prompt_only.py --concurrency 4

    # Custom prompt template:
    python baml/track_a_prompt_only.py --prompt-template examples/prompt_template.txt
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict

import pandas as pd
import baml_py
from baml_client import b
from baml_client.types import PerturbEffect
from baml_client.sync_client import BamlCallOptions

from mlgenx import parse_answer

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "data" / "test.csv"
SEEDS = [42, 43, 44]

_DEFAULT_UP, _DEFAULT_DOWN = parse_answer("")

_EFFECT_TO_PRED: dict[PerturbEffect, tuple[float, float]] = {
    PerturbEffect.Up:       (1.0, 0.0),
    PerturbEffect.Down:     (0.0, 1.0),
    PerturbEffect.NoEffect: (0.0, 0.0),
}


# ---------------------------------------------------------------------------
# Prompt loading (same as prompt_only baseline)
# ---------------------------------------------------------------------------

def load_prompts_csv(path: Path) -> Dict[str, str]:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(path)
    missing = {"id", "prompt"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return dict(zip(df["id"].astype(str), df["prompt"].astype(str)))


def load_prompt_template(path: Path) -> str:
    text = path.read_text()
    for req in ("{pert}", "{gene}"):
        if req not in text:
            raise ValueError(f"Template {path} must contain {req}")
    return text


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
    parser = argparse.ArgumentParser(description="Track A: BAML prompt-only (3 seeds)")
    parser.add_argument("--prompts-csv", type=Path, default=None)
    parser.add_argument("--prompt-template", type=Path, default=None)
    parser.add_argument("--api-base", default="http://localhost:11434/v1",
                        help="vLLM/Ollama base URL")
    parser.add_argument("--api-key", default="ollama")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument(
        "--reasoning-effort", default="medium",
        choices=["low", "medium", "high"],
    )
    parser.add_argument("--test-csv", type=Path, default=TEST_CSV)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "track_a" / "baml")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N rows (useful for quick testing).",
    )
    args = parser.parse_args()

    prompts_map: Dict[str, str] | None = None
    template: str | None = None

    if args.prompts_csv:
        prompts_map = load_prompts_csv(args.prompts_csv)
        print(f"Loaded {len(prompts_map)} prompts from {args.prompts_csv}")
    if args.prompt_template:
        template = load_prompt_template(args.prompt_template)
        print(f"Loaded prompt template from {args.prompt_template}")
    if not prompts_map and not template:
        print("Using BAML zero-shot prompt (track_a.baml)")

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

    def make_client_registry(seed: int) -> baml_py.ClientRegistry:
        cr = baml_py.ClientRegistry()
        cr.add_llm_client("GPT", "openai-generic", {
            "base_url": args.api_base,
            "api_key": args.api_key,
            "model": args.model,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_tokens": args.max_tokens,
            "seed": seed,
            "reasoning_effort": args.reasoning_effort,
        })
        cr.set_primary("GPT")
        return cr

    def process_row(idx: int, row: pd.Series) -> None:
        nonlocal new_count
        rid = str(row["id"])
        pert, gene = row["pert"], row["gene"]

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

            cr = make_client_registry(seed)
            effect: PerturbEffect | None = None

            for attempt in range(args.max_retries + 1):
                try:
                    effect = b.PredictPerturbEffect(
                        pert=pert,
                        gene=gene,
                        baml_options=BamlCallOptions(client_registry=cr),
                    )
                    break
                except Exception as e:
                    print(f"  [{rid}] seed={seed} attempt={attempt+1} error={e}")
                    if attempt < args.max_retries:
                        time.sleep(2 ** attempt)

            if effect is not None:
                pred_up, pred_down = _EFFECT_TO_PRED[effect]
            else:
                pred_up, pred_down = _DEFAULT_UP, _DEFAULT_DOWN

            cached[key_up] = pred_up
            cached[key_down] = pred_down
            cached[f"effect_seed{seed}"] = effect.value if effect else "unknown"

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
                f"pred_down={cached['prediction_down']:.3f} "
                f"effect={cached.get(f'effect_seed{SEEDS[0]}', '?')}"
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
    print(f"Collected {total} rows ({new_count} new calls)")

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
            "effect_seed42": c.get("effect_seed42", "unknown"),
            "effect_seed43": c.get("effect_seed43", "unknown"),
            "effect_seed44": c.get("effect_seed44", "unknown"),
            "model_name": c.get("model_name", args.model),
        })

    sub_df = pd.DataFrame(rows_out)
    sub_path = args.output_dir / "submission.csv"
    sub_df.to_csv(sub_path, index=False)

    prompt_path = args.output_dir / "prompt.txt"
    prompt_path.write_text(
        "# Track A (BAML) -- prompt from baml_src/track_a.baml\n"
        f"# Model: {args.model}\n"
        f"# Client: GPT (openai-generic @ {args.api_base})\n"
    )

    zip_path = args.output_dir / "submission_track_a.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_path, "submission.csv")
        zf.write(prompt_path, "prompt.txt")

    print(f"Wrote {sub_path}")
    print(f"Wrote {zip_path}  <-- upload this to Kaggle")


if __name__ == "__main__":
    main()
