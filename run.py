#!/usr/bin/env python3
"""
NePTune — unified entrypoint.

Reads the config file, infers the task from the 'dataset' field, and dispatches
to the appropriate runner.  All unknown arguments are forwarded as-is.

Usage
-----
# VQA (CLEVR-Humans / CLEVR) — parallel across all available GPUs
python run.py --config configs/clevr-humans.json

# VQA on specific GPUs
python run.py --config configs/clevr-humans.json --gpus 0 1 2 3

# Evaluate results with LLM judge
python run.py --config configs/clevr-humans.json --evaluate

# CLEVR-Transfer (RPM / Puzzle)
python run.py --config configs/clevr-rpm.json
python run.py --config configs/clevr-puzzle.json

# CLEVR-Transfer (Referring Expressions)
python run.py --config configs/clevr-refexps.json

# Referring Expression Grounding (RefCOCO / RefGTA)
python run.py --config configs/refadv.json
python run.py --config configs/refgta.json
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List


# ---------------------------------------------------------------------------
# Dataset → runner routing table
# ---------------------------------------------------------------------------

VQA_DATASETS       = {"clevr-humans", "clevr"}
RPM_PUZZLE_DATASETS = {"rpm", "puzzle"}
REFEXP_EXT_DATASETS = {"refexps"}
REFEXP_DATASETS     = {
    "refcoco-adv", "refgta", "refcocog",
    "refcocoA", "refcocoB", "refcoco+val",
    "refcoco+A", "refcoco+B", "refcocoval",
}


ROOT = os.path.dirname(os.path.abspath(__file__))


def run(cmd: List[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NePTune unified entrypoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to the JSON config file (e.g. configs/clevr-humans.json)."
    )

    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run LLM-judge evaluation on an existing results file."
    )

    # ---- VQA-parallel options ----
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=None,
        help="GPU indices for parallel VQA (default: all CUDA devices)."
    )

    # ---- evaluate options ----
    parser.add_argument(
        "--workers", type=int, default=256,
        help="Parallel threads for evaluation (default: 256)."
    )
    parser.add_argument(
        "--judge_model", type=str, default="deepseek-chat",
        help="Model ID for the LLM judge (default: deepseek-chat)."
    )
    parser.add_argument(
        "--judge_provider", type=str, default="deepseek",
        choices=["deepseek", "openai", "vllm"],
        help="Provider for the LLM judge (default: deepseek)."
    )


    # Capture any extra args and forward them to the target script
    args, extra = parser.parse_known_args()

    # Load config to determine routing
    with open(args.config) as f:
        config = json.load(f)

    dataset = config.get("dataset", "").lower()

    # -----------------------------------------------------------------------
    # --evaluate: always routes to analysis/evaluate_clevr_human.py
    # -----------------------------------------------------------------------
    if args.evaluate:
        # Derive results path the same way run_vqa.py does
        vlm_model = config.get("vlm_model_name", "")
        results_file = os.path.join(
            "experiments", dataset, vlm_model,
            config.get("results_save_file", "results.json"),
        )
        run([
            sys.executable, "analysis/evaluate_clevr_human.py",
            "--model_id", args.judge_model,
            "--provider", args.judge_provider,
            "--workers", str(args.workers),
            "--vlm_output_data_path", results_file,
            *extra,
        ])
        return

    # -----------------------------------------------------------------------
    # Standard run — route by dataset
    # -----------------------------------------------------------------------
    if dataset in VQA_DATASETS:
        cmd = [
            sys.executable, os.path.join("runners", "run_vqa_parallel.py"),
            "--config_file", args.config,
        ]
        if args.gpus is not None:
            cmd += ["--gpus", *[str(g) for g in args.gpus]]
        run([*cmd, *extra])

    elif dataset in RPM_PUZZLE_DATASETS:
        run([
            sys.executable, os.path.join("runners", "run_clevr_extensions_rpm_puzzle.py"),
            "--config_file", args.config,
            *extra,
        ])

    elif dataset in REFEXP_EXT_DATASETS:
        run([
            sys.executable, os.path.join("runners", "run_clevr_extensions_ref.py"),
            "--config_file", args.config,
            *extra,
        ])

    elif dataset in REFEXP_DATASETS:
        run([
            sys.executable, os.path.join("runners", "run_ref_exp_fast.py"),
            "--config_file", args.config,
            *extra,
        ])

    else:
        print(f"Error: Unknown dataset '{dataset}' in config.")
        print(f"  VQA datasets       : {VQA_DATASETS}")
        print(f"  RPM/Puzzle datasets: {RPM_PUZZLE_DATASETS}")
        print(f"  RefExp-ext datasets: {REFEXP_EXT_DATASETS}")
        print(f"  RefExp datasets    : {REFEXP_DATASETS}")
        sys.exit(1)


if __name__ == "__main__":
    main()
