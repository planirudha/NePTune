#!/usr/bin/env python3
"""
Parallel VQA runner across multiple GPUs.

Launches one run_vqa.py process per GPU, each handling its own split of the
dataset and writing to its own results file. When all splits finish, merges
them into a single combined results file.

Usage:
    python run_vqa_parallel.py --config_file configs/clevr-humans.json
    python run_vqa_parallel.py --config_file configs/clevr-humans.json --gpus 0 1 2 3
    python run_vqa_parallel.py --config_file configs/clevr.json --gpus 0 1 2 3 4 5 6 7
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Split results helpers
# ---------------------------------------------------------------------------

def split_results_path(base_path: str, split_index: int) -> str:
    """Derive a per-split results file path from the base path."""
    root, ext = os.path.splitext(base_path)
    return f"{root}_split{split_index}{ext}"


def load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def merge_split_files(split_paths: List[str], output_path: str, description: str) -> Tuple[int, int]:
    """Merge per-split result files into one combined file.

    Returns (total_merged, n_splits_found).
    """
    combined: List[Dict] = []
    seen_ids = set()
    n_found = 0

    for path in split_paths:
        data = load_json(path)
        if data is None:
            print(f"  WARNING: split file not found: {path}")
            continue
        n_found += 1
        items = data.get("results", data) if isinstance(data, dict) else data
        for item in items:
            item_id = str(item.get("id", ""))
            if item_id and item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            combined.append(item)

    # Sort by original_dataset_index so order matches the original dataset
    combined.sort(key=lambda r: r.get("original_dataset_index", 0))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "description": description,
                "timestamp": datetime.datetime.now().isoformat(),
                "results": combined,
            },
            f,
            indent=2,
        )
    return len(combined), n_found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run run_vqa.py in parallel across multiple GPUs."
    )
    parser.add_argument(
        "--config_file", default="configs/clevr-humans.json",
        help="Config file forwarded to each run_vqa.py worker."
    )
    parser.add_argument(
        "--gpus", nargs="+", type=int, default=list(range(8)),
        help="GPU indices to use (default: 0-7)."
    )
    parser.add_argument(
        "--keep_split_files", action="store_true",
        help="Keep per-split result files after merging (deleted by default)."
    )
    parser.add_argument(
        "--base_results_file", type=str, default=None,
        help="Existing merged results file whose IDs should be skipped by each worker "
             "(used in refinement loops to avoid re-processing already-correct items)."
    )
    parser.add_argument(
        "--enable_vlm_direct_prediction", action="store_true", default=False,
        help="Forward --enable_vlm_direct_prediction to each run_vqa.py worker."
    )
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = json.load(f)

    n_gpus = len(args.gpus)
    dataset    = config.get("dataset", "clevr-humans")
    vlm_model  = config.get("vlm_model_name", "intern25_8b")
    base_results = os.path.join(
        "experiments", dataset, vlm_model,
        config.get("results_save_file", "results.json"),
    )

    print(f"Config       : {args.config_file}")
    print(f"GPUs         : {args.gpus}")
    print(f"Splits       : {n_gpus}")
    print(f"Results base : {base_results}")
    print()

    # Build per-split result file paths
    split_files = [split_results_path(base_results, i) for i in range(n_gpus)]

    MAX_RETRIES = 3

    def make_worker(local_rank: int, gpu_id: int, attempt: int) -> Tuple[subprocess.Popen, object, int, int]:
        cmd = [
            sys.executable, os.path.join("runners", "run_vqa.py"),
            "--config_file", args.config_file,
            "--num_splits", str(n_gpus),
            "--split_index", str(local_rank),
            "--results_save_file", os.path.basename(split_files[local_rank]),
        ]
        if args.base_results_file:
            cmd += ["--base_results_file", args.base_results_file]
        if args.enable_vlm_direct_prediction:
            cmd += ["--enable_vlm_direct_prediction"]
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

        log_path = split_results_path(base_results, local_rank).replace(".json", f".attempt{attempt}.log")
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_file = open(log_path, "w")

        print(f"[GPU {gpu_id}] split {local_rank}/{n_gpus} attempt {attempt}  →  {split_files[local_rank]}")
        print(f"           log: {log_path}")

        proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        return proc, log_file, gpu_id, local_rank

    # Launch one worker per GPU
    retry_counts: List[int] = [0] * n_gpus
    processes: List[Tuple] = []
    for local_rank, gpu_id in enumerate(args.gpus):
        proc, log_file, gpu_id, rank = make_worker(local_rank, gpu_id, attempt=0)
        processes.append((proc, log_file, gpu_id, rank))
        time.sleep(1)

    print(f"\nAll {n_gpus} workers launched. Waiting for completion...\n")

    # Poll until all workers finish, restarting failed ones up to MAX_RETRIES
    t_start = time.time()
    remaining = list(range(n_gpus))
    permanently_failed: List[Tuple[int, int]] = []

    while remaining:
        time.sleep(10)
        still_running = []
        for idx in remaining:
            proc, log_file, gpu_id, rank = processes[idx]
            ret = proc.poll()
            if ret is None:
                still_running.append(idx)
            elif ret == 0:
                elapsed = time.time() - t_start
                print(f"  [GPU {gpu_id}] split {rank} finished — OK  ({elapsed:.0f}s elapsed)")
                log_file.close()
            else:
                elapsed = time.time() - t_start
                retry_counts[idx] += 1
                log_file.close()
                if retry_counts[idx] <= MAX_RETRIES:
                    print(f"  [GPU {gpu_id}] split {rank} FAILED (exit {ret}) — restarting "
                          f"(attempt {retry_counts[idx]}/{MAX_RETRIES})  ({elapsed:.0f}s elapsed)")
                    time.sleep(5)
                    proc, log_file, gpu_id, rank = make_worker(rank, gpu_id, attempt=retry_counts[idx])
                    processes[idx] = (proc, log_file, gpu_id, rank)
                    still_running.append(idx)
                else:
                    print(f"  [GPU {gpu_id}] split {rank} FAILED permanently after "
                          f"{MAX_RETRIES} retries  ({elapsed:.0f}s elapsed)")
                    permanently_failed.append((gpu_id, rank))
        remaining = still_running
        if remaining:
            elapsed = time.time() - t_start
            print(f"  {len(remaining)} worker(s) still running... ({elapsed:.0f}s)")

    total_elapsed = time.time() - t_start
    print(f"\nAll workers done in {total_elapsed:.0f}s.")

    if permanently_failed:
        print(f"\nWARNING: {len(permanently_failed)} split(s) failed permanently: {permanently_failed}")
        print("Merging available results anyway.\n")

    # Merge split files; if a base_results_file was used to skip already-correct
    # items, include it in the merge so those items are preserved in the output.
    paths_to_merge = list(split_files)
    if args.base_results_file and os.path.exists(args.base_results_file):
        paths_to_merge = [args.base_results_file] + paths_to_merge
        print(f"Including base results file in merge: {args.base_results_file}")

    print(f"Merging {len(paths_to_merge)} file(s) into: {base_results}")
    n_merged, n_found = merge_split_files(
        paths_to_merge,
        base_results,
        description=config.get("desc", f"VQA {dataset} parallel run"),
    )
    print(f"Merged {n_merged} results from {n_found}/{len(paths_to_merge)} files → {base_results}")

    # Clean up split files
    if not args.keep_split_files:
        for path in split_files:
            if os.path.exists(path):
                os.remove(path)
                print(f"  Removed: {path}")

    # Print quick accuracy summary if 'correct' fields are present
    with open(base_results) as f:
        merged = json.load(f)
    results_list = merged.get("results", [])
    evaluated = [r for r in results_list if isinstance(r.get("correct_final_answer"), bool)]
    if evaluated:
        n_correct = sum(1 for r in evaluated if r["correct_final_answer"])
        print(f"\nQuick accuracy (correct_final_answer): "
              f"{n_correct}/{len(evaluated)} = {100*n_correct/len(evaluated):.2f}%")


if __name__ == "__main__":
    main()
