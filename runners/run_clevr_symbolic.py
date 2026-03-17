"""run_clevr_symbolic.py

CLEVR evaluation using pre-generated symbolic NePTune programs.

Instead of calling an LLM to generate code at runtime, this script loads a
pre-built question→code map (produced by LEFT/scripts/gen-clevr-neptune-program.py)
and uses those symbolic programs directly for execution.

Usage:
    python run_clevr_symbolic.py \
        --symbolic_programs  /path/to/clevr.json \
        --vlm_model_name     intern25_1b \
        --results_save_file  results.json \
        --num_samples        1000
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import traceback
import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import argparse
from copy import deepcopy

import torch
from PIL import Image, UnidentifiedImageError
import numpy as np
from tqdm import tqdm

from vision_utils import object_detector, expand_bboxes

from vision_agents import get_model
from utils import unified_postprocess

from probabilistic_tensor import ProbabilisticTensor, and_op, or_op
from code_base_template import build_logic_executor_code, build_score_fn, build_query_fn

from local_datasets import get_dataset
ProbabilisticTensor.training = False
ProbabilisticTensor.set_and_op("min")


# ==============================================================================
# Configuration and Setup
# ==============================================================================

def load_config_from_file(config_file_path: str) -> Dict:
    """Loads configurations from a JSON file."""
    if config_file_path and os.path.exists(config_file_path):
        try:
            with open(config_file_path, 'r') as f:
                print(f"Loading configuration from: {config_file_path}")
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {config_file_path}: {e}. Using defaults.")
        except Exception as e:
            print(f"Could not load config file {config_file_path}: {e}. Using defaults.")
    return {}


def setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CLEVR evaluation with pre-generated symbolic NePTune programs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--config_file', type=str, default="config_clevr_symbolic.json",
        help='Path to the JSON configuration file.',
    )
    parser.add_argument(
        '--symbolic_programs', type=str,
        help='Path to the pre-generated question→code JSON map (clevr.json).',
    )
    parser.add_argument(
        '--dataset', type=str, default='clevr',
        choices=['clevr', 'clevr-humans'],
        help='CLEVR dataset split to evaluate.',
    )
    parser.add_argument(
        '--results_save_file', type=str,
        help='Path to the results JSON file (supports incremental resuming).',
    )
    parser.add_argument(
        '--desc', type=str,
        help='Human-readable description for this run.',
    )
    parser.add_argument(
        '--vlm_model_name', type=str,
        help='Name of the VLM model (passed to get_model).',
    )
    parser.add_argument(
        '--od_model', type=str,
        help="Object detector model ID (e.g., 'dino', 'florence'). Only used when GT bboxes are unavailable.",
    )
    parser.add_argument(
        '--od_box_threshold', type=float,
        help="OD confidence threshold.",
    )
    parser.add_argument(
        '--use_gt_bboxes',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use ground-truth bounding boxes from the dataset instead of running OD.',
    )
    parser.add_argument(
        '--save_periodically',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save results periodically during the run.',
    )
    parser.add_argument(
        '--save_interval', type=int,
        help='How often to save (number of processed samples).',
    )
    # --- Split arguments ---
    parser.add_argument(
        '--num_splits', type=int, default=1,
        help='Total number of splits for distributed processing.',
    )
    parser.add_argument(
        '--split_index', type=int, default=0,
        help='Index of the current split (0 to num_splits-1).',
    )
    parser.add_argument(
        '--num_samples', type=int, default=1000,
        help='Number of dataset samples to load.',
    )
    # --- Fast-cache arguments ---
    parser.add_argument(
        '--enable_fast_cache',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable two-pass fast score caching (dry-run then prewarm VLM).',
    )
    parser.add_argument(
        '--use_vlm_batch',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use batched VLM scoring during fast-cache prewarm.',
    )

    parser.set_defaults(
        symbolic_programs="data/clevr/val/clevr.json",
        results_save_file="results.json",
        desc="CLEVR Symbolic VQA run",
        vlm_model_name="intern25_1b",
        od_model="dino",
        od_box_threshold=0.25,
        use_gt_bboxes=True,
        save_periodically=True,
        save_interval=10,
        enable_fast_cache=True,
        use_vlm_batch=True,
    )
    return parser


def apply_config_file_to_parser_defaults(parser: argparse.ArgumentParser, config_data: Dict):
    action_map = {action.dest: action for action in parser._actions}
    temp_defaults = {}
    for key, value in config_data.items():
        if key in action_map:
            temp_defaults[key] = value
        else:
            print(f"Config key '{key}' not found in parser arguments. Ignoring.")
    parser.set_defaults(**temp_defaults)


# ==============================================================================
# Symbolic Program Loader
# ==============================================================================

def load_symbolic_programs(path: str) -> Dict[str, str]:
    """Load the pre-generated question→code map from disk.

    Returns a dict mapping question string → NePTune code snippet (the body
    that goes inside logic_executor, *not* the full def block).
    """
    print(f"Loading symbolic programs from: {path}")
    with open(path, 'r') as f:
        programs = json.load(f)
    print(f"  Loaded {len(programs):,} question→program entries.")
    return programs


# ==============================================================================
# Model Initialization
# ==============================================================================

def initialize_models(args: argparse.Namespace):
    """Initialize OD and VLM. No code generator needed."""
    try:
        print("\n" + "=" * 30 + "\n--- Initializing Models ---")

        print(f"Initializing OD ({args.od_model})..."); st = time.time()
        od_predictor = object_detector(args.od_model, threshold=args.od_box_threshold)
        print(f"OD initialized in {time.time()-st:.2f}s")

        print(f"Initializing VLM ({args.vlm_model_name})..."); st = time.time()
        vl_model = get_model(args.vlm_model_name, wrapper=ProbabilisticTensor)
        if vl_model is None:
            raise RuntimeError(f"Failed to load VLM: {args.vlm_model_name}")
        print(f"VLM initialized in {time.time()-st:.2f}s")

        print("--- Models Initialized ---")
        return od_predictor, vl_model
    except Exception as e:
        print(f"\nError during model initialization: {e}")
        traceback.print_exc()
        return None, None


# ==============================================================================
# VQACache – dry-run stub for fast-cache prewarm
# ==============================================================================

class VQACache:
    """Records every unique (question, num_objects, type) score call without
    invoking the real VLM.  Used in the fast-cache prewarm pass."""

    def __init__(self, device):
        self.memory: set = set()
        self.device = device

    def score(self, image, bboxes, question, num_objects, *args, **kwargs):
        self.memory.add((question, num_objects, kwargs.get('type', None)))
        n = len(bboxes)
        tensor_object = torch.rand(*[[n] * num_objects], device=self.device)
        return ProbabilisticTensor(tensor_object)

    def query(self, question, object_id=None, *args, **kwargs):
        return ""


# ==============================================================================
# Code Execution
# ==============================================================================

def execute_generated_code(
    code_str: str,
    question: str,
    vl_model_inst: Any,
    proposals: Optional[List[List[float]]],
    img: Image.Image,
    history: Optional[List[List[str]]],
    *,
    enable_fast_cache: bool = False,
    use_vlm_batch: bool = False,
) -> Tuple[Optional[str], Optional[Dict], Optional[str], Dict]:
    """Execute a logic_executor code string and return the answer string.

    Supports an optional two-pass fast-cache prewarm path when the VLM
    exposes ``cache_scores`` / ``clean_cache`` APIs.

    Returns
    -------
    result_answer_str, cache, error_msg, fast_cache_stats
    """
    spatial_relations = {}
    if proposals and hasattr(vl_model_inst, 'calculate_all_spatial_metrics'):
        spatial_relations = vl_model_inst.calculate_all_spatial_metrics(
            img, proposals, wrapper=ProbabilisticTensor
        )
    print(code_str)
    exec_context = {
        "ProbabilisticTensor": ProbabilisticTensor,
        "and_op": and_op,
        "or_op": or_op,
        **spatial_relations,
    }
    print(f"Executing symbolic program for: {question[:70]}...")

    error_msg = None
    cache = None
    result_answer_str = None
    fast_cache_stats: Dict[str, Any] = {
        "fast_cache_used": False,
        "fast_cache_unique_keys": 0,
        "fast_cache_prewarm_time": -1.0,
        "fast_cache_fallback_reason": None,
    }

    has_cache_api = (
        enable_fast_cache
        and bool(proposals)
        and hasattr(vl_model_inst, "cache_scores")
        and hasattr(vl_model_inst, "clean_cache")
    )

    if enable_fast_cache:
        if not bool(proposals):
            print("[fast-cache] SKIP — no proposals; running single-pass.")
        elif not has_cache_api:
            missing = [m for m in ("cache_scores", "clean_cache") if not hasattr(vl_model_inst, m)]
            print(f"[fast-cache] SKIP — model missing API(s): {missing}; running single-pass.")
        else:
            print(f"[fast-cache] ENABLED — {len(proposals)} proposal(s), use_batch={use_vlm_batch}.")
    else:
        print("[fast-cache] DISABLED — running single-pass.")

    try:
        exec(code_str, exec_context)
        logic_executor_func = exec_context["logic_executor"]

        ran_fast_path = False
        if has_cache_api:
            try:
                # Pass 1: dry-run with stub to collect score keys
                vqa_cache = VQACache(getattr(vl_model_inst, "device", "cpu"))
                cache_score_fn = lambda q, num_objects, type=None: vqa_cache.score(
                    img, proposals, q, num_objects, type=type
                )
                cache_query_fn = lambda q, object_id=None, mask=False, mask_outside=False, \
                                         draw_bbox=False, extract_object=False: vqa_cache.query(q, object_id)

                logic_executor_func(
                    question=question,
                    score=cache_score_fn,
                    query=cache_query_fn,
                    all_bounding_boxes=proposals,
                    image=img,
                    history=history,
                )

                n_keys = len(vqa_cache.memory)
                fast_cache_stats["fast_cache_unique_keys"] = n_keys
                print(
                    f"[fast-cache] dry-run done — {n_keys} unique score key(s): "
                    f"{sorted(vqa_cache.memory)}"
                )

                # Pass 1b: prewarm
                vl_model_inst.clean_cache()
                t_prewarm = time.time()
                vl_model_inst.cache_scores(
                    img, proposals, list(vqa_cache.memory),
                    with_mask=False, use_batch=use_vlm_batch,
                )
                fast_cache_stats["fast_cache_prewarm_time"] = time.time() - t_prewarm
                print(
                    f"[fast-cache] prewarm finished in "
                    f"{fast_cache_stats['fast_cache_prewarm_time']:.2f}s."
                )
                ran_fast_path = True

            except Exception as prewarm_err:
                reason = f"{type(prewarm_err).__name__}: {prewarm_err}"
                print(f"[fast-cache] prewarm failed — falling back to single-pass. {reason}")
                fast_cache_stats["fast_cache_fallback_reason"] = reason
                try:
                    vl_model_inst.clean_cache()
                except Exception:
                    pass

        # Pass 2: real execution
        mode = "fast-cache" if ran_fast_path else "single-pass"
        print(f"[fast-cache] executing in {mode} mode.")
        score_fn = build_score_fn(vl_model_inst, img, proposals, history)
        query_fn = build_query_fn(vl_model_inst, img, proposals)
        ProbabilisticTensor.start_cache()
        t_exec = time.time()
        result_answer_str = logic_executor_func(
            question=question,
            score=score_fn,
            query=query_fn,
            all_bounding_boxes=proposals,
            image=img,
            history=history,
        )
        if isinstance(result_answer_str, ProbabilisticTensor):
            result_answer_str = result_answer_str.tensor.item()
        cache = ProbabilisticTensor.end_cache()
        print(f"[fast-cache] logic_executor finished in {time.time() - t_exec:.2f}s ({mode}).")

        if ran_fast_path:
            fast_cache_stats["fast_cache_used"] = True

        if not isinstance(result_answer_str, str) and result_answer_str is not None:
            print(
                f"Warning: code returned {type(result_answer_str).__name__}, "
                f"converting: '{result_answer_str}'"
            )
            result_answer_str = str(result_answer_str)
        elif result_answer_str is None:
            print("Warning: code returned None.")

    except Exception as exec_err:
        error_msg = (
            f"Code Execution Error: {type(exec_err).__name__}: {exec_err}\n"
            f"Code:\n{code_str[:500]}..."
        )
        print(f"Code execution failed: {exec_err}")
        traceback.print_exc()

    return result_answer_str, cache, error_msg, fast_cache_stats


# ==============================================================================
# Results I/O
# ==============================================================================

def load_previous_results(results_file_path: str) -> Tuple[List[Dict], set, Dict[str, int]]:
    results: List[Dict] = []
    processed_ids: set = set()
    stats = {"correct_final": 0, "total_final": 0}

    if os.path.exists(results_file_path):
        print(f"Loading previous results from {results_file_path}...")
        try:
            with open(results_file_path, 'r') as f:
                loaded_data = json.load(f)
            if not isinstance(loaded_data, dict) or not isinstance(
                loaded_data.get("results"), list
            ):
                print("Results file format error. Starting fresh.")
                return [], set(), stats

            results = loaded_data["results"]
            for r in results:
                item_id = r.get("id")
                if item_id is not None:
                    processed_ids.add(str(item_id))
                    stats["total_final"] += 1
                    if r.get("correct_final_answer"):
                        stats["correct_final"] += 1

            print(f"Loaded {len(results)} results ({len(processed_ids)} unique IDs).")
            if stats["total_final"] > 0:
                print(
                    f"  Loaded accuracy: "
                    f"{stats['correct_final']/stats['total_final']:.2%} "
                    f"({stats['correct_final']}/{stats['total_final']})"
                )
        except Exception as e:
            print(f"Error loading results file: {e}. Starting fresh.")
            return [], set(), {"correct_final": 0, "total_final": 0}
    else:
        print("No previous results file found. Starting fresh.")
    return results, processed_ids, stats


def save_results_to_file(
    results_list: List[Dict],
    file_path: str,
    indent: Optional[int] = 2,
    description: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
):
    desc_to_save = description or "CLEVR Symbolic VQA Results"
    if args and args.num_splits > 1:
        desc_to_save += f" (Split {args.split_index + 1}/{args.num_splits})"
    label = "intermediate" if indent == 4 else "final"
    print(f"\nSaving {label} results ({len(results_list)} entries) to: {file_path}")
    try:
        with open(file_path, "w") as f:
            json.dump(
                {
                    "description": desc_to_save,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "results": results_list,
                },
                f,
                indent=indent,
            )
        print("Results saved.")
    except Exception as e:
        print(f"Error saving results: {e}")


def print_summary(
    start_time: float,
    samples_processed: int,
    skipped: int,
    total_original: int,
    num_in_split: int,
    newly_correct: int,
    newly_processed: int,
    args: argparse.Namespace,
):
    print("\n" + "=" * 30 + "\n--- Processing Summary (This Run) ---")
    elapsed = time.time() - start_time
    if args.num_splits > 1:
        print(f"Split {args.split_index + 1} of {args.num_splits}")
    print(f"Total dataset entries:          {total_original}")
    print(f"Entries in this split:          {num_in_split}")
    print(f"Skipped (already processed):    {skipped}")
    print(f"Newly processed this run:       {samples_processed}")
    if newly_processed > 0:
        print(
            f"  Accuracy this run: "
            f"{newly_correct/newly_processed:.2%} ({newly_correct}/{newly_processed})"
        )
        print(f"Total time: {elapsed:.2f}s  |  Avg per sample: {elapsed/samples_processed:.3f}s")
    print("--- End Summary ---")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    parser = setup_arg_parser()
    temp_args, _ = parser.parse_known_args()
    file_config = load_config_from_file(temp_args.config_file)
    if file_config:
        apply_config_file_to_parser_defaults(parser, file_config)
    args = parser.parse_args()

    # --- Validate splits ---
    if args.num_splits < 1:
        print(f"Error: --num_splits must be >= 1. Got {args.num_splits}")
        exit(1)
    if not (0 <= args.split_index < args.num_splits):
        print(
            f"Error: --split_index ({args.split_index}) out of range "
            f"[0, {args.num_splits - 1}]."
        )
        exit(1)

    # --- Load symbolic programs ---
    symbolic_programs: Dict[str, str] = load_symbolic_programs(args.symbolic_programs)

    # --- Load dataset ---
    print(f"\nLoading dataset: {args.dataset} ({args.num_samples} samples)...")
    full_dataset = get_dataset(args.dataset, num_samples=args.num_samples)
    if not full_dataset:
        print(f"Failed to load dataset '{args.dataset}'. Exiting.")
        exit(1)
    original_dataset_size = len(full_dataset)
    print(f"Dataset loaded: {original_dataset_size} samples.")

    # --- Determine split slice ---
    if args.num_splits == 1:
        dataset_to_process = full_dataset
        dataset_offset = 0
        print("Processing full dataset (num_splits=1).")
    else:
        print(f"Processing split {args.split_index + 1} of {args.num_splits}.")
        chunk_size = (original_dataset_size + args.num_splits - 1) // args.num_splits
        start_idx = args.split_index * chunk_size
        end_idx = min((args.split_index + 1) * chunk_size, original_dataset_size)
        if start_idx >= original_dataset_size:
            print("Split index out of range. Nothing to process.")
            exit(0)
        dataset_to_process = full_dataset[start_idx:end_idx]
        dataset_offset = start_idx
        print(f"Processing indices {start_idx}–{end_idx-1} ({len(dataset_to_process)} samples).")

    if not dataset_to_process:
        print("No samples to process. Exiting.")
        exit(0)

    # --- Results path ---
    if args.results_save_file:
        args.results_save_file = os.path.join(
            "experiments", args.dataset, args.vlm_model_name, args.results_save_file
        )
        results_dir = os.path.dirname(args.results_save_file)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {args.results_save_file}")

    results, processed_ids, cumulative_stats = load_previous_results(args.results_save_file)

    # --- Initialize models ---
    od_predictor, vl_model = initialize_models(args)
    if od_predictor is None or vl_model is None:
        print("Exiting due to model initialization failure.")
        exit(1)

    # --- Main processing loop ---
    print("\n" + "=" * 30 + "\n--- Starting CLEVR Symbolic Processing Loop ---")
    start_time_processing = time.time()
    samples_processed_this_run = 0
    samples_skipped_this_run = 0
    newly_correct_this_run = 0
    newly_processed_this_run = 0

    # Stats for programs not found in the symbolic map
    missing_program_count = 0

    progress_bar = tqdm(
        dataset_to_process,
        desc=(
            f"Split {args.split_index+1}/{args.num_splits} CLEVR"
            if args.num_splits > 1
            else "CLEVR Symbolic"
        ),
        total=len(dataset_to_process),
        dynamic_ncols=True,
    )

    for index_in_split, item_data in enumerate(progress_bar):
        item_id = item_data.get("id")
        original_item_index = dataset_offset + index_in_split

        if item_id is None:
            item_id = item_data.get("image_path", f"item_{original_item_index}")
            item_data["id"] = item_id

        if str(item_id) in processed_ids:
            samples_skipped_this_run += 1
            continue

        samples_processed_this_run += 1
        if hasattr(vl_model, "clean_cache"):
            vl_model.clean_cache()

        current_question = item_data.get("query")
        gt_text_answer = item_data.get("answer")

        if current_question is None or gt_text_answer is None:
            print(
                f"[{item_id}] Skipping — missing 'query' or 'answer'."
            )
            samples_skipped_this_run += 1
            processed_ids.add(str(item_id))
            results.append({
                "id": str(item_id),
                "error": "Missing question or GT answer",
                "original_index": original_item_index,
            })
            continue

        # Build per-sample result record (no PIL Image or masks stored)
        sample_result = {
            k: v for k, v in item_data.items()
            if k not in ("image", "masks")
        }
        sample_result["id"] = str(item_id)
        sample_result["original_question"] = current_question
        sample_result["ground_truth_answer"] = gt_text_answer
        sample_result["original_dataset_index"] = original_item_index
        sample_result.update({
            "error": "",
            "program_code": None,
            "symbolic_program_found": False,
            "code_generated_answer": None,
            "final_answer_text": None,
            "correct_final_answer": None,
            "execution_successful": False,
            "cache": None,
            "num_proposals": 0,
            "fast_cache_used": False,
            "fast_cache_unique_keys": 0,
            "fast_cache_prewarm_time": -1.0,
            "fast_cache_fallback_reason": None,
        })

        gt_answer_processed = unified_postprocess(gt_text_answer)

        try:
            # --- Load image ---
            image_input = item_data.get("image")
            if not isinstance(image_input, Image.Image):
                image_path = item_data.get("image_path", "")
                if image_path:
                    try:
                        image_input = Image.open(image_path).convert("RGB")
                    except FileNotFoundError:
                        raise ValueError(f"Image not found: {image_path}")
                    except UnidentifiedImageError:
                        raise ValueError(f"Cannot open image: {image_path}")
                else:
                    raise ValueError(f"Item {item_id} has no valid image or image_path.")

            # --- Look up symbolic program ---
            code_snippet = symbolic_programs.get(current_question)

            if code_snippet is None:
                missing_program_count += 1
                sample_result["error"] = (
                    f"No symbolic program found for question: {current_question[:100]}..."
                )
                print(f"[{item_id}] WARNING: no symbolic program found. "
                      f"(Total missing so far: {missing_program_count})")
            else:
                sample_result["symbolic_program_found"] = True
                code_to_execute = build_logic_executor_code(code_snippet)
                sample_result["program_code"] = code_to_execute

                # --- Bounding boxes: GT (default) or OD ---
                if args.use_gt_bboxes and item_data.get("bboxes") is not None:
                    all_proposals = item_data["bboxes"]
                    sample_result["num_proposals"] = len(all_proposals)
                    print(
                        f"[{item_id}] Using GT bboxes: {len(all_proposals)} objects."
                    )
                else:
                    # Fall back to OD with a generic "object" prompt for CLEVR
                    od_prompt = "object"
                    try:
                        proposal_tensor = od_predictor.predict(
                            image_input, text=od_prompt, apply_custom_nms=True
                        )
                        all_proposals = (
                            proposal_tensor.cpu().numpy().tolist()
                            if proposal_tensor is not None and proposal_tensor.numel() > 0
                            else []
                        )
                        sample_result["num_proposals"] = len(all_proposals)
                        print(
                            f"[{item_id}] OD generated {len(all_proposals)} proposals."
                        )
                    except Exception as od_err:
                        print(f"[{item_id}] OD failed: {od_err}")
                        all_proposals = []
                        sample_result["error"] = (
                            str(sample_result.get("error", "")) + f" OD Error: {od_err};"
                        ).strip()

                # --- Execute symbolic program ---
                newly_processed_this_run += 1
                code_ans, exec_cache, exec_error, fc_stats = execute_generated_code(
                    code_to_execute,
                    current_question,
                    vl_model,
                    expand_bboxes(all_proposals, image_input.width, image_input.height, scale=1.1),
                    image_input,
                    None,
                    enable_fast_cache=args.enable_fast_cache,
                    use_vlm_batch=args.use_vlm_batch,
                )
                sample_result["cache"] = exec_cache
                sample_result.update(fc_stats)
                sample_result["code_generated_answer"] = code_ans

                print(
                    f"[{item_id}] fast-cache: used={fc_stats['fast_cache_used']}, "
                    f"keys={fc_stats['fast_cache_unique_keys']}, "
                    + (
                        f"prewarm={fc_stats['fast_cache_prewarm_time']:.2f}s"
                        if fc_stats['fast_cache_prewarm_time'] >= 0
                        else f"fallback={fc_stats['fast_cache_fallback_reason']}"
                    )
                )

                if exec_error:
                    sample_result["error"] = (
                        str(sample_result.get("error", "")) + f" ExecError: {exec_error};"
                    ).strip()
                    sample_result["execution_successful"] = False
                elif code_ans is None:
                    sample_result["error"] = (
                        str(sample_result.get("error", "")) + " CodeReturnedNone;"
                    ).strip()
                    sample_result["execution_successful"] = True
                else:
                    sample_result["execution_successful"] = True
                    sample_result["final_answer_text"] = code_ans

                    # --- Accuracy ---
                    pred_processed = unified_postprocess(code_ans)
                    is_correct = pred_processed == gt_answer_processed
                    sample_result["correct_final_answer"] = is_correct
                    if is_correct:
                        newly_correct_this_run += 1
                    print(
                        f"[{item_id}] Pred: '{pred_processed}' | GT: '{gt_answer_processed}' | "
                        + ("CORRECT" if is_correct else "WRONG")
                    )

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = f"Runtime Error for {item_id}: {type(e).__name__}: {e}"
            sample_result["error"] = (
                (sample_result.get("error", "") + f"; {error_msg}").lstrip("; ")
            )
            sample_result["correct_final_answer"] = False
            print(f"[{item_id}] {error_msg}\n{tb_str}")

        results.append(sample_result)
        processed_ids.add(str(item_id))

        # Update progress bar postfix
        err_count = sum(
            1 for r in results[-samples_processed_this_run:]
            if r.get("error")
        )
        postfix: Dict[str, Any] = {"Err": err_count}
        if newly_processed_this_run > 0:
            postfix["Acc"] = f"{newly_correct_this_run/newly_processed_this_run:.2%}"
        progress_bar.set_postfix(postfix)

        if (
            args.save_periodically
            and samples_processed_this_run > 0
            and samples_processed_this_run % args.save_interval == 0
        ):
            save_results_to_file(
                results, args.results_save_file, indent=4,
                description=args.desc, args=args
            )

    progress_bar.close()

    if missing_program_count > 0:
        print(
            f"\nWARNING: {missing_program_count} questions had no symbolic program "
            f"in {args.symbolic_programs}."
        )

    print_summary(
        start_time_processing,
        samples_processed_this_run,
        samples_skipped_this_run,
        original_dataset_size,
        len(dataset_to_process),
        newly_correct_this_run,
        newly_processed_this_run,
        args,
    )

    if args.results_save_file:
        save_results_to_file(
            results, args.results_save_file, indent=2,
            description=args.desc, args=args
        )
