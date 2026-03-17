from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os
import json
import time
import re
import traceback
import datetime
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import argparse # Added for command-line arguments
from copy import deepcopy
# --- Vision and NLP Libraries ---
import torch
from PIL import Image, UnidentifiedImageError
# import matplotlib.pyplot as plt # Not used directly in the provided snippet for plotting
# import matplotlib.patches as patches # Not used directly
import numpy as np
# from openai import OpenAI # Not used if DeepSeek is the primary
import spacy
from tqdm import tqdm # For progress bar
from vision_utils import object_detector, expand_bboxes # Assuming object_detector is correctly imported
from vision_agents import get_model # Assuming get_model is correctly imported
from code_generator import CodeGenerator
from utils import unified_postprocess # Assuming these are in utils.py, crucial for VQA

from probabilistic_tensor import ProbabilisticTensor, and_op, or_op
from code_base_template import build_logic_executor_code, build_score_fn, build_query_fn


# --- Regex patterns ---
# The generated code inside logic_executor should now aim to return a string answer for VQA.
# --- Regex patterns ---


# ==============================================================================
# Configuration and Setup Functions
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
    """Sets up and returns the argument parser."""
    parser = argparse.ArgumentParser(description="Run Incremental Adversarial VQA Evaluation.", # Updated description
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--config_file', type=str,
        default="config_vqa.json", # Default VQA config file name
        help='Path to the JSON configuration file.'
    )
    parser.add_argument(
        '--dataset', type=str,
        help='Dataset name',
        choices=['clevr-humans', 'clevr'] # Example VQA datasets
    )
    parser.add_argument(
        '--prompt_file_address', type=str,
        help='Path to the code generation prompt template for VQA.'
    )
    parser.add_argument(
        '--results_save_file', type=str,
        help='Results file for incremental runs.'
    )
    parser.add_argument(
        '--program_cache_address', type=str,
        help='Path to program cache for VQA code generator.'
    )
    parser.add_argument(
        '--desc', type=str,
        help='Description for the run.'
    )
    parser.add_argument(
        '--vlm_model_name', type=str,
        help='Name of the VLM model.'
    )
    parser.add_argument(
        '--code_gen_model_name', type=str,
        help='Name of the Code Generation model.'
    )
    parser.add_argument(
        '--code_gen_provider', type=str,
        help="Provider for code generation model (e.g. 'deepseek', 'openrouter', 'openai')."
    )
    parser.add_argument(
        '--enable_reasoning',
        action=argparse.BooleanOptionalAction,
        help='Enable chain-of-thought reasoning for code generation (OpenRouter only).'
    )
    parser.add_argument(
        '--od_model', type=str,
        help="Object Detector model ID (e.g., 'dino', 'florence')."
    )
    parser.add_argument(
        '--od_box_threshold', type=float,
        help="Object Detector box confidence threshold."
    )
    parser.add_argument(
        '--enable_vlm_direct_prediction',
        action=argparse.BooleanOptionalAction,
        help='Enable VLM direct answer prediction.'
    )
    parser.add_argument(
        '--save_periodically', # Corrected attribute name from add_periodically
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable periodic saving of results.'
    )
    parser.add_argument(
        '--update_cache',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Update program cache for code generator.'
    )
    parser.add_argument(
        '--save_interval', type=int,
        help='How often to save periodically (number of samples).'
    )
    parser.add_argument(
        '--deepseek_api_key', type=str,
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help='DeepSeek API Key. Overrides environment variable and config file if set.'
    )
    parser.add_argument(
        '--base_results_file', type=str, default=None,
        help='Path to an existing merged results file whose IDs should be skipped '
             '(used in refinement loops to avoid re-processing already-correct items).'
    )
    # --- New arguments for splitting ---
    parser.add_argument(
        '--num_splits', type=int, default=1,
        help='Total number of splits for distributed processing. Default is 1 (no splitting).'
    )
    parser.add_argument(
        '--split_index', type=int, default=0,
        help='Index of the current split (0 to num_splits-1). Default is 0.'
    )
    parser.add_argument(
        '--num_samples',
        type=int, default=1000,
        help='Number of samples to process in each split.'
    )
    parser.add_argument(
        '--enable_fast_cache',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable two-pass fast score caching before code execution (does NOT batch dataset items).'
    )
    parser.add_argument(
        '--use_vlm_batch',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Use batched VLM scoring during cache prewarm (only active when --enable_fast_cache is set).'
    )

    parser.set_defaults(
        dataset='clevr-humans', # Default VQA dataset
        prompt_file_address="prompts/clevr-humans-prompt.txt", # VQA specific prompt
        results_save_file="results.json",
        program_cache_address="programs/clevr_cache.json",
        desc='VQA run',
        vlm_model_name="intern25_1b",
        code_gen_model_name="deepseek-chat",
        code_gen_provider="deepseek",
        enable_reasoning=False,
        od_model="dino",
        num_samples=1000, # Default number of samples per split
        od_box_threshold=0.25,
        save_periodically=True, # Explicitly set default
        update_cache=True,     # Explicitly set default
        save_interval=2, # Adjusted save interval
        deepseek_api_key=os.environ.get("DEEPSEEK_API_KEY"), # Use env var or None
        enable_fast_cache=False,
        use_vlm_batch=False,
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


def load_previous_results(results_file_path: str, args: argparse.Namespace) -> Tuple[List[Dict], set, Dict[str, int]]:
    results = []
    processed_ids = set()
    # Stats now refer to VQA answer correctness
    stats = {
        "correct_final": 0, "total_final": 0,
        "correct_vlm_direct": 0, "total_vlm_direct": 0,
        "correct_vlm_correction": 0, "total_vlm_correction": 0
    }
    if os.path.exists(results_file_path):
        print(f"Loading previous results from {results_file_path}...")
        try:
            with open(results_file_path, 'r') as f:
                loaded_data = json.load(f)
            if not isinstance(loaded_data, dict) or not isinstance(loaded_data.get("results", None), list):
                print("Results file format error. Starting fresh.")
                return [], set(), stats

            results = loaded_data.get("results", [])
            valid_ids = set()
            temp_stats = stats.copy()

            for r in results:
                item_id = r.get("id", None)
                if item_id:
                    valid_ids.add(str(item_id)) # Ensure IDs are strings for consistent set lookups
                    # Check for VQA correctness, 'correct' field should exist for valid, non-error entries
                else:
                    print(f"Skipping invalid entry in results file (missing id): {r}")

            processed_ids = valid_ids
            stats = temp_stats
            print(f"Loaded {len(results)} results. Found {len(processed_ids)} processed unique IDs.")
            if stats["total_final"] > 0: print(f"  Cumulative Final Acc from loaded: {stats['correct_final']/stats['total_final']:.2%} ({stats['correct_final']}/{stats['total_final']})")
            if stats["total_vlm_direct"] > 0: print(f"  Cumulative VLM Direct Acc from loaded: {stats['correct_vlm_direct']/stats['total_vlm_direct']:.2%} ({stats['correct_vlm_direct']}/{stats['total_vlm_direct']})")
            if stats["total_vlm_correction"] > 0: print(f"  Cumulative VLM Correction Acc from loaded: {stats['correct_vlm_correction']/stats['total_vlm_correction']:.2%} ({stats['correct_vlm_correction']}/{stats['total_vlm_correction']})")

        except Exception as e:
            print(f"Error loading/processing results file {results_file_path}: {e}. Starting fresh run.")
            return [], set(), {k: 0 for k in stats}
    else:
        print("No previous results file found. Starting fresh run.")
    return results, processed_ids, stats

def initialize_models(api_key: Optional[str], args: argparse.Namespace) -> Tuple[Optional[CodeGenerator], Optional[Any], Optional[Any]]:
    try:
        print("\n" + "=" * 30 + "\n--- Initializing Models ---")
        if not api_key and args.code_gen_model_name not in ["mock", "local_llm"]:
            print("DeepSeek API key is not set. Code generation might fail if using an API-based model.")

        code_generator = CodeGenerator(api_key=api_key,
                                       model_name=args.code_gen_model_name,
                                       program_cache_address=args.program_cache_address,
                                       prompt_file_address=args.prompt_file_address,
                                       update_cache=args.update_cache,
                                       provider=getattr(args, "code_gen_provider", "deepseek"),
                                       enable_reasoning=getattr(args, "enable_reasoning", False))
        print("Code Generator initialized.")

        print(f"Initializing OD ({args.od_model})..."); st_od = time.time()
        od_predictor = object_detector(args.od_model, threshold=args.od_box_threshold)
        print(f"OD initialized in {time.time()-st_od:.2f}s")

        print(f"Initializing VLM ({args.vlm_model_name})..."); st_vlm = time.time()
        # Wrapper might not be ProbabilisticTensor directly for VLM if its methods return strings for VQA
        vl_model = get_model(args.vlm_model_name, wrapper=ProbabilisticTensor)
        if vl_model is None:
            raise RuntimeError(f"Failed to load VLM: {args.vlm_model_name}")
        print(f"VLM initialized in {time.time()-st_vlm:.2f}s")
        print("--- Models Initialized ---")
        return code_generator, od_predictor, vl_model
    except Exception as e:
        print(f"\nError: Failed during model initialization: {e}")
        traceback.print_exc()
        return None, None, None

def generate_code_and_object_queries(code_generator_model: CodeGenerator, question: str, image: Image.Image, vl_model: Any, enable_vlm_suggestion: bool) -> Tuple[Optional[str], str, Optional[List[List[str]]]]:
    """Generates code snippet and object/concept queries for OD for a VQA task."""
    code_snippet, objects_str = code_generator_model.generate_code(question) # question is the new 'query'
    if not code_snippet or code_snippet.startswith("# Code generation failed"):
        return None, "", None

    objects_for_od = []
    if objects_str: # comma-separated concepts/objects from code gen
        objects_for_od = [o.strip() for o in objects_str.split(",") if o.strip()] if "," in objects_str else [objects_str.strip()]

    # For VQA, suggested concepts might be more useful than just object names
    concept_queries_base = set(objects_for_od)
    # Add nouns from the question for OD, if spacy is available
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Lightweight
        doc = nlp(question)
        nouns = [token.lemma_.lower() for token in doc if token.pos_ == "NOUN"]
        concept_queries_base.update(nouns)
    except ImportError:
        print("spaCy not installed or model not found. Skipping noun extraction from question for OD.")
    except Exception as e:
        print(f"Error during spacy noun extraction: {e}")


    concept_queries_final = {q.strip().lower().replace("'", "") for q in concept_queries_base if q.strip()}
    # Sort by length for GroundingDINO or similar ODs
    gd_prompt = ". ".join(sorted(list(concept_queries_final), key=len, reverse=True)) + "."
    if not gd_prompt.strip() or gd_prompt == '.': gd_prompt = "object. entity." # Fallback for OD
    
    return code_snippet, gd_prompt



class VQACache:
    """Lightweight dry-run stub that records every unique (question, num_objects, type)
    score call made by a logic_executor without invoking the real VLM.  Used in the
    fast-cache prewarm pass so all required scores can be batched in one GPU call."""

    def __init__(self, device):
        self.memory: set = set()
        self.device = device

    def score(self, image, bboxes, question, num_objects, *args, **kwargs):
        self.memory.add((question, num_objects, kwargs.get('type', None)))
        n = len(bboxes)
        tensor_object = torch.rand(*[[n] * num_objects], device=self.device)
        return ProbabilisticTensor(tensor_object)

    def query(self, question, object_id=None, *args, **kwargs):
        # Stub — returns empty string so dry-run completes without VLM call
        return ""


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
    """Executes the dynamically generated code for VQA, expecting a string answer.

    When *enable_fast_cache* is True and the VLM exposes ``cache_scores`` /
    ``clean_cache`` APIs a two-pass strategy is used:

      1. Dry-run with ``VQACache`` stub — collects every unique
         (question, num_objects, type) score key the program needs.
      2. ``vl_model_inst.cache_scores(...)`` — precomputes all those scores in
         one (optionally batched) VLM call.
      3. Real execution via ``build_score_fn`` / ``build_query_fn`` which read
         from the warm cache.

    Falls back transparently to single-pass if fast-cache is disabled, the
    model APIs are absent, or the prewarm step raises any exception.

    Returns
    -------
    result_answer_str, cache, error_msg, fast_cache_stats
    """
    spatial_relations = {}
    if proposals and hasattr(vl_model_inst, 'calculate_all_spatial_metrics'):
        spatial_relations = vl_model_inst.calculate_all_spatial_metrics(img, proposals, wrapper=ProbabilisticTensor)

    exec_context = {"ProbabilisticTensor": ProbabilisticTensor, "and_op": and_op, "or_op": or_op, **spatial_relations}
    print(f"Executing code for VQA question: {question[:50]}...\n{code_str}")

    error_msg = None
    cache = None
    result_answer_str = None
    fast_cache_stats: Dict[str, Any] = {
        "fast_cache_used": False,
        "fast_cache_unique_keys": 0,
        "fast_cache_prewarm_time": -1.0,
        "fast_cache_fallback_reason": None,
    }

    # Determine whether the fast-cache path can be attempted
    has_cache_api = (
        enable_fast_cache
        and bool(proposals)  # nothing to cache if no proposals
        and hasattr(vl_model_inst, "cache_scores")
        and hasattr(vl_model_inst, "clean_cache")
    )

    if enable_fast_cache:
        if not bool(proposals):
            print("[fast-cache] SKIP — no proposals available; running single-pass.")
        elif not hasattr(vl_model_inst, "cache_scores") or not hasattr(vl_model_inst, "clean_cache"):
            missing = [m for m in ("cache_scores", "clean_cache") if not hasattr(vl_model_inst, m)]
            print(f"[fast-cache] SKIP — model missing API(s): {missing}; running single-pass.")
        else:
            print(
                f"[fast-cache] ENABLED — {len(proposals)} proposal(s), "
                f"use_batch={use_vlm_batch}."
            )
    else:
        print("[fast-cache] DISABLED — running single-pass execution.")

    try:
        exec(code_str, exec_context)
        logic_executor_func = exec_context["logic_executor"]

        ran_fast_path = False
        if has_cache_api:
            try:
                # === Pass 1: dry-run with stub cache to collect score keys ===
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
                ratio = f"{n_keys}/{len(proposals)} proposals" if proposals else "N/A"
                print(
                    f"[fast-cache] dry-run done — {n_keys} unique score key(s) collected "
                    f"({ratio}): {sorted(vqa_cache.memory)}"
                )

                # === Prewarm — batch all collected score keys in one VLM call ===
                vl_model_inst.clean_cache()
                t_prewarm = time.time()
                vl_model_inst.cache_scores(
                    img, proposals, list(vqa_cache.memory),
                    with_mask=False, use_batch=use_vlm_batch,
                )
                fast_cache_stats["fast_cache_prewarm_time"] = time.time() - t_prewarm
                print(
                    f"VQA cache prewarm finished in {fast_cache_stats['fast_cache_prewarm_time']:.2f}s "
                    f"(use_batch={use_vlm_batch})."
                )
                ran_fast_path = True

            except Exception as prewarm_err:
                reason = f"{type(prewarm_err).__name__}: {prewarm_err}"
                print(f"Fast-cache prewarm failed — falling back to single-pass. {reason}")
                fast_cache_stats["fast_cache_fallback_reason"] = reason
                # Leave model cache clean for the real pass
                try:
                    vl_model_inst.clean_cache()
                except Exception:
                    pass

        # === Pass 2 (real execution, or single-pass if fast-cache skipped) ===
        mode = "fast-cache" if ran_fast_path else "single-pass"
        print(f"[fast-cache] starting real execution in {mode} mode.")
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
        cache = ProbabilisticTensor.end_cache()
        print(f"[fast-cache] logic_executor finished in {time.time() - t_exec:.2f}s ({mode}).")

        if ran_fast_path:
            fast_cache_stats["fast_cache_used"] = True

        if not isinstance(result_answer_str, str) and result_answer_str is not None:
            print(
                f"Warning: Generated code returned type {type(result_answer_str)}, expected str. "
                f"Converting: '{str(result_answer_str)}'"
            )
            result_answer_str = str(result_answer_str)
        elif result_answer_str is None:
            print("Warning: Generated code returned None.")

    except Exception as exec_err:
        error_msg = f"Code Execution Error: {type(exec_err).__name__}: {exec_err}\nCode:\n{code_str[:500]}..."
        print(f"Code execution failed: {exec_err}")
        traceback.print_exc()

    return result_answer_str, cache, error_msg, fast_cache_stats



# ==============================================================================
# Summary and Saving Functions
# ==============================================================================
def print_processing_summary_this_run(start_time: float, samples_processed: int, skipped: int, total_original_dataset_samples: int,
                                      newly_correct: Dict[str, int], newly_processed: Dict[str, int], args: argparse.Namespace,
                                      num_in_split: int): # Added num_in_split
    print("\n" + "=" * 30 + "\n--- Processing Summary (This Run) ---")
    total_time_this_run = time.time() - start_time
    if args.num_splits > 1:
        print(f"Split {args.split_index + 1} of {args.num_splits}")
    print(f"Total entries in original dataset file: {total_original_dataset_samples}")
    print(f"Entries assigned to this split: {num_in_split}")
    print(f"Entries skipped in this split (already processed or missing ID): {skipped}")
    print(f"Entries newly processed THIS RUN (in this split): {samples_processed}")


    if samples_processed > 0:
        if newly_processed.get("final", 0) > 0:
            run_final_accuracy = (newly_correct.get("final", 0) / newly_processed["final"] * 100)
            print(f"  Accuracy (Final Pipeline) THIS RUN: {run_final_accuracy:.2f}% ({newly_correct.get('final',0)}/{newly_processed['final']})")

        if args.enable_vlm_direct_prediction and newly_processed.get("vlm_direct", 0) > 0:
            run_vlm_accuracy = (newly_correct.get("vlm_direct", 0) / newly_processed["vlm_direct"] * 100)
            print(f"  Accuracy (VLM Direct) THIS RUN: {run_vlm_accuracy:.2f}% ({newly_correct.get('vlm_direct',0)}/{newly_processed['vlm_direct']})")

        print(f"Processing time THIS RUN: {total_time_this_run:.2f} seconds")
        if samples_processed > 0 : print(f"Avg time per new sample THIS RUN: {total_time_this_run / samples_processed:.3f} seconds")
    print("--- End of Processing Summary (This Run) ---")

def save_results_to_file(results_list: List[Dict], file_path: str, indent: Optional[int] = 2, description: Optional[str] = None, args: Optional[argparse.Namespace] = None):
    desc_to_save = description if description else "VQA Evaluation Results"
    if args and args.num_splits > 1:
        desc_to_save += f" (Split {args.split_index + 1}/{args.num_splits})"

    print(f"\nSaving {'intermediate ' if indent==4 else 'final combined '}results ({len(results_list)} total entries) to: {file_path}")
    try:
        with open(file_path, "w") as f:
            json.dump(
                {
                    "description": desc_to_save,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "results": results_list
                }, f, indent=indent)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error: Failed to save results to {file_path}: {e}")


from local_datasets import get_dataset # Assuming this is your dataset loading utility
# ==============================================================================
# Main Execution Logic
# ==============================================================================
if __name__ == '__main__':
    parser = setup_arg_parser()
    temp_args, _ = parser.parse_known_args()
    file_config = load_config_from_file(temp_args.config_file)
    if file_config:
        apply_config_file_to_parser_defaults(parser, file_config)
    args = parser.parse_args()

    # --- Validate split arguments ---
    if args.num_splits < 1:
        print(f"Error: --num_splits must be at least 1. Got {args.num_splits}")
        exit(1)
    if not (0 <= args.split_index < args.num_splits):
        print(f"Error: --split_index ({args.split_index}) must be between 0 and --num_splits-1 ({args.num_splits-1}).")
        exit(1)

    print(f"Loading dataset: {args.dataset}...")
    full_dataset = get_dataset(args.dataset, num_samples=args.num_samples) # Must provide 'image', 'query' (question), 'answer' (text)
    if not full_dataset:
        print(f"Failed to load dataset '{args.dataset}'. Exiting.")
        exit(1)
    
    original_dataset_size = len(full_dataset)
    print(f"Full dataset loaded. Total samples: {original_dataset_size}")

    # --- Determine the subset of the dataset for this split ---
    if args.num_splits == 1:
        dataset_to_process = full_dataset
        dataset_offset = 0 # For fallback ID generation
        print("Processing the full dataset (num_splits=1).")
    else:
        print(f"Processing split {args.split_index + 1} of {args.num_splits}.")
        chunk_size = (original_dataset_size + args.num_splits - 1) // args.num_splits # Ceiling division
        start_idx = args.split_index * chunk_size
        end_idx = min((args.split_index + 1) * chunk_size, original_dataset_size)
        
        if start_idx >= original_dataset_size:
            print(f"Split index {args.split_index} is out of bounds for dataset size {original_dataset_size} with {args.num_splits} splits. No data to process for this split.")
            dataset_to_process = []
            dataset_offset = start_idx
        else:
            dataset_to_process = [full_dataset[i] for i in range(start_idx, end_idx)]
            dataset_offset = start_idx # Store the starting index of this split in the original dataset
            print(f"This split will process samples from index {start_idx} to {end_idx-1} (total {len(dataset_to_process)} samples).")

    if not dataset_to_process:
        print("No samples to process for this split. Exiting.")
        # Optionally save an empty results file or log this state
        # For now, just exit.
        exit(0)


    # Ensure results directory exists
    if args.results_save_file:
        args.results_save_file = os.path.join("experiments", args.dataset, args.vlm_model_name, args.results_save_file)
        results_dir = os.path.dirname(args.results_save_file)
        if results_dir: # Ensure it's not an empty path if file is in current dir
             os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {args.results_save_file}")

    results, processed_ids, cumulative_stats = load_previous_results(args.results_save_file, args)

    # Load IDs from base results file (e.g. already-correct items from a refinement loop)
    if args.base_results_file and os.path.exists(args.base_results_file):
        try:
            with open(args.base_results_file) as _bf:
                _base = json.load(_bf)
            _base_items = _base.get("results", _base) if isinstance(_base, dict) else _base
            _base_ids = {str(r["id"]) for r in _base_items if r.get("id")}
            new_skips = _base_ids - processed_ids
            processed_ids |= _base_ids
            print(f"Loaded {len(_base_ids)} IDs from base results file "
                  f"({len(new_skips)} additional items will be skipped).")
        except Exception as _e:
            print(f"Warning: could not load base results file {args.base_results_file}: {_e}")
    # Unpack cumulative stats (these are counts, not accuracies directly)
    correct_final, total_final = cumulative_stats["correct_final"], cumulative_stats["total_final"]
    correct_vlm_direct, total_vlm_direct = cumulative_stats["correct_vlm_direct"], cumulative_stats["total_vlm_direct"]


    code_generator, od_predictor, vl_model = initialize_models(args.deepseek_api_key, args)
    if not all([code_generator, od_predictor, vl_model]):
        print("Exiting due to model initialization failure.")
        exit(1)

    print("\n" + "=" * 30 + "\n--- Starting VQA Processing Loop ---")
    if args.num_splits > 1:
        print(f"--- Running for Split {args.split_index + 1}/{args.num_splits} ---")
    start_time_processing = time.time()
    samples_processed_this_run = 0
    samples_skipped_this_run = 0 

    newly_correct_counts = defaultdict(int)
    newly_processed_counts = defaultdict(int)

    progress_bar_desc = "Processing VQA Samples"
    if args.num_splits > 1:
        progress_bar_desc = f"Split {args.split_index+1}/{args.num_splits} VQA"

    progress_bar = tqdm(dataset_to_process, desc=progress_bar_desc, total=len(dataset_to_process), dynamic_ncols=True)

    for index_in_split, item_data in enumerate(progress_bar):
        item_id = item_data.get("id")
        original_item_index = dataset_offset + index_in_split # Global index in the original full_dataset

        if item_id is None:
            # Fallback ID uses original_item_index for global uniqueness across splits
            item_id = item_data.get("image_path", f"item_{original_item_index}")
            print(f"Warning: Item at original index {original_item_index} (split index {index_in_split}) missing 'id'. Using generated ID: {item_id}")
            item_data["id"] = item_id # Add it to item_data for consistency


        if str(item_id) in processed_ids: # Ensure comparison is consistent (e.g. string vs string)
            samples_skipped_this_run += 1
            continue

        samples_processed_this_run += 1
        if hasattr(vl_model, "clean_cache"):
            vl_model.clean_cache()
        current_question = item_data.get("query") 
        gt_text_answer = item_data.get("answer") 

        if current_question is None or gt_text_answer is None:
            print(f"Skipping item {item_id} (original index {original_item_index}) due to missing 'query' or 'answer'.")
            samples_skipped_this_run +=1 
            processed_ids.add(str(item_id))
            results.append({"id": str(item_id), "error": "Missing question or GT answer", "query": current_question, "gt_answer": gt_text_answer, "original_index": original_item_index})
            continue

        sample_result = deepcopy(item_data)
        if "image" in sample_result: 
            del sample_result["image"]
        if "masks" in sample_result:
            del sample_result["masks"]
        sample_result["original_question"] = current_question
        sample_result["ground_truth_answer"] = gt_text_answer
        sample_result["original_dataset_index"] = original_item_index # Store original index


        sample_result.update({
            "error": "",
            "program_code": None,
            "vlm_direct_answer": None, "vlm_direct_correct": None,
            "code_generated_answer": None, 
            "verified_code_answer": None, 
            "final_answer_text": None, "correct_final_answer": None, 
            "execution_successful": False,
            "cache": None, 
            "od_prompt_used": None, 
            "num_proposals": 0,
            # Fast-cache diagnostics (populated by execute_generated_code)
            "fast_cache_used": False,
            "fast_cache_unique_keys": 0,
            "fast_cache_prewarm_time": -1.0,
            "fast_cache_fallback_reason": None,
        })
        
        gt_answer_processed = unified_postprocess(gt_text_answer)

        try:
            image_input = item_data["image"] 
            if not isinstance(image_input, Image.Image):
                if "image_path" in item_data and isinstance(item_data["image_path"], str):
                    try:
                        print(f"[{item_id}] Loading image from path: {item_data['image_path']}")
                        image_input = Image.open(item_data["image_path"]).convert("RGB")
                    except FileNotFoundError:
                        raise ValueError(f"Image file not found: {item_data['image_path']}")
                    except UnidentifiedImageError:
                        raise ValueError(f"Cannot identify image file: {item_data['image_path']}")
                else:
                    raise ValueError(f"Item {item_id} has invalid 'image' field. Expected PIL Image or valid 'image_path'.")

            newly_processed_counts["final"] += 1 

            final_answer_candidate = None
            MAX_CODE_RETRIES = 2
            last_exec_error = None
            last_code_snippet = None

            for code_attempt in range(MAX_CODE_RETRIES + 1):
                force_regen = code_attempt > 0
                if force_regen:
                    print(f"[{item_id}] Execution error on attempt {code_attempt}. Force-regenerating code with error context...")
                    # Force-regenerate passing previous code + error so the model can fix the bug
                    code_generator.generate_code(
                        current_question,
                        force_generate=True,
                        prev_code=last_code_snippet,
                        exec_error=last_exec_error,
                    )

                code_snippet, od_prompt = generate_code_and_object_queries(
                    code_generator, current_question, image_input, vl_model, None
                )

                sample_result["od_prompt_used"] = od_prompt

                if not code_snippet:
                    sample_result["error"] = f"Code generation failed for question: {current_question[:100]}..."
                    print(f"[{item_id}] {sample_result['error']}")
                    break

                code_to_execute = build_logic_executor_code(code_snippet)
                sample_result["program_code"] = code_to_execute

                all_proposals_from_od = None
                if item_data.get("bboxes", None) is None:
                    if od_prompt and od_prompt.strip() != ".":
                        try:
                            proposal_tensor = od_predictor.predict(image_input, text=od_prompt, apply_custom_nms=True)
                            all_proposals_from_od = proposal_tensor.cpu().numpy().tolist() if proposal_tensor is not None and proposal_tensor.numel() > 0 else []
                            sample_result["num_proposals"] = len(all_proposals_from_od)
                            print(f"[{item_id}] OD generated {len(all_proposals_from_od)} proposals using prompt: '{od_prompt[:100]}...'")
                        except Exception as od_err:
                            print(f"[{item_id}] OD proposal generation failed: {od_err}")
                            sample_result["error"] = (str(sample_result.get("error", "")) + " OD Error; ").strip()
                else:
                    all_proposals_from_od = item_data["bboxes"]
                    if isinstance(all_proposals_from_od, list) and all(isinstance(p, list) for p in all_proposals_from_od):
                        sample_result["num_proposals"] = len(all_proposals_from_od)
                        print(f"[{item_id}] Using provided OD proposals: {len(all_proposals_from_od)} proposals.")
                    else:
                        print(f"[{item_id}] Invalid 'bboxes' format. Expected list of lists, got {type(all_proposals_from_od)}.")
                all_proposals_from_od = expand_bboxes(all_proposals_from_od, image_input.width, image_input.height, scale=1.1)
                code_generated_ans_str, exec_cache, exec_error, fast_cache_result = execute_generated_code(
                    code_to_execute, current_question, vl_model, all_proposals_from_od, image_input, None,
                    enable_fast_cache=args.enable_fast_cache,
                    use_vlm_batch=args.use_vlm_batch,
                )
                sample_result["cache"] = exec_cache
                sample_result.update(fast_cache_result)
                sample_result["code_generated_answer"] = code_generated_ans_str
                print(
                    f"[{item_id}] fast-cache summary — "
                    f"used={fast_cache_result['fast_cache_used']}, "
                    f"keys={fast_cache_result['fast_cache_unique_keys']}, "
                    f"prewarm={fast_cache_result['fast_cache_prewarm_time']:.2f}s" if fast_cache_result['fast_cache_prewarm_time'] >= 0
                    else f"[{item_id}] fast-cache summary — "
                    f"used={fast_cache_result['fast_cache_used']}, "
                    f"keys={fast_cache_result['fast_cache_unique_keys']}, "
                    f"fallback={fast_cache_result['fast_cache_fallback_reason']}"
                )

                if exec_error:
                    sample_result["error"] = (str(sample_result.get("error", "")) + f" ExecError(attempt {code_attempt+1}): {str(exec_error)}; ").strip()
                    sample_result["execution_successful"] = False
                    last_exec_error = exec_error
                    last_code_snippet = code_snippet
                    if code_attempt < MAX_CODE_RETRIES:
                        continue  # retry with force-regenerated code
                    # Exhausted retries
                elif code_generated_ans_str is None:
                    sample_result["error"] = (str(sample_result.get("error", "")) + " CodeReturnedNone; ").strip()
                    sample_result["execution_successful"] = True
                else:
                    sample_result["execution_successful"] = True
                    final_answer_candidate = code_generated_ans_str
                    print(f"[{item_id}] Code Generated Answer: '{code_generated_ans_str}' (attempt {code_attempt+1})")
                break

            print("Ground Truth Answer Processed:", gt_answer_processed)
            sample_result["final_answer_text"] = final_answer_candidate

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"Runtime Error in main loop for {item_id}: {type(e).__name__}: {e}"
            if not sample_result.get("error"):
                 sample_result["error"] = error_message
            else: 
                 sample_result["error"] += f"; MainLoopError: {error_message}"
            print(f"[{item_id}] {error_message}\nTraceback: {tb_str}")
            
            if not ("total_final" in locals() and newly_processed_counts["final"] > 0 and total_final >= newly_processed_counts["final"]):
                total_final += 1 # Increment if not already counted in this attempt
            sample_result["correct_final_answer"] = False
            
            if args.enable_vlm_direct_prediction and sample_result.get("vlm_direct_answer") is None:
                if not ("total_vlm_direct" in locals() and newly_processed_counts["vlm_direct"] > 0 and total_vlm_direct >= newly_processed_counts["vlm_direct"]):
                    total_vlm_direct +=1
                sample_result["vlm_direct_correct"] = False
            
        
        results.append(sample_result)
        processed_ids.add(str(item_id))

        postfix_dict = {}
        current_run_errors = sum(1 for r_idx in range(len(results) - samples_processed_this_run, len(results)) if results[r_idx].get('error'))
        postfix_dict["Err (Run)"] = current_run_errors
        progress_bar.set_postfix(postfix_dict)

        if args.save_periodically and samples_processed_this_run > 0 and \
           samples_processed_this_run % args.save_interval == 0:
            save_results_to_file(results, args.results_save_file, indent=4, description=args.desc, args=args)

    progress_bar.close() 
    print_processing_summary_this_run(start_time_processing, samples_processed_this_run, samples_skipped_this_run,
                                      original_dataset_size, newly_correct_counts, newly_processed_counts, args,
                                      num_in_split=len(dataset_to_process))

    if args.results_save_file: 
        save_results_to_file(results, args.results_save_file, indent=2, description=args.desc, args=args)