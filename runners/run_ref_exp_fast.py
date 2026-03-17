# ==============================================================================
# Imports
# ==============================================================================
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import os
import json
import time
# import pickle # No longer needed for refs.p
import re
# import random # Not used currently
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import argparse # Added for command-line arguments
from copy import deepcopy
# --- Vision and NLP Libraries ---
import torch
from PIL import Image
# import matplotlib.pyplot as plt # Not used directly in the provided snippet for plotting
# import matplotlib.patches as patches # Not used directly
import numpy as np
# from openai import OpenAI # Not used if DeepSeek is the primary
from tqdm import tqdm # For progress bar
from vision_utils import object_detector # Assuming object_detector is correctly imported
from vision_agents import get_model # Assuming get_model is correctly imported
from code_generator import CodeGenerator
from utils import calculate_iou # Assuming these are in utils.py
from local_datasets import get_dataset # Assuming this is your dataset loading utility
from probabilistic_tensor import ProbabilisticTensor, and_op, or_op


ProbabilisticTensor.and_op = "mul"
from code_base_template import build_logic_executor_code, build_score_fn, build_query_fn

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
    parser = argparse.ArgumentParser(description="Run Incremental Adversarial RefCOCO Evaluation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter) # Shows defaults in help

    # Config file argument first
    parser.add_argument(
        '--config_file', type=str,
        default="config.json", # Default config file name
        help='Path to the JSON configuration file.'
    )

    # Path Arguments
    # --------------
    parser.add_argument(
        '--dataset', type=str,
        # Default will be set by config file or remain 'refcoco-adv'
        help='Dataset name',
        choices=[
            'refgta', 'refcoco-adv', 'refcocog',
            'refcocoA', 'refcocoB', 'refcoco+val',
            'refcoco+A', 'refcoco+B', 'refcocoval'
        ]
    )
    parser.add_argument(
        '--num_samples', type=int, default=-1,
        help='Number of samples to process from the dataset.'
    )
    parser.add_argument(
        '--prompt_file_address', type=str,
        help='Path to the code generation prompt template.'
    )
    parser.add_argument(
        '--results_save_file', type=str,
        help='Results file for incremental runs.'
    )
    parser.add_argument(
        '--program_cache_address', type=str,
        help='Path to program cache for code generator.'
    )
    parser.add_argument(
        '--desc', type=str,
        help='Description for the run.'
    )

    # Model Settings
    # --------------
    parser.add_argument(
        '--vlm_model_name', type=str,
        help='Name of the VLM model.'
    )
    parser.add_argument(
        '--code_gen_model_name', type=str,
        help='Name of the Code Generation model.'
    )
    parser.add_argument(
        '--od_model', type=str, # Changed from od_model_id for consistency
        help="Object Detector model ID."
    )
    parser.add_argument(
        '--od_box_threshold', type=float,
        help="Object Detector box confidence threshold."
    )
    parser.add_argument(
        '--include_mask', action='store_true',
        help='Use with_mask scoring method for VLM.'
    )

    # Execution Controls (Booleans)
    # -----------------------------
    parser.add_argument(
        '--enable_vlm_object_suggestion',
        action=argparse.BooleanOptionalAction, # Allows --enable_vlm_object_suggestion / --no-enable_vlm_object_suggestion
        help='Enable VLM to suggest additional object names for OD.'
    )
    parser.add_argument(
        '--enable_vlm_direct_prediction', # Renamed for consistency with BooleanOptionalAction
        action=argparse.BooleanOptionalAction,
        help='Enable VLM direct prediction.'
    )
    parser.add_argument(
        '--enable_gd_direct_prediction', # Renamed
        action=argparse.BooleanOptionalAction,
        help='Enable Grounding DINO direct prediction.'
    )
    parser.add_argument(
        '--enable_vlm_verification', # Renamed
        action=argparse.BooleanOptionalAction,
        help='Enable VLM verification step.'
    )
    parser.add_argument(
        '--enable_vlm_correction', # Renamed
        action=argparse.BooleanOptionalAction,
        help='Enable VLM correction step.'
    )

    # Save Settings
    # -------------
    parser.add_argument(
        '--save_interval', type=int,
        help='How often to save periodically (number of samples).'
    )

    # Evaluation Settings
    # -------------------
    parser.add_argument(
        '--iou_threshold', type=float,
        help='IoU threshold for correctness.'
    )
    parser.add_argument(
        '--vlm_score_threshold', type=float,
        help='VLM score threshold for verification success.'
    )
    parser.add_argument(
        '--use_vlm_batch', type=bool,
        help='Use VLM batch processing.'
    )

    # API Keys
    # --------
    parser.add_argument(
        '--deepseek_api_key', type=str,
        default=os.environ.get("DEEPSEEK_API_KEY"), # Default to environment variable
        help='DeepSeek API Key. Overrides environment variable and config file if set.'
    )

    # Set initial defaults (these can be overridden by config file, then by CLI)
    # It's generally cleaner to let the config file specify these if they are not CLI-overridden.
    # For clarity, we list what they *would* be if not for the config file.
    parser.set_defaults(
        dataset='refcoco-adv',
        image_dir="data/images/",
        prompt_file_address="prompts/refexp.txt",
        results_save_file="results_test.json",
        program_cache_address="programs/refcoco-adv.json",
        desc='',
        vlm_model_name="intern",
        code_gen_model_name="deepseek-chat",
        od_model="dino",
        od_box_threshold=0.25,
        enable_vlm_object_suggestion=False,
        enable_vlm_direct_prediction=True,
        enable_gd_direct_prediction=True,
        enable_vlm_verification=False,
        enable_vlm_correction=False,
        save_periodically=True,
        update_cache=True,
        save_interval=1,
        num_samples=-1,
        iou_threshold=0.5,
        vlm_score_threshold=0.5,
        use_vlm_batch=False,
        deepseek_api_key=None
    )
    return parser

def apply_config_file_to_parser_defaults(parser: argparse.ArgumentParser, config_data: Dict):
    """Applies loaded config_data to parser's defaults."""
    # Create a mapping from dest name to action
    # This is necessary because some arguments might have different 'dest' than their flag name
    action_map = {action.dest: action for action in parser._actions}

    temp_defaults = {}
    for key, value in config_data.items():
        if key in action_map:
            # Handle BooleanOptionalAction specifically if needed, but set_defaults generally works.
            # If the key corresponds to an action, its type is already defined.
            # Type conversion should happen during parsing if CLI arg is provided,
            # or here if we want to be strict (but set_defaults is usually fine).
            temp_defaults[key] = value
        else:
            print(f"Config key '{key}' not found in parser arguments. Ignoring.")
    parser.set_defaults(**temp_defaults)


def load_previous_results(results_file_path: str, args: argparse.Namespace) -> Tuple[List[Dict], set, Dict[str, int]]:
    """Loads previous results from a JSON file and calculates initial statistics."""
    results = []
    processed_ids = set()
    stats = {
        "correct_final": 0, "total_final": 0,
        "correct_vlm_direct": 0, "total_vlm_direct": 0,
        "correct_od_direct": 0, "total_od_direct": 0,
        "correct_vlm_correction": 0, "total_vlm_correction": 0,
        "correct_execution": 0, "total_execution": 0
    }
    if os.path.exists(results_file_path):
        print(f"Loading previous results from {results_file_path}...")
        try:
            with open(results_file_path, 'r') as f:
                loaded_data = json.load(f)
            if not isinstance(loaded_data, dict) and not isinstance(loaded_data.get("results", None), list):
                print("Results file format error. Starting fresh.")
                return [], set(), stats

            results = loaded_data.get("results", [])
            valid_ids = set()
            temp_stats = stats.copy()

            for r in results:
                item_id = r.get('id')
                if item_id:
                    valid_ids.add(item_id)
                    if r.get("iou") is not None and not r.get("error"):
                        if r.get("correct") is True: temp_stats["correct_final"] += 1
                        temp_stats["total_final"] += 1
                    if args.enable_vlm_direct_prediction and r.get("vlm_direct_iou") is not None and not r.get("error"):
                        if r.get("vlm_direct_correct") is True: temp_stats["correct_vlm_direct"] += 1
                        temp_stats["total_vlm_direct"] += 1
                    if args.enable_gd_direct_prediction and r.get("od_direct_iou") is not None and not r.get("error"):
                        if r.get("od_direct_correct") is True: temp_stats["correct_od_direct"] += 1
                        temp_stats["total_od_direct"] += 1
                    if args.enable_vlm_correction and r.get("vlm_correction_iou") is not None and not r.get("error"):
                        if r.get("vlm_correction_correct") is True: temp_stats["correct_vlm_correction"] += 1
                        temp_stats["total_vlm_correction"] += 1
                    if r.get("execution_correct") is True: temp_stats["correct_execution"] += 1
                    temp_stats["total_execution"] += 1
                    
                else:
                    print(f"Skipping invalid entry in results file (missing id): {r}")

            processed_ids = valid_ids
            stats = temp_stats
            print(f"Loaded {len(results)} results. Found {len(processed_ids)} processed unique IDs.")
            if stats["total_final"] > 0: print(f"  Cumulative Final Acc from loaded: {stats['correct_final']/stats['total_final']:.2%} ({stats['correct_final']}/{stats['total_final']})")
            if stats["total_vlm_direct"] > 0: print(f"  Cumulative VLM Direct Acc from loaded: {stats['correct_vlm_direct']/stats['total_vlm_direct']:.2%} ({stats['correct_vlm_direct']}/{stats['total_vlm_direct']})")
            if stats["total_od_direct"] > 0: print(f"  Cumulative OD Direct Acc from loaded: {stats['correct_od_direct']/stats['total_od_direct']:.2%} ({stats['correct_od_direct']}/{stats['total_od_direct']})")
            if stats["total_vlm_correction"] > 0: print(f"  Cumulative VLM Correction Acc from loaded: {stats['correct_vlm_correction']/stats['total_vlm_correction']:.2%} ({stats['correct_vlm_correction']}/{stats['total_vlm_correction']})")
            if stats["total_execution"] > 0: print(f"  Cumulative Execution Success Acc from loaded: {stats['correct_execution']/stats['total_execution']:.2%} ({stats['correct_execution']}/{stats['total_execution']})")
        except Exception as e:
            print(f"Error loading/processing results file {results_file_path}: {e}. Starting fresh run.")
            return [], set(), {k: 0 for k in stats} # Reset all stats
    else:
        print("No previous results file found. Starting fresh run.")
    print(type(results))
    return results, processed_ids, stats

def initialize_models(api_key: Optional[str], args: argparse.Namespace) -> Tuple[Optional[CodeGenerator], Optional[Any], Optional[Any]]:
    """Initializes and returns the CodeGenerator, OD, and VLM models."""
    # try:
    print("\n" + "=" * 30 + "\n--- Initializing Models ---")
    if not api_key and args.code_gen_model_name not in ["mock", "local_llm"]: # Or other non-API models
        print("DeepSeek API key is not set. Code generation might fail if using an API-based model.")
        # Consider raising an error or allowing a specific "no-api" mode if appropriate
        # For now, we'll let it proceed and potentially fail at the CodeGenerator init or first call.

    code_generator = CodeGenerator(api_key=api_key,
                                    model_name=args.code_gen_model_name,
                                    program_cache_address=args.program_cache_address,
                                    prompt_file_address=args.prompt_file_address,
                                    update_cache=args.update_cache)
    print("Code Generator initialized.")

    print(f"Initializing OD ({args.od_model})..."); st_od = time.time()
    od_predictor = object_detector(args.od_model, threshold=args.od_box_threshold)
    print(f"OD initialized in {time.time()-st_od:.2f}s")

    print(f"Initializing VLM ({args.vlm_model_name})..."); st_vlm = time.time()
    vl_model = get_model(args.vlm_model_name, wrapper=ProbabilisticTensor)
    if vl_model is None:
        raise RuntimeError(f"Failed to load VLM: {args.vlm_model_name}")
    print(f"VLM initialized in {time.time()-st_vlm:.2f}s")
    print("--- Models Initialized ---")
    return code_generator, od_predictor, vl_model
    # except Exception as e:
    #     print(f"\nError: Failed during model initialization: {e}")
    #     return None, None, None

def generate_code_and_object_queries(code_generator_model: CodeGenerator, query: str, image: Image.Image, vl_model: Any, enable_vlm_suggestion: bool, caption: Optional[str]) -> Tuple[Optional[str], str, Optional[List[List[str]]]]:
    """Generates code snippet and object queries for OD."""
    code_snippet, objects_str = code_generator_model.generate_code_json(query, caption=caption, force_generate=False)
    if not code_snippet or code_snippet.startswith("# Code generation failed"):
        return None, "", None # Indicate failure clearly

    objects = []
    if objects_str:
        objects = [o.strip() for o in objects_str.split(",") if o.strip()] if "," in objects_str else [objects_str.strip()]

    object_queries_base = set(objects)
    object_queries_base.add(query) # Add the main query itself

    vlm_history = None
    if enable_vlm_suggestion:
        vlm_objects_prompt = f"""Extract potential object names relevant to the query: "{query}" for the program:\n{code_snippet}\nReturn a Python list of strings like ['obj1', 'obj2']."""
        # Ensure vl_model.query can handle this type of prompt and returns a string
        vlm_suggested_objects_str = vl_model.query(image, None, None, vlm_objects_prompt)
        vlm_history = [[vlm_objects_prompt, vlm_suggested_objects_str]]
        concepts_match = re.search(r"\[(.*?)\]", vlm_suggested_objects_str)
        if concepts_match:
            try:
                concepts_str = concepts_match.group(1)
                # Handle cases like "['obj1', 'obj2']" or "'obj1', 'obj2'"
                concepts = [c.strip().strip("'\"") for c in concepts_str.split(",") if c.strip()]
                object_queries_base.update(concepts)
            except Exception as e:
                print(f"Parsing VLM suggested objects failed: '{vlm_suggested_objects_str}' with error: {e}")

    object_queries_final = {q.strip().lower().replace("'", "").replace(".", "") for q in object_queries_base if q.strip()}
    # Sort by length to potentially give GroundingDINO better context for multi-object phrases
    gd_prompt = ". ".join(sorted(list(object_queries_final), key=len, reverse=True)) + "."
    if not gd_prompt.strip() or gd_prompt == '.': gd_prompt = "object." # Fallback
    elif len(gd_prompt) > 250 : gd_prompt = query # Fallback if prompt is too long

    return code_snippet, gd_prompt, vlm_history


def run_vlm_direct_prediction(vl_model: Any, image: Image.Image, query: str) -> Tuple[Optional[List[float]], int, int]: # Removed iou, correct
    """Runs VLM direct prediction."""
    # The prompt might need adjustment based on the VLM's expected format
    try:
        vlm_direct_box = vl_model.ground(image, query)
        newly_processed = 1 # Attempted
        return vlm_direct_box, 0, newly_processed # 0 for newly_correct as it's calculated outside
    except Exception as e:
        print(f"VLM direct prediction failed: {e}")
        return None, 0, 1 # Still count as attempted


def run_od_direct_prediction(od_predictor: Any, image: Image.Image, query: str) -> Tuple[Optional[List[float]], int, int]: # Removed iou, correct
    """Runs OD direct prediction."""
    try:
        
        gd_boxes_raw_tensor = od_predictor.predict_for_comparison(image, query)
        gd_box = gd_boxes_raw_tensor.detach().cpu().numpy().tolist()[0] if gd_boxes_raw_tensor.numel() > 0 else None

        newly_processed = 1 # Attempted
        return gd_box, 0, newly_processed # 0 for newly_correct as it's calculated outside
    except Exception as e:
        print(f"OD direct prediction failed: {e}")
        return [], 0, 1 # Still count as attempted

class Cache:
    def __init__(self, device, *args, **kwargs):
        self.memory = set()
        self.device = device

    def score(self, image, bboxes, question, num_objects, *args, **kwargs):
        self.memory.add((question, num_objects, kwargs.get('type', None)))
        n = len(bboxes)
        tensor_object = torch.rand(*[[n]*num_objects], device=self.device)
        return ProbabilisticTensor(tensor_object)         # vector of length n

    def query(self, question, num_objects, *args, **kwargs):
        # later you can decide what to return here
        return ""
def execute_generated_code(code_str: str, query: str, vl_model_inst: Any, proposals: List[List[float]], img: Image.Image, history: Optional[List[List[str]]], with_mask, use_vlm_batch=False) -> Tuple[Optional[ProbabilisticTensor], Optional[Dict], Optional[str]]:
    """Executes the dynamically generated code."""
    spatial_relations = vl_model_inst.calculate_all_spatial_metrics(img, proposals, wrapper=ProbabilisticTensor)
    exec_context = {"ProbabilisticTensor": ProbabilisticTensor, "and_op": and_op, "or_op": or_op, **spatial_relations}
    print(f"Executing code:\n{code_str}")
    error_msg = None
    cache = None
    result_tensor = None
    cache_obj = Cache(vl_model_inst.device)
    execution_time = -1.0
    try:
        print("Starting cache execution...")
        start_exec = time.time()
        exec(code_str, exec_context)
        logic_executor_func = exec_context["logic_executor"]
        
        cache_score_fn = lambda x, num_objects, type=None: cache_obj.score(
            image, proposals, x, num_objects, type=type
        )
        cache_query_fn = lambda x, object_id=None, mask=False, mask_outside=False, draw_bbox=False, extract_object=False: cache_obj.query(
            x, object_id
        )
        logic_executor_func(query, cache_score_fn, cache_query_fn, proposals, image, history)
        vl_model_inst.clean_cache()
        print("Caching scores for unique (question, num_objects, type) combinations...")
        start_cache = time.time()
        vl_model_inst.cache_scores(image, proposals, list(cache_obj.memory), with_mask=with_mask, use_batch=use_vlm_batch)
        end_cache = time.time()
        print(f"Cached {len(cache_obj.memory)} unique scores in {end_cache - start_cache:.2f}s")
        score_fn = build_score_fn(vl_model_inst, img, proposals, history, with_mask=with_mask)
        query_fn = build_query_fn(vl_model_inst, img, proposals)
        ProbabilisticTensor.start_cache()
        result_tensor = logic_executor_func(question=query, score=score_fn, query=query_fn,
                            all_bounding_boxes=proposals, image=img, history=history)
        cache = ProbabilisticTensor.end_cache()
        execution_time = time.time() - start_exec
        print(f"Code executed in {execution_time:.2f}s. Result tensor shape: {result_tensor.tensor.shape if result_tensor else 'None'}")
    except Exception as exec_err:
        import traceback
        error_msg = f"Code Execution Error: {type(exec_err).__name__}: {exec_err}\nCode:\n{code_str[:500]}..." # Include part of code
        print(f"Code execution failed: {exec_err}")
        print(traceback.format_exc())
    return result_tensor, cache, error_msg, execution_time


# ==============================================================================
# Summary and Saving Functions
# ==============================================================================
def print_processing_summary_this_run(start_time: float, samples_processed: int, skipped: int, total_dataset_samples: int,
                                      newly_correct: Dict[str, int], newly_processed: Dict[str, int], args: argparse.Namespace):
    """Prints the summary of the processing done in the current run."""
    print("\n" + "=" * 30 + "\n--- Processing Summary (This Run) ---")
    total_time_this_run = time.time() - start_time
    print(f"Total entries in dataset file: {total_dataset_samples}")
    print(f"Entries skipped (already processed): {skipped}")
    print(f"Entries newly processed THIS RUN: {samples_processed}")
    # Note: Cannot easily count errors "this run" from the global results list without start/end index.
    # Will rely on logs for per-sample errors.

    if samples_processed > 0:
        if newly_processed.get("final", 0) > 0:
            run_final_accuracy = (newly_correct.get("final", 0) / newly_processed["final"] * 100)
            print(f"  Accuracy (Final) THIS RUN: {run_final_accuracy:.2f}% ({newly_correct.get('final',0)}/{newly_processed['final']})")

        if args.enable_vlm_direct_prediction and newly_processed.get("vlm_direct", 0) > 0:
            run_vlm_accuracy = (newly_correct.get("vlm_direct", 0) / newly_processed["vlm_direct"] * 100)
            print(f"  Accuracy (VLM Direct) THIS RUN: {run_vlm_accuracy:.2f}% ({newly_correct.get('vlm_direct',0)}/{newly_processed['vlm_direct']})")

        if args.enable_gd_direct_prediction and newly_processed.get("od_direct", 0) > 0:
            run_od_accuracy = (newly_correct.get("od_direct", 0) / newly_processed["od_direct"] * 100)
            print(f"  Accuracy (OD Direct) THIS RUN: {run_od_accuracy:.2f}% ({newly_correct.get('od_direct',0)}/{newly_processed['od_direct']})")

        if args.enable_vlm_correction and newly_processed.get("vlm_correction", 0) > 0:
            run_vlm_corr_accuracy = (newly_correct.get("vlm_correction", 0) / newly_processed["vlm_correction"] * 100)
            print(f"  Accuracy (VLM Correction) THIS RUN: {run_vlm_corr_accuracy:.2f}% ({newly_correct.get('vlm_correction',0)}/{newly_processed['vlm_correction']})")

        print(f"Processing time THIS RUN: {total_time_this_run:.2f} seconds")
        if samples_processed > 0 : print(f"Avg time per new sample THIS RUN: {total_time_this_run / samples_processed:.3f} seconds")
    print("--- End of Processing Summary (This Run) ---")

def save_results_to_file(results_list: List[Dict], file_path: str, args: argparse.Namespace, indent: Optional[int] = 2, description: Optional[str] = None):
    """Saves the results list to a JSON file."""
    print(f"\nSaving {'intermediate ' if indent==4 else 'final combined '}results ({len(results_list)} total entries) to: {file_path}")
    
    # Calculate cumulative API usage
    cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    has_usage = False
    for r in results_list:
        if "api_usage" in r and r["api_usage"]:
            has_usage = True
            for k in cumulative_usage:
                cumulative_usage[k] += r["api_usage"].get(k, 0)

    try:
        config_to_save = vars(deepcopy(args)) # Use deepcopy if you want to avoid modifying original args if you convert it elsewhere or it has complex objects. For basic types, vars(args) is often enough.
        data_to_save = {
            "description": description,
            "results": results_list,
            "config": config_to_save
        }
        if has_usage:
            data_to_save["cumulative_api_usage"] = cumulative_usage
            print(f"Cumulative API Usage: {cumulative_usage}")

        with open(file_path, "w") as f:
            json.dump(data_to_save, f, indent=indent)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error: Failed to save results to {file_path}: {e}")

def print_cumulative_summary(results_list: List[Dict], args: argparse.Namespace, total_dataset_samples: int, all_processed_ids: set):
    """Prints the final cumulative summary based on all results."""
    print("\n" + "=" * 30 + "\n--- Cumulative Results Summary (Based on Updated File) ---")
    final_num_results_total = len(results_list)
    final_num_errors_total = sum(1 for r in results_list if r.get("error"))

    # Calculate cumulative API usage
    cumulative_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}
    has_usage = False
    for r in results_list:
        if "api_usage" in r and r["api_usage"]:
            has_usage = True
            for k in cumulative_usage:
                cumulative_usage[k] += r["api_usage"].get(k, 0)
    
    if has_usage:
        print(f"Cumulative API Usage: {cumulative_usage}")

    metrics = {
        "Final": ("iou", "correct", "Final Pipeline Results:"),
        "VLM Direct": ("vlm_direct_iou", "vlm_direct_correct", f"VLM Direct Prediction Results ({args.vlm_model_name}):"),
        "OD Direct": ("od_direct_iou", "od_direct_correct", f"OD Direct Prediction Results (GD - {args.od_model}):"), # od_model not od_model_id
        "VLM Correction": ("vlm_correction_iou", "vlm_correction_correct", f"VLM Correction Results ({args.vlm_model_name}):"),
        "Execution": ("execution_iou", "execution_correct", "Execution Results:")
    }

    print(f"Dataset: {args.dataset}")
    print(f"Total entries in dataset file: {total_dataset_samples}")
    print(f"Total result entries in file: {final_num_results_total}")
    print(f"Total processed unique IDs tracked: {len(all_processed_ids)}")
    print(f"Total entries with errors logged: {final_num_errors_total}")
    print("-" * 20)

    for key, (iou_key, correct_key, title) in metrics.items():
        enabled = False
        if key == "Final": enabled = True
        elif key == "VLM Direct" and args.enable_vlm_direct_prediction: enabled = True
        elif key == "OD Direct" and args.enable_gd_direct_prediction: enabled = True
        elif key == "VLM Correction" and args.enable_vlm_correction: enabled = True

        if enabled:
            valid_results = [r for r in results_list if not r.get("error") and r.get(iou_key) is not None]
            num_successful = len(valid_results)
            num_correct = sum(1 for r in valid_results if r.get(correct_key) is True)
            accuracy = (num_correct / num_successful * 100) if num_successful > 0 else 0.0
            valid_ious = [r[iou_key] for r in valid_results if iou_key in r and isinstance(r[iou_key], (int, float))]
            average_iou = np.mean(valid_ious) if valid_ious else 0.0

            print(title)
            print(f"  Valid Executions/Predictions (IoU calculable): {num_successful}")
            print(f"  Correct Predictions (IoU >= {args.iou_threshold}): {num_correct}")
            print(f"  Overall Cumulative Accuracy ({key}): {accuracy:.2f}%")
            print(f"  Overall Average IoU ({key}, {len(valid_ious)} valid): {average_iou:.4f}")
            print("-" * 20)
    print("=" * 30 + "\n--- Run Finished ---")

def batched_dataset_generator(dataset, processed_ids, batch_size, vl_model):
    unprocessed = [item for item in dataset if item.get("id") not in processed_ids and item.get("id") is not None]
    for i in range(0, len(unprocessed), batch_size):
        batch = unprocessed[i:i+batch_size]
        images = [item['image'] for item in batch]
        captions = [None] * len(batch)
        # try:
        #      if hasattr(vl_model, '_query_batch'):
        #         captions = vl_model._query_batch(images, ["Please provide a detailed caption for this image."]*len(images))
        #      else:
        #         captions = [vl_model._query(img, "Please provide a detailed caption for this image.") for img in images]
        # except Exception as e:
        #     print(f"Batch captioning error: {e}")
        
        for item, caption in zip(batch, captions):
            yield item, caption

# ==============================================================================
# Main Execution Logic
# ==============================================================================
if __name__ == '__main__':
    # 1. Create parser
    parser = setup_arg_parser()

    # 2. Load config from file (if specified and exists)
    # We need to parse known args to get config_file path first
    temp_args, _ = parser.parse_known_args()
    file_config = load_config_from_file(temp_args.config_file)

    # 3. Apply file_config to parser defaults
    if file_config:
        apply_config_file_to_parser_defaults(parser, file_config)

    # 4. Parse all arguments (CLI will override file-based defaults)
    args = parser.parse_args()

    # --- Dataset Loading (using the configured dataset name) ---
    print(f"Loading dataset: {args.dataset}...")
    dataset = get_dataset(args.dataset, num_samples=args.num_samples)
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    args.results_save_file = os.path.join("experiments", args.dataset, args.vlm_model_name, args.results_save_file)
    ## if doens't exist, create the directory
    os.makedirs(os.path.dirname(args.results_save_file), exist_ok=True)
    print(f"Results will be saved to: {args.results_save_file}")
    results, processed_ids, cumulative_stats = load_previous_results(args.results_save_file, args)
    correct_final, total_final = cumulative_stats["correct_final"], cumulative_stats["total_final"]
    correct_vlm_direct, total_vlm_direct = cumulative_stats["correct_vlm_direct"], cumulative_stats["total_vlm_direct"]
    correct_od_direct, total_od_direct = cumulative_stats["correct_od_direct"], cumulative_stats["total_od_direct"]
    correct_vlm_correction, total_vlm_correction = cumulative_stats["correct_vlm_correction"], cumulative_stats["total_vlm_correction"]

    code_generator, od_predictor, vl_model = initialize_models(args.deepseek_api_key, args)
    if not all([code_generator, od_predictor, vl_model]):
        print("Exiting due to model initialization failure.")
        exit(1)

    print("\n" + "=" * 30 + "\n--- Starting Processing Loop ---")
    start_time_processing = time.time()
    samples_processed_this_run = 0
    samples_skipped = 0

    newly_correct_counts = defaultdict(int)
    newly_processed_counts = defaultdict(int)

    # dataset_items should now be populated
    print(f"Previously processed IDs: {processed_ids}")
    
    BATCH_SIZE = 8
    dataset_gen = batched_dataset_generator(dataset, processed_ids, BATCH_SIZE, vl_model)
    total_unprocessed = sum(1 for item in dataset if item.get("id") not in processed_ids and item.get("id") is not None)
    samples_skipped = len(dataset) - total_unprocessed

    progress_bar = tqdm(dataset_gen, desc="Processing Samples", total=total_unprocessed, dynamic_ncols=True)
    
    for index_in_loop, (item_data, precomputed_caption) in enumerate(progress_bar): # Use index_in_loop for unique identification if needed for tqdm
        item_id = item_data.get("id")
        vl_model.clean_cache()
        if hasattr(vl_model, "reset_usage"):
            vl_model.reset_usage()
        
        samples_processed_this_run += 1

        sample_result = deepcopy(item_data) # Start with a copy of the item data
        # Remove potentially large objects or non-serializable items before processing starts
        if "image" in sample_result:
            del sample_result["image"]

        sample_result.update({
            "error": None,
            "program_code": None,
            "vlm_direct_bboxes_xyxy": None,
            "od_direct_bbox_xyxy": None,
            "vlm_correction_bboxes_xyxy": None,
            "vlm_correction_iou": None,
            "vlm_correction_correct": None,
            "vlm_direct_iou": None,
            "vlm_direct_correct": None,
            "od_direct_iou": None,
            "od_direct_correct": None,
            "final_bbox_xyxy": None,
            "iou": None,
            "correct": None,
            "execution_successful": False, # Assume false until proven true
            "num_proposals": 0,
            "all_proposal_bboxes": [],
            "cache": None,
            "gd_prompt_used": None
        })
        actual_bbox_xyxy = None

        try:
            image, query, actual_bbox_xyxy = item_data["image"], item_data["query"], item_data["answer"]
            sample_result["gt_bbox_xyxy"] = actual_bbox_xyxy
            sample_result["query"] = query # This is the actual expression

            # This sample is valid for processing (has GT)
            newly_processed_counts["final"] += 1 # For the main pipeline attempt
            caption = precomputed_caption
            code_snippet, gd_prompt, vlm_history_obj_suggestion = generate_code_and_object_queries(
                code_generator, query, image, vl_model, args.enable_vlm_object_suggestion, caption=None
            )
            sample_result["gd_prompt_used"] = gd_prompt # Save the GD prompt

            if not code_snippet:
                error_message = f"Code generation failed for query: {query[:100]}..."
                sample_result["error"] = error_message
                print(f"[{item_id}] {index_in_loop}  {error_message}")
            else:
                code_to_execute = build_logic_executor_code(code_snippet)
                
                sample_result["program_code"] = code_to_execute # Save the generated program

                vlm_direct_final_box = None # For fallback
                if args.enable_vlm_direct_prediction:
                    newly_processed_counts["vlm_direct"] +=1
                    total_vlm_direct += 1 # Count attempt
                    vlm_direct_box, _, _ = run_vlm_direct_prediction(vl_model, image, query)
                    sample_result["vlm_direct_bboxes_xyxy"] = [vlm_direct_box] if vlm_direct_box else None
                    if vlm_direct_box:
                        vlm_iou = calculate_iou(vlm_direct_box, actual_bbox_xyxy)
                        sample_result["vlm_direct_iou"] = vlm_iou
                        is_correct_vlm = (vlm_iou >= args.iou_threshold)
                        sample_result["vlm_direct_correct"] = is_correct_vlm
                        if is_correct_vlm: correct_vlm_direct += 1; newly_correct_counts["vlm_direct"] += 1
                        vlm_direct_final_box = vlm_direct_box
                    else:
                        sample_result["vlm_direct_correct"] = False; sample_result["vlm_direct_iou"] = 0.0
                    print(f"[{item_id}] {index_in_loop}  VLM Direct Prediction. IoU: {sample_result['vlm_direct_iou']:.2f}, Correct: {sample_result['vlm_direct_correct']}")

                final_selected_bbox = vlm_direct_final_box
                if args.enable_gd_direct_prediction:
                    newly_processed_counts["od_direct"] += 1
                    total_od_direct += 1 # Count attempt
                    gd_box, _, _ = run_od_direct_prediction(od_predictor, image, query)
                    sample_result["od_direct_bbox_xyxy"] = gd_box if gd_box else None # Store as single box or None
                    if gd_box:
                        od_iou = calculate_iou(gd_box, actual_bbox_xyxy)
                        sample_result["od_direct_iou"] = od_iou
                        is_correct_od = (od_iou >= args.iou_threshold)
                        sample_result["od_direct_correct"] = is_correct_od
                        if is_correct_od: correct_od_direct += 1; newly_correct_counts["od_direct"] += 1
                    else:
                        sample_result["od_direct_correct"] = False; sample_result["od_direct_iou"] = 0.0
                    print(f"[{item_id}] {index_in_loop}  OD Direct Prediction. IoU: {sample_result['od_direct_iou']:.2f}, Correct: {sample_result['od_direct_correct']}")


                # Proposal Generation
                extracted_objects_for_vlm_query = [o.strip() for o in gd_prompt.split('.') if o.strip() and len(o.strip()) > 1] # Basic split
                if not extracted_objects_for_vlm_query: extracted_objects_for_vlm_query = [query] # Fallback

                vlm_proposed_objects = [vlm_direct_final_box] if vlm_direct_final_box else [] # Start with VLM direct box if available
                history_for_code_exec = None
                if args.enable_vlm_object_suggestion: # Reuse the flag or a new one for this stage
                    try:
                        vlm_object_query, vlm_object_query_answer = vl_model.suggest_objects(image, query, extracted_objects_for_vlm_query)
                        # Assuming suggest_objects might return structured data or update history
                        history_for_code_exec = [[vlm_object_query, vlm_object_query_answer]]
                         # If suggest_objects also returns bbox suggestions, integrate them here
                        # for label, bboxes_for_label in vlm_object_query_answer_all_objects.items():
                        #     vlm_proposed_objects.extend(bboxes_for_label) # Assuming bboxes_for_label is a list of xyxy boxes
                    except Exception as vlm_sugg_err:
                        print(f"[{item_id}] {index_in_loop}  VLM object suggestion failed: {vlm_sugg_err}")


                try:
                    proposal_tensor = od_predictor.predict(image, text=gd_prompt, apply_custom_nms=True, extra_bboxes=vlm_proposed_objects if vlm_proposed_objects else None)
                    all_proposal_bboxes = proposal_tensor.cpu().numpy().tolist()
                except Exception as od_predict_err:
                    print(f"[{item_id}] {index_in_loop}  OD proposal prediction failed: {od_predict_err}")
                    all_proposal_bboxes = [] # Ensure it's an empty list on failure


                sample_result["num_proposals"] = len(all_proposal_bboxes)
                sample_result["all_proposal_bboxes"] = all_proposal_bboxes
                print(f"[{item_id}] {index_in_loop}  Generated {len(all_proposal_bboxes)} proposals.")


                final_selected_bbox_from_code = None
                result_tensor_from_code_exec = None

                if not all_proposal_bboxes:
                    print(f"[{item_id}] {index_in_loop}  No proposals for query '{query[:50]}...'. Falling back to VLM direct if enabled.")
                    final_selected_bbox_from_code = vlm_direct_final_box # Fallback
                    sample_result["execution_successful"] = False
                    if not sample_result["error"]: sample_result["error"] = "No proposals generated by OD."
                elif len(all_proposal_bboxes) == 1:
                    final_selected_bbox_from_code = all_proposal_bboxes[0]
                    sample_result["execution_successful"] = True # Technically code wasn't run, but a single proposal is a direct choice
                    print(f"[{item_id}] {index_in_loop}  Only 1 proposal, skipping code execution, using the single proposal.")
                else:
                    result_tensor_from_code_exec, exec_cache, exec_error, execution_time = execute_generated_code(
                        code_to_execute, query, vl_model, all_proposal_bboxes, image, history_for_code_exec, args.include_mask, 
                        use_vlm_batch=args.use_vlm_batch
                    )
                    sample_result["cache"] = exec_cache
                    sample_result["code_execution_time"] = execution_time
                    if exec_error:
                        sample_result["error"] = exec_error # Overwrites previous "No proposals" if code exec also fails

                    if result_tensor_from_code_exec is None or not isinstance(result_tensor_from_code_exec, ProbabilisticTensor) or result_tensor_from_code_exec.tensor is None or result_tensor_from_code_exec.tensor.numel() == 0 or result_tensor_from_code_exec.tensor.shape[-1] != len(all_proposal_bboxes):
                        print(f"[{item_id}] {index_in_loop}  Code exec returned invalid or empty tensor ({type(result_tensor_from_code_exec)}, shape {result_tensor_from_code_exec.tensor.shape if result_tensor_from_code_exec and isinstance(result_tensor_from_code_exec, ProbabilisticTensor) and result_tensor_from_code_exec.tensor is not None else 'N/A'}) for query '{query[:50]}...'. Fallback to VLM direct if enabled.")
                        final_selected_bbox_from_code = vlm_direct_final_box # Fallback
                        sample_result["execution_successful"] = False
                        if not sample_result["error"]: sample_result["error"] = f"Bad Exec Result: {type(result_tensor_from_code_exec)}"
                    else:
                        sample_result["execution_successful"] = True
                        try:
                            # Ensure tensor is 1D for argmax
                            tensor_for_argmax = result_tensor_from_code_exec.tensor.squeeze()

                            if tensor_for_argmax.ndim == 0: # Scalar tensor
                                best_index_exec = 0
                            else:
                                best_index_exec = tensor_for_argmax.argmax().item() # .item() to get Python number

                            if 0 <= best_index_exec < len(all_proposal_bboxes):
                                final_selected_bbox_from_code = all_proposal_bboxes[best_index_exec]
                                print(f"[{item_id}] {index_in_loop}  Code execution successful. Best box index: {best_index_exec}.")
                            else:
                                print(f"[{item_id}] {index_in_loop}  Argmax index {best_index_exec} out of bounds (proposals: {len(all_proposal_bboxes)}). Fallback.")
                                final_selected_bbox_from_code = vlm_direct_final_box # Fallback
                                sample_result["execution_successful"] = False
                                if not sample_result["error"]: sample_result["error"] = "Argmax index out of bounds."
                        except Exception as argmax_err:
                            print(f"[{item_id}] {index_in_loop}  Error getting argmax: {argmax_err}. Fallback.")
                            final_selected_bbox_from_code = vlm_direct_final_box # Fallback
                            sample_result["execution_successful"] = False
                            if not sample_result["error"]: sample_result["error"] = f"Argmax error: {argmax_err}"

                final_selected_bbox = final_selected_bbox_from_code
                if final_selected_bbox:
                    final_selected_bbox_tensor = torch.tensor(final_selected_bbox).unsqueeze(0)
                    final_selected_bbox_xyxy = final_selected_bbox_tensor.cpu().numpy().tolist()[0]
                    final_selected_bbox_xyxy = [float(coord) for coord in final_selected_bbox_xyxy] # Ensure float
                    sample_result["execution_selected_bbox_xyxy"] = final_selected_bbox_xyxy
                    final_iou = calculate_iou(final_selected_bbox_xyxy, actual_bbox_xyxy)
                    sample_result["execution_iou"] = final_iou
                    is_correct_final = (final_iou >= args.iou_threshold)
                    sample_result["execution_correct"] = is_correct_final
                    # if is_correct_final: correct_final += 1; newly_correct_counts["final"] += 1
                    print(f"[{item_id}] {index_in_loop}  Code execution IoU: {final_iou:.2f}, Correct: {is_correct_final}")

                # Final Pipeline Result (based on `final_selected_bbox` after code exec and optional verification)
                sample_result["final_bbox_xyxy"] = final_selected_bbox
                if final_selected_bbox and actual_bbox_xyxy:
                    final_iou = calculate_iou(final_selected_bbox, actual_bbox_xyxy)
                    sample_result["iou"] = final_iou
                    is_correct_final = (final_iou >= args.iou_threshold)
                    sample_result["correct"] = is_correct_final
                    total_final += 1 # Successfully processed for final pipeline
                    if is_correct_final: correct_final += 1; newly_correct_counts["final"] += 1
                    print(f"[{item_id}] {index_in_loop}  Final Pipeline Result. IoU: {final_iou:.2f}, Correct: {is_correct_final}")

                elif actual_bbox_xyxy: # GT exists, but no final box from pipeline
                    sample_result["iou"] = 0.0; sample_result["correct"] = False
                    total_final += 1 # Count as processed for final pipeline (failure case)
                    print(f"[{item_id}] {index_in_loop}  Final Pipeline Result: No box produced. IoU: 0.0, Correct: False")
                else: # No GT, cannot calculate final IoU/Correctness
                     sample_result["iou"] = None
                     sample_result["correct"] = False
                     # This case should ideally not happen if filtering by 'answer' != None initially
                     print(f"[{item_id}] {index_in_loop}  Final Pipeline Result: No GT available. Cannot calculate IoU/Correctness.")


        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"Runtime Error in main loop for {item_id}: {type(e).__name__}: {e}"
            if not sample_result.get("error"): # Don't overwrite more specific errors from code_exec etc.
                 sample_result["error"] = error_message
            print(f"[{item_id}] {index_in_loop}  {error_message}\nTraceback: {tb_str}") # Log full traceback

            # Ensure correctness is False on any error for all metrics IF GT was available
            if actual_bbox_xyxy is not None: # Check if GT was loaded before error
                # For final pipeline
                if sample_result.get("iou") is None: total_final +=1 # Count as an attempt if not already
                sample_result["correct"] = False
                sample_result["iou"] = sample_result.get("iou", 0.0) # Set IOU to 0 if not calculated

                # For VLM direct
                if args.enable_vlm_direct_prediction:
                    if sample_result.get("vlm_direct_iou") is None: # Check if successfully processed (had iou)
                        # If total_vlm_direct wasn't incremented because run_vlm_direct_prediction failed before the counter, increment it here.
                        # A simpler approach is to increment *before* the call and handle failure.
                        # Based on `run_vlm_direct_prediction`'s return (newly_processed = 1), it *does* count the attempt.
                        pass # Counter was already incremented
                    sample_result["vlm_direct_correct"] = False
                    sample_result["vlm_direct_iou"] = sample_result.get("vlm_direct_iou", 0.0) # Keep existing iou if available, else 0

                # For OD direct
                if args.enable_gd_direct_prediction:
                    if sample_result.get("od_direct_iou") is None: # Check if successfully processed (had iou)
                         pass # Counter was already incremented
                    sample_result["od_direct_correct"] = False
                    sample_result["od_direct_iou"] = sample_result.get("od_direct_iou", 0.0)

                # For VLM correction
                if args.enable_vlm_correction:
                     if sample_result.get("vlm_correction_iou") is None: # Check if successfully processed (had iou)
                          pass # Counter was already incremented
                     sample_result["vlm_correction_correct"] = False
                     sample_result["vlm_correction_iou"] = sample_result.get("vlm_correction_iou", 0.0)
        
        if hasattr(vl_model, "usage_stats"):
             sample_result["api_usage"] = vl_model.usage_stats.copy()

        results.append(sample_result)
        processed_ids.add(item_id) # Use item_id here

        postfix_dict = {}
        if total_final > 0: postfix_dict["Acc F"] = f"{(correct_final / total_final * 100):.1f}% ({correct_final}/{total_final})"
        else: postfix_dict["Acc F"] = "N/A"

        if args.enable_vlm_direct_prediction:
            if total_vlm_direct > 0: postfix_dict["VLM_D"] = f"{(correct_vlm_direct / total_vlm_direct * 100):.1f}% ({correct_vlm_direct}/{total_vlm_direct})"
            else: postfix_dict["VLM_D"] = "N/A"
        if args.enable_gd_direct_prediction:
            if total_od_direct > 0: postfix_dict["OD_D"] = f"{(correct_od_direct / total_od_direct * 100):.1f}% ({correct_od_direct}/{total_od_direct})"
            else: postfix_dict["OD_D"] = "N/A"
        if args.enable_vlm_correction:
            if total_vlm_correction > 0: postfix_dict["VLM_C"] = f"{(correct_vlm_correction / total_vlm_correction * 100):.1f}% ({correct_vlm_correction}/{total_vlm_correction})"
            else: postfix_dict["VLM_C"] = "N/A"
        # Count errors *this run* by looking at the last `samples_processed_this_run` results
        postfix_dict["Err (Run)"] = sum(1 for r in results[len(results) - samples_processed_this_run:] if r.get('error')) if samples_processed_this_run > 0 else 0

        progress_bar.set_postfix(postfix_dict)

        if args.save_periodically and samples_processed_this_run > 0 and \
           samples_processed_this_run % args.save_interval == 0:
            save_results_to_file(results, args.results_save_file, args, indent=4, description=args.desc) # Intermediate save

    # --- Post-Processing & Summary ---
    print_processing_summary_this_run(start_time_processing, samples_processed_this_run, samples_skipped,
                                      len(dataset), newly_correct_counts, newly_processed_counts, args)

    save_results_to_file(results, args.results_save_file, args, indent=4, description=args.desc) # Final save

    print_cumulative_summary(results, args, len(dataset), processed_ids)