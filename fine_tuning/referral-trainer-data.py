"""
referral-trainer-data.py

Generates training metadata for the NePTune fine-tuning pipeline on RefGTA.

For each sample:
  1. Loads image + expression from the dataset
  2. Generates a NePTune code snippet via the LLM code generator
  3. Runs Grounding DINO to get candidate bounding boxes
  4. Finds the ground-truth box index by IoU matching
  5. Saves metadata to OUTPUT_METADATA_FILE

Usage:
    python fine_tuning/referral-trainer-data.py [--num_samples 1000] [--force_regen]
"""

import os
import json
import time
import traceback
import argparse
from typing import Optional

import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import spacy
import sys
sys.path.append("../") # Ensure parent directory is in path for imports
from local_datasets import get_dataset
from vision_utils import object_detector
from code_generator import CodeGenerator

# ---------------------------------------------------------------------------
# spaCy model
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_nouns_and_phrases(text, return_phrases=True, return_individual_nouns=True):
    doc = nlp(text)
    result = {}
    if return_phrases:
        result["noun_phrases"] = [chunk.text for chunk in doc.noun_chunks if chunk.text.strip()]
    if return_individual_nouns:
        result["nouns"] = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop]
    return result


def calculate_iou(b1, b2):
    if b1 is None or b2 is None:
        return 0.0
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter) if inter > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate NePTune fine-tuning metadata for RefGTA.")
    parser.add_argument("--dataset", type=str, default="refgta")
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--code_gen_model", type=str, default="deepseek/deepseek-v3.2")
    parser.add_argument("--code_gen_provider", type=str, default="openrouter")
    parser.add_argument("--od_model", type=str, default="dino")
    parser.add_argument("--od_threshold", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--force_regen", action="store_true", help="Regenerate code even if cached.")
    parser.add_argument("--output", type=str, default=None, help="Output metadata JSON path.")
    parser.add_argument("--program_cache", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default="prompts/refexp.txt")
    args = parser.parse_args()

    output_file = args.output or f"data/training_{args.dataset}.json"
    program_cache = args.program_cache or f"programs/{args.dataset}_tune.json"

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # --- Load dataset ---
    print(f"Loading dataset: {args.dataset}")
    dataset = get_dataset(args.dataset)

    # --- Initialize OD ---
    print(f"Initializing OD ({args.od_model})...")
    try:
        od_model = object_detector(args.od_model, threshold=args.od_threshold)
        if od_model is None:
            raise ValueError("Object detector initialization returned None.")
    except Exception as e:
        print(f"Error: Failed to initialize OD model: {e}")
        traceback.print_exc()
        return

    # --- Initialize Code Generator ---
    print(f"Initializing Code Generator ({args.code_gen_model} via {args.code_gen_provider})...")
    try:
        code_generator = CodeGenerator(
            api_key=None,
            model_name=args.code_gen_model,
            program_cache_address=program_cache,
            prompt_file_address=args.prompt_file,
            update_cache=True,
            provider=args.code_gen_provider,
        )
    except Exception as e:
        print(f"Error: Failed to initialize Code Generator: {e}")
        traceback.print_exc()
        return

    print("--- Initialization Complete ---")

    training_metadata = []
    processed_count = 0
    error_count = 0

    total = min(args.num_samples, len(dataset))
    start_idx = len(dataset) - total
    ## starting from the end to avoid overlap with the test data
    print(f"\n--- Generating metadata for {total} samples (indices {start_idx}–{len(dataset)-1}) ---")

    for index in tqdm(range(start_idx, len(dataset)), desc="Generating Metadata"):
        if processed_count >= args.num_samples:
            break

        expression = dataset[index]["query"]
        image = dataset[index]["image"].convert("RGB")
        gt_bbox_xyxy = dataset[index]["answer"]
        image_width, image_height = image.size

        # Generate code snippet
        code_snippet, object_names = code_generator.generate_code(
            expression, force_generate=args.force_regen
        )

        # Build OD prompt from nouns/phrases in expression
        extraction = extract_nouns_and_phrases(expression)
        queries = set(extraction.get("nouns", []) + extraction.get("noun_phrases", []))
        if object_names:
            queries.update(object_names.split(","))
        queries.add(expression.strip().lower())
        od_prompt = ". ".join(sorted({q.strip().lower().replace("'", "") for q in queries if q.strip()})) + "."
        if not od_prompt.strip() or od_prompt == ".":
            od_prompt = "object."

        try:
            boxes_tensor = od_model.predict(image, od_prompt, apply_custom_nms=True, extra_bboxes=[gt_bbox_xyxy])
            candidate_bboxes = boxes_tensor.detach().cpu().numpy().tolist()
            max_w = image_width * 0.97
            max_h = image_height * 0.97
            candidate_bboxes = [b for b in candidate_bboxes if (b[2] - b[0]) < max_w and (b[3] - b[1]) < max_h]
        except Exception as e:
            print(f"W: Skipping index {index}: OD failed: {e}")
            error_count += 1
            continue

        if not candidate_bboxes:
            print(f"W: Skipping index {index}: No candidate boxes.")
            error_count += 1
            continue

        # Find GT index by IoU
        gt_index = -1
        max_iou = -1.0
        for i, box in enumerate(candidate_bboxes):
            iou = calculate_iou(box, gt_bbox_xyxy)
            if iou > max_iou and iou >= args.iou_threshold:
                max_iou = iou
                gt_index = i

        if gt_index == -1:
            print(f"W: Skipping index {index}: GT IoU too low ({max_iou:.2f}).")
            error_count += 1
            continue

        training_metadata.append({
            "image_path": dataset[index]["image_path"],
            "expression": expression,
            "candidate_bboxes": candidate_bboxes,
            "gt_index": gt_index,
            "code_snippet": code_snippet,
            "original_index": index,
            "gt_bbox_xyxy": gt_bbox_xyxy,
            "max_iou_for_gt": max_iou,
        })
        processed_count += 1

    print(f"\n--- Metadata Generation Finished ---")
    print(f"Successfully processed: {processed_count} samples")
    print(f"Skipped due to errors:  {error_count} samples")

    if training_metadata:
        print(f"Saving {len(training_metadata)} entries to: {output_file}")
        with open(output_file, "w") as f:
            json.dump(training_metadata, f, indent=2)
        print("Saved.")
    else:
        print("No metadata generated.")


if __name__ == "__main__":
    main()
