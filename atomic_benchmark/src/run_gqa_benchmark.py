#!/usr/bin/env python
import os
import torch
import argparse
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
from vision_agents import MarkerV2, get_model
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run yes/no GQA benchmark with vision-language model")
    parser.add_argument("--model", "-m", required=True, help="Model name or path to load (passed to get_model)")
    parser.add_argument("--bench", "-b", default="benchmarks/gqa_benchmark_templated_test.json", help="Path to benchmark JSON file")
    parser.add_argument("--out", "-o", default=None, help="Output JSON file (default: auto-generated)")
    parser.add_argument("--relation_only", action="store_true", help="Use only relation questions")
    parser.add_argument("--use_non_abstract", action="store_true", help="Use only non-abstract questions")
    parser.add_argument("--use_mixed_abstraction", action="store_true", help="Use only mixed abstraction questions")
    parser.add_argument("--multi_image", action="store_true", help="Use multiple images", default=False)
    parser.add_argument("--with_mask", action="store_true", help="Use with_mask scoring method", default=False)
    parser.add_argument("--no_cache", action="store_true", help="Do not cache images", default=False)
    parser.add_argument("--draw_mask", action="store_true", help="Draw mask on images", default=False)
    parser.add_argument("--image-root", "-i", default=os.path.join(os.environ.get("GQA_DIR", "data/gqa"), "images"), help="Directory containing GQA images")
    args = parser.parse_args()

    random.seed(425)

    with open(args.bench, "r") as f:
        new_data = json.load(f)

    model = get_model(args.model)
    yes_no_questions = [item for item in new_data if "yes" in item.get("answer", "").lower() or "no" in item.get("answer", "").lower()]
    if args.relation_only:
        yes_no_questions = [item for item in yes_no_questions if "relation" == item["type"]]
    if args.with_mask:
        out_file = args.out or f"{os.path.splitext(args.bench)[0]}_{args.model}_concept_with_mask_results.json"
    else:
        out_file = args.out or f"{os.path.splitext(args.bench)[0]}_{args.model}_concept_results.json"

    # Load previous results if available
    results = []
    seen = set()
    if os.path.exists(out_file) and not args.no_cache:
        with open(out_file, "r") as f:
            results = json.load(f)
        for entry in results:
            seen.add(entry["imageId"] + entry["question"])  # Composite key to detect duplicates

    # Track accuracy
    total = len(results)
    correct = sum(1 for r in results if r["short_answer"].lower() in r["predicted_answer"].lower())

    pbar = tqdm(yes_no_questions, desc="Processing", unit="item", dynamic_ncols=True)
    for index, item in enumerate(pbar):
        key = item["imageId"] + item["question"]
        if key in seen:
            continue  # Already processed

        image_path = os.path.join(args.image_root, f"{item['imageId']}.jpg")
        image = Image.open(image_path).convert("RGB")
        marked_image = model.marker.mark_objects(image, item["bboxes"], draw_bbox=True, draw_text=True, masks=args.draw_mask, overlay_masks=args.draw_mask)
        question = item["question"]

        if args.use_non_abstract and "non_abstract_question" in item:
            question = item["non_abstract_question"]
            target_image = image
        elif args.use_mixed_abstraction and "mixed_question" in item:
            question = item["mixed_question"]
            target_image = marked_image
        else:
            question = item["question"]
            target_image = marked_image
        if args.multi_image:
            question = "These are two similar images, "+ question
            target_image = [image, marked_image]
        else:
            target_image = marked_image
        
        if args.with_mask:
            try:
                score_val = model._score_with_mask(image, question, item["bboxes"], type=item["type2"])
            except ValueError as e:
                print(f"Error in _score_with_mask: {e}")
                score_val = 0
        else:
            question += "\nAnswer the question using a single word or phrase."
            score_val = model._score(target_image, question)
        
        if isinstance(score_val, torch.Tensor):
            score_val = score_val.detach().cpu().item()
        answer = item["answer"].lower()
        pred = "Yes" if score_val >= 0.5 else "No"
        print(f"Question: {question}, answer: {answer}, Score: {score_val:.4f}")

        if pred.lower() in item["answer"].lower():
            correct += 1
        total += 1
        accuracy = 100 * correct / total

        pbar.set_postfix({
            "Accuracy": f"{accuracy:.2f}%",
            "Correct": correct,
            "Total": total
        })

        result_entry = {
            **item,
            "predicted_answer": pred,
            "score": score_val
        }
        results.append(result_entry)
        if index % 100 == 0 and not args.no_cache:
            with open(out_file, "w") as f:
                json.dump(results, f, indent=4)
    # Save after each item to prevent loss on interruption
    if not args.no_cache:
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    main()
