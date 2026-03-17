import argparse
import json
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def load_records(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def compute_metrics(records: List[dict], threshold: float = 0.5) -> Dict[str, dict]:
    """Compute metrics matching the notebook approach using sklearn."""
    # Collect all predictions and ground truths for each concept
    concept_metrics = {}
    total_pred_pos = 0
    total_gt_pos = 0

    for rec in records:
        concepts = rec.get("concepts", {})
        for concept, payload in concepts.items():
            if concept not in concept_metrics:
                concept_metrics[concept] = {'predictions': [], 'ground_truth': []}
            
            scores = payload.get("scores", [])
            gt = payload.get("gt", [])
            
            if gt is None or len(scores) != len(gt):
                continue
            
            # Convert scores to binary predictions (threshold at 0.5)
            predictions = np.array(scores)
            predictions[predictions >= threshold] = 1.0
            predictions[predictions < threshold] = 0.0
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            total_pred_pos += sum(predictions)
            gt = np.array(gt)
            ## convert N by N 2D gt to a N times N 1d if needed
            if gt.ndim > 1:
                gt = gt.flatten()

            total_gt_pos += np.sum(gt)
            
            concept_metrics[concept]['predictions'].extend(predictions)
            concept_metrics[concept]['ground_truth'].extend(gt)

    # Calculate metrics for each concept
    results = {}
    for concept, data in concept_metrics.items():
        y_true = np.array(data['ground_truth'])
        y_pred = np.array(data['predictions'])
        
        # Handle edge cases explicitly to avoid being dropped from averages
        gt_pos = int(y_true.sum())
        pred_pos = int(y_pred.sum())
        if pred_pos == 0 and gt_pos == 0:
            precision = recall = f1 = 0.0
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results[concept] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(y_true),
            'gt_pos': gt_pos,
            'pred_pos': pred_pos,
        }

    # Calculate averages
    if results:
        avg_precision = np.mean([m['precision'] for m in results.values()])
        avg_recall = np.mean([m['recall'] for m in results.values()])
        avg_f1 = np.mean([m['f1'] for m in results.values()])
        total_support = sum([m['support'] for m in results.values()])
    else:
        avg_precision = avg_recall = avg_f1 = total_support = 0

    return {
        "per_concept": results,
        "average": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "total_support": total_support,
            "num_concepts": len(results)
        },
        "global_counts": {
            "total_pred_pos": total_pred_pos,
            "total_gt_pos": total_gt_pos,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compute per-concept and overall F1 from VLM vs GT JSONL.")
    parser.add_argument("--input", default="vlm_atomic_results_new.jsonl", help="Path to JSONL produced by atomic_concept_evaluation.py")
    parser.add_argument("--output", default=None, help="Optional path to save metrics JSON; if omitted, only prints")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold to binarize VLM scores")
    args = parser.parse_args()

    records = load_records(args.input)
    metrics = compute_metrics(records, threshold=args.threshold)

    # Print results in notebook format
    print("=" * 80)
    print(f"{'Concept':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("=" * 80)

    for concept in sorted(metrics["per_concept"].keys()):
        m = metrics["per_concept"][concept]
        print(f"{concept:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['support']:>10}")

    # Print averages
    avg = metrics["average"]
    print("=" * 80)
    print(f"{'Average':<20} {avg['precision']:>10.3f} {avg['recall']:>10.3f} "
          f"{avg['f1']:>10.3f} {avg['total_support']:>10}")
    print("=" * 80)
    
    print(f"\nTotal concepts evaluated: {avg['num_concepts']}")
    print(f"Total predictions: {avg['total_support']}")
    print(f"Threshold: {args.threshold}")

    counts = metrics.get("global_counts", {})
    if counts:
        print(f"Total predicted positives: {counts.get('total_pred_pos', 0)}")
        print(f"Total ground-truth positives: {counts.get('total_gt_pos', 0)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
