import argparse
import json
import math
import os
import torch
from typing import Dict, List, Tuple
from copy import deepcopy
import numpy as np
from tqdm import tqdm

from local_datasets import get_dataset
from probabilistic_tensor import ProbabilisticTensor
from vision_agents import get_model
from vision_utils import expand_bboxes


SCENE_FILE = os.path.join(os.getenv("CLEVR_DIR", "data/clevr"), "val", "scenes.json")
DEFAULT_OUTPUT = "vlm_atomic_results_new.jsonl"
DEFAULT_DATASET_NAME = "clevr"

def load_scenes(scene_file: str = SCENE_FILE) -> List[dict]:
    with open(scene_file, "r") as f:
        data = json.load(f)
    return data["scenes"]


def build_scene_index(scenes: List[dict]) -> Dict[str, dict]:
    return {scene["image_filename"]: scene for scene in scenes}


def collect_global_concepts(scenes: List[dict]) -> Dict[str, List[str]]:
    shapes, sizes, colors, materials = set(), set(), set(), set()
    for scene in scenes:
        for obj in scene["objects"]:
            shapes.add(obj["shape"])
            sizes.add(obj["size"])
            colors.add(obj["color"])
            materials.add(obj["material"])
    def _clean(values: set) -> List[str]:
        return [v for v in values if v != "None"]
    return {
        "shapes": _clean(shapes),
        "sizes": _clean(sizes),
        "colors": _clean(colors),
        "materials": _clean(materials),
    }


def align_bboxes_to_objects(scene: dict, bboxes: List[List[float]]) -> List[List[float]]:
    """Reorder bboxes to match object order based on pixel_coords."""
    objects = scene["objects"]
    object_centers = []
    for obj in objects:
        px, py = obj["pixel_coords"][:2]
        object_centers.append((px, py))

    bbox_centers = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        bbox_centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

    aligned_bboxes: List[List[float]] = []
    used = set()

    for obj_center in object_centers:
        min_dist = float("inf")
        best_idx = -1
        for idx, bbox_center in enumerate(bbox_centers):
            if idx in used:
                continue
            dist = math.sqrt((obj_center[0] - bbox_center[0]) ** 2 + (obj_center[1] - bbox_center[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        if best_idx == -1:
            aligned_bboxes.append([0, 0, 0, 0])
        else:
            aligned_bboxes.append(bboxes[best_idx])
            used.add(best_idx)

    return aligned_bboxes




def extract_concept_scores(scene: dict, global_concepts: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
    objects = scene["objects"]
    n_objects = len(objects)

    all_concepts: List[str] = []
    for concept_list in [global_concepts.get("shapes", []), global_concepts.get("sizes", []),
                         global_concepts.get("colors", []), global_concepts.get("materials", [])]:
        all_concepts.extend(concept_list)
    all_concepts = [c for c in all_concepts if c != "None"]

    result: Dict[str, List[int]] = {}
    for concept in all_concepts:
        scores = []
        for obj in objects:
            match = 1 if (
                obj["shape"] == concept or obj["size"] == concept or obj["color"] == concept or obj["material"] == concept
            ) else 0
            scores.append(match)
        result[concept] = scores

    if "relationships" in scene:
        relationships = scene["relationships"]
        for rel_type, rel_lists in relationships.items():
            scores = [[0 for _ in range(n_objects)] for _ in range(n_objects)]
            for obj_idx, target_indices in enumerate(rel_lists):
                for target_idx in target_indices:
                    if obj_idx != target_idx:
                        # scores[obj_idx][target_idx] = 1
                        scores[target_idx][obj_idx] = 1
            result[rel_type] = scores
    ## same_shape, same_size, same_color, same_material
    for attr in ["shape", "size", "color", "material"]:
        scores = [[0 for _ in range(n_objects)] for _ in range(n_objects)]
        for i in range(n_objects):
            for j in range(n_objects):
                if i != j and objects[i][attr] == objects[j][attr]:
                    scores[i][j] = 1
        result[f"same_{attr}"] = scores

    return result

dataset_prompt = """ The possible colors are: gray, blue, purple, brown, green, cyan, red, and yellow.
The possible sizes are: Small, Large
The possible materials are: Rubber and Metal
The possible shapes are: Sphere, Cube, Cylinder\n\n
"""
dataset_prompt = """"""
def evaluate_atomic_concepts(
    dataset_name: str = DEFAULT_DATASET_NAME,
    scene_file: str = SCENE_FILE,
    output_path: str = DEFAULT_OUTPUT,
    model_name: str = "qwen2-sam",
    num_scenes: int = None,
) -> None:
    scenes = load_scenes(scene_file)
    scene_index = build_scene_index(scenes)
    global_concepts = collect_global_concepts(scenes)

    dataset = get_dataset(dataset_name)
    vl_model = get_model(model_name)
    concept_to_query = {}
    for key in ["attribute","relation"]:
        for concept in vl_model.concept2query[key].keys():
            concept_to_query[concept] = vl_model.concept2query[key][concept]["query"]    
    seen_images = set()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tqdm_obj = tqdm(dataset, desc="Evaluating atomic concepts")
    with open(output_path, "w") as f_out:
        for item in dataset:
            image_filename = item["original_item"].get("image_filename")
            if image_filename in seen_images:
                continue
            seen_images.add(image_filename)
            scene = scene_index.get(image_filename)
            if scene is None:
                continue
            if num_scenes is not None and len(seen_images) > num_scenes:
                break
            concept_scores_gt = extract_concept_scores(scene, global_concepts)
            aligned_bboxes = align_bboxes_to_objects(scene, deepcopy(item["bboxes"]))
            print(f"Image: {image_filename}, Aligned BBoxes: {aligned_bboxes}")
            sizes = [(bbox[2]-bbox[0])*(bbox[3]-bbox[1]) for bbox in aligned_bboxes ]
            print(f"Image: {image_filename}, BBox sizes: {sizes}")
            if min(sizes) == 0:
                print("Skipping image with zero-area bbox")
                continue
            concepts_payload = {}
            aligned_bboxes = expand_bboxes(
                aligned_bboxes,
                image_width=item["image"].width,
                image_height=item["image"].height,
                scale=1.2,
                min_box_size=28,
            )

            croped_images = [item['image'].copy().crop(bbox) for bbox in aligned_bboxes]
            marked_images = [vl_model.marker.mark_objects(item['image'], [bbox], draw_bbox=True) for bbox in aligned_bboxes]
            ## save all cropped images
            if int(os.getenv("SAVE_DEBUG_IMAGES", "0")) == 1:
                for idx, croped_image in enumerate(croped_images):
                    debug_image_path = f"debug_atomic/{image_filename.replace('/', '_')}_obj{idx}.jpg"
                    os.makedirs(os.path.dirname(debug_image_path), exist_ok=True)
                    croped_image.save(debug_image_path)
                    debug_marked_image_path = f"debug_atomic/{image_filename.replace('/', '_')}_obj{idx}_marked.jpg"
                    marked_images[idx].save(debug_marked_image_path)
            for concept_index, (concept, query) in enumerate(concept_to_query.items()):
                tqdm_obj.set_postfix({"processing_image": image_filename, "concept": f"{concept}-{query} ({concept_index+1}/{len(concept_to_query)})"})
                # if concept in ["small", "large"]:
                #     scores = [float(vl_model._score(marked_image, dataset_prompt +  query + "\nAnswer the question using a single word or phrase.").cpu().detach().numpy().item()) for marked_image in marked_images]  
                # else:
                #     scores = [float(vl_model._score(croped_image, dataset_prompt +  query + "\nAnswer the question using a single word or phrase.").cpu().detach().numpy().item()) for croped_image in croped_images]
                scores = vl_model.score(item['image'], aligned_bboxes, dataset_prompt +  query + "\nAnswer the question using a single word or phrase.", num_objects=1 if concept not in ["left", "right", "front", "behind", "same_shape", "same_size", "same_color", "same_material"] else 2, draw_bbox=True)
                # print(f"Scores: {scores}")
                ## save bboxes
                # Convert scores to list of floats
                if isinstance(scores, torch.Tensor):
                    score_list = scores.detach().cpu().numpy().tolist()
                elif isinstance(scores, np.ndarray):
                    score_list = scores.tolist()
                else:
                    score_list = list(scores)
                
                gt_vec = concept_scores_gt.get(concept)
                if gt_vec is None:
                    continue
                concepts_payload[concept] = {
                    "scores": score_list,
                    "gt": gt_vec,
                }

            record = {
                "index": len(seen_images),
                "image_filename": image_filename,
                "concepts": concepts_payload,
            }
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            tqdm_obj.update(1)

    print(f"Saved VLM vs GT records to {output_path} ({len(seen_images)} unique scenes)")


def main():
    parser = argparse.ArgumentParser(description="Run atomic concept evaluation with VLM scores and GT concepts.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="Dataset name for local_datasets.get_dataset")
    parser.add_argument("--scene-file", default=SCENE_FILE, help="Path to scenes.json")
    parser.add_argument("--num_scenes", type=int, default=None, help="Optional limit on number of dataset samples to process")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--model", default="intern", help="Vision-language model name for vision_agents.get_model")
    # parser.add_argument("--model", default="intern", help="Vision-language model name for vision_agents.get_model")
    args = parser.parse_args()

    evaluate_atomic_concepts(
        dataset_name=args.dataset,
        scene_file=args.scene_file,
        output_path=args.output,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
