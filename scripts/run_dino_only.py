#!/usr/bin/env python3
"""
Run only the Grounding DINO stage of the NePTune pipeline over a folder of images
and save the detected boxes (plus optional visualisations).

Uses the same detector as runners/run_vqa.py: vision_utils.Dino.predict with
apply_custom_nms=True, then expand_bboxes(scale=1.1), which is what the VLM receives.

Usage
-----
python scripts/run_dino_only.py --num_images 20 --save_vis
python scripts/run_dino_only.py --num_images -1 --prompt "object. thing. shape."
"""
import argparse
import json
import os
import sys

from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vision_utils import object_detector, expand_bboxes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grounding DINO only and save boxes.")
    parser.add_argument("--images_dir", type=str, default="data/clevr/val/images")
    parser.add_argument("--scenes_file", type=str, default="data/clevr/val/scenes.json",
                        help="Optional CLEVR scenes file; adds the ground-truth object count per image.")
    parser.add_argument("--num_images", type=int, default=20, help="-1 = all images.")
    parser.add_argument("--prompt", type=str, default="object. thing.",
                        help="Grounding DINO text prompt ('.'-separated phrases).")
    parser.add_argument("--threshold", type=float, default=0.25, help="Box threshold (configs/clevr.json: od_box_threshold).")
    parser.add_argument("--expand_scale", type=float, default=1.1, help="Box expansion applied before the VLM in run_vqa.py.")
    parser.add_argument("--output_dir", type=str, default="experiments/clevr/dino_only")
    parser.add_argument("--save_vis", action="store_true", help="Also save images with boxes drawn.")
    return parser.parse_args()


def load_gt_counts(scenes_file: str) -> dict:
    if not scenes_file or not os.path.exists(scenes_file):
        return {}
    with open(scenes_file) as f:
        scenes = json.load(f)["scenes"]
    return {s["image_filename"]: len(s["objects"]) for s in scenes}


def draw_boxes(image: Image.Image, boxes: list, color: str = "red") -> Image.Image:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, y1 + 1), str(i), fill=color)
    return vis


def main() -> None:
    args = parse_args()
    image_files = sorted(f for f in os.listdir(args.images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if args.num_images > 0:
        image_files = image_files[:args.num_images]

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "vis")
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    gt_counts = load_gt_counts(args.scenes_file)
    od = object_detector("dino", threshold=args.threshold)

    detections = {}
    for filename in tqdm(image_files, desc="Grounding DINO"):
        image = Image.open(os.path.join(args.images_dir, filename)).convert("RGB")
        boxes = od.predict(image, text=args.prompt, apply_custom_nms=True)
        boxes = boxes.cpu().numpy().tolist() if boxes.numel() > 0 else []
        expanded = expand_bboxes(boxes, image.width, image.height, scale=args.expand_scale)

        detections[filename] = {
            "num_boxes": len(boxes),
            "gt_num_objects": gt_counts.get(filename),
            "boxes_xyxy": boxes,
            "boxes_expanded_xyxy": expanded,
        }
        if args.save_vis:
            draw_boxes(image, boxes).save(os.path.join(vis_dir, filename))

    output_file = os.path.join(args.output_dir, "detections.json")
    with open(output_file, "w") as f:
        json.dump({"prompt": args.prompt, "threshold": args.threshold,
                   "expand_scale": args.expand_scale, "detections": detections}, f, indent=2)

    with_gt = [d for d in detections.values() if d["gt_num_objects"] is not None]
    if with_gt:
        exact = sum(d["num_boxes"] == d["gt_num_objects"] for d in with_gt)
        print(f"Box count == GT object count: {exact}/{len(with_gt)} images")
    print(f"Saved detections for {len(detections)} images to {output_file}")
    if args.save_vis:
        print(f"Visualisations: {vis_dir}/")


if __name__ == "__main__":
    main()
