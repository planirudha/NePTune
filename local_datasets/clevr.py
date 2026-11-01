import os
import json
from PIL import Image
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from pycocotools import mask as coco_mask # You may need to install pycocotools: pip install pycocotools

class CLEVRDataset(Dataset):
    """
    A PyTorch Dataset for loading questions, images, and bounding boxes
    from the CLEVR dataset.
    """
    def __init__(self, questions_path: str, scenes_path: str, images_dir: str, num_samples: Optional[int] = None):
        """
        Args:
            questions_path (str): Path to the CLEVR questions JSON file.
            scenes_path (str): Path to the CLEVR scenes JSON file.
            images_dir (str): Path to the directory containing the images.
            num_samples (int, optional): If specified, limits the dataset to the first `num_samples`.
        """
        # Load questions and scenes annotations
        print("Loading CLEVR questions...")
        with open(questions_path, 'r') as f:
            self.questions_data = json.load(f)['questions']
        
        print("Loading CLEVR scenes...")
        with open(scenes_path, 'r') as f:
            self.scenes_data = json.load(f)['scenes']

        self.images_dir = images_dir
        
        # Limit the number of samples if specified
        if num_samples is not None:
            self.questions_data = self.questions_data[:num_samples]

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.questions_data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves a single data point from the dataset.

        Args:
            idx (int): The index of the data point.

        Returns:
            A dictionary containing the image, question, answer, bboxes, and other metadata.
        """
        # Get the question item for the given index
        question_item = self.questions_data[idx]
        image_index = question_item['image_index']
        
        # --- Image Loading ---
        image_filename = question_item['image_filename']
        image_path = os.path.join(self.images_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        # --- Bounding Box Extraction ---
        scene_info = self.scenes_data[image_index]
        bounding_boxes = []
        for obj in scene_info.get('objects_detection', []):
            encoded_mask = obj['mask']
            # Convert COCO mask format to a bounding box [x, y, width, height]
            bbox_coco = coco_mask.toBbox(encoded_mask)
            # Convert to [x_min, y_min, x_max, y_max] format
            bbox = [bbox_coco[0], bbox_coco[1], bbox_coco[0] + bbox_coco[2], bbox_coco[1] + bbox_coco[3]]
            bounding_boxes.append(bbox)
        
        return {
            'id': idx,
            'query': question_item['question'],
            'answer': question_item.get('answer', 'unknown'),
            'image': image,
            'image_path': image_path,
            'image_index': image_index,
            # None (not []) when the scene has no Mask R-CNN detections, so run_vqa.py falls back to the OD model
            'bboxes': bounding_boxes if 'objects_detection' in scene_info else None,
            'original_item': question_item,
        }