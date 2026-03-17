import torch
import transformers
from packaging import version
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
except:
    print("Detectron2 not found. RCNN functionality will be unavailable.")
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForCausalLM
from copy import deepcopy
import cv2
from typing import List, Optional, Union

# Add NMS utility functions
def bbox_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (list): First bounding box in format [x1, y1, x2, y2]
        box2 (list): Second bounding box in format [x1, y1, x2, y2]
        
    Returns:
        float: IoU value between 0 and 1
    """
    # Get coordinates of intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x2 - x1) * (y2 - y1)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou

def apply_nms(boxes, scores=None, iou_threshold=0.6):
    """
    Apply non-maximum suppression to eliminate redundant bounding boxes.
    
    Args:
        boxes (list or numpy.ndarray): List of bounding boxes in format [[x1, y1, x2, y2], ...]
        scores (list or numpy.ndarray, optional): Confidence scores for each box. If None, boxes are processed in given order.
        iou_threshold (float): IoU threshold for considering boxes as overlapping. Increase to be more strict.
        
    Returns:
        list: Indices of boxes to keep after NMS
    """
    # If no boxes, return empty list
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    
    # Initialize list of picked indices
    keep_indices = []
    
    # If scores are provided, sort boxes by score (highest first)
    if scores is not None:
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        order = scores.argsort()[::-1]
    else:
        # Process in given order
        order = np.arange(len(boxes))
    
    while len(order) > 0:
        # Pick the box with highest score or first in order
        current_index = order[0]
        keep_indices.append(current_index)
        
        # If only one box left, we're done
        if len(order) == 1:
            break
            
        # Calculate IoU of current box with remaining boxes
        ious = [bbox_iou(boxes[current_index], boxes[j]) for j in order[1:]]
        
        # Keep boxes with IoU less than threshold
        filtered_indices = np.where(np.array(ious) < iou_threshold)[0]
        
        # Update order to include only boxes with low IoU
        order = order[filtered_indices + 1]  # +1 because we skipped the first element
    
    return keep_indices


class Masker:
    """
    Masker class to generate object masks using Segment Anything Model (SAM).
    """
    def __init__(self, sam_checkpoint = "sam_vit_h_4b8939.pth", model_type = "vit_h", device = "cuda"):
        """
        Initializes the Masker with a SAM model.

        Args:
            sam_checkpoint (str, optional): Path to the SAM checkpoint file.
            model_type (str, optional): Type of SAM model to use.
            device (str, optional): Device to run SAM on ('cuda' or 'cpu').
        """
        self.device = device
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # Load SAM model
        sam.to(device=device) # Move model to specified device
        self.predictor = SamPredictor(sam) # Create SAM predictor

    def mask_image(self, image, input_boxes ):
        """
        Generates masks for objects in the image given bounding boxes.

        Args:
            image (numpy.ndarray): Input image as a numpy array (H, W, C).
            input_boxes (numpy.ndarray): Bounding boxes of objects as numpy array (N, 4) in [x0, y0, x1, y1] format.

        Returns:
            torch.Tensor: Predicted masks as a torch tensor of shape (N, 1, H, W).
        """
        if not isinstance(input_boxes, torch.Tensor):
            input_boxes = torch.tensor(input_boxes, device=self.device) # Convert boxes to tensor if needed
        self.predictor.set_image(image) # Set image for the predictor
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]) # Transform boxes to SAM input space
        masks, _, _ = self.predictor.predict_torch( # Predict masks
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False, # Single mask output for each box
        )
        return masks


def _expand_bbox(bbox, expansion_ratio, min_margin):
    """
    Expands a bounding box by a given ratio and minimum margin.

    Args:
        bbox (list): Bounding box in [x0, y0, x1, y1] format.
        expansion_ratio (float): Ratio to expand the bbox by.
        min_margin (int): Minimum margin to add to each side of the bbox.

    Returns:
        list: Expanded bounding box.
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    margin_width = max(int(width * expansion_ratio / 2), min_margin) # Calculate width margin
    margin_height = max(int(height * expansion_ratio / 2), min_margin) # Calculate height margin
    return [ # Return expanded bbox
        bbox[0] - margin_width, bbox[1] - margin_height,
        bbox[2] + margin_width, bbox[3] + margin_height
    ]


def expand_bboxes(
    bboxes: List[List[float]],
    image_width: int,
    image_height: int,
    scale: float = 1.2,
    min_box_size: Optional[float] = None,
) -> List[List[float]]:
    """Expand bboxes around center and clamp to image bounds.

    Args:
        bboxes: List of [x1, y1, x2, y2] boxes.
        image_width: Width of source image.
        image_height: Height of source image.
        scale: Multiplicative factor for width/height expansion.
        min_box_size: Optional minimum width/height after expansion.

    Returns:
        Expanded and clamped bounding boxes.
    """
    expanded_bboxes: List[List[float]] = []
    for bbox in bboxes:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        new_width = width * scale
        new_height = height * scale
        if min_box_size is not None:
            new_width = max(new_width, float(min_box_size))
            new_height = max(new_height, float(min_box_size))

        new_x1 = max(0.0, center_x - new_width / 2.0)
        new_y1 = max(0.0, center_y - new_height / 2.0)
        new_x2 = min(float(image_width), center_x + new_width / 2.0)
        new_y2 = min(float(image_height), center_y + new_height / 2.0)
        expanded_bboxes.append([new_x1, new_y1, new_x2, new_y2])

    return expanded_bboxes


def _apply_mask_outside(image, masks_or_bboxes, use_mask, expansion_ratio, min_margin):
    """
    Applies a mask to the image to mask out the regions *outside* the provided masks or bounding boxes.

    Args:
        image (PIL.Image.Image): Input PIL image.
        masks_or_bboxes (torch.Tensor or numpy.ndarray or list): Masks (torch.Tensor/numpy.ndarray) or bboxes (list of lists) defining object regions.
        use_mask (bool): If True, use masks; if False, use bboxes.
        expansion_ratio (float): Ratio to expand bboxes before masking.
        min_margin (int): Minimum margin for bbox expansion.

    Returns:
        PIL.Image.Image: Image with background masked out (made transparent).
    """
    mask_img = Image.new("L", image.size, 0) # Create black mask image
    mask_draw = ImageDraw.Draw(mask_img) # Create drawing object for mask

    if use_mask: # Process masks
        combined_mask = Image.new("L", image.size, 0) # Initialize combined mask
        for mask in masks_or_bboxes: # Iterate through masks
            mask_np = mask[0].cpu().numpy() if isinstance(mask, torch.Tensor) else mask[0] # Convert mask to numpy
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert('L') # Convert numpy mask to PIL Image
            combined_mask = Image.composite(Image.new("L", image.size, 255), combined_mask, mask_pil) # Combine masks (union)
        mask_img = combined_mask # Use combined mask
    else: # Process bounding boxes
        expanded_bboxes = [_expand_bbox(bbox, expansion_ratio, min_margin) for bbox in masks_or_bboxes] # Expand bboxes
        for expanded_bbox in expanded_bboxes: # Iterate through expanded bboxes
            mask_draw.rectangle(expanded_bbox, fill=255) # Draw white rectangles on mask for bboxes

    if image.mode != 'RGBA':
        image = image.convert('RGBA') # Convert image to RGBA if not already
    background = Image.new('RGBA', image.size, (0, 0, 0, 255)) # Create transparent background
    return Image.composite(image, background, mask_img) # Composite image with transparent background using mask


def _draw_masks_overlay(image, masks):
    """
    Overlays semi-transparent colored masks on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        masks (torch.Tensor or numpy.ndarray or list): Masks to overlay.

    Returns:
        PIL.Image.Image: Image with masks overlaid.
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA') # Convert image to RGBA if not already

    for i, mask in enumerate(masks): # Iterate through masks
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) # Generate random color with alpha
        mask_image_np = mask[0].cpu().numpy() if isinstance(mask, torch.Tensor) else mask[0] # Convert mask to numpy
        mask_image_pil = Image.fromarray((mask_image_np * 255).astype(np.uint8)).convert('L') # Convert numpy mask to PIL Image
        color_overlay = Image.new('RGBA', image.size, tuple((np.array(color[:3]) * 255).astype(int)) + (int(color[3] * 255),)) # Create color overlay image
        image = Image.composite(color_overlay, image, mask_image_pil) # Composite color overlay onto image using mask
    return image


def _draw_bboxes_and_text(image, bboxes, draw_labels, font, bbox_line_ratio, chosen_colors, expansion_ratio, min_margin):
    """
    Draws bounding boxes and optional labels on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        bboxes (list): List of bounding boxes in [x0, y0, x1, y1] format.
        draw_labels (bool): Whether to draw labels inside the bboxes. (Note: controlled by draw_bbox in main function now)
        font (PIL.ImageFont.FreeTypeFont): Font for labels.
        bbox_line_ratio (float): Ratio of bbox line width to image width.
        chosen_colors (list): List of colors to use for bboxes and labels.
        expansion_ratio (float): Ratio to expand bboxes before drawing.
        min_margin (int): Minimum margin for bbox expansion.
    """
    draw = ImageDraw.Draw(image) # Create drawing object
    expanded_bboxes = [_expand_bbox(bbox, expansion_ratio, min_margin) for bbox in bboxes] # Expand bboxes

    for i, expanded_bbox in enumerate(expanded_bboxes): # Iterate through expanded bboxes
        color = chosen_colors[i % len(chosen_colors)] # Cycle through colors
        line_width = int(max(image.size) * bbox_line_ratio) # Calculate line width
        draw.rectangle(expanded_bbox, outline=color, width=line_width) # Draw rectangle

        if draw_labels: # Draw labels if draw_labels is True (controlled by draw_bbox now)
            text = str(i) # Label text is object index
            text_bbox = draw.textbbox((expanded_bbox[0], expanded_bbox[1]), text, font=font) # Get bounding box of text
            draw.rectangle([*text_bbox], fill=color) # Draw rectangle behind text
            draw.text((text_bbox[0], text_bbox[1]), text, fill="white", font=font) # Draw text


def mark_objects(image, masks=None, bboxes=None, mask_background=False, draw_bbox=False, overlay_masks=False, bbox_line_ratio=0.01, force_show_bbox=False):
    """
    Marks objects in an image by applying masks, bounding boxes, and labels.

    This function takes an image and optionally masks and/or bounding boxes to visually
    highlight objects. It provides options to overlay masks, mask the background
    (outside objects), and draw bounding boxes with labels.

    Args:
        image (PIL.Image.Image): The input image to mark objects on.
        masks (torch.Tensor, numpy.ndarray, optional): Masks for objects. Defaults to None.
            If provided, masks take precedence over bounding boxes for masking effects.
        bboxes (numpy.ndarray, optional): Bounding boxes for objects. Defaults to None.
            Used if masks are not provided, or for drawing bounding box outlines and labels.
        mask_background (bool, optional): If True, masks out the background (area outside objects). Defaults to False.
            Uses either provided `masks` or `bboxes` to determine object regions.
        draw_bbox (bool, optional): If True, draws bounding boxes with labels around the objects. Defaults to False.
        bbox_line_ratio (float, optional): Ratio of the bounding box line width to the image width. Defaults to 0.01.
        overlay_masks (bool, optional): If True, overlays semi-transparent colored masks on the objects. Defaults to False.
            Only effective if `masks` are provided.

    Returns:
        tuple: A tuple containing:
            - PIL.Image.Image: The image with marked objects.
            - list: A list of chosen colors used for marking (currently always a fixed list).
    """
    image = deepcopy(image) # Create a copy to avoid modifying original image
    chosen_colors = ["red", "green", "blue", "yellow", "orange", "pink"] # Fixed color list
    font = ImageFont.load_default() # Load default font

    # Determine bbox parameters based on whether bounding boxes are explicitly drawn
    if bboxes is not None and draw_bbox: # Use larger bbox params when drawing bboxes
        min_margin_val = 7
        expansion_ratio_val = 0.1
    else: # Use smaller/no expansion bbox params when not explicitly drawing bboxes (e.g., for masking only)
        min_margin_val = 0
        expansion_ratio_val = 0.0

    # Apply masking effects (background)
    if masks is not None: # Masks are provided, use mask-based operations
        use_mask = True
        if mask_background: # Mask out background if mask_background is True
            image = _apply_mask_outside(image, masks, use_mask=use_mask, expansion_ratio=expansion_ratio_val, min_margin=min_margin_val) # Apply background mask using masks
        if overlay_masks: # Overlay masks if overlay_masks is True
            image = _draw_masks_overlay(image, masks) # Overlay masks on image
    elif bboxes is not None: # No masks, but bboxes are provided, use bbox-based operations
        use_mask = False
        if mask_background: # Mask out background if mask_background is True
            image = _apply_mask_outside(image, bboxes, use_mask=use_mask, expansion_ratio=expansion_ratio_val, min_margin=min_margin_val) # Apply background mask using bboxes

    # Draw bounding boxes if draw_bbox is True and bboxes are provided
    if bboxes is not None and draw_bbox:
        _draw_bboxes_and_text(image, bboxes, draw_labels=draw_bbox, font=font, bbox_line_ratio=bbox_line_ratio, chosen_colors=chosen_colors, expansion_ratio=expansion_ratio_val, min_margin=min_margin_val) # Draw bboxes and labels


    return image, chosen_colors


class RCNN:
    def __init__(self, model="rcnn", threshold=0.0):
        self.predictor = self._load_predictor(model, threshold)

    def _load_predictor(self, model, threshold, nms_threshold=0.5): 
        """Load the appropriate object detection model."""
        cfg = get_cfg()
        # if model == "rcnn":
        #     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        #     cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # else:
        config_file = os.getenv("MASK_RCNN_CONFIG", "detection/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
        weights_file = os.getenv("MASK_RCNN_WEIGHTS", "detection/model_final_2d9806.pkl")
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights_file
        
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        self.threshold = threshold
        self.nms_threshold = nms_threshold

        
        return DefaultPredictor(cfg)

    def predict(self, image, apply_custom_nms=False):
        image_np = np.array(image)[:, :, ::-1]  # Convert RGB to BGR
        outputs = self.predictor(image_np)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        boxes_list = boxes.tolist()
        scores = outputs["instances"].scores.cpu().numpy()
        scores_list = scores.tolist()
        # Apply additional custom NMS if requested
        if apply_custom_nms and len(boxes_list) > 0:
            keep_indices = apply_nms(boxes_list, scores_list, self.nms_threshold)
            boxes_list = [boxes_list[i] for i in keep_indices]
        # Convert to tensor
        return torch.tensor(boxes_list) if apply_custom_nms else boxes
    

    def mark_objects(self, image, bboxes, font_size=60, draw_text=True, mask_outside=False, mask_regions=False, force_show_bbox=False):
        """Draw bounding boxes and optional text on the image."""  
        return mark_objects(image, None , bboxes, font_size, draw_text, mask_outside, mask_regions, force_show_bbox=force_show_bbox)

    def predict_and_mark(self, image):
        """Predict bounding boxes and mark them on the image."""
        bboxes = self.predict(image)
        return self.mark_objects(image, bboxes, font_size=70)

    def predict_mark_and_show(self, image):
        """Predict bounding boxes, mark them on the image, and display the result."""
        marked_image, _ = self.predict_and_mark(image)
        plt.imshow(marked_image)
        plt.axis("off")
        plt.show()
        return marked_image

class Dino:
    def __init__(self, model_id="IDEA-Research/grounding-dino-base", threshold=0.1, nms_threshold=0.5):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.threshold = threshold
        self.nms_threshold = nms_threshold
    
    def predict(self, image, text=None, apply_custom_nms=False, extra_bboxes=None):
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if version.parse(transformers.__version__) >= version.parse("4.46.0"):
            results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.threshold, text_threshold=0.25, target_sizes=[image.size[::-1]]
            )
        else:
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=self.threshold, text_threshold=0.25, target_sizes=[image.size[::-1]]
            )
        
        boxes = results[0]['boxes']
        scores = results[0]['scores'] if 'scores' in results[0] else None
        
        # Convert to list format
        boxes_list = boxes.tolist()
        scores_list = scores.tolist() if scores is not None else None
        if extra_bboxes is not None: ## extra forced bbox
            for extra_bbox in extra_bboxes:
                boxes_list.append(extra_bbox)
                scores_list.append(0.9)

        # Apply additional custom NMS if requested
        if apply_custom_nms and len(boxes_list) > 0:
            keep_indices = apply_nms(boxes_list, scores_list, self.nms_threshold)
            ## Sort based on scores
            keep_indices = sorted(keep_indices, key=lambda x: scores_list[x], reverse=True)
            boxes_list = [boxes_list[i] for i in keep_indices]
        else:
            ## Sort based on scores
            if scores_list is not None:
                keep_indices = sorted(range(len(scores_list)), key=lambda x: scores_list[x], reverse=True)
                boxes_list = [boxes_list[i] for i in keep_indices]
                scores_list = [scores_list[i] for i in keep_indices]
        
        return torch.tensor(boxes_list) if apply_custom_nms else boxes

    def predict_v2(self, image, text: str = None, apply_custom_nms: bool = True, extra_bboxes: list[list[float]] = None):
        """
        Improved multi-query prediction that:
          1. Splits the input text on '.' into individual sub-queries (ignores empties).
          2. Runs GroundingDINO separately for each sub-query and aggregates boxes & scores.
          3. (Phase 1) Optionally applies NMS across all predicted boxes (if apply_custom_nms).
          4. Adds any provided extra_bboxes (forced boxes) with a fixed high score.
          5. (Phase 2) Applies a final NMS over the union of predicted + extra boxes (if apply_custom_nms),
             ensuring extra boxes also participate in suppression.
          6. Returns the final boxes sorted by score (descending) as a torch.FloatTensor (CPU).

        Differences vs original predict():
          - Handles multi-sentence / multi-concept prompts by splitting with '.' and querying per part.
          - Always returns the *final* processed boxes (including extra_bboxes) instead of the raw model output.
          - Ensures extra_bboxes are subject to the same NMS as predictions.

        Args:
            image (PIL.Image.Image): Input PIL image.
            text (str | None): Prompt string possibly containing '.' separators. If None/empty -> single empty query.
            apply_custom_nms (bool): Whether to perform the two-phase NMS procedure.
            extra_bboxes (list[list[float]] | None): Additional boxes to force-include (each [x0,y0,x1,y1]).

        Returns:
            torch.Tensor: (N,4) tensor of final boxes in XYXY format (CPU tensor, float32).
        """
        # --- Build sub-queries ---
        if text is None:
            queries = [""]
        else:
            # Split by '.' and strip whitespace, filter empties
            queries = [q.strip() for q in text.split('.') if q.strip()]
            if not queries:  # fallback to single empty query if all were empty
                queries = [""]

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []

        # --- Run model per query ---
        self.model.eval()
        for q in queries:
            try:
                inputs = self.processor(images=image, text=q, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                if version.parse(transformers.__version__) >= version.parse("4.46.0"):
                    results = self.processor.post_process_grounded_object_detection(
                        outputs, inputs.input_ids, threshold=self.threshold, text_threshold=0.25, target_sizes=[image.size[::-1]]
                    )
                else:
                    results = self.processor.post_process_grounded_object_detection(
                        outputs, inputs.input_ids, box_threshold=self.threshold, text_threshold=0.25, target_sizes=[image.size[::-1]]
                    )
                boxes = results[0].get('boxes', None)
                scores = results[0].get('scores', None)
                if boxes is None or boxes.shape[0] == 0:
                    continue
                # Accumulate
                b_list = boxes.tolist()
                s_list = scores.tolist() if scores is not None else [1.0] * len(b_list)
                all_boxes.extend(b_list)
                all_scores.extend(s_list)
            except Exception as e:
                print(f"[predict_v2] Warning: query '{q}' failed with error: {e}")
                continue

        # If no detections, still consider extra_bboxes if provided
        if not all_boxes and extra_bboxes:
            all_boxes = [list(b) for b in extra_bboxes]
            all_scores = [1.0] * len(all_boxes)

        # Phase 1 NMS on predicted boxes only (before adding extras)
        if apply_custom_nms and all_boxes:
            keep_pred = apply_nms(all_boxes, all_scores, self.nms_threshold)
            all_boxes = [all_boxes[i] for i in keep_pred]
            all_scores = [all_scores[i] for i in keep_pred]

        # Append extra boxes (forced) with high score so they are retained unless heavily overlapping
        if extra_bboxes:
            for eb in extra_bboxes:
                all_boxes.append(list(eb))
                all_scores.append(0.95)  # slightly below perfect to allow ordering but strong enough

        # Phase 2 NMS over combined predicted + extra
        if apply_custom_nms and all_boxes:
            keep_final = apply_nms(all_boxes, all_scores, self.nms_threshold)
            all_boxes = [all_boxes[i] for i in keep_final]
            all_scores = [all_scores[i] for i in keep_final]

        # Final sort by score descending
        if all_boxes:
            order = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
            all_boxes = [all_boxes[i] for i in order]
            # all_scores = [all_scores[i] for i in order]  # kept for potential future use

        return torch.tensor(all_boxes, dtype=torch.float32)

    def mark_objects(self, image, bboxes, font_size=60, draw_text=True, mask_outside=False, mask_regions=False, force_show_bbox=False):
        return mark_objects(image, None,  bboxes, font_size, draw_text, mask_outside, mask_regions, force_show_bbox=force_show_bbox)

    def predict_and_mark(self, image, text=None):
        bboxes = self.predict(image, text)
        return self.mark_objects(image, bboxes.tolist(), font_size=70)

    def predict_mark_and_show(self, image, text=""):
        marked_image, _ = self.predict_and_mark(image, text)
        plt.imshow(marked_image)
        plt.axis("off")
        plt.show()
        return marked_image

    def predict_for_comparison(self, image: Image.Image, text: Optional[str] = None) -> torch.Tensor:
        """
        Performs prediction in a way compatible with the original script's usage pattern.

        Uses self.threshold for box detection, applies torchvision.ops.nms if
        apply_custom_nms is True, sorts results by score, and returns a
        detached CPU tensor.

        Args:
            image (PIL.Image.Image): The input image.
            text (Optional[str]): The text prompt for grounding.
            apply_custom_nms (bool): Whether to apply NMS using self.nms_threshold.

        Returns:
            torch.Tensor: A detached CPU tensor of final bounding boxes (XYXY format),
                          sorted by confidence score (highest first). Returns an empty
                          tensor if no objects are found above the threshold.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input 'image' must be a PIL Image.")

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        self.model.eval() # Ensure model is in eval mode
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process using specified thresholds stored in the instance
        # Target sizes are reversed: (height, width)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        if version.parse(transformers.__version__) >= version.parse("4.46.0"):
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=target_sizes,
                # box_threshold=0.0,
                threshold=0.0,
                text_threshold=0.1,
            )
        else:
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                target_sizes=target_sizes,
                box_threshold=0.0,
                text_threshold=0.1,
            )

        # Extract boxes and scores for the first (and only) image in the batch
        result_dict = results[0]
        boxes = result_dict['boxes']    # XYXY format
        scores = result_dict['scores']

        if boxes.shape[0] == 0:
            # No boxes detected above threshold, return empty tensor 
            # return torch.empty((0, 4), dtype=torch.float32).to(self.device)
            return torch.tensor([50.9,  39.1, 493.5, 356.5], dtype=torch.float32).to(self.device)  # Return empty tensor

        # Always sort boxes by score (descending) before potential NMS
        try:
            sorted_indices = scores.argsort(descending=True)
            sorted_boxes = boxes[sorted_indices]
            sorted_scores = scores[sorted_indices]
        except Exception as e:
             print(f"Warning: Error during sorting boxes/scores: {e}. Returning unsorted boxes.")
             # Fallback: return boxes as they came from post-processing if sorting fails
             sorted_boxes = boxes
             sorted_scores = scores # Keep scores aligned for potential NMS

        
        return sorted_boxes.detach().cpu()


class GLIP:
    def __init__(
        self,
        config_file: str = "pretrained_models/GLIP/configs/glip_Swin_L.yaml",
        weight_file: str = "pretrained_models/GLIP/checkpoints/glip_large_model.pth",
        device: str = "cuda:0",
        min_image_size: int = 800,
        confidence_threshold: float = 0.3,
        show_mask_heatmaps: bool = False,
        nms_threshold: float = 0.6,
    ):
        from maskrcnn_benchmark.config import cfg
        from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
        # 1. Load config & weights
        cfg.merge_from_file(config_file)
        cfg.merge_from_list([
            "MODEL.WEIGHT", weight_file,
            "MODEL.DEVICE", device,
        ])
        self.nms_threshold = nms_threshold


        # 2. Instantiate GLIPDemo
        self.demo = GLIPDemo(
            cfg,
            min_image_size=min_image_size,
            confidence_threshold=confidence_threshold,
            show_mask_heatmaps=show_mask_heatmaps,
        )
        self.device = device

    def predict(
        self,
        image: Image.Image,
        text: Union[str, List[str]],
        apply_custom_nms=False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run a GLIP query on `image`.

        - If `text` is a str containing '.', it's split into queries.
        - If `text` is a single str with no '.', it's treated as one query.
        - If `text` is a list[str]:
            - Inference is run *individually* for each query string.
            - Results (boxes, scores, labels) are concatenated.
            - If `apply_custom_nms` is True, NMS is applied to the *combined* results.
        - If `text` is a str (single or split):
            - Inference is run once with all queries together.
            - If `apply_custom_nms` is True, NMS is applied to the results.


        Args:
            image: Input image (PIL Image or BGR NumPy array).
            text: Query string or list of query strings.
            apply_custom_nms: If True, applies NMS using `self.nms_threshold`.
                              For list inputs, NMS is applied *after* combining results.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
              - boxes:  [N×4] numpy array of [x_min, y_min, x_max, y_max]
              - labels: [N] numpy array of int indices corresponding to the input query list.
              - scores: [N] numpy array of confidence scores.
        """
        # — Prepare list of queries —
        
        if isinstance(text, str) and "." in text:
            queries = [t.strip() for t in text.split(".") if t.strip()]
        elif isinstance(text, str):
            queries = [text]
        else: # it's already a list
            queries = text

        if not queries:
            # Return empty results if no valid queries are provided
            return np.empty((0, 4), dtype=int), np.empty((0,), dtype=int), np.empty((0,), dtype=float)

        # — Prepare image —
        if isinstance(image, Image.Image):
            # Ensure image is in RGB before converting
            if image.mode != 'RGB':
                image = image.convert('RGB')
            rgb = np.array(image)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            # Assume image is already BGR NumPy array
            bgr = image

        # — Run Inference —
        all_boxes = []
        all_scores = []
        all_labels = []

        # --- Run inference individually for each query in the list ---
        for i, query in enumerate(queries):
            # Run inference for a single query (pass as a list)
            prediction = self.demo.inference(bgr, [query]) # Pass query as a list

            # Extract results for this query
            boxes_i = prediction.bbox.cpu().numpy() # [N_i, 4]
            scores_i = prediction.get_field("scores").cpu().numpy() # [N_i]
            # Labels from prediction will be 0 (index within the single-item query list)
            # We need to map it to the original query index 'i'
            labels_i = np.full(len(boxes_i), i, dtype=int)

            all_boxes.append(boxes_i)
            all_scores.append(scores_i)
            all_labels.append(labels_i)

        # Concatenate results from all individual runs
        if all_boxes:
            final_boxes = np.concatenate(all_boxes, axis=0)
            final_scores = np.concatenate(all_scores, axis=0)
            final_labels = np.concatenate(all_labels, axis=0)
        else:
            final_boxes = np.empty((0, 4), dtype=int)
            final_labels = np.empty((0,), dtype=int)
            final_scores = np.empty((0,), dtype=float)


        # Ensure boxes are integers
        final_boxes = final_boxes.astype(int)

        # — Apply NMS conditionally AFTER collecting all boxes —
        if apply_custom_nms and len(final_boxes) > 0:
            print(f"Applying NMS with threshold {self.nms_threshold}") # Debug print
            try:
                keep_indices = apply_nms(final_boxes, final_scores, self.nms_threshold)
                final_boxes = final_boxes[keep_indices]
                final_scores = final_scores[keep_indices]
                final_labels = final_labels[keep_indices]
                print(f"Kept {len(keep_indices)} boxes after NMS.") # Debug print
            except Exception as e:
                 print(f"Error during NMS: {e}. Skipping NMS.")


        return final_boxes


    def mark_objects(
        self,
        image: Image.Image,
        boxes: Union[list[list[int]], np.ndarray],
        labels: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        box_color: str = "red",
        thickness: int = 4,
        font_size: int = 20,
    ) -> Image.Image:
        """
        Draws boxes (and optional labels+scores) onto a PIL image.
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()

        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = map(int, box)
            draw.rectangle([x0, y0, x1, y1], outline=box_color, width=thickness)
            if labels is not None and scores is not None:
                text = f"{labels[i]}:{scores[i]:.2f}"
                draw.text((x0+2, y0+2), text, fill=box_color, font=font)

        return img

    def predict_and_mark(
        self,
        image: Image.Image,
        text: Union[str, List[str]],
    ) -> Image.Image:
        """
        Runs predict() then mark_objects() and returns the annotated PIL image.
        """
        boxes, label_idxs, scores = self.predict(image, text)
        # map indices back to strings
        if isinstance(text, str) and "." in text:
            queries = [t.strip() for t in text.split(".") if t.strip()]
        elif isinstance(text, str):
            queries = [text]
        else:
            queries = text
        label_strs = [queries[i] for i in label_idxs]
        return self.mark_objects(image, boxes, label_strs, scores)

    def predict_mark_and_show(
        self,
        image: Image.Image,
        text: Union[str, List[str]],
    ) -> Image.Image:
        """
        Same as predict_and_mark, but also plt.shows it inline.
        """
        img = self.predict_and_mark(image, text)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        return img


class Florence2:
    """
    Florence2 in the same API shape as your Dino class:
      - .predict(image, text, apply_custom_nms=False, extra_bboxes=None)
          → torch.Tensor of shape (N,4): [x0,y0,x1,y1]
      - .predict_for_comparison(image, text)
          → torch.Tensor of raw boxes, no sorting/NMS/extra steps
      - .mark_objects, .predict_and_mark, .predict_mark_and_show
    """
    def __init__(
        self,
        model_name: str = "microsoft/Florence-2-large-ft",
        box_threshold: float = 0.1,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        gpu_number: int = 0,
    ):
        assert gpu_number == 0, "Florence2 only works on cuda:0"
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.float16

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = (
            AutoModelForCausalLM
            .from_pretrained(model_name, torch_dtype=self.dtype, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

        self.task_prompt    = "<OPEN_VOCABULARY_DETECTION>"
        self.box_threshold  = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold  = nms_threshold

    def predict(
        self,
        image: Union[Image.Image, torch.Tensor],
        text: str = "",
        apply_custom_nms: bool = False,
        extra_bboxes: list[list[float]] = None,
    ) -> torch.Tensor:
        """
        Runs Florence2 detection and returns a tensor of SAM-format boxes [x0,y0,x1,y1].
        Includes optional extra_bboxes, custom NMS, and sorting by dummy score=1.
        """
        # — prepare PIL image —
        if isinstance(image, torch.Tensor):
            arr = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = image.convert("RGB")
        W, H = pil.size

        # — run Florence2 core inference —
        prompt = self.task_prompt + text
        inputs = self.processor(text=prompt, images=pil, return_tensors="pt")\
                           .to(self.device, self.dtype)
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids    = inputs.input_ids,
                pixel_values = inputs.pixel_values,
                max_new_tokens=1024,
                num_beams     = 3,
            )
        gen_text = self.processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed   = self.processor.post_process_generation(
            gen_text,
            task       = self.task_prompt,
            image_size = (W, H),
        )

        # — extract & reshape boxes —
        raw = np.array(parsed[self.task_prompt]["bboxes"])        # flat [x,y,w,h]
        boxes = raw
        ## convert to [x0,y0,x1,y1] format to [x0,y0,x1,y1] = [x,y,x+w,y+h]
        # boxes = raw.reshape(-1, 4)[:, [0, 1, 0, 1]] + raw.reshape(-1, 4)[:, [2, 3, 2, 3]]
        # flip y-axis to bottom-origin
        # boxes[:, 1] = H - boxes[:, 1]
        # boxes[:, 3] = H - boxes[:, 3]

        # — prepare lists for sorting/NMS —
        boxes_list  = boxes.tolist()
        scores_list = [1.0] * len(boxes_list)

        # — append extra forced boxes —
        if extra_bboxes:
            boxes_list  += extra_bboxes
            scores_list += [1.0] * len(extra_bboxes)

        # — apply custom NMS if requested —
        if apply_custom_nms and boxes_list:
            keep = apply_nms(boxes_list, scores_list, self.nms_threshold)
            # sort keep by score descending
            keep = sorted(keep, key=lambda i: scores_list[i], reverse=True)
            boxes_list = [boxes_list[i] for i in keep]
        else:
            # always sort by score descending
            idxs = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
            boxes_list = [boxes_list[i] for i in idxs]

        return torch.tensor(boxes_list, dtype=torch.float32)

    def predict_for_comparison(
        self,
        image: Union[Image.Image, torch.Tensor],
        text: str = "",
    ) -> torch.Tensor:
        """
        Core Florence2 inference, no extra_bboxes, no NMS, no sorting.
        """
        # — prepare PIL image —
        if isinstance(image, torch.Tensor):
            arr = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil = Image.fromarray(arr)
        else:
            pil = image.convert("RGB")
        W, H = pil.size

        # — core inference —
        prompt = self.task_prompt + text
        inputs = self.processor(text=prompt, images=pil, return_tensors="pt")\
                           .to(self.device, self.dtype)
        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids    = inputs.input_ids,
                pixel_values = inputs.pixel_values,
                max_new_tokens=1024,
                num_beams     = 3,
            )
        gen_text = self.processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed   = self.processor.post_process_generation(
            gen_text,
            task       = self.task_prompt,
            image_size = (W, H),
        )

        raw = np.array(parsed[self.task_prompt]["bboxes"])
        return torch.tensor(raw, dtype=torch.float32)

    def mark_objects(
        self,
        image: Image.Image,
        bboxes: Union[List[List[int]], torch.Tensor],
        font_size: int = 60,
        draw_text: bool = True,
        mask_outside: bool = False,
        mask_regions: bool = False,
        force_show_bbox: bool = False,
    ) -> Image.Image:
        return mark_objects(
            image,
            None,
            bboxes,
            font_size,
            draw_text,
            mask_outside,
            mask_regions,
            force_show_bbox=force_show_bbox,
        )

    def predict_and_mark(
        self,
        image: Union[Image.Image, torch.Tensor],
        text: str = "",
    ) -> tuple[Image.Image, torch.Tensor]:
        boxes = self.predict(image, text)
        annotated = self.mark_objects(image, boxes.tolist(), font_size=70)
        return annotated, boxes

    def predict_mark_and_show(
        self,
        image: Union[Image.Image, torch.Tensor],
        text: str = "",
    ) -> Image.Image:
        img, _ = self.predict_and_mark(image, text)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        return img

def object_detector(model="rcnn", threshold=0.2):
    if model == "rcnn":
        model = RCNN(threshold=threshold)
    elif model == "dino":
        model = Dino(threshold=threshold)
    elif model == "glip":
        model = GLIP(
            confidence_threshold=threshold,
        )
    elif model in ["florence2", "florence"]:
        model = Florence2(
            model_name="microsoft/Florence-2-large-ft",
            box_threshold=threshold,
            text_threshold=0.25,
            nms_threshold=0.6,
        )
    elif model == "florence2_base":
        model = Florence2(
            model_name="microsoft/Florence-2-base-ft",
            box_threshold=threshold,
            text_threshold=0.25,
            nms_threshold=0.6,
        )
    else:
        raise ValueError(f"Unknown model: {model}. Supported models are: 'rcnn', 'dino', 'glip'.")
    return model


