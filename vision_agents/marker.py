import numpy as np
from PIL import Image
import torch
import numpy as np
from copy import deepcopy, copy
from PIL import Image, ImageDraw, ImageFont

import numpy as np
import torch
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    build_sam2 = None
    SAM2ImagePredictor = None
    print("sam2 not installed. Skipping.")

class Marker:
    def __init__(self):
        # Load a font or fallback to the default if unavailable
        self.font = ImageFont.load_default()
        # self.colors = ["red", "green", "blue", "yellow", "orange", "pink"]
        self.colors = ["red", "green", "blue", "#4B0082", "orange", "pink"]


    def mark_objects(self, image, bboxes, font_size=60, draw_text=True, mask_outside=False, mask_regions=False, expansion_ratio=0.1, min_margin=7, box_width_ratio=0.01, small_box_line_width_coeff=0.5, small_area_threshold_ratio=0.01):
        """Draw bounding boxes and optional text on the image, making them 10% bigger."""
        image = deepcopy(image)  # Avoid altering the original image
        chosen_colors = []
        img_width, img_height = image.size
        image_area = img_width * img_height
        small_area_threshold = image_area * small_area_threshold_ratio # Calculate pixel threshold

        
        # Precompute expanded bboxes
        expanded_bboxes = []
        for bbox in bboxes:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            margin_width = max(int(width * expansion_ratio / 2), min_margin)
            margin_height = max(int(height * expansion_ratio / 2), min_margin)
            expanded_bbox = [
                bbox[0] - margin_width, bbox[1] - margin_height,
                bbox[2] + margin_width, bbox[3] + margin_height
            ]
            expanded_bboxes.append(expanded_bbox)

        if mask_regions:
            mask = Image.new("L", image.size, 255)
            mask_draw = ImageDraw.Draw(mask)
            for expanded_bbox in expanded_bboxes:
                mask_draw.rectangle(expanded_bbox, fill=0)  # Mask the regions

            if image.mode != 'RGBA':
                image = image.convert('RGBA')
            ## reverse the mask
            mask = Image.fromarray(255 - np.array(mask))
            # Create a mask that keeps the regions transparent and the rest opaque
            transparent_mask = Image.new('RGBA', image.size, (0, 0, 0, 255))
            image = Image.composite(transparent_mask, image, mask)
        
        if mask_outside:
            mask = Image.new("L", image.size, 0)
            mask_draw = ImageDraw.Draw(mask)
            for expanded_bbox in expanded_bboxes:
                mask_draw.rectangle(expanded_bbox, fill=255)

            if image.mode != 'RGBA':
                image = image.convert('RGBA')

            background = Image.new('RGBA', image.size, (0, 0, 0, 255))
            image = Image.composite(image, background, mask)
        
        if not mask_regions:
            draw = ImageDraw.Draw(image)
            for i, expanded_bbox in enumerate(expanded_bboxes):
                color = self.colors[i % 6]
                chosen_colors.append(color)
                base_line_width = int(image.size[0] * box_width_ratio)
                original_bbox = bboxes[i]
                bbox_width = original_bbox[2] - original_bbox[0]
                bbox_height = original_bbox[3] - original_bbox[1]
                bbox_area = bbox_width * bbox_height

                # Adjust line width if bbox area is small
                line_width = base_line_width
                if bbox_area > 0 and bbox_area < small_area_threshold:
                    line_width = int(base_line_width * small_box_line_width_coeff)

                # Ensure minimum line width of 1 pixel
                line_width = max(1, line_width)

                
                # Draw the expanded rectangle
                draw.rectangle(expanded_bbox, outline=color, width=line_width)

                if draw_text:
                    text = str(i)
                    text_bbox = draw.textbbox((expanded_bbox[0], expanded_bbox[1]), text, font=self.fontfont)
                    draw.rectangle([*text_bbox], fill=color)
                    draw.text((text_bbox[0], text_bbox[1]), text, fill="white", font=self.fontfont)

        return image, chosen_colors




# --- Masker Class and Helper Functions (same as before) ---
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
        from segment_anything import sam_model_registry, SamPredictor
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
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        ### list of tensors
        if not isinstance(input_boxes, torch.Tensor):
            if isinstance(input_boxes[0], torch.Tensor):
                input_boxes = torch.stack(input_boxes).to(self.device)
            else:
                input_boxes = torch.tensor(input_boxes, device=self.device) # Convert boxes to tensor if needed
        input_boxes = input_boxes.to(self.device)
        self.predictor.set_image(image) # Set image for the predictor
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2]) # Transform boxes to SAM input space
        masks, _, _ = self.predictor.predict_torch( # Predict masks
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False, # Single mask output for each box
        )
        return masks



class MaskerSAM2:
    """
    MaskerSAM2 class to generate object masks using Segment Anything Model v2 (SAM2).
    """
    def __init__(
        self,
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint = "sam_res/checkpoints/sam2.1_hiera_large.pt",
        device: str = "cuda",
    ):
        """
        Initializes the MaskerSAM2 with a SAM2 model.

        Args:
            model_cfg (str): Path to the SAM2 model config file.
            checkpoint (str): Path to the SAM2 checkpoint file.
            device (str): Device to run SAM2 on ('cuda' or 'cpu').
        """
        try:
            self.device = device
            # Build and load SAM2 model
            sam = build_sam2(model_cfg, checkpoint)
            sam.to(device=self.device)
            # Create SAM2 image predictor
            self.predictor = SAM2ImagePredictor(sam)
        except Exception as e:
            print("Error initializing SAM2 model:", e)
            self.device = None
            self.predictor = None

    def mask_image(self, image, input_boxes: np.ndarray) -> torch.Tensor:
        """
        Generates masks for objects in the image given bounding boxes.

        Args:
            image (numpy.ndarray): Input image as a numpy array (H, W, C).
            input_boxes (np.ndarray): Bounding boxes as numpy array (N, 4) in [x0, y0, x1, y1].

        Returns:
            torch.Tensor: Predicted masks of shape (N, 1, H, W).
        """
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)

        # Convert boxes to tensor on the correct device
        if not isinstance(input_boxes, torch.Tensor):
            if len(input_boxes) > 0 and isinstance(input_boxes[0], torch.Tensor):
                input_boxes = torch.stack(input_boxes).to(device=self.device)
            else:
                input_boxes = torch.tensor(input_boxes, device=self.device, dtype=torch.float32)
        else:
            input_boxes = input_boxes.to(device=self.device, dtype=torch.float32)

        # Set the image for prediction
        self.predictor.set_image(image)

        # Predict masks with inference mode and automatic mixed precision
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            masks, _, _ = self.predictor.predict(
                box=input_boxes,
                multimask_output=False
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


from PIL import ImageColor

from PIL import Image, ImageColor
import numpy as np
import torch
from PIL import Image, ImageColor, ImageChops, ImageFilter

def _draw_masks_overlay(image, masks, colors, alpha=0.1):
    """
    Overlays semi-transparent colored masks on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        masks (torch.Tensor or numpy.ndarray or list): Masks to overlay.
        colors (list): List of color specs, either:
                       - CSS-style strings ("red", "#FF00FF", "rgb(255,0,0)") or
                       - 3-tuples of ints (R, G, B)
        alpha (float): Transparency between 0 (invisible) and 1 (opaque).

    Returns:
        PIL.Image.Image: Image with masks overlaid.
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    for i, mask in enumerate(masks):
        rgb = ImageColor.getrgb(colors[i % len(colors)]) \
            if isinstance(colors[i % len(colors)], str) \
            else tuple(colors[i % len(colors)])
        # make a scratch RGBA canvas
        overlay = Image.new('RGBA', image.size, rgb + (0,))
        # turn your binary mask into a *partial* alpha‐channel
        m = mask.cpu().numpy()[0] if isinstance(mask, torch.Tensor) else mask[0]
        mask_pil = Image.fromarray((m * alpha * 255).astype(np.uint8)).convert('L')
        overlay.putalpha(mask_pil)
        # blend
        image = Image.alpha_composite(image, overlay)
               # 3)  optional bold outline
        edge_thickness = 3
        draw_edge = edge_thickness > 0
        if draw_edge:
            # a) dilate & erode to get morphological gradient (edge ring)
            dil = mask_pil.filter(ImageFilter.MaxFilter(size=2 * edge_thickness + 1))
            ero = mask_pil.filter(ImageFilter.MinFilter(size=2 * edge_thickness + 1))
            outline = ImageChops.difference(dil, ero)          # 'L' mask, 0/255

            # b) solid-colour overlay restricted to outline
            edge_rgba = ImageColor.getrgb(colors[i % len(colors)]) + (255,)
            edge_ovl = Image.new("RGBA", image.size, edge_rgba)
            edge_ovl.putalpha(outline)
            image = Image.alpha_composite(image, edge_ovl)

    return image.convert("RGB") # Convert back to RGB for final output



def _draw_black_masks_overlay(image, masks):
    """
    Overlays black masks on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        masks (torch.Tensor or numpy.ndarray or list): Masks to overlay.

    Returns:
        PIL.Image.Image: Image with masks overlaid.
    """
    if image.mode != 'RGBA':
        image = image.convert('RGBA') # Convert image to RGBA if not already

    for i, mask in enumerate(masks): # Iterate through masks
        mask_image_np = mask[0].cpu().numpy() if isinstance(mask, torch.Tensor) else mask[0] # Convert mask to numpy
        mask_image_pil = Image.fromarray((mask_image_np * 255).astype(np.uint8)).convert('L') # Convert numpy mask to PIL Image
        color_overlay = Image.new('RGBA', image.size, (0, 0, 0, 255)) # Create black overlay image
        image = Image.composite(color_overlay, image, mask_image_pil) # Composite color overlay onto image using mask
    return image


def _draw_bboxes_and_text(image, bboxes, draw_labels, font, box_width_ratio, chosen_colors, expansion_ratio, min_margin, small_area_threshold_ratio=0.05, small_box_line_width_coeff=0.5, extra_thick=False):
    """
    Draws bounding boxes and optional labels on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        bboxes (list): List of bounding boxes in [x0, y0, x1, y1] format.
        draw_labels (bool): Whether to draw labels inside the bboxes. (Note: controlled by draw_bbox in main function now)
        font (PIL.ImageFont.FreeTypeFont): Font for labels.
        box_width_ratio (float): Ratio of bbox line width to image width.
        chosen_colors (list): List of colors to use for bboxes and labels.
        expansion_ratio (float): Ratio to expand bboxes before drawing.
        min_margin (int): Minimum margin for bbox expansion.
    """
    img_width, img_height = image.size
    image_area = img_width * img_height
    small_area_threshold = image_area * small_area_threshold_ratio # Calculate pixel threshold
    draw = ImageDraw.Draw(image) # Create drawing object
    expanded_bboxes = [_expand_bbox(bbox, expansion_ratio, min_margin) for bbox in bboxes] # Expand bboxes

    for i, expanded_bbox in enumerate(expanded_bboxes): # Iterate through expanded bboxes
        color = chosen_colors[i % len(chosen_colors)] # Cycle through colors
        base_line_width = int(max(image.size) * box_width_ratio)
        original_bbox = bboxes[i]
        bbox_width = original_bbox[2] - original_bbox[0]
        bbox_height = original_bbox[3] - original_bbox[1]
        bbox_area = bbox_width * bbox_height

        # Adjust line width if bbox area is small
        line_width = base_line_width
        if bbox_area > 0 and bbox_area < small_area_threshold:
            line_width = int(base_line_width * small_box_line_width_coeff)
        # Ensure minimum line width of 1 pixel
        if extra_thick:
            line_width = max(2, int(line_width*1.5))
        else:
            line_width = max(1, line_width)
        
        # Draw the expanded rectangle
        draw.rectangle(expanded_bbox, outline=color, width=line_width) # Draw rectangle

        if draw_labels: # Draw labels if draw_labels is True (controlled by draw_bbox now)
            text = str(i) # Label text is object index
            text_bbox = draw.textbbox((expanded_bbox[0], expanded_bbox[1]), text, font=font) # Get bounding box of text
            draw.rectangle([*text_bbox], fill=color) # Draw rectangle behind text
            draw.text((text_bbox[0], text_bbox[1]), text, fill="white", font=font) # Draw text

def _draw_black_bbox(image, bboxes):
    """
    Draws black bounding boxes on the image.

    Args:
        image (PIL.Image.Image): Input PIL image.
        bboxes (list): List of bounding boxes in [x0, y0, x1, y1] format.

    Returns:
        PIL.Image.Image: Image with black bounding boxes drawn.
    """
    draw = ImageDraw.Draw(image)  # Create drawing object
    for bbox in bboxes:  # Iterate through bboxes
        # Convert tensor or numpy array to list of ints for PIL
        if isinstance(bbox, torch.Tensor):
            coords = bbox.cpu().tolist()
        elif isinstance(bbox, np.ndarray):
            coords = bbox.tolist()
        else:
            coords = list(bbox)
        draw.rectangle(coords, outline="black", width=3, fill="black")  # Draw black rectangle
    return image



class MarkerV2:
    def __init__(self):
        # Load a font or fallback to the default if unavailable
        self.font = ImageFont.load_default()
        self.colors = ["red", "green", "blue", "#4B0082", "orange", "pink"]

        self.masker = MaskerSAM2()
        # self.masker_2 = Masker()

        self.box_width_ratio = 0.01
    

    def mark_objects(self, image, bboxes=None, masks=None, mask_background=False, draw_bbox=False, overlay_masks=False, draw_text=False, mask_regions=False, mask_bbox=False, extra_thick=False, crop=False):
        image = copy(image) # Create a copy to avoid modifying original image
        
        # Determine bbox parameters based on whether bounding boxes are explicitly drawn
        # if bboxes is not None and draw_bbox: # Use larger bbox params when drawing bboxes
        #     min_margin_val = 7
        #     expansion_ratio_val = 0.1
        # else: # Use smaller/no expansion bbox params when not explicitly drawing bboxes (e.g., for masking only)
        #     min_margin_val = 0
        #     expansion_ratio_val = 0.0
        min_margin_val = 0
        expansion_ratio_val = 0.0
        # Apply masking effects (background)
        if masks is True: # Masks are provided, use mask-based operations

            masks = self.masker.mask_image(image, bboxes)
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().detach()
            if isinstance(masks, np.ndarray):
                masks = torch.tensor(masks).cpu().detach()
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(0)
            
            use_mask = True
            if mask_background: # Mask out background if mask_background is True
                image = _apply_mask_outside(image, masks, use_mask=use_mask, expansion_ratio=expansion_ratio_val, min_margin=min_margin_val) # Apply background mask using masks
            if overlay_masks: # Overlay masks if overlay_masks is True
                image = _draw_masks_overlay(image, masks, self.colors) # Overlay masks on image
            if mask_regions:
                image = _draw_black_masks_overlay(image, masks)
            if mask_bbox:
                image = _draw_black_bbox(image, bboxes)
        elif bboxes is not None: # No masks, but bboxes are provided, use bbox-based operations
            use_mask = False
            if mask_background: # Mask out background if mask_background is True
                image = _apply_mask_outside(image, bboxes, use_mask=use_mask, expansion_ratio=expansion_ratio_val, min_margin=min_margin_val) # Apply background mask using bboxes

        # Draw bounding boxes if draw_bbox is True and bboxes are provided
        if bboxes is not None and draw_bbox:
            _draw_bboxes_and_text(image, bboxes, draw_labels=draw_text, font=self.font, box_width_ratio=self.box_width_ratio, chosen_colors=self.colors, expansion_ratio=expansion_ratio_val, min_margin=min_margin_val, extra_thick=extra_thick) # Draw bboxes and labels
        

        return image.convert("RGB")

