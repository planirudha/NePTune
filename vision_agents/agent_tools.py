import torch
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import spacy
from typing import List, Tuple, Dict, Any, Union
try:
    import lmdeploy
except ImportError:
    print("lmdeploy not installed. Skipping.")
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForCausalLM
from .marker import Marker, MarkerV2
from .depth_estimator import DepthEstimator
   
class Assets:
    def __init__(self):
        self._translator = None
        self._depth_estimator = None
        self._marker = None
        self._nlp = None
        self.final_question_cache = []

    @property
    def depth_estimator(self):
        if self._depth_estimator is None:
            self._depth_estimator = DepthEstimator()
        return self._depth_estimator

    @property
    def marker(self):
        if self._marker is None:
            self._marker = MarkerV2()
        return self._marker

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except ImportError:
                print("spacy not installed. Cannot initialize nlp.")
                return None
        return self._nlp

    def extract_nouns_and_phrases(self, text, return_phrases=True, return_individual_nouns=True):
        # Keep unchanged
        doc = self.nlp(text)
        result = {}
        if return_phrases: 
            result["noun_phrases"] = [chunk.text for chunk in doc.noun_chunks if chunk.text.strip()]
        if return_individual_nouns:
            result["nouns"] = [token.text for token in doc if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop]
        return result
    
    
    def extract_object_with_mask(self, image_pil: Image.Image, mask_input, bbox=None, bbox_color=None) -> Image.Image:
        """
        Extracts an object from a PIL image using a mask.

        The resulting image will have the object pixels from the original image
        and a transparent background elsewhere.

        Args:
            image_pil: The input PIL Image (should ideally be RGB or RGBA).
            mask_input: The mask defining the object. Can be:
                - A PIL Image (mode 'L' or '1'). Assumed white object, black background.
                - A NumPy array (2D HxW, boolean or integer 0/1).
                - A PyTorch Tensor (2D HxW, boolean or numeric 0/1).

        Returns:
            A new PIL Image (RGBA) with the extracted object on a transparent background.
        """
        # 1. Convert input PIL image to NumPy array
        # Ensure image is RGBA for transparency handling
        image_rgba = image_pil.convert("RGBA")
        image_np = np.array(image_rgba)
        H, W, _ = image_np.shape

        # 2. Process the mask_input into a 2D boolean NumPy array
        mask_np = None
        if isinstance(mask_input, Image.Image):
            mask_pil = mask_input.convert('L') # Convert to grayscale
            if mask_pil.size != (W, H):
                mask_pil = mask_pil.resize((W,H), Image.NEAREST) # Ensure same size
            mask_np = np.array(mask_pil)
            mask_bool = mask_np > 128 # Threshold grayscale mask (adjust if needed)
        elif isinstance(mask_input, np.ndarray):
            if mask_input.ndim == 3: # Handle masks like (H, W, 1)
                mask_input = mask_input.squeeze()
            if mask_input.shape != (H, W):
                # Basic resize if shape mismatch, might need interpolation adjustment
                mask_img = Image.fromarray(mask_input.astype(np.uint8)*255) # Convert to PIL for resize
                mask_img = mask_img.resize((W,H), Image.NEAREST)
                mask_np = np.array(mask_img)
            else:
                mask_np = mask_input
            # Convert to boolean (assuming non-zero means True)
            mask_bool = mask_np.astype(bool)
        elif 'torch' in globals() and isinstance(mask_input, torch.Tensor):
            mask_tensor = mask_input.cpu().detach()
            if mask_tensor.ndim == 3: # Handle masks like (1, H, W) or (H, W, 1)
                mask_tensor = mask_tensor.squeeze()
            mask_np = mask_tensor.numpy()
            if mask_np.shape != (H, W):
                # Basic resize
                mask_img = Image.fromarray(mask_np.astype(np.uint8)*255) # Convert to PIL for resize
                mask_img = mask_img.resize((W,H), Image.NEAREST)
                mask_np = np.array(mask_img)
            # Convert to boolean (handle probabilities > 0.5 or integer masks > 0)
            if np.issubdtype(mask_np.dtype, np.floating):
                mask_bool = mask_np > 0.5
            else:
                mask_bool = mask_np > 0
        else:
            raise TypeError("Unsupported mask type. Provide PIL Image, NumPy array, or PyTorch Tensor.")

        if mask_bool.shape != (H, W):
            raise ValueError(f"Mask shape {mask_bool.shape} doesn't match image shape {(H, W)}")

        # 3. Create a transparent background canvas
        output_np = np.zeros_like(image_np) # Shape (H, W, 4), filled with 0s

        # 4. Copy original pixels where mask is True
        output_np[mask_bool] = image_np[mask_bool]

        # 5. Ensure Alpha channel is 255 where mask is true, 0 otherwise
        # (This is already handled by the copy above since image_np is RGBA
        # and output_np was initialized to zeros)
        # If input was RGB, we would do:
        # output_np[mask_bool, 3] = 255

        # 6. Convert the result back to a PIL Image
        extracted_image_pil = Image.fromarray(output_np, 'RGBA')
        ## let's draw the bounding boxes
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Draw the bounding box on the extracted image
            draw = ImageDraw.Draw(extracted_image_pil)
            draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=3)
        return extracted_image_pil
