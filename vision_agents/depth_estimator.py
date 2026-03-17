import os
import torch
import numpy as np
from PIL import Image
import cv2


class DepthEstimator:
    def __init__(self, model_path='depth_anything_v2/depth_anything_v2_vits.pth'):
        """
        Initializes the DepthEstimator with the Depth Anything model.

        Args:
            model_path (str, optional): Path to the model weights. 
                                       Defaults to 'depth_anything_v2/depth_anything_v2_vits.pth'.
        """
        self.model_path = model_path
        self.initialized = False
    def init_class(self):
        if self.initialized:
            return
        from .depth_anything_v2 import DepthAnythingV2
        self.model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384])
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Depth Anything V2 weights not found at '{self.model_path}'. "
                "Download them with:\n"
                "  wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth \\\n"
                "       -O depth_anything_v2/depth_anything_v2_vits.pth"
            )
        self.model.load_state_dict(torch.load(self.model_path))            
        self.model.eval()
        self.model.cuda()
        self.initialized = True

    def estimate_depth(self, pil_image: Image.Image) -> Image.Image:
        """
        Estimates the depth map from a PIL image.

        Args:
            pil_image (PIL.Image.Image): The input PIL image.

        Returns:
            PIL.Image.Image: The resulting depth map as a PIL image.

        Raises:
             TypeError: If the input is not a PIL Image.
        """
        self.init_class()
        if not isinstance(pil_image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        # Convert PIL image to OpenCV format
        image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            depth = self.model.infer_image(image_cv)  # HxW raw depth map

        # Normalize depth map
        normalized_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        grayscale_depth = normalized_depth.astype(np.uint8)

        # Repeat single channel to create a three-channel image
        colored_depth = np.stack([grayscale_depth, grayscale_depth, grayscale_depth], axis=-1)

        depth_image = Image.fromarray(colored_depth)
        return depth_image

    def get_depth_for_bboxes(self, image, bboxes):
        self.init_class()
        ## Return the depth median of each bounding box
        depth_image = self.estimate_depth(image)
        depth_values = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Crop the depth image to the bounding box
            cropped_depth = depth_image.crop((x1, y1, x2, y2))
            # Convert to numpy array and calculate median
            depth_array = np.array(cropped_depth)
            median_depth = np.median(depth_array)
            depth_values.append(median_depth)
        return depth_values
    def get_depth_for_masks(self, image, masks):
        """
        Return the median depth value for each object mask.

        Args:
            image (PIL.Image.Image): The original image.
            masks (list of np.ndarray): Binary masks for each object. [num_objects, 1, H, W]

        Returns:
            list of float: Median depth value within each mask region.
        """
        self.init_class()
        # get depth map as a PIL image (3‐channel grayscale)
        depth_image = self.estimate_depth(image)
        # convert to single‐channel numpy array
        depth_np = np.array(depth_image)

        depth_values = []
        for mask in masks:
            # Convert mask to binary
            if mask.shape[0] == 1:
                mask = mask.squeeze(0).astype(np.uint8)  # Remove the channel dimension
            mask = (mask > 0).astype(np.uint8)  # Convert to binary mask
            # Get the masked depth values
            masked_depth = depth_np[mask == 1]
            # Calculate the median depth value
            if masked_depth.size > 0:
                median_depth = np.median(masked_depth)
            else:
                median_depth = 0

            depth_values.append(median_depth)
        return depth_values
            