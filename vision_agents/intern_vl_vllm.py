import torch
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
try:
    from vllm import LLM, SamplingParams
    from vllm.assets.image import ImageAsset
except ImportError:
    raise ImportError("Please install vllm: pip install vllm")
from typing import Optional, List, Tuple
import logging
logging.getLogger().setLevel(logging.ERROR)
"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.

Modified from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union
import re
import json
import numpy as np
from PIL import Image
import torch # Import torch only if you might pass a tensor mask
from PIL import ImageDraw

BBOX_PATTERN_VLM = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')

import torch
from vllm import LLM


import torch
from typing import Optional, List

from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor, # Wrapper base-class
    RequestLogitsProcessor, # Request-level logitsproc type annotation
    LogitsProcessor
)

class DummyPerReqLogitsProcessor:
    """The request-level logits processor masks out all logits except the
    token id identified by `target_token`"""

    def __init__(self, target_tokens: List[int]) -> None:
        """Specify `target_token`"""
        self.target_tokens = target_tokens
        # print("USING PER REQUEST LOGITS PROCESSOR FOR TARGET TOKEN", target_tokens)

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # val_to_keep = logits[self.target_tokens].item()
        # logits[:] = float("-inf")
        # for target_token in self.target_tokens:
        #     logits[target_token] = val_to_keep

        # New approach: set everything to a very low value except target tokens
        mask = torch.ones_like(logits) * -1e9  # Very low value
        for target_token in self.target_tokens:
            # print("Keeping token", target_token, "with logit", logits[target_token].item())
            mask[target_token] = 0  # Keep the logit for target tokens
        logits = logits + mask
        
        return logits
from typing import Any, Optional
import logging
import numpy as np

class WrappedPerReqLogitsProcessor(AdapterLogitsProcessor):
    """Example of wrapping a fake request-level logit processor to create a
    batch-level logits processor"""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> Optional[RequestLogitsProcessor]:
        """This method returns a new request-level logits processor, customized
        to the `target_token` value associated with a particular request.

        Returns None if the logits processor should not be applied to the
        particular request. To use the logits processor the request must have
        a "target_token" custom argument with an integer value.

        Args:
        params: per-request sampling params

        Returns:
        `Callable` request logits processor, or None
        """
        target_tokens: Optional[Any] = params.extra_args and params.extra_args.get(
            "target_tokens"
        )
        if target_tokens is None:
            return None
        return DummyPerReqLogitsProcessor(target_tokens)


def make_llm(model_path,
             yes_token_id=None,   # "Yes" token in InternVL tokenizer
             no_token_id=None,    # "No" token in InternVL tokenizer
                # raise/lower based on VRAM & your output length
             ):

    # 1) Pick the best dtype for your GPU (BF16 on A100/H100, else FP16)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = "bfloat16" if bf16_ok else "float16"
    want_len=4096*4        # tighten this if you can — it’s a big speed lever
    target_util=0.85
    target_concurrency=8  # bump up until you hit VRAM or no longer see speedup
    # 2) Try FP8 KV cache for speed+capacity; fall back safely if unsupported
    if "1b" in model_path.lower():
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            enable_prefix_caching=True,
            max_model_len=want_len,
            max_num_batched_tokens=4096*target_concurrency,
            mm_processor_cache_gb=5,
            gpu_memory_utilization=0.9,
            logprobs_mode='processed_logits',
            logits_processors=[WrappedPerReqLogitsProcessor],
        )
    else:
        
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            enable_prefix_caching=True,
            max_model_len=want_len,
            max_num_batched_tokens=4096*target_concurrency,
            gpu_memory_utilization=0.85,
            logprobs_mode='processed_logits',
            logits_processors=[WrappedPerReqLogitsProcessor],

        )


    # llm = LLM(
    #     model=model_path,
    #     trust_remote_code=True,
    #     dtype=dtype,
    #     enable_prefix_caching=True,
    #     max_num_seqs=target_concurrency,
    #     max_seq_len_to_capture=want_len*4,
    #     max_num_batched_tokens=want_len*4,
    #     max_model_len=want_len,
    #     gpu_memory_utilization=target_util,
    # )
    return llm


# Example: tight context, short answers, heavy batching for max throughput



################################################################################
# Example: InternVL class
################################################################################
from .agent import Agent

class InternVL(Agent):
    """
    An example agent that mirrors your QwenVL style but uses 
    InternVL for scoring yes/no and generating text answers.
    """
    concept2query = {
    "attribute": {
        # Material-related attributes
        "metal": {
            "query": "Is the main object in the red bounding box metallic?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        "rubber": {
            "query": "Is the main object in the red bounding box matte?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        # Size-related attributes
        "large": {
            "query": "Is the main object in the red bounding box large?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "context"
        },
        "small": {
            "query": "Is the main object in the red bounding box small?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "context"
        },
        # Color-related attributes
        "brown": {
            "query": "Is the main object in the red bounding box brown colored?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        "green": {
            "query": "Is the main object in the red bounding box green colored?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        "purple": {
            "query": "Is the main object in the red bounding box purple colored?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        "gray": {
            "query": "Is the main object in the red bounding box gray colored?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        "yellow": {
            "query": "Is the main object in the red bounding box yellow colored?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        "blue": {
            "query": "Is the marked object in the red bounding box blue colored?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        "red": {
            "query": "Is the main object in the red bounding box red colored?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        "cyan": {
            "query": "Is the main object in the red bounding box cyan colored?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "property"
        },
        # Shape-related attributes
        "sphere": {
            "query": "Is the main object in the red bounding box a sphere?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "class"
            
        },
        "cube": {
            "query": "Is the main object in the red bounding box a cube?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "class"
        },
        "cylinder": {
            "query": "Is the main object in the red bounding box a cylinder?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "class"
        },
    },
    "relation": {
        "left": {
            "query": "Is the main object in the red bounding box to the left of the main object in the green bounding box?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "relation"
        },
        "behind": {
            "query": "Is the main object in the red bounding box behind of the main object in the green bounding box?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "relation"

        },
        "front": {
            "query": "Is the main object in the red bounding box in front of the main object in the green bounding box?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "relation"
        },
        "right": {
            "query": "Is the main object in the red bounding box to the right of the main object in the green bounding box?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "relation"
        },
        "same_color": {
            "query": "Is the main object in the red bounding box the same color as the main object in the green bounding box?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "relation"
        },
        "same_material": {
            "query": "Is the main object in the red bounding box the same material as the main object in the green bounding box?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "relation"
        },
        "same_shape": {
            "query": "Is the main object in the red bounding box the same shape as the main object in the green bounding box?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "relation"

        },
        "same_size": {
            "query": "Is the main object in the red bounding box the same size as the main object in the green bounding box?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False,
            "type": "relation"
        },
    },
    "multi_relation": {
        "between": {
            "query": "Is the main object in the red bounding box between the main objects in the green and blue bounding boxes?",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False
        },
    },
    "description": {
        "color": {
            "query": "What is the color of the main object in the red bounding box? (e.g., red, blue, purple, green, yellow, cyan, gray or brown)",
            "mask": False,
            "draw_bbox": True,
            "depth_estimation": False

        },
        "shape": {
            "query": "What is the shape of the main object in the red bounding box? (e.g. sphere, cube, or cylinder)",
            # "query": "What is the shape of the main object in the red bounding box? (e.g. car, bus, airplane, bike or motorcycle)",
            "mask": False,
            "draw_bbox": True,
            "depth_estimation": False
        },
        "size": {
            "query": "What is the size of the main object in the red bounding box? (e.g., small or large)",
            "mask": False,
            "draw_bbox": True,
            "depth_estimation": False
        },
        "material": {
            "query": "What is the material of the main object in the red bounding box? (e.g., rubber or metal)",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False
        },
        "texture": {
            "query": "What is the texture of the main object in the red bounding box? (e.g., dotted, checkered, stripped or none)",
            "mask": False,
            "draw_bbox": True,
            "pass_bbox": False,
            "depth_estimation": False
        }
    }
    }

    def __init__(
        self,
        model_path: str = "OpenGVLab/InternVL2_5-8B-MPO",
        device: str = "cuda",
        trust_remote_code: bool = True,
        llm=None,
        *args,
        **kwargs,
    ):
        """
        :param model_path: The HF Hub repo or local path for the InternVL model
        :param device: "cuda" or "cpu"
        :param load_4bit: Whether to load in 4-bit precision
        :param max_num_patches: Maximum number of patches/tiles for an image
        :param trust_remote_code: Whether to trust the remote code
        """
        super().__init__(*args, **kwargs)
        device_id = 0 if torch.cuda.device_count() > 1 else 0
        self.device = f"cuda:{device_id}"
        # 1) Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            model_max_length=1,
            trust_remote_code=trust_remote_code
        )
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_token_id  = self.tokenizer.convert_tokens_to_ids("No")
        # 2) Load model       
        if llm is not None:
            self.model = llm
        else: 
            # self.model  = LLM(
            #     model=model_path,
            #     trust_remote_code=True,
            #     max_model_len=12_000,   
            #     gpu_memory_utilization=0.8,
            # )
            # self.model = make_llm(
            #     model_path,
            #     want_len=12_000,          # tighten this if you can — it’s a big speed lever
            #     target_util=0.85,
            #     target_concurrency=64,  # bump up until you hit VRAM or no longer see speedup
            # )
            self.model = make_llm(
                model_path,
                yes_token_id=self.yes_token_id,
                no_token_id=self.no_token_id,
            )


    def _score(self, image_paths: Union[str, 'PIL.Image', list], question: str, target_token="Yes", candidates: list=None, temperature=0.5, target_tokens=None):
        """
        1) Loads the image, 
        2) builds a prompt with a single turn, 
        3) asks for 1 new token, 
        4) gets its logits for "Yes" vs "no" 
        5) returns probability of "Yes" or "no".
        """
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        if candidates is not None:
            question = f"{question}\n If you have to classify among one of these objects {candidates}, {question}"
        else:
            question = f"{question}"
        # Build the prompt with a single user question
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)        
        batch_inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image_paths},
        }]


        # We only have 1 step of generation, so the relevant logits are generation_output.scores[0]
        # This is shape: [batch_size, vocab_size]
        if target_tokens is not None:
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            for o in outputs:
                # from pprint import pprint
                # pprint(o.outputs)
                # Convert from logprob (log-space) to logits
                logprobs = np.array([o.outputs[0].logprobs[-1][x].logprob for x in token_ids if x in o.outputs[0].logprobs[-1]])

                # Apply softmax
                exp_logits = np.exp(logprobs - np.max(logprobs))  # numerical stability
                probs = exp_logits / exp_logits.sum()

                # return torch.tensor(probs, device=self.device)
                return torch.tensor(probs, device=self.device)
        else:
            target_tokens = ["Yes", "No"]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": [self.yes_token_id, self.no_token_id]},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            for o in outputs:
                # Extract logits for Yes and No tokens [the logprob mode is raw_logits]
                try:
                    yes_logit = o.outputs[0].logprobs[-1][self.yes_token_id].logprob
                except:
                    yes_logit = -100.0  # very low logprob if "Yes" is not in the top-k
                try:
                    no_logit = o.outputs[0].logprobs[-1][self.no_token_id].logprob
                except:
                    no_logit = -100.0  # very low logprob if "No" is not in the top-k

                # Convert from logprob (log-space) to logits
                logits = np.array([yes_logit, no_logit])

                # Apply softmax
                exp_logits = np.exp(logits - np.max(logits))  # numerical stability
                probs = exp_logits / exp_logits.sum()
                # return torch.tensor(probs[0], device=self.device)    
                return torch.tensor(probs[0], device=self.device)

    def _score_with_mask(self, image: 'PIL.Image', question: str, bboxes: list, candidates: list=None, type="default", history=None, target_tokens=None):
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy().tolist()
        marked_image = self.marker.mark_objects(image, bboxes, draw_bbox=True, draw_text=False, overlay_masks=False, masks=False)
        masks = self.marker.masker.mask_image(image, bboxes)
        extra_images = [
                self.extract_object_with_mask(image, masks[i]).crop((bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]))
                for i in range(len(bboxes))
            ]
        if type in ["class", "property"]:
            question = question.replace("the red bounding box", "Image-2")
        if type == "relation":
            min_x = min([bbox[0] for bbox in bboxes])
            min_y = min([bbox[1] for bbox in bboxes])
            max_x = max([bbox[2] for bbox in bboxes])
            max_y = max([bbox[3] for bbox in bboxes])
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image.size[0], max_x + padding)
            max_y = min(image.size[1], max_y + padding)
            marked_image = marked_image.crop((min_x, min_y, max_x, max_y))

        image_list = [marked_image, *extra_images]

        image_prompt = "Image-1: <image>\n"
        for idx in range(1, len(image_list)):
            image_prompt += f"Image-{idx+1}: <image>\n"
        num_extra_images = len(extra_images)
        if num_extra_images:
            image_prompt += "Image-1 is the source image with bounding boxes."
        for i in range(num_extra_images):
            image_prompt += (
                f"\nImage-{i+2} is the zoomed-in of the extracted main object in the "
                f"{self.marker.colors[i % len(self.marker.colors)]} bounding box from Image-1."
            )

        if num_extra_images > 1:
            image_prompt += (
                "\nAnswer the question, based on the provided images (for object/attribute "
                "recognition red box: Image-2, green box: Image-3).\n"
            )
        elif num_extra_images == 1:
            image_prompt += (
                "\nAnswer the question, based on the provided images (for object/attribute "
                "recognition red box: Image-2).\n"
            )

        if candidates is not None:
            question = f"{image_prompt}\nIf you have to classify among one of these objects {candidates}, {question}"
        else:
            question = f"{image_prompt}\nWith the most strict bounding boxes, {question}"
        question += "\nAnswer the question using a single word or phrase."

        messages = [{'role': 'user', 'content': question}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        batch_inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image_list},
        }]
            

        if target_tokens is None:
            target_tokens = ["Yes", "No"]
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            output = outputs[0] # We have a batch size of 1


            # Extract logprobs for Yes and No tokens
            try:
                yes_logit = output.outputs[0].logprobs[-1][self.yes_token_id].logprob
            except:
                yes_logit = -100.0  # very low logprob if "Yes" is not in the top-k
            try:
                no_logit = output.outputs[0].logprobs[-1][self.no_token_id].logprob
            except:
                no_logit = -100.0  # very low logprob if "No" is not in the top-k

            # Convert from logprob (log-space) to logits
            logits = np.array([yes_logit, no_logit])

            # Apply softmax
            exp_logits = np.exp(logits - np.max(logits))  # numerical stability
            probs = exp_logits / exp_logits.sum()
            # return torch.tensor(probs[0], device=self.device)    
            return torch.tensor(probs[0], device=self.device)
        else:
            # Score against provided target tokens
            target_tokens = [target_tokens] if isinstance(target_tokens, str) else target_tokens
            target_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": target_token_ids},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)
            output = outputs[0] # We have a batch size of 1
            logprobs = []
            for token_id in target_token_ids:
                if token_id in output.outputs[0].logprobs[-1]:
                    logprobs.append(output.outputs[0].logprobs[-1][token_id].logprob)
                else:
                    logprobs.append(-100.0) # Assign a very low logprob if not in top-k
            
            logprobs = np.array(logprobs)
            if not logprobs.size:
                return torch.tensor([])

            exp_logits = np.exp(logprobs - np.max(logprobs))
            probs = exp_logits / exp_logits.sum()
            return torch.tensor(probs, device=self.device)
    def _score_batch_with_mask(self, batch_images: List[List['PIL.Image']], questions: str, candidates: list=None, types:List[str]=None, history=None, target_tokens=None):
        if types is None:
            types = ["default"] * len(batch_images)
        if isinstance(types, str):
            types = [types] * len(batch_images)
        batch_inputs = []
        for image_list, question, type in zip(batch_images, questions, types):
            if type in ["class", "property"]:
                question = question.replace("the red bounding box", "Image-2")

            image_prompt = "Image-1: <image>\n"
            for idx in range(1, len(image_list)):
                image_prompt += f"Image-{idx+1}: <image>\n"

            num_extra_images = len(image_list) - 1
            if num_extra_images:
                image_prompt += "Image-1 is the source image with bounding boxes."
            for i in range(num_extra_images):
                image_prompt += (
                    f"\nImage-{i+2} is the zoomed-in of the extracted main object in the "
                    f"{self.marker.colors[i % len(self.marker.colors)]} bounding box from Image-1."
                )

            if num_extra_images > 1:
                image_prompt += (
                    "\nAnswer the question, based on the provided images (for object/attribute "
                    "recognition red box: Image-2, green box: Image-3).\n"
                )
            elif num_extra_images == 1:
                image_prompt += (
                    "\nAnswer the question, based on the provided images (for object/attribute "
                    "recognition red box: Image-2).\n"
                )

            if candidates is not None:
                question = f"{image_prompt}\nIf you have to classify among one of these objects {candidates}, {question}"
            else:
                question = f"{image_prompt}\nWith the most strict bounding boxes, {question}"
            question += "\nAnswer the question using a single word or phrase."

            messages = [{'role': 'user', 'content': question}]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_list},
            })
        

        # We only have 1 step of generation, so the relevant logits are generation_output.scores[0]
        # This is shape: [batch_size, vocab_size]
        if target_tokens is not None:
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=1,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

            probs = []
            for o in outputs:
                # Getting the logits
                logits = np.array([o.outputs[0].logprobs[-1][x].logprob for x in token_ids if x in o.outputs[0].logprobs[-1]])

                # Apply softmax
                exp_logits = np.exp(logits - np.max(logits))  # numerical stability
                probs.append(exp_logits / exp_logits.sum())
            return torch.tensor(probs, device=self.device)
        else:
            probs = []
            target_tokens = ["Yes", "No"]
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=1,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

            for o in outputs:
                # Extract logprobs for Yes and No tokens
                yes_logit = o.outputs[0].logprobs[-1][self.yes_token_id].logprob
                try:
                    no_logit = o.outputs[0].logprobs[-1][self.no_token_id].logprob
                except:
                    no_logit = -100.0  # very low logit if "No" is not in the top-k

                # Convert from logit (log-space) to logits
                logits = np.array([yes_logit, no_logit])

                # Apply softmax
                exp_logits = np.exp(logits - np.max(logits))  # numerical stability
                temp_probs = exp_logits / exp_logits.sum()
                probs.append(torch.tensor(temp_probs[0]))
            # return torch.tensor(probs[0], device=self.device)    
            return torch.stack(probs).to(self.device)
    
    def _score_batch(self, image_paths: List[str], questions: List[str], candidates: List[str]=None, target_tokens: Optional[List[str]]=None, target_token="Yes", temperature=0.5):
        """
        1) Loads the image, 
        2) builds a prompt with a single turn, 
        3) asks for 1 new token, 
        4) gets its logits for "Yes" vs "no" 
        5) returns probability of "Yes" or "no".
        """
        batch_inputs = []
        for i, (image_path, question) in enumerate(zip(image_paths, questions)):
            if not isinstance(image_path, list):
                image_path = [image_path]
            if candidates is not None:
                question = f"{question}\n If you have to classify among one of these objects {candidates}, {question}"
            else:
                question = f"{question}"
            # Build the prompt with a single user question
            messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
            prompt = self.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)        
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_path},
            })
        
        # We only have 1 step of generation, so the relevant logits are generation_output.scores[0]
        # This is shape: [batch_size, vocab_size]
        if target_tokens is not None:
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

            probs = []
            for o in outputs:
                # Convert from logprob (log-space) to logits
                logits = np.array([o.outputs[0].logprobs[-1][x].logprob for x in token_ids if x in o.outputs[0].logprobs[-1]])

                # Apply softmax
                exp_logits = np.exp(logits - np.max(logits))  # numerical stability
                probs = exp_logits / exp_logits.sum()

                # return torch.tensor(probs, device=self.device)
                probs.append(torch.tensor(probs))
            return torch.stack(probs).to(self.device)
        else:
            probs = []
            target_tokens = ["Yes", "No"]
            token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=20,
                extra_args={"target_tokens": token_ids},
            )
            outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

            for o in outputs:
                # Extract logits for Yes and No tokens
                yes_logit = o.outputs[0].logprobs[-1][self.yes_token_id].logprob
                no_logit = o.outputs[0].logprobs[-1][self.no_token_id].logprob
                # Convert from logit (log-space) to logits
                logits = np.array([yes_logit, no_logit])

                # Apply softmax
                exp_logits = np.exp(logits - np.max(logits))  # numerical stability
                temp_probs = exp_logits / exp_logits.sum()
                probs.append(torch.tensor(temp_probs[0]))
            return probs
            # return torch.tensor(probs[0], device=self.device)    
            # return torch.stack(probs).to(self.device)

    def _query(
        self, 
        image_paths: Union[str, 'PIL.Image', list],
        question: str, 
        max_new_tokens=128,
        history=None
    ) -> str:
        """
        For general question-answering with the model. 
        Returns the decoded text from the model's generation.
        """
        # Build the prompt with a single user question
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)        
        batch_inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"image": image_paths},
        }]
        sampling_params = SamplingParams(
            temperature=0,
            
            max_tokens=max_new_tokens,
        )
        outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

        return outputs[0].outputs[0].text    
    def _text_only_query(
        self,
        question: str,
        max_new_tokens=1024,
        ) -> str:
        """
        For text-only question-answering with the model.
        Returns the decoded text from the model's generation.
        """
        # Build the prompt with a single user question
        messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
        prompt = self.tokenizer.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)        
        batch_inputs = [{
            "prompt": prompt,
        }]
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
        )
        outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

        return outputs[0].outputs[0].text    
        

    def _query_batch(
        self, 
        image_paths: Union[str, 'PIL.Image', list], 
        questions: Union[str, 'PIL.Image', list], 
        max_new_tokens=128,
        batch_size=10,  # Set your desired batch size here
    ) -> Union[str, list]:
        """
        General question-answering with the model.
        Processes the inputs in batches and returns the decoded text(s)
        from the model's generation.

        Parameters:
        image_paths: A single image path/PIL.Image or a list of them.
        questions: A single question or a list of questions corresponding to the images.
        max_new_tokens: Maximum tokens to generate.
        batch_size: Number of images (and questions) to process at a time.

        Returns:
        A single response if there is only one image/question, or a list of responses.
        """
        # Ensure image_paths is a list.
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        
        # If questions is not a list, replicate it for each image.
        if not isinstance(questions, list):
            questions = [questions] * len(image_paths)
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
        )
        batch_inputs = []
        all_responses = []
        for question, image_path in zip(questions, image_paths):
            # Process in batches.
            # Build the prompt with a single user question
            messages = [{'role': 'user', 'content': f"<image>\n{question}"}]
            prompt = self.tokenizer.apply_chat_template(messages,
                                                    tokenize=False,
                                                    add_generation_prompt=True)        
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"image": image_path},
            })

        outputs = self.model.generate(batch_inputs, sampling_params=sampling_params, use_tqdm=False)

        return [x.outputs[0].text for x in outputs]

    def _normalize_coordinates(self, boxes, image_width, image_height):
        normalized_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            normalized_box = [
                round((x1 / image_width) * 1000),
                round((y1 / image_height) * 1000),
                round((x2 / image_width) * 1000),
                round((y2 / image_height) * 1000)
            ]
            normalized_boxes.append(normalized_box)
        return normalized_boxes

    def _denormalize_coordinates(self, boxes, image_width, image_height):
        denormalized_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            denormalized_box = [
                round((x1 / 1000) * image_width),
                round((y1 / 1000) * image_height),
                round((x2 / 1000) * image_width),
                round((y2 / 1000) * image_height)
            ]
            denormalized_boxes.append(denormalized_box)
        return denormalized_boxes
    
    def _parse_vlm_bbox_output(self, text_response: str, width: int, height: int) -> Optional[List[float]]:
        """Parses VLM text output for the first [[x1,y1,x2,y2]] pattern."""
        # Keep from previous adaptation (adjust regex/normalization if needed)
        try:
            bbox_matches = BBOX_PATTERN_VLM.findall(text_response)
            if bbox_matches:
                coords_str = bbox_matches[0]
                nx1, ny1, nx2, ny2 = [float(c.strip()) for c in coords_str]
                if float(nx1) < 1 and float(nx2) < 1 and float(ny1) < 1 and float(ny2) < 1:
                    # Denormalize assuming [0, 1] range from VLM
                    x1 = round(nx1 * width)
                    y1 = round(ny1 * height)
                    x2 = round(nx2 * width)
                    y2 = round(ny2 * height)
                else: 
                    # Denormalize assuming [0, 1000] range from VLM
                    x1 = round((nx1 / 1000) * width)
                    y1 = round((ny1 / 1000) * height)
                    x2 = round((nx2 / 1000) * width)
                    y2 = round((ny2 / 1000) * height)
                # Basic sanity check on coordinates
                if x1 < x2 and y1 < y2 and x1 >= 0 and y1 >= 0 and x2 <= width and y2 <= height:
                    return [x1, y1, x2, y2]
                else:
                    print(f"W: Parsed VLM box has invalid coordinates after denormalization: {[x1, y1, x2, y2]}")
                    return None
        except Exception:
            pass # Ignore parsing errors silently or log them
        return None

    def _parse_labeled_bboxes(self, text: str, image) -> Dict[str, List[List[int]]]:
        """
        Parses a labeled bounding box text and returns a dictionary with labels and their bounding boxes.

        Args:
            text (str): The input string containing labels and associated bounding boxes.
            image (PIL.Image): The image corresponding to the bounding boxes.

        Returns:
            Dict[str, List[List[int]]]: A dictionary mapping labels to lists of bounding boxes.
        """
        # Match label and bbox string
        pattern = r'([\w\s-]+?)\s*(\[\[.*?\]\])'
        matches = re.findall(pattern, text, re.DOTALL)

        label_dict = {}
        for label, group in matches:
            # Extract individual bounding boxes and convert to list of integers
            bboxes = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', group)
            bboxes = [list(map(int, bbox)) for bbox in bboxes]
            bboxes = self._denormalize_bbox(bboxes, image.width, image.height)
            label_dict[label.strip()] = bboxes

        return label_dict

    def ground(self, image, phrase):
        vlm_bbox_prompt = f"Please provide the bounding box coordinates of the region this sentence describes: <ref> {phrase} </ref>"
        vlm_direct_answer = self._query(image, vlm_bbox_prompt)
        vlm_direct_box = self._parse_vlm_bbox_output(vlm_direct_answer, image.size[0], image.size[1])
        return vlm_direct_box

    def suggest_objects(self, image, query, objects):
        vlm_object_query = f"Please provide the bounding box coordinates and label of all objects like {', '.join(objects)} related to this query: \"{query}\" in the following image."
        vlm_direct_answer = self._query(image, vlm_object_query, max_new_tokens=1024)
        return vlm_object_query, vlm_direct_answer
