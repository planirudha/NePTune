import torch
import warnings
import os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
import re

from .agent import Agent
        
BBOX_PATTERN_VLM = re.compile(r'\[*\[(.*?),(.*?),(.*?),(.*?)\]\]*')

class OvisGemma(Agent):### transformers==4.20.1
#     concept2query = {
#     "attribute": {
#         # Material-related attributes
#         "metal": {
#             "query": "Is the main object metallic?",
#             "mask": True,
#             "draw_bbox": False,
#             "pass_bbox": False,
#             "depth_estimation": False,
#         },
#         "rubber": {
#             "query": "Is the main object in the red bounding box matte?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         # Size-related attributes
#         "large": {
#             "query": "Is the main object in the red bounding box large?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "small": {
#             "query": "Is the main object in the red bounding box small?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         # Color-related attributes
#         "brown": {
#             "query": "Is the main object brown colored?",
#             "mask": True,
#             "draw_bbox": False,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "green": {
#             "query": "Is the object in the red bounding box green colored?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "purple": {
#             "query": "Is the main object in the red bounding box purple colored?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "gray": {
#             "query": "Is the main object in the red bounding box gray colored?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "yellow": {
#             "query": "Is the main object yellow colored?",
#             "mask": True,
#             "draw_bbox": False,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "blue": {
#             "query": "Is the main object in the red bounding box blue colored?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "red": {
#             "query": "Is the main object red colored?",
#             "mask": True,
#             "draw_bbox": False,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "cyan": {
#             "query": "Is the main object in the red bounding box cyan colored?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         # Shape-related attributes
#         "sphere": {
#             "query": "Is the main object in the red bounding box a sphere?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
            
#         },
#         "cube": {
#             "query": "Is the main object a cube?",
#             "mask": True,
#             "draw_bbox": False,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "cylinder": {
#             "query": "Is the main object a cylinder?",
#             "mask": True,
#             "draw_bbox": False,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#     },
#     "relation": {
#         "left": {
#             "query": "Is the main object in the red bounding box left of the main object in the green bounding box?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "behind": {
#             "query": "Is the main object in the red bounding box further behind than the main object in the green bounding box?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False

#         },
#         "front": {
#             "query": "Is the main object in the red bounding box front of the main object in the green bounding box?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "right": {
#             "query": "Is the main object in the red bounding box right of the main object in the green bounding box?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "same_color": {
#             "query": "Are the two main non-masked objects the similar color?",
#             "mask": True,
#             "draw_bbox": False,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "same_material": {
#             "query": "Are the two main objects made of the similar material?",
#             "mask": True,
#             "draw_bbox": False,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#         "same_shape": {
#             "query": "Is the main object in the red bounding box similar shape as the main object in the green bounding box?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": True,
#             "depth_estimation": False,
#             "add_description": False
#         },
#         "same_size": {
#             "query": "Is the main object in the red bounding box similar size as the main object in the green bounding box?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False,
#             "add_description": False
#         },
#     },
#     "multi_relation": {
#         "between": {
#             "query": "Is the main object in the red bounding box between the main objects in the green and blue bounding boxes?",
#             "mask": False,
#             "draw_bbox": True,
#             "pass_bbox": False,
#             "depth_estimation": False
#         },
#     },
#     "description": {
#         "color": {
#             "query": "What is the color of the main object? (e.g., red, blue, purple, green, yellow, cyan, gray or brown)",
#             "mask": True,
#             "draw_bbox": False,
#             "depth_estimation": False

#         },
#         "shape": {
#             "query": "What is the shape of the main object? (e.g. sphere, cube, or cylinder)",
#             "mask": True,
#             "draw_bbox": False,
#             "depth_estimation": False
#         },
#         "size": {
#             "query": "What is the size of the main object in the red bounding box? (e.g., small or large)",
#             "mask": False,
#             "draw_bbox": True,
#             "depth_estimation": False
#         },
#         "material": {
#             "query": "What is the material of the main object? (e.g., rubber or metal)",
#             "mask": True,
#             "draw_bbox": False,
#             "depth_estimation": False
#         },
#     }
# }

    concept2query = {
        "attribute": {
            # Material-related attributes
            "metal": {
                "query": "Is the main object in the red bounding box metallic?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            "rubber": {
                "query": "Is the main object in the red bounding box matte?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            # Size-related attributes
            "large": {
                "query": "Is the main object in the red bounding box large?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "context",
            },
            "small": {
                "query": "Is the main object in the red bounding box small?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "context",
            },
            # Color-related attributes
            "brown": {
                "query": "Is the main object in the red bounding box brown colored?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            "green": {
                "query": "Is the main object in the red bounding box green colored?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            "purple": {
                "query": "Is the main object in the red bounding box purple colored?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            "gray": {
                "query": "Is the main object in the red bounding box gray colored?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            "yellow": {
                "query": "Is the main object in the red bounding box yellow colored?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            "blue": {
                "query": "Is the main object in the red bounding box blue colored?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            "red": {
                "query": "Is the main object in the red bounding box red colored?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            "cyan": {
                "query": "Is the main object in the red bounding box cyan colored?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "property",
            },
            # Shape-related attributes
            "sphere": {
                "query": "Is the main object in the red bounding box a sphere?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "class",
            },
            "cube": {
                "query": "Is the main object in the red bounding box a cube?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "class",
            },
            "cylinder": {
                "query": "Is the main object in the red bounding box a cylinder?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "class",
            },
        },
        "relation": {
            "left": {
                "query": "Is the main object in the red bounding box left of the main object in the green bounding box?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",
            },
            "behind": {
                "query": "Is the object in the red bounding box behind the object in the green bounding box?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox":  False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",
            },
            "front": {
                "query": "Is the main object in the red bounding box front of the main object in the green bounding box?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox":  False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",
            },
            "right": {
                "query": "Is the main object in the red bounding box right of the main object in the green bounding box?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox":  False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",
            },
            "same_color": {
                "query": "Is the main object in the red bounding box the same color as the main object in the green bounding box?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",
            },
            "same_material": {
                "query": "Is the main object in the red bounding box the same material as the main object the green bounding box?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",
            },
            "same_shape": {
                "query": "Is the main object in the red bounding box the same shape as the main object the green bounding box?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",
            },
            "same_size": {
                "query": "Is the main object in the red bounding box the same size as the main object the green bounding box?",
                # "query": "Are the two main objects similar size?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",

            },
        },
        "multi_relation": {
            "between": {
                "query": "Is the main object in the red bounding box between the main objects in the green and blue bounding boxes?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False
            },
        },
        "description": {
            "color": {
                "query": "What is the color of the main object? (e.g., red, blue, purple, green, yellow, cyan, gray or brown)",
                "mask": True,
                "draw_bbox": False,
                "depth_estimation": False
            },
            "shape": {
                "query": "What is the shape of the main object? (e.g. sphere, cube, or cylinder)",
                "mask": True,
                "draw_bbox": False,
                "depth_estimation": False
            },
            "size": {
                "query": "What is the size of the main object in the red bounding box? (e.g., small or large)",
                "mask": False,
                "draw_bbox": True,
                "depth_estimation": False
            },
            "material": {
                "query": "What is the material of the main object? (e.g., rubber or metal)",
                "mask": False,
                "draw_bbox": True,
                "depth_estimation": False
            },
        }
    }
    def __init__(self,
                model_name="AIDC-AI/Ovis1.6-Gemma2-9B",
                device="cuda",
                *args,
                **kwargs):
        """
        :param model_path: The HF Hub repo or local path for the InternVL model
        :param device: "cuda" or "cpu"
        :param load_4bit: Whether to load in 4-bit precision
        :param max_num_patches: Maximum number of patches/tiles for an image
        :param trust_remote_code: Whether to trust the remote code
        """
        super().__init__(*args, **kwargs)
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            multimodal_max_length=8192*3,
            trust_remote_code=True,
            # llm_attn_implementation='flash_attention_2',
        ).to(device)
        self.max_partition = 9
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.device = device
        self.model.eval()
        if getattr(self.model, 'llm', None) and hasattr(self.model.llm, '_cache'):
            self.model.llm._cache.max_batch_size = 1

    def _score(self, images, text, target_token="Yes", temperature = 0.5, target_tokens=None, *args, **kwargs):
        if not isinstance(images, list):
            images = [images]
        
        query = f'<image>\n{text}'
        # Preprocess inputs
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id).unsqueeze(0).to(device=self.device)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        pixel_values = [pixel_values.to(dtype=self.dtype, device=self.device)]
        
        # Prepare target tokens
        target_token_ids = self.text_tokenizer(["Yes", "No"], add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        yes_token_id = target_token_ids[0][0].item()
        no_token_id = target_token_ids[1][0].item()

        # Generate outputs with scores
        # gen_kwargs = dict(
        #     max_new_tokens=1,
        #     do_sample=False,
        #     output_scores=True,
        #     return_dict_in_generate=True,
        #     pixel_values=pixel_values,
        #     attention_mask=attention_mask,
        # )

        with torch.inference_mode():
            # _, inputs_embeds, labels, attention_mask = self.model.merge_multimodal(
            # text_input_ids=input_ids,
            # text_attention_masks=gen_kwargs.pop('attention_mask'),
            # text_labels=None,
            # pixel_values=gen_kwargs.pop('pixel_values'),
            # left_padding=True
            # )
            # outputs = self.model.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs)
            gen_kwargs = dict(
                max_new_tokens=1,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                output_scores=True,
                return_dict_in_generate=True,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            outputs = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)


        # # Get logits of the generated token
        # next_token_logits = outputs.scores[0][0]  # Shape: [vocab_size]

        # # Extract logits for 'Yes' and 'No'
        # yes_no_logits = next_token_logits[[yes_token_id, no_token_id]]  # Shape: [2]
        # # Temperature scaling
        
        if target_tokens is None:            
            # 2) Extract final-step logits for each output
            top_k = torch.topk(outputs.scores[0][0], k=3)

            other_token_ids = []
            for index in top_k.indices:
                token = self.text_tokenizer.decode(index)
                if len(other_token_ids) < 0 and token.lower() not in ["yes", "no"]:
                    other_token_ids.append(int(index))
            # Get logits of the generated token
            next_token_logits = outputs.scores[0][0]  # Shape: [vocab_size]
            # Extract logits for 'Yes' and 'No'
            yes_no_logits = next_token_logits[[yes_token_id, no_token_id, *other_token_ids]]  # Shape: [2]
            yes_no_logits = yes_no_logits / 1  # Scale logits by the temperature
            probs = torch.softmax(yes_no_logits, dim=-1)
            return probs[0]
        else:
            target_tokens = [target_tokens] if isinstance(target_tokens, str) else target_tokens
            target_token_ids = self.text_tokenizer.convert_tokens_to_ids(target_tokens)
            ## squeeze the target_token_ids to match the shape of outputs.scores
            target_token_ids = torch.tensor(target_token_ids).unsqueeze(0).detach().numpy()
            ## convert to list
            target_token_ids = target_token_ids.tolist()[0]
            print(target_token_ids)
            target_tokens_score = outputs.scores[0][0][target_token_ids]  # shape [num_target_tokens]
            target_prob = torch.softmax(target_tokens_score, dim=-1)
            return target_prob
    
    def _score_with_mask(self, image: 'PIL.Image', question: str, bboxes: list, candidates: list=None, type="default", history=None, target_tokens=None):
        if getattr(self.model.llm,"_cache", False) and getattr(self.model.llm._cache, "max_batch_size", None) != 100:
            self.model.llm._cache.max_batch_size = 100
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy().tolist()
        zero_image = Image.new("RGB", image.size, (0, 0, 0)) 
        marked_image = self.marker.mark_objects(image, bboxes, draw_bbox=True, draw_text=False, overlay_masks=False, masks=False)
        marked_zero_image = self.marker.mark_objects(zero_image, bboxes, draw_bbox=True, draw_text=False, overlay_masks=False, masks=False)
        masks = self.marker.masker.mask_image(image, bboxes)
        extra_images = [
                self.extract_object_with_mask(image, masks[i]).crop((bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]))
                for i in range(len(bboxes))
            ]
        if type in ["class", "property"]:
            question = question.replace("red bounding box", "Image-3")

        image_paths = [marked_image, marked_zero_image, *extra_images]
        image_prompt = "Image-1: <image>\n"
        image_prompt = "Image-2: <image>\n"
        for i in range(1, len(extra_images)+1):
            image_prompt += f"Image-{i+2}: <image>\n"
        if len(extra_images) > 0:
            image_prompt += "Image-1 is the source image."
            image_prompt += f"\nImage-2 shows the position of bounding boxes."

        for i in range(len(extra_images)):
            image_prompt += f"\nImage-{i+3} is the extracted object of the {self.marker.colors[i]} bounding box from Image-1."
            # image_prompt += f"\nImage-{i+2} is the zoomed-in of the extracted main object in the {self.marker.colors[i % len(self.marker.colors)]} bounding box from Image-1."

        if len(bboxes) > 1:
            image_prompt += f"\nAnswer the question, based on the provided images (for object/attribute recognition red box: Image-3, green box: Image-4).\n"
        elif len(bboxes) > 0:
            image_prompt += f"\nAnswer the question, based on the provided images (for object/attribute recognition red box: Image-3).\n"        

        question = image_prompt + question 
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(question, image_paths, max_partition=self.max_partition  * len(image_paths))
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id).unsqueeze(0).to(device=self.device)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        pixel_values = [pixel_values.to(dtype=self.dtype, device=self.device)]
        
        # Prepare target tokens
        target_token_ids = self.text_tokenizer(["Yes", "No"], add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        yes_token_id = target_token_ids[0][0].item()
        no_token_id = target_token_ids[1][0].item()

        # Generate outputs with scores
        gen_kwargs = dict(
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
        )

        with torch.inference_mode():
            # _, inputs_embeds, labels, attention_mask = self.model.merge_multimodal(
            # text_input_ids=input_ids,
            # text_attention_masks=gen_kwargs.pop('attention_mask'),
            # text_labels=None,
            # pixel_values=gen_kwargs.pop('pixel_values'),
            # left_padding=True
            # )
            # outputs = self.model.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs)

            gen_kwargs = dict(
                max_new_tokens=1,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                output_scores=True,
                return_dict_in_generate=True,
                eos_token_id=self.model.generation_config.eos_token_id,
                pad_token_id=self.text_tokenizer.pad_token_id,
                use_cache=True
            )
            outputs = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)



        # # Get logits of the generated token
        # next_token_logits = outputs.scores[0][0]  # Shape: [vocab_size]

        # # Extract logits for 'Yes' and 'No'
        # yes_no_logits = next_token_logits[[yes_token_id, no_token_id]]  # Shape: [2]
        # # Temperature scaling
        
        # yes_no_logits = yes_no_logits / temperature  # Scale logits by the temperature
        # probs = torch.softmax(yes_no_logits, dim=-1)
        # yes_prob, no_prob = probs[0], probs[1]
        # # if target_token == "Yes":
        #     return yes_prob
        # else:
        #     return no_prob
         # 2) Extract final-step logits for each output
        top_k = torch.topk(outputs.scores[0][0], k=3)
        other_token_ids = []
        for index in top_k.indices:
            token = self.text_tokenizer.decode(index)
            # print(token, outputs.scores[0][0][index])
            if len(other_token_ids) < 0 and token.lower() not in ["yes", "no"]:
                other_token_ids.append(int(index))
        # Get logits of the generated token
        next_token_logits = outputs.scores[0][0]  # Shape: [vocab_size]
        # Extract logits for 'Yes' and 'No'
        if target_tokens is not None:
            target_tokens = [target_tokens] if isinstance(target_tokens, str) else target_tokens
            target_token_ids = self.text_tokenizer.convert_tokens_to_ids(target_tokens)
            ## squeeze the target_token_ids to match the shape of outputs.scores
            target_token_ids = torch.tensor(target_token_ids).unsqueeze(0).detach().numpy().tolist()[0]
            target_tokens_score = outputs.scores[0][0][target_token_ids]  # shape [num_target_tokens]
            target_prob = torch.softmax(target_tokens_score, dim=-1)
            return target_prob
        else:
            yes_no_logits = next_token_logits[[yes_token_id, no_token_id, *other_token_ids]]  # Shape: [2]
            yes_no_logits = yes_no_logits / 1  # Scale logits by the temperature
            probs = torch.softmax(yes_no_logits, dim=-1)
            return probs[0]

    def _score_contrastive(self, image_masked, image_not_masked, text, target_token="Yes", alpha=1):
        ## Similar to _score and _score_batch but with two images and calculate the difference of scores and the probability yes
        query = f'<image>\n{text} You must answer only with Yes or No.'
        # Preprocess inputs
        if getattr(self.model.llm,"_cache", False) and getattr(self.model.llm._cache, "max_batch_size", None) != 100:
            self.model.llm._cache.max_batch_size = 100
        prompt, input_ids_masked, pixel_values_masked = self.model.preprocess_inputs(query, [image_masked])
        prompt, input_ids_not_masked, pixel_values_not_masked = self.model.preprocess_inputs(query, [image_not_masked])
        attention_mask_masked = torch.ne(input_ids_masked, self.text_tokenizer.pad_token_id).unsqueeze(0).to(device=self.device)
        attention_mask_not_masked = torch.ne(input_ids_not_masked, self.text_tokenizer.pad_token_id).unsqueeze(0).to(device=self.device)
        input_ids_masked = input_ids_masked.unsqueeze(0).to(device=self.device)
        input_ids_not_masked = input_ids_not_masked.unsqueeze(0).to(device=self.device)
        pixel_values_masked = [pixel_values_masked.to(dtype=self.dtype, device=self.device)]
        pixel_values_not_masked = [pixel_values_not_masked.to(dtype=self.dtype, device=self.device)]

        # Prepare target tokens
        target_tokens = ["Yes", "No"]
        target_token_ids = self.text_tokenizer(target_tokens, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        yes_token_id = target_token_ids[0][0].item()
        no_token_id = target_token_ids[1][0].item()

        # Generate outputs with scores
        gen_kwargs_masked = dict(
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pixel_values=pixel_values_masked,
            attention_mask=attention_mask_masked,
        )
        gen_kwargs_not_masked = dict(
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            pixel_values=pixel_values_not_masked,
            attention_mask=attention_mask_not_masked,
        )

        with torch.inference_mode():
            _, inputs_embeds_masked, labels_masked, attention_mask_masked = self.model.merge_multimodal(
            text_input_ids=input_ids_masked,
            text_attention_masks=gen_kwargs_masked.pop('attention_mask'),
            text_labels=None,
            pixel_values=gen_kwargs_masked.pop('pixel_values'),
            left_padding=True
            )
            outputs_masked = self.model.llm.generate(inputs=None, inputs_embeds=inputs_embeds_masked, attention_mask=attention_mask_masked, **gen_kwargs_masked)

            _, inputs_embeds_not_masked, labels_not_masked, attention_mask_not_masked = self.model.merge_multimodal(
            text_input_ids=input_ids_not_masked,
            text_attention_masks=gen_kwargs_not_masked.pop('attention_mask'),
            text_labels=None,
            pixel_values=gen_kwargs_not_masked.pop('pixel_values'),
            left_padding=True
            )
            outputs_not_masked = self.model.llm.generate(inputs=None, inputs_embeds=inputs_embeds_not_masked, attention_mask=attention_mask_not_masked, **gen_kwargs_not_masked)

        # Get logits of the generated token

        next_token_logits_masked = outputs_masked.scores[0][0]  # Shape: [vocab_size]
        next_token_logits_not_masked = outputs_not_masked.scores[0][0]  # Shape: [vocab_size]

        # Extract logits for 'Yes' and 'No'
        yes_no_logits_masked = next_token_logits_masked[[yes_token_id, no_token_id]]  # Shape: [2]
        yes_no_logits_not_masked = next_token_logits_not_masked[[yes_token_id, no_token_id]]  # Shape: [2]
        contrastive_yes_no_logits = (1 + alpha) * yes_no_logits_not_masked - alpha * yes_no_logits_masked
        probs = torch.softmax(contrastive_yes_no_logits, dim=-1)
        yes_prob, no_prob = probs[0], probs[1]
        if target_token == "Yes":
            return yes_prob
        else:
            return no_prob
        
    def _query(self, images, text, max_new_tokens=128):
        if getattr(self.model.llm,"_cache", False) and getattr(self.model.llm._cache, "max_batch_size", None) != 100:
            self.model.llm._cache.max_batch_size = 100
        if not isinstance(images, list):
            images = [images]
        
        query = f'{"<image>"*len(images)}\n{text}'
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, images)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id).unsqueeze(0).to(self.dtype).to(device=self.device)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        pixel_values = [pixel_values.to(dtype=self.dtype, device=self.device)]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True
        )

        
        if getattr(self.model.llm,"_cache", False) and getattr(self.model.llm._cache, "max_batch_size", None) != 1024:
            self.model.llm._cache.max_batch_size = 128
        # if getattr(self.model.llm,"_cache", False) and getattr(self.model.llm._cache, "batch_size", None) != 1:
        #     self.model.llm._cache.batch_size = 1

        with torch.inference_mode():
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        # Remove the prompt from the output to get the assistant's response
        # assistant_response = output_text[len(prompt):].strip()
        return output
    
    def _text_only_query(
        self,
        query: str,
        max_new_tokens=1024,
        ) -> str:
        """
        For text-only question-answering with the model.
        Returns the decoded text from the model's generation.
        """
        
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [])
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id).unsqueeze(0).to(self.dtype).to(device=self.device)
        input_ids = input_ids.unsqueeze(0).to(device=self.device)
        pixel_values = [pixel_values.to(dtype=self.dtype, device=self.device)]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True
        )

        
        if getattr(self.model.llm,"_cache", False) and getattr(self.model.llm._cache, "max_batch_size", None) != 1024:
            self.model.llm._cache.max_batch_size = 128
        
        with torch.inference_mode():
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)
        

    def _score_batch(self, images, texts, target_token="Yes"):
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []

        for image, text in zip(images, texts):
            query = f'<image>\n{text} You must answer only with Yes or No.'
            prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
            pixel_values = [pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]
            batch_input_ids.append(input_ids.squeeze())
            batch_attention_mask.append(attention_mask.squeeze())
            batch_pixel_values.append(pixel_values)

        # Pad input_ids and attention_mask with left-padding
        pad_batch_input_ids = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in batch_input_ids],batch_first=True, padding_value=0.0).flip(dims=[1])
        pad_batch_input_ids =  pad_batch_input_ids[:,-self.model.config.multimodal_max_length:]
        pad_batch_attention_mask = torch.nn.utils.rnn.pad_sequence([i.flip(dims=[0]) for i in batch_attention_mask],batch_first=True, padding_value=False).flip(dims=[1]).to(dtype=self.model.dtype)
        pad_batch_attention_mask = pad_batch_attention_mask[:,-self.model.config.multimodal_max_length:].to(dtype=self.model.dtype)
        pad_batch_pixel_values = [item for sublist in batch_pixel_values for item in sublist]


        # Prepare target tokens
        target_tokens = ["Yes", "No"]
        target_token_ids = self.text_tokenizer(target_tokens, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        yes_token_id = target_token_ids[0][0].item()
        no_token_id = target_token_ids[1][0].item()
        if getattr(self.model.llm,"_cache", False) and getattr(self.model.llm._cache, "max_batch_size", None) != 100:
            self.model.llm._cache.max_batch_size = 100
        # Generate outputs with scores
        gen_kwargs = dict(
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            # use_cache=False,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            pixel_values=pad_batch_pixel_values,
            attention_mask=pad_batch_attention_mask,
            
            
        )
        with torch.inference_mode():
            _, inputs_embeds, labels, attention_mask = self.model.merge_multimodal(
            text_input_ids=pad_batch_input_ids,
            text_attention_masks=gen_kwargs.pop('attention_mask'),
            text_labels=None,
            pixel_values=gen_kwargs.pop('pixel_values'),
            left_padding=True
            )
            outputs = self.model.llm.generate(inputs=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **gen_kwargs)


        # Extract logits of the generated tokens
        next_token_logits = outputs.scores[0]  # Shape: [batch_size, vocab_size]

        # Extract logits for 'Yes' and 'No' for each example
        yes_no_logits = next_token_logits[:, [yes_token_id, no_token_id]]  # Shape: [batch_size, 2]
        probs = torch.softmax(yes_no_logits, dim=-1)
        yes_probs = probs[:, 0].tolist()
        no_probs = probs[:, 1].tolist()

        if target_token == "Yes":
            return yes_probs
        else:
            return no_probs


    def _query_batch(self, images, texts, max_new_tokens=128):
        """
        Process a batch of image-text pairs and generate responses.

        Parameters:
            images (PIL.Image or list of PIL.Image): One or more images.
            texts (str or list of str): One or more text prompts corresponding to the images.
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            List[str]: The generated output for each image-text pair.
        """
        # Ensure that images and texts are lists
        if not isinstance(images, list):
            images = [images]
        if not isinstance(texts, list):
            texts = [texts]
        assert len(images) == len(texts), "Number of images and texts must be equal."

        # Lists to hold individual preprocessed inputs
        batch_input_ids = []
        batch_attention_mask = []
        batch_pixel_values = []

        # Process each image-text pair individually
        for image, text in zip(images, texts):
            # Build the query string for the current pair
            query = f'<image>\n{text}'

            # Preprocess inputs; note that we pass the image as a singleton list.
            # This returns the prompt string, input_ids tensor, and pixel_values tensor.
            prompt, input_ids, pixel_values = self.model.preprocess_inputs(query, [image])

            # Create attention mask by marking non-pad tokens.
            # (Assumes that self.text_tokenizer.pad_token_id is the pad value.)
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)

            # Move the tensors to the proper device/dtype.
            input_ids = input_ids.to(device=self.device)
            attention_mask = attention_mask.to(device=self.device)
            # Wrap pixel_values in a list (to match the expected format) and move it.
            pixel_values = [pixel_values.to(dtype=self.dtype, device=self.device)]

            # Append these to the batch lists.
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_pixel_values.append(pixel_values)

        # Pad and truncate the input_ids.
        # The trick with .flip() left-pads the sequences (i.e. pads on the left)
        pad_batch_input_ids = torch.nn.utils.rnn.pad_sequence(
            [i.flip(dims=[0]) for i in batch_input_ids],
            batch_first=True,
            padding_value=self.text_tokenizer.pad_token_id
        ).flip(dims=[1])
        pad_batch_input_ids = pad_batch_input_ids[:, -self.model.config.multimodal_max_length:]

        # Similarly pad/truncate the attention masks.
        pad_batch_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [i.flip(dims=[0]) for i in batch_attention_mask],
            batch_first=True,
            padding_value=False
        ).flip(dims=[1])
        pad_batch_attention_mask = pad_batch_attention_mask[:, -self.model.config.multimodal_max_length:]

        # Flatten the list of pixel values (each is wrapped in a list).
        pad_batch_pixel_values = [item for sublist in batch_pixel_values for item in sublist]

        # (Optional) Ensure the model's cache supports the batch size.
        if getattr(self.model.llm, "_cache", False) and getattr(self.model.llm._cache, "max_batch_size", None) != 1024:
            self.model.llm._cache.max_batch_size = 1024

        # Set up generation keyword arguments.
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True
        )

        # Generate outputs for the entire batch.
        with torch.inference_mode():
            output_ids = self.model.generate(
                pad_batch_input_ids,
                pixel_values=pad_batch_pixel_values,
                attention_mask=pad_batch_attention_mask,
                **gen_kwargs
            )

        # Decode each generated sequence and return the list of outputs.
        outputs = [self.text_tokenizer.decode(seq, skip_special_tokens=True) for seq in output_ids]
        return outputs

    def _normalize_coordinates(box, image_width, image_height):
        return box

    def _parse_vlm_bbox_output(self, text_response: str, width: int, height: int) -> Optional[List[float]]:
        """Parses VLM text output for the first [[x1,y1,x2,y2]] pattern."""
        # Keep from previous adaptation (adjust regex/normalization if needed)
        try:
            bbox_matches = BBOX_PATTERN_VLM.findall(text_response)
            if bbox_matches:
                coords_str = bbox_matches[0]
                nx1, ny1, nx2, ny2 = [float(c.strip()) for c in coords_str]
                # print(f"Parsed VLM box in function Before: {[nx1, ny1, nx2, ny2]}")

                if float(nx1) <= 1 and float(nx2) <= 1 and float(ny1) <= 1 and float(ny2) <= 1:
                    # Denormalize assuming [0, 1] range from VLM
                    x1 = round(nx1 * width)
                    y1 = round(ny1 * height)
                    x2 = round(nx2 * width)
                    y2 = round(ny2 * height)
                    # print(f"Parsed VLM box in function after: {[x1, y1, x2, y2]}")
                else: 
                    # Denormalize assuming [0, 1000] range from VLM
                    x1 = round((nx1 / 1000) * width)
                    y1 = round((ny1 / 1000) * height)
                    x2 = round((nx2 / 1000) * width)
                    y2 = round((ny2 / 1000) * height)
                # Basic sanity check on coordinates
                # print(f"Parsed VLM box in function: {[x1, y1, x2, y2]}")
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
        
        vlm_bbox_prompt = f"Please provide the bounding box coordinates of the region this sentence describes:<ref>{phrase}</ref>"
        vlm_direct_answer = self._query(image, vlm_bbox_prompt)
        # print(f"## ground VLM direct answer: {vlm_direct_answer}")
        vlm_direct_box = self._parse_vlm_bbox_output(vlm_direct_answer, image.size[0], image.size[1])
        # print(f"## ground VLM direct parsed: {vlm_direct_answer}")
        return vlm_direct_box

    def suggest_objects(self, image, query, objects):
        vlm_object_query = f"Please provide the bounding box coordinates and label of all objects like {', '.join(objects)} related to this query: \"{query}\" in the following image."
        vlm_direct_answer = self._query(image, vlm_object_query, max_new_tokens=1024)
        return vlm_object_query, vlm_direct_answer 