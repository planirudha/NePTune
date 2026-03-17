import torch
import re
from PIL import Image
from .agent import Agent
from qwen_vl_utils import process_vision_info


class QwenVL2(Agent):
#     concept2query = {
#         "attribute": {
#             # Material-related attributes
#             "metal": {
#                 "query": "Is the main object in the red bounding box metallic?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "rubber": {
#                 "query": "Is the main object in the red bounding box matte?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             # Size-related attributes
#             "large": {
#                 "query": "Is the main object in the red bounding box large?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False,
#             },
#             "small": {
#                 "query": "Is the main object in the red bounding box small?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             # Color-related attributes
#             "brown": {
#                 "query": "Is the main object in the red bounding box brown colored?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "green": {
#                 "query": "Is the main object in the red bounding box green colored?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "purple": {
#                 "query": "Is the main object in the red bounding box purple colored?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "gray": {
#                 "query": "Is the main object in the red bounding box gray colored?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "yellow": {
#                 "query": "Is the main object in the red bounding box yellow colored?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "blue": {
#                 "query": "Is the main object in the red bounding box blue colored?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "red": {
#                 "query": "Is the main object in the red bounding box red colored?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "cyan": {
#                 "query": "Is the main object in the red bounding box cyan colored?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             # Shape-related attributes
#             "sphere": {
#                 "query": "Is the main object in the red bounding box a sphere?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "cube": {
#                 "query": "Is the main object in the red bounding box a cube?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "cylinder": {
#                 "query": "Is the main object in the red bounding box a cylinder?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#         },
#         "relation": {
#             "left": {
#                 "query": "Is the main object in the red bounding box left of the main object in the green bounding box?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "behind": {
#                 "query": "Is the object in the red bounding box behind the object in the green bounding box?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox":  False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "front": {
#                 "query": "Is the main object in the red bounding box front of the main object in the green bounding box?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox":  False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "right": {
#                 "query": "Is the main object in the red bounding box right of the main object in the green bounding box?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox":  False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "same_color": {
#                 "query": "Are the two main objects a similar color?",
#                 "mask": True,
#                 "draw_bbox": False,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "same_material": {
#                 "query": "Is the main object in the red bounding box the same material as the main object the green bounding box?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "same_shape": {
#                 "query": "Are the two main objects similar shape?",
#                 "mask": True,
#                 "draw_bbox": False,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#             "same_size": {
#                 "query": "Is the main object in the red bounding box the same size as the main object the green bounding box?",
#                 # "query": "Are the two main objects similar size?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False

#             },
#         },
#         "multi_relation": {
#             "between": {
#                 "query": "Is the main object in the red bounding box between the main objects in the green and blue bounding boxes?",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "pass_bbox": False,
#                 "depth_estimation": False,
#                 "add_description": False
#             },
#         },
#         "description": {
#             "color": {
#                 "query": "What is the color of the main object? (e.g., red, blue, purple, green, yellow, cyan, gray or brown)",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "depth_estimation": False
#             },
#             "shape": {
#                 "query": "What is the shape of the main object? (e.g. sphere, cube, or cylinder)",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "depth_estimation": False
#             },
#             "size": {
#                 "query": "What is the size of the main object in the red bounding box? (e.g., small or large)",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "depth_estimation": False
#             },
#             "material": {
#                 "query": "What is the material of the main object? (e.g., rubber or metal)",
#                 "mask": False,
#                 "draw_bbox": True,
#                 "depth_estimation": False
#             },
#         }
#     }
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
                "query": "Is the main object in the red bounding box the same material as the main object in the green bounding box?",
                "mask": False,
                "draw_bbox": True,
                "pass_bbox": False,
                "depth_estimation": False,
                "add_description": False,
                "type": "relation",
            },
            "same_shape": {
                "query": "Is the main object in the red bounding box the same base shape as the main object the green bounding box?",
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
    def __init__(self, model_name="Qwen/Qwen2-VL-7B-Instruct", device="cuda", *args, **kwargs):
        super().__init__(*args,**kwargs)  # Initialize the parent class properly
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True,
            attn_implementation="flash_attention_2",

        )
        self.vlm_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.model.cuda()
        self.device = "cuda"
    
    def _score(self, images, text, target_token="Yes", candidates: list=None, temperature=0.5, target_tokens=None, guidance_scale=None):
        if not isinstance(images, list):
            images = [images]
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{
                        "type": "image",
                        "image": img,
                    } for img in images],
                    {"type": "text", "text": text},
                ],
            }
        ]
        # Preprocess the inputs
        text_prompt = self.vlm_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        inputs = self.vlm_processor(
            text=[text_prompt], images=images, padding=True, return_tensors="pt"
        ).to(self.device)



        with torch.no_grad():
            ouput = self.model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, do_sample=False, top_k=None, top_p=None, temperature=None)
        logits = ouput.scores[0]
        ## print the top-k tokens
        top_k = torch.topk(logits, k=10)
        for index in top_k.indices[0]:
            token = self.vlm_processor.tokenizer.decode(index)
            # print(token, logits[0][index].item())
        
        if target_tokens is None:
            yes_target_id = self.vlm_processor.tokenizer("Yes", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item()
            no_target_id = self.vlm_processor.tokenizer("No", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item()        
            yes_or_no = logits[0,[yes_target_id,no_target_id]]
            yes_or_no = yes_or_no / 1
            yes_prob, no_prob = torch.softmax(yes_or_no, dim=-1)
            if target_token != "Yes":
                return no_prob
            return yes_prob
        else:
            target_ids = [self.vlm_processor.tokenizer(token, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item() for token in target_tokens]
            target_logits = logits[0,target_ids]
            target_logits = target_logits / temperature
            target_probs = torch.softmax(target_logits, dim=-1)
            return target_probs
    def _score_with_mask(self, image: 'PIL.Image', question: str, bboxes: list, candidates: list=None, type="default", history=None, target_tokens=None):
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
            question = question.replace("the red bounding box", "Image-3")
            # print("Updated question", type, question)
        if type == "relation":
            ## crop the marked image to the bounding box
            min_x = min([bbox[0] for bbox in bboxes])
            min_y = min([bbox[1] for bbox in bboxes])
            max_x = max([bbox[2] for bbox in bboxes])
            max_y = max([bbox[3] for bbox in bboxes])
            ## add some padding
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(image.size[0], max_x + padding)
            max_y = min(image.size[1], max_y + padding)
            
            marked_image = marked_image.crop((min_x, min_y, max_x, max_y))


        # image_paths = [marked_image, *extra_images]
        image_paths = [marked_image, marked_zero_image, *extra_images]
        image_prompt = ""
        if len(extra_images) > 0:
            image_prompt += "Image-1 is the source image."
            image_prompt += f"\nImage-2 shows the position of bounding boxes."
        for i in range(len(extra_images)):
            # image_prompt += f"\nImage-{i+3} is the extracted object of the {self.marker.colors[i]} bounding box from Image-1."
            image_prompt += f"\nImage-{i+3} is the zoomed-in of the extracted main object in the {self.marker.colors[i % len(self.marker.colors)]} bounding box from Image-1."

        if len(bboxes) > 1:
            image_prompt += f"\nAnswer the question, based on the provided images (for object/attribute recognition red box: Image-3, green box: Image-4).\n"
        elif len(bboxes) > 0:
            image_prompt += f"\nAnswer the question, based on the provided images (for object/attribute recognition red box: Image-3).\n"

        image_prompt += f"\nQuestion: {question}\n"
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{
                        "type": "image",
                        "image": img,
                    } for img in image_paths],
                    {"type": "text", "text": image_prompt},
                ],
            }
        ]
        # Preprocess the inputs
        text_prompt = self.vlm_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, add_vision_id=True)

        inputs = self.vlm_processor(
            text=[text_prompt], images=image_paths, padding=True, return_tensors="pt"
        ).to(self.device)


        with torch.no_grad():
            ouput = self.model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, do_sample=False, top_k=None, top_p=None, temperature=None)
        logits = ouput.scores[0]
        top_k = torch.topk(logits, k=10)
        # for index in top_k.indices[0]:
        #     token = self.vlm_processor.tokenizer.decode(index)
        #     print(token, logits[0][index].item())
        if target_tokens is None:
            yes_target_id = self.vlm_processor.tokenizer("Yes", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item()
            no_target_id = self.vlm_processor.tokenizer("No", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item()        
            yes_or_no = logits[0, [yes_target_id,no_target_id]]
            yes_or_no = yes_or_no / 1
            yes_prob, no_prob = torch.softmax(yes_or_no, dim=-1)
            return yes_prob
        else:
            target_ids = [self.vlm_processor.tokenizer(token, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item() for token in target_tokens]
            target_logits = logits[0, target_ids]
            target_logits = target_logits / 1
            target_probs = torch.softmax(target_logits, dim=-1)
            return target_probs

    def _score_batch(self, batch_images, texts, target_token="Yes", temperature = 0.5, guidance_scale=None):
        if not isinstance(batch_images, list):
            batch_images = [batch_images]
        if not isinstance(texts, list):
            texts = [texts]
        conversations = []
        for text, images in zip(texts, batch_images):
            if not isinstance(images, list):
                images = [images] 
            conversation = [
                {
                    "role": "user",
                    "content": [
                        *[{
                            "type": "image",

                        } for _ in images],
                        {"type": "text", "text": text},
                    ],
                }
            ]
            conversations.append(conversation)

        # Preprocess the inputs
        texts_prompts = [self.vlm_processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]

        inputs = self.vlm_processor(
            text=texts_prompts, images=batch_images, padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, output_scores=True, do_sample=False, top_k=None, top_p=None, temperature=None)
        

        logits = output.scores[0]
        yes_target_id = self.vlm_processor.tokenizer("Yes", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item()
        no_target_id = self.vlm_processor.tokenizer("No", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item()        
        yes_or_no = logits[:,[yes_target_id,no_target_id]].float()            
        yes_or_no = yes_or_no / temperature
        yes_no_probs = torch.softmax(yes_or_no, dim=1)
        if target_token != "Yes":
            return yes_no_probs[:,1]
        return yes_no_probs[:,0]

    def _score_contrastive(self, image_masked, image_not_masked, text, target_token="Yes", alpha=1):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]



        # Preprocess the inputs
        text_prompt = self.vlm_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

        # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

        inputs_masked = self.vlm_processor(
            text=[text_prompt], images=[image_masked], padding=True, return_tensors="pt"
        ).to(self.device)

        inputs_not_masked = self.vlm_processor(
            text=[text_prompt], images=[image_not_masked], padding=True, return_tensors="pt"
        ).to(self.device)
        yes_target_id = self.vlm_processor.tokenizer("Yes", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item()
        no_target_id = self.vlm_processor.tokenizer("No", add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)[0][0].item()

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                res_masked = self.model(**inputs_masked)
                res_not_masked = self.model(**inputs_not_masked)
                logits_masked = res_masked.logits
                logits_not_masked = res_not_masked.logits
                shift_logits_masked = logits_masked[..., :-1, :].contiguous()
                shift_logits_not_masked = logits_not_masked[..., :-1, :].contiguous()                
                new_logits =  (1 + alpha) * shift_logits_not_masked - alpha * shift_logits_masked
                yes_or_no = torch.tensor([new_logits[0, -1][yes_target_id], new_logits[0, -1][no_target_id]]).float()            
                yes_prob, no_prob = torch.softmax(yes_or_no, dim=0)
        if target_token != "Yes":
            return no_prob
        return yes_prob

    def _query(self, image, text, max_new_tokens=128):
        if not isinstance(image, list):
            image = [image]
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{
                        "type": "image",
                        "image": img,
                    } for img in image],
                    {"type": "text", "text": text},
                ],
            }
        ]
        text_prompt = self.vlm_processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.vlm_processor(
            text=[text_prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)
        with torch.amp.autocast(self.device):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        return output_text[0]
    
    def _query_batch(self, images, texts, max_new_tokens=128, do_sample=True):
        conversations = []
        for text in texts:
            conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
            conversations.append(conversation)
        # Preparation for batch inference
        texts_prompts = [self.vlm_processor.apply_chat_template(msg, add_generation_prompt=True) for msg in conversations]
        inputs = self.vlm_processor(
            text=texts_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to('cuda')

        # Batch Inference
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=1)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.vlm_processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text


    def ground(self, image, phrase):
        if not isinstance(image, list):
            image = [image]
        text = f"<ref>{phrase}</ref><box>"
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{
                        "type": "image",
                        "image": img,
                    } for img in image],
                    {"type": "text", "text": text},
                ],
            }
        ]
        text_prompt = self.vlm_processor.apply_chat_template(conversation, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = self.vlm_processor(
            text=[text_prompt],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.device)
        with torch.amp.autocast(self.device):
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        ## extract the numbers from small spehre(571,681),(681,837)
        pattern = r"(\d+)"
        matches = re.findall(pattern, output_text[0])

        if len(matches) == 4:
            coordinates = [int(matches[0]), int(matches[1]), int(matches[2]), int(matches[3])]
        else:
            coordinates = [0, 0, 1000, 1000]
        ## coordinates are normalized to [0, 1000] from the original image size
        x1, y1, x2, y2 = [c / 1000 for c in coordinates]
        ## now multiply by the image size
        image_size = image[0].size
        x1, y1 = int(x1 * image_size[0]), int(y1 * image_size[1])
        x2, y2 = int(x2 * image_size[0]), int(y2 * image_size[1])

        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_size[0], x2), min(image_size[1], y2)
        return [x1, y1, x2, y2]  # Return as a list of bounding boxes