import torch
import os
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List, Tuple
import logging
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

class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = '{system_message}'
    # The system message
    system_message: str = ''
    # The names of two roles
    roles: Tuple[str] = ('USER', 'ASSISTANT')
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = '\n'
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ': '  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = '' if system_prompt == '' else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ': '
                        + message.replace('\r\n', '\n').replace('\n\n', '\n')
                    )
                    ret += '\n\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = '[INST] '
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + ' '
                    else:
                        ret += tag + ' ' + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == 'chatglm2' else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ''

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f'[Round {i//2 + round_add_n}]{self.sep}'

                if message:
                    ret += f'{role}：{message}{self.sep}'
                else:
                    ret += f'{role}：'
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = '' if system_prompt == '' else system_prompt + self.sep + '\n'
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + message + self.sep + '\n'
                else:
                    ret += role + '\n'
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ''
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + '\n' + ' ' + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ':' + message + seps[i % 2] + '\n'
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ':\n' + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += '\n\n'
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + '<s>' + message + '</s>'
                else:
                    ret += role + ': ' + '<s>'
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ':\n' + message + self.sep
                else:
                    ret += role + ':\n'
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ''
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ': ' + message + self.sep
                else:
                    ret += role + ':'

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ': ' + message + seps[i % 2]
                else:
                    ret += role + ':'
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f'Invalid style: {self.sep_style}')

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{'role': 'system', 'content': self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({'role': 'user', 'content': msg})
            else:
                if msg is not None:
                    ret.append({'role': 'assistant', 'content': msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            'template_name': self.name,
            'system_message': self.system_message,
            'roles': self.roles,
            'messages': self.messages,
            'offset': self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f'{template.name} has been registered.'

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# Both Hermes-2 and internlm2-chat are chatml-format conversation templates. The difference
# is that during training, the preprocessing function for the Hermes-2 template doesn't add
# <s> at the beginning of the tokenized sequence, while the internlm2-chat template does.
# Therefore, they are completely equivalent during inference.
register_conv_template(
    Conversation(
        name='Hermes-2',
        system_template='<|im_start|>system\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>',
        stop_str='<|endoftext|>',
    )
)


register_conv_template(
    Conversation(
        name='internlm2-chat',
        system_template='<|im_start|>system\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>',
    )
)


register_conv_template(
    Conversation(
        name='phi3-chat',
        system_template='<|system|>\n{system_message}',
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        system_message='你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。',
        roles=('<|user|>\n', '<|assistant|>\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|end|>',
    )
)


register_conv_template(
    Conversation(
        name='internvl2_5',
        system_template='<|im_start|>system\n{system_message}',
        system_message='你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。',
        roles=('<|im_start|>user\n', '<|im_start|>assistant\n'),
        sep_style=SeparatorStyle.MPT,
        sep='<|im_end|>\n',
    )
)






# 1. Put the relevant image-processing functions here or import them
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(
    image: Image.Image,
    min_num=1,
    max_num=24,
    image_size=448,
    use_thumbnail=False
) -> List[Image.Image]:
    """
    Splits the image into multiple tiles if needed. 
    The code is from your snippet, unchanged (except it returns a list of PIL images).
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # 1) find the closest aspect ratio, 2) resize the image, 3) crop into blocks
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                # tie-breaker
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    target_ratios = set(
        (i, j) 
        for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images

def load_image(image_file: str, input_size=448, max_num=12):
    if isinstance(image_file, Image.Image):
        image = image_file
    else:
        image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)  # shape: [num_tiles, 3, H, W]
    return pixel_values


# 2. Import or define your conversation templates and relevant logic
#    (this is your big conversation.py snippet).
#    For brevity, I will assume you have the same snippet in conversation.py
#    or you keep it inline. We only need the get_conv_template, etc.
# from conversation import get_conv_template, conv_templates, SeparatorStyle, Conversation
# We'll assume those are already defined in your environment.


################################################################################
# Example: InternVL class
################################################################################
from .agent import Agent

class InternVLTuned(Agent):
    """
    An example agent that mirrors your QwenVL style but uses 
    InternVL for scoring yes/no and generating text answers.
    """
    # concept2query = {
    # "attribute": {
    #     # Material-related attributes
    #     "metal": {
    #         "query": "Is the main object metallic?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False,
    #     },
    #     "rubber": {
    #         "query": "Is the main object matte?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     # Size-related attributes
    #     "large": {
    #         "query": "Is the main object in the red bounding box large?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "small": {
    #         "query": "Is the main object in the red bounding box small?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     # Color-related attributes
    #     "brown": {
    #         "query": "Is the main object brown colored?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "green": {
    #         "query": "Is the main marked object green colored?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "purple": {
    #         "query": "Is the main object purple colored?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "gray": {
    #         "query": "Is the main marked object gray colored?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "yellow": {
    #         "query": "Is the main object yellow colored?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "blue": {
    #         "query": "Is the marked object blue colored?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "red": {
    #         "query": "Is the main object red colored?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "cyan": {
    #         "query": "Is the main object cyan colored?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     # Shape-related attributes
    #     "sphere": {
    #         "query": "Is the main object a sphere?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
            
    #     },
    #     "cube": {
    #         "query": "Is the main object a cube?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "cylinder": {
    #         "query": "Is the main marked object a cylinder?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    # },
    # "relation": {
    #     "left": {
    #         "query": "Is the main object in the red bounding box left of the main object in the green bounding box?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "behind": {
    #         "query": "Is the main object in the red bounding box behind the main object in the green bounding box?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False

    #     },
    #     "front": {
    #         "query": "Is the main object in the red bounding box front of the main object in the green bounding box?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "right": {
    #         "query": "Is the main object in the red bounding box right of the main object in the green bounding box?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "same_color": {
    #         "query": "Are the two main non-masked objects a similar color?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "same_material": {
    #         "query": "Does the object in the red bounding box have similar material as the object in the green bounding box?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    #     "same_shape": {
    #         "query": "Are the two main non-masked objects similar shape?",
    #         "mask": True,
    #         "draw_bbox": False,
    #         "pass_bbox": False,
    #         "depth_estimation": False

    #     },
    #     "same_size": {
    #         "query": "Are the main object in the red bounding box the similar size as the main object the green bounding box?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    # },
    # "multi_relation": {
    #     "between": {
    #         "query": "Is the main object in the red bounding box between the main objects in the green and blue bounding boxes?",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "pass_bbox": False,
    #         "depth_estimation": False
    #     },
    # },
    # "description": {
    #     "color": {
    #         "query": "What is the color of the main object? (e.g., red, blue, purple, green, yellow, cyan, gray or brown)",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "depth_estimation": False

    #     },
    #     "shape": {
    #         "query": "What is the shape of the main object? (e.g. sphere, cube, or cylinder)",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "depth_estimation": False
    #     },
    #     "size": {
    #         "query": "What is the size of the main object in the red bounding box? (e.g., small or large)",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "depth_estimation": False
    #     },
    #     "material": {
    #         "query": "What is the material of the main object? (e.g., rubber or metal)",
    #         "mask": False,
    #         "draw_bbox": True,
    #         "depth_estimation": False
    #     },
    # }
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
        lora_path: str = None,
        device: str = "cuda",
        load_4bit: bool = False,
        max_num_patches: int = 24,
        trust_remote_code: bool = True,
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
        # Use the provided lora_path if given, otherwise fall back to model_path
        lora_path = lora_path if lora_path is not None else model_path
        print(f"Loading InternVL tuned model from {lora_path} ...")
        device_id = 0 if device == "cuda" else -1
        self.device = f"cuda:{device_id}"
        self.max_num_patches = max_num_patches
        logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

        # 1) Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            lora_path, 
            model_max_length=8192,
            trust_remote_code=trust_remote_code
        )

        # 2) Load base InternVL model
        self.model = AutoModel.from_pretrained(
            lora_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=trust_remote_code,
            device_map={"": device_id},
        ).eval()

        # self.model.cuda()
        # InternVL uses special tokens for images
        # We'll store them for convenience.
        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN   = "</img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

        # We'll store a conversation template pointer if the model has one
        if hasattr(self.model, "template"):
            self.template = get_conv_template(self.model.template)
            self.template.system_message = self.model.system_message
            # sep token
            self.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.template.sep.strip())
        else:
            self.template = None
            self.eos_token_id = self.tokenizer.eos_token_id

    def _build_prompt(
        self, 
        question: str, 
        history: Optional[List[Tuple[str,str]]] = None,
        pixel_values: Optional[torch.Tensor] = None, 
        num_patches_list: Optional[List[int]] = None
    ) -> str:
        """
        Builds the conversation prompt string given a user question 
        and an optional chat history. 
        InternVL expects <image> tokens replaced with <img><IMG_CONTEXT>...</img>.
        """
        
        if self.template is None:
            raise ValueError("No conversation template found for InternVL. Cannot build prompt.")

        # Clear old messages from the template
        self.template.messages = []

        # Append old conversation if any
        if history is not None:
            for (old_q, old_a) in history:
                self.template.append_message(self.template.roles[0], old_q)
                self.template.append_message(self.template.roles[1], old_a)

        # If you have pixel_values, that means we are doing an image question
        # If question doesn't already have <image>, prepend it.
        if pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        # Now add the new user turn and a placeholder for assistant
        self.template.append_message(self.template.roles[0], question)
        self.template.append_message(self.template.roles[1], None)

        prompt = self.template.get_prompt()

        # If we do have an image, replace the first occurrence of <image> with 
        # a block of <img><IMG_CONTEXT_TOKEN * num_image_tokens></img>.
        if pixel_values is not None:
            if num_patches_list is None:
                num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            assert pixel_values is None or len(pixel_values) == sum(num_patches_list)        
            for num_patches in num_patches_list:
                image_tokens = self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + self.IMG_END_TOKEN
                prompt = prompt.replace('<image>', image_tokens, 1)

        return prompt

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
        num_patches_list = []
        # Prepare the pixel_values
        image_prompt = ""
        if len(image_paths) == 1:
            image_prompt = "<image>\n"
            pixel_values = load_image(image_paths[0], max_num=self.max_num_patches).to(torch.bfloat16).to(self.device)
            num_patches_list = None
        else:
            pixel_values = []
            for image_id,image_path in enumerate(image_paths):
                pixel_values.append(load_image(image_path, max_num=self.max_num_patches).to(torch.bfloat16).to(self.device))
                num_patches_list.append(pixel_values[image_id].shape[0])
                image_prompt = f"Image-{image_id}: <image> " + image_prompt
            pixel_values = torch.cat(pixel_values, dim=0)
        if candidates is not None:
            question = f"{question}\n If you have to classify among one of these objects {candidates}, {question}"
        else:
            question = f"{image_prompt}\n{question}"
        # Build the prompt with a single user question
        prompt = self._build_prompt(question, history=None, pixel_values=pixel_values, num_patches_list=num_patches_list)

        # Tokenize
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)

        # Generate exactly 1 new token to see if it's "Yes" or "no"
        generation_config = dict(
            max_new_tokens=1, 
            do_sample=False, 
            return_dict_in_generate=True, 
            output_scores=True,
            eos_token_id=self.eos_token_id,
            # guidance_scale=guidance_scale
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )

        # We only have 1 step of generation, so the relevant logits are generation_output.scores[0]
        # This is shape: [batch_size, vocab_size]
        scores = generation_output.scores[0]  # shape [1, vocab_size]
        top_k = torch.topk(scores, k=10)
        if target_tokens is None:
            yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
            no_token_id  = self.tokenizer.convert_tokens_to_ids("No")
            other_token_ids = []
            for index in top_k.indices[0]:
                token = self.tokenizer.decode(index)
                # print(token, scores[0][index].item())

                if len(other_token_ids) < 1 and token.lower() not in ["yes", "no"]: ## 3 for CLEVR
                    # print("ADDED", token)
                    other_token_ids.append(int(index))
            yes_no_logits = scores[:, [yes_token_id, no_token_id, *other_token_ids]]

            yes_no_logits = yes_no_logits / 1
            yes_no_probs  = torch.softmax(yes_no_logits, dim=-1)
            yes_prob = yes_no_probs[0][0]
            
            return yes_prob if target_token.lower() == "yes" else 1 - yes_prob
        else:
            target_tokens = [target_tokens] if isinstance(target_tokens, str) else target_tokens
            target_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            target_token_ids = [token_id for token_id in target_token_ids if token_id != -1]  # filter out invalid tokens
            if not target_token_ids:
                return 0.0
            target_tokens_score = scores[:, target_token_ids]  # shape [1, num_target_tokens]
            target_prob = torch.softmax(target_tokens_score, dim=-1)
            return target_prob



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
        image_prompt = "Image-1: <image>\n"
        for i in range(0, len(extra_images)+1):
            image_prompt += f"Image-{i+2}: <image>\n"
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
        num_patches_list = []
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 6))
        # plt.subplot(1, len(extra_images)+2, 1)
        # plt.imshow(marked_image)
        # plt.axis("off")
        # plt.title("Marked Image")
        # if len(extra_images) > 0:
        #     plt.subplot(1, len(extra_images)+2, 2)
        #     plt.imshow(marked_zero_image)
        #     plt.axis("off")
        #     plt.title("Marked Image with BBoxes")
        #     plt.subplot(1, len(extra_images)+2, 3)
        # for i, extra_image in enumerate(extra_images):
        #     plt.subplot(1, len(extra_images)+2, i+3)
        #     plt.imshow(extra_image)
        #     plt.axis("off")
        #     plt.title(f"Image {i+1}")
        # plt.show()
        pixel_values = []
        for image_id,image_path in enumerate(image_paths):
            pixel_values.append(load_image(image_path, max_num=self.max_num_patches).to(torch.bfloat16).to("cuda:0"))
            num_patches_list.append(pixel_values[image_id].shape[0])
        pixel_values = torch.cat(pixel_values, dim=0)
        if candidates is not None:
            question = f"{image_prompt}\nIf you have to classify among one of these objects {candidates}, {question}"
        else:
            question = f"{image_prompt}\nWith the most strict bounding boxes, {question}"
        question += "\nAnswer the question using a single word or phrase."
        # print(question)
        # print(history)
        prompt = self._build_prompt(question, history=history, pixel_values=pixel_values, num_patches_list=num_patches_list)
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to("cuda:0")
        attention_mask = model_inputs["attention_mask"].to("cuda:0")
        generation_config = dict(
            max_new_tokens=1, 
            do_sample=False, 
            return_dict_in_generate=True, 
            output_scores=True,
            eos_token_id=self.eos_token_id,
            # guidance_scale=guidance_scale
        )

        with torch.no_grad():
            generation_output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        scores = generation_output.scores[0]  # shape [1, vocab_size]
        top_k = torch.topk(scores, k=10)
        if target_tokens is None:
            yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
            no_token_id  = self.tokenizer.convert_tokens_to_ids("No")
            other_token_ids = []
            
            for index in top_k.indices[0]:
                token = self.tokenizer.decode(index)
                # print(token, scores[0][index].item())
                if len(other_token_ids) < 0 and token.lower() not in ["yes", "no"]:
                    other_token_ids.append(int(index))
            yes_no_logits = scores[:, [yes_token_id, no_token_id, *other_token_ids]]

            yes_no_logits = yes_no_logits / 1
            yes_no_probs  = torch.softmax(yes_no_logits, dim=-1)
            yes_prob = yes_no_probs[0][0]
            
            return yes_prob
        else:
            target_tokens = [target_tokens] if isinstance(target_tokens, str) else target_tokens
            target_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in target_tokens]
            target_token_ids = [token_id for token_id in target_token_ids if token_id != -1]  # filter out invalid tokens
            if not target_token_ids:
                return 0.0
            target_tokens_score = scores[:, target_token_ids]  # shape [1, num_target_tokens]
            target_prob = torch.softmax(target_tokens_score, dim=-1)
            return target_prob

    def _score_batch(self, image_paths: List[str], questions: List[str], target_token="Yes", temperature=0.5):
        prompts = []
        pixel_values = []
        

        with torch.no_grad():
            for image_path, question in zip(image_paths, questions):
                pixel_value = load_image(image_path, max_num=self.max_num_patches).to(torch.bfloat16).to(self.device)
                pixel_values.append(pixel_value)
                prompt = self._build_prompt(question, history=None, pixel_values=pixel_value)
                prompts.append(prompt)

            model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = model_inputs["input_ids"].to(self.device)
            attention_mask = model_inputs["attention_mask"].to(self.device)
                
            generation_config = dict(
                max_new_tokens=1, 
                do_sample=False,
                return_dict_in_generate=True, 
                output_scores=True,
                eos_token_id=self.eos_token_id
            )
            generation_output = self.model.generate(
                pixel_values=torch.cat(pixel_values, dim=0),
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
        generation = torch.stack(generation_output.scores, dim=0)

        # We only have 1 step of generation, so the relevant logits are generation_output.scores[0]
        # This is shape: [batch_size, vocab_size]
        yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        no_token_id  = self.tokenizer.convert_tokens_to_ids("No")


        yes_no_logits = generation[:,:, [yes_token_id, no_token_id]]  # shape [1, 2]
        yes_no_logits = yes_no_logits / temperature
        yes_no_probs  = torch.softmax(yes_no_logits, dim=-1)[0][:,0]
        return yes_no_probs

    def _score_contrastive(
        self, 
        image_masked_path: str, 
        image_not_masked_path: str,
        question: str, 
        target_token="Yes", 
        alpha=1.0,
        apply_alpha_on="probs",
    ):
        """
        Compares two images (masked vs. unmasked) by 
        computing a difference in logits for "Yes"/"no" tokens.
        This is an example logic adapted from QwenVL. 
        """

        # 1) Load both images
        pixel_values_masked = load_image(image_masked_path, max_num=self.max_num_patches).to(torch.bfloat16).to(self.device)
        pixel_values_not_masked = load_image(image_not_masked_path, max_num=self.max_num_patches).to(torch.bfloat16).to(self.device)

        # 2) Build prompts for both
        prompt_masked = self._build_prompt(question, history=None, pixel_values=pixel_values_masked)
        prompt_not_masked = self._build_prompt(question, history=None, pixel_values=pixel_values_not_masked)

        # 3) Tokenize
        inputs_masked = self.tokenizer(prompt_masked, return_tensors="pt").to(self.device)
        inputs_not_masked = self.tokenizer(prompt_not_masked, return_tensors="pt").to(self.device)

        # 4) Forward pass (1 new token). We only need logits, so set max_new_tokens=1
        gen_cfg = dict(
            max_new_tokens=1, 
            do_sample=False, 
            return_dict_in_generate=True, 
            output_scores=True,
            eos_token_id=self.eos_token_id
        )
        with torch.no_grad():
            out_masked = self.model.generate(
                pixel_values=pixel_values_masked,
                input_ids=inputs_masked["input_ids"], 
                attention_mask=inputs_masked["attention_mask"],
                **gen_cfg
            )
            out_not_masked = self.model.generate(
                pixel_values=pixel_values_not_masked,
                input_ids=inputs_not_masked["input_ids"], 
                attention_mask=inputs_not_masked["attention_mask"],
                **gen_cfg
            )

        logits_masked = out_masked.scores[0]       # [1, vocab_size]
        logits_not_masked = out_not_masked.scores[0]  # [1, vocab_size]

        yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        no_token_id  = self.tokenizer.convert_tokens_to_ids("no")

        # Combine logits
        # new_logits = (1 + alpha) * logits_not_masked - alpha * logits_masked
        # This is your optional contrastive formula from the snippet:
        # Inside _score_contrastive, replace the selected block with:

        # add a flag to choose where to apply alpha, e.g. at method signature:
        # def _score_contrastive(..., alpha=1.0, apply_alpha_on="logits"):

        # compute indices
        yes_token_id = self.tokenizer.convert_tokens_to_ids("Yes")
        no_token_id  = self.tokenizer.convert_tokens_to_ids("No")

        # now branch
        if apply_alpha_on == "logits":
            # combine before softmax
            combined = (1 + alpha) * logits_not_masked - alpha * logits_masked
            logits2 = combined[:, [yes_token_id, no_token_id]]    # [1,2]
            probs  = torch.softmax(logits2, dim=-1)
        elif apply_alpha_on == "probs":
            # softmax first then combine
            p_not    = torch.softmax(logits_not_masked[:, [yes_token_id, no_token_id]], dim=-1)
            p_masked = torch.softmax(logits_masked[:,    [yes_token_id, no_token_id]], dim=-1)
            probs    = (1 + alpha) * p_not - alpha * p_masked
            ## normalize with softmax
            probs = torch.softmax(probs, dim=-1)
            probs = torch.clamp(probs, min=0.0)
        else:
            raise ValueError("apply_alpha_on must be 'logits' or 'probs'")

        yes_prob, no_prob = probs[0]
        return yes_prob if target_token.lower() == "yes" else no_prob

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

        # pixel_values = load_image(image_path, max_num=self.max_num_patches).to(torch.bfloat16).to(self.device)
        # prompt = self._build_prompt(question, history=history, pixel_values=pixel_values)

        # # Tokenize
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # # Generate
        # gen_cfg = dict(
        #     max_new_tokens=max_new_tokens,
        #     do_sample=False,
        #     temperature=0,
        #     eos_token_id=self.eos_token_id
        # )
        # with torch.no_grad():
        #     generation_output = self.model.generate(
        #         pixel_values=pixel_values,
        #         input_ids=inputs["input_ids"],
        #         attention_mask=inputs["attention_mask"],
        #         **gen_cfg
        #     )

        # # Decode
        # # The entire sequence = prompt tokens + newly generated tokens.
        # # We only want the part after the prompt, or you can decode all and strip the system prompt.
        # output_ids = generation_output[0]
        # output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # # The conversation template often appends a special separator at the end,
        # # so let's remove everything after the first occurrence of the separator token
        # if self.template is not None:
        #     sep_str = self.template.sep.strip()
        #     output_text = output_text.split(sep_str)[0].strip()

        # # If you want just the newly generated text, subtract the prompt length:
        # # new_tokens = output_ids[len(inputs["input_ids"][0]):]
        # # output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # return output_text
        if not isinstance(image_paths, list):
            image_paths = [image_paths]
        num_patches_list = []
        # Prepare the pixel_values
        if len(image_paths) == 1:
            pixel_values = load_image(image_paths[0], max_num=self.max_num_patches).to(torch.bfloat16).to(self.device)
            num_patches_list = None
        else:
            pixel_values = []
            for image_id, image_path in enumerate(image_paths):
                pixel_values.append(load_image(image_path, max_num=self.max_num_patches).to(torch.bfloat16).to(self.device))
                num_patches_list.append(pixel_values[image_id].shape[0])
            pixel_values = torch.cat(pixel_values, dim=0)

        # return output_text
        with torch.inference_mode():
            question = f"<image>\n{question}"
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0)
            response = self.model.chat(self.tokenizer, pixel_values, question, generation_config, num_patches_list=num_patches_list, history=history, return_history=False)
        return response
    
    def _text_only_query(
        self,
        question: str,
        max_new_tokens=1024,
        ) -> str:
        """
        For text-only question-answering with the model.
        Returns the decoded text from the model's generation.
        """
        with torch.inference_mode():
            generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, temperature=0)
            response = self.model.chat(self.tokenizer, None, question, generation_config,history=None, return_history=False)
        return response
        

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
        
        all_responses = []
        # Process in batches.
        for i in range(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i : i + batch_size]
            batch_questions = questions[i : i + batch_size]
            
            # Load and process images for the current batch.
            pixel_values = [
                load_image(image_path, max_num=self.max_num_patches)
                .to(torch.bfloat16)
                .to(self.device)
                for image_path in batch_image_paths
            ]
            num_patches_list = [img.shape[0] for img in pixel_values]
            # Concatenate the patch tensors along the batch dimension.
            pixel_values = torch.cat(pixel_values, dim=0)
            
            # Run inference on the batch.
            with torch.inference_mode():
                generation_config = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0
                )
                responses = self.model.batch_chat(
                    self.tokenizer, 
                    pixel_values, 
                    questions=batch_questions, 
                    generation_config=generation_config, 
                    history=None, 
                    return_history=False, 
                    num_patches_list=num_patches_list
                )
            all_responses.extend(responses)
        
        # If only one response, return it as a string; otherwise, return a list.
        return all_responses[0] if len(all_responses) == 1 else all_responses

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
