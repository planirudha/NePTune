import re
import json
import torch

import numpy as np
from typing import List, Dict, Optional


def calculate_iou(bbox1_xyxy, bbox2_xyxy):
    # Keep unchanged
    if not bbox1_xyxy or not bbox2_xyxy or len(bbox1_xyxy)!=4 or len(bbox2_xyxy)!=4: return 0.0
    x1=max(bbox1_xyxy[0], bbox2_xyxy[0]); y1=max(bbox1_xyxy[1], bbox2_xyxy[1])
    x2=min(bbox1_xyxy[2], bbox2_xyxy[2]); y2=min(bbox1_xyxy[3], bbox2_xyxy[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1_xyxy[2]-bbox1_xyxy[0])*(bbox1_xyxy[3]-bbox1_xyxy[1])
    area2 = (bbox2_xyxy[2]-bbox2_xyxy[0])*(bbox2_xyxy[3]-bbox2_xyxy[1])
    if area1 <= 0 or area2 <= 0: return 0.0
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def coco_to_xyxy(bbox_coco, width, height):
    # Keep unchanged
    x, y, w, h = bbox_coco
    if w <= 0 or h <= 0: return None
    x1=max(0, float(x)); y1=max(0, float(y))
    x2=min(float(width), float(x) + float(w)); y2=min(float(height), float(y) + float(h))
    # Return as integers or floats? IoU usually uses floats. Keep as floats for precision.
    return [x1, y1, x2, y2]

def parse_bbox_map(bbox_map_str: str) -> Optional[Dict[str, List[float]]]:
    """Safely parses the bbox_candidates_map JSON string."""
    # Keep from previous adaptation
    try:
        bbox_map = json.loads(bbox_map_str)
        # Basic validation: check if it's a dict and values are lists of 4 numbers
        if not isinstance(bbox_map, dict): return None
        for k, v in bbox_map.items():
            if not isinstance(v, list) or len(v) != 4: return None
            # Ensure numbers are convertible to float
            _ = [float(coord) for coord in v]
        return bbox_map
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    except Exception:
        return None


def unified_postprocess(prediction):
    # Internal configuration (can be moved outside if needed globally)
            # For evaluation
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
                    "couldnt": "couldn't", "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
                    "didnt": "didn't","doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't",
                    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
                    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd",
                    "howll": "how'll", "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm",
                    "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've",
                    "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
                    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
                    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
                    "oclock": "o'clock", "oughtnt": "oughtn't", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at",
                    "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
                    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
                    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
                    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've",
                    "somebodyll": "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
                    "someoned've": "someone'd've", "someone'dve": "someone'd've", "someonell": "someone'll",
                    "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've",
                    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's",
                    "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
                    "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've",
                    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've",
                    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've", "we'dve": "we'd've",
                    "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're",
                    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
                    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've",
                    "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've",
                    "whyll": "why'll", "whyre": "why're", "whys": "why's", "wont": "won't",
                    "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
                    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
                    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've",
                    "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll",
                    "youre": "you're", "youve": "you've"}
    manual_map = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
    }
    
    articles = ['a', 'an', 'the']
    punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_',
             '-', '>', '<', '@', '`', ',', '?', '!']
    period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
    comma_strip = re.compile(r"(\d)(\,)(\d)")

    # 1. General preprocessing
    try:
        if type(prediction).__name__ == 'ProbabilisticTensor':
            prediction = prediction.arg

        if isinstance(prediction, list):
            prediction = prediction[0] if len(prediction) > 0 else "no"

        if isinstance(prediction, torch.Tensor):
            prediction = prediction.item()

        if prediction is None:
            prediction = "no"

        if isinstance(prediction, bool):
            prediction = "yes" if prediction else "no"
        elif isinstance(prediction, int):
            prediction = str(prediction)
            print("No answer is a number, so this will be wrong")
        elif isinstance(prediction, str):
            prediction = prediction.strip()
            if prediction.endswith('s') and prediction.strip().lower() not in ['yes', 'no']:
                prediction = prediction[:-1]
    except Exception:
        prediction = str(prediction)

    prediction = str(prediction)
    prediction = prediction.replace('\n', ' ').replace('\t', ' ').strip().lower()
    if prediction == 'true':
        prediction = 'yes'
    elif prediction == 'false':
        prediction = 'no'

    # 2. processPunctuation logic
    out_text = prediction
    for p in punct:
        if (p + ' ' in out_text or ' ' + p in out_text) or re.search(comma_strip, out_text) is not None:
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = period_strip.sub("", out_text)

    # 3. processDigitArticle logic
    temp_words = out_text.lower().split()
    out_words = []
    for word in temp_words:
        word = manual_map.get(word, word)
        if word not in articles:
            out_words.append(word)
    out_words = [contractions.get(word, word) for word in out_words]
    
    return ' '.join(out_words)

