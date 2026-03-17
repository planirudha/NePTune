import json
import os
import random
from collections import defaultdict
import time
import re # Import regex for question parsing

# Set random seed for reproducibility
random.seed(42)

# --- Constants ---

# Maximum attempts to find a truly negative target object for a given source/relation
MAX_NEGATIVE_TARGET_ATTEMPTS = 10

# --- Global Data Definitions (Attributes and Inverse Relations) ---

# Attribute Lists (derived from user context)
ATTRIBUTE_LISTS = {
    # Colors
    "colors": ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "silver", "gold", "tan", "beige", "maroon", "navy", "teal", "ivory", "cream colored", "light blue", "dark blue", "light brown", "dark brown", "light colored", "dark colored", "black and white", "rainbow colored", "pastel", "vibrant", "neon", "calico", "tabby"],
    # Materials
    "materials": ["wood", "metal", "plastic", "glass", "fabric", "leather", "paper", "ceramic", "rubber", "stone", "brick", "concrete", "cardboard", "pine", "copper", "granite", "wool", "chalk", "denim", "brass", "cloth", "mesh", "cotton", "clay", "straw", "aluminum", "asphalt", "chrome", "marble", "bronze", "iron", "wire", "silk", "gravel", "steel", "porcelain", "wicker", "styrofoam", "tin", "vinyl", "crystal", "knit", "hardwood", "plush", "lace", "jeans"],
    # Sizes
    "sizes": ["massive", "oversized", "large", "small", "little", "short", "long", "tall", "wide", "narrow", "giant", "miniature", "deep", "shallow", "low", "high", "thick", "thin", "full", "empty", "half full", "huge", "tiny", "fat", "skinny", "chubby", "elongated"],
    # Actions/States
    "actions": ["splashing", "tied", "parked", "reading", "posing", "jumping", "squatting", "walking", "hitting", "resting", "crashing", "swimming", "blowing", "eating", "performing trick", "waiting", "batting", "skateboarding", "riding", "standing", "breaking", "bending", "playing", "docked", "grilled", "ski", "talking", "surfing", "pointing", "skiing", "laughing", "grazing", "sitting", "running", "cooking", "staring", "sleeping", "flying", "looking up", "driving", "skating", "waving", "lying", "smiling", "kneeling", "sliding", "looking down", "spinning", "burning", "drinking", "crouching", "wrapped", "clumped", "folded", "perched", "reflected", "outstretched", "painted", "covered", "written", "decorated", "grouped", "attached", "mounted", "pulled back", "crossed", "piled", "stacked", "spread", "suspended", "open", "closed", "shut", "connected", "filled", "used", "worn", "held by", "holding", "following", "followed by", "pushing", "pushed by", "decorating", "decorated by", "pulling", "pulled by", "surrounding", "surrounded by", "wearing", "worn on", "reflecting in", "reflected on", "collared", "gloved"],
    # Other descriptive attributes
    "other_attributes": [
        "comfortable", "military", "shaggy", "modern", "upholstered", "cracked", "rimmed", "dry", "fake", "commercial",
        "wild", "decorative", "male", "indoors", "upside down", "wooded", "speckled", "female", "power",
        "soft", "shadowed", "crumbled", "disposable", "scarce", "recessed", "wii", "beautiful", "warm", "hairy",
        "crowded", "seasoned", "floppy", "feathered", "halved", "foreign", "padded", "jagged", "tasty", "sprinkled", "sleepy", "sealed", "sad", "dirty", "spiral",
        "sweet", "printed", "curved", "melted", "lined", "roman", "lighted", "strawberry", "railroad", "rippled", "textured",
        "frozen", "diamond", "barren", "computer", "bathroom", "shaved", "spraying", "blond", "flat", "dangling", "baked", "curly", "greasy", "fuzzy", "hazy", "support",
        "manicured", "sleeveless", "unripe", "arched", "scrambled", "snow", "iced", "peeling",
        "snowy", "unpeeled", "steamed", "kitchen", "scattered", "banana", "curious", "damaged",
        "blooming", "winter", "old", "creamy", "tail", "illuminated", "nike", "horizontal", "down", "roasted", "industrial", "delicious", "browned", "abundant",
        "bamboo", "cloudy", "smooth", "light", "framed", "fine", "rolled", "potted", "water", "blank", "real",
        "patchy", "broken", "grated", "unpaved", "rough", "steep", "breakable", "cloudless", "floral", "chain-link", "bare", "ocean", "barbed", "oblong",
        "office", "tangled", "barefoot", "baseball", "diced", "ruffled", "fluffy", "exterior", "raw",
        "deciduous", "crooked", "glowing", "rectangular", "on", "striped", "rocky", "vintage", "crouched", "cordless", "inflatable",
        "angled", "turned", "young", "soap", "capital", "cut", "toilet", "adult", "dusty", "short sleeved", "raised", "dull", "baby", "shingled", "chipped", "funny", "mowed",
        "woven", "polar", "braided", "public", "beaded", "heavy", "powerful", "urban", "carved",
        "coarse", "wired", "caucasian", "floating", "octagonal", "paneled", "oriental", "groomed", "khaki", "opaque", "furry", "fire", "rotten",
        "displayed", "eaten", "glossy", "adidas", "packed", "loose", "peeled", "muddy", "gas", "plain", "abandoned", "choppy", "traffic", "misty",
        "overgrown", "telephone", "fenced", "angry", "chocolate", "fancy", "wispy", "knotted", "sloped",
        "faded", "slanted", "square", "paved", "pepper", "gloomy", "carpeted", "shining",
        "crispy", "vertical", "sculpted", "inflated", "leafy", "packaged", "sheer", "vast", "hanging",
        "rusty", "tiled", "triangular", "having meeting", "electric", "fried", "fallen",
        "winding", "soda", "cylindrical", "healthy", "coffee", "still",
        "chinese", "old fashioned", "busy", "shirtless", "bushy", "cobblestone", "wedding",
        "digital", "artificial", "simple", "checkered", "strong", "chopped", "antique",
        "translucent", "rugged", "evergreen", "typical", "upper", "homemade", "quilted", "tropical",
        "rock", "hollow", "wavy", "directional", "dried", "ugly", "immature", "professional", "muscular", "apple", "bunched", "cushioned", "clear",
        "pointy", "messy", "fresh", "safety", "wine", "shredded", "uncooked", "christmas", "birthday", "soccer", "lush", "up", "torn", "bent", "patched",
        "rustic", "elevated", "assorted", "american", "edged", "weathered", "street", "stainless steel",
        "stuffed", "snowboarding", "shaded", "garbage", "outdoor", "protective", "stained",
        "flowered", "uneven", "elderly", "powdered", "tennis", "puffy", "mature", "dotted", "leafless", "transparent", "frosted", "neat", "drawn", "regular",
        "beer", "unhealthy", "mixed", "portable", "toy", "corded", "overhead", "unlit",
        "polished", "analog", "ornamental", "complete", "brunette", "hard", "discolored",
        "tomato", "cooked", "alert", "foggy", "designed", "formal", "rainy", "glazed", "trimmed", "sturdy", "calm", "wireless", "long sleeved",
        "unoccupied", "athletic", "french", "vacant", "bright", "folding",
        "spiky", "boiled", "ridged", "pale", "curvy", "trash", "foamy", "reflective", "twisted", "sunlit", "wrinkled", "domed", "polo",
        "fluorescent", "electronic", "pretty", "lower", "incomplete", "ripe", "off",
        "sandy", "murky", "ornate", "toasted", "shaped", "asian", "tinted", "cluttered",
        "uncomfortable", "balding", "blurry", "wrinkly", "crumpled", "bald", "lit", "happy", "sharp",
        "forested", "stormy", "patterned", "wet", "palm", "crisp", "shiny", "straight", "intricate", "tight", "abstract", "new",
        "double decker", "cheese", "colorful", "made", "handmade", "irregular", "burnt",
        "vanilla", "juicy", "license", "sunny", "plaid", "curled", "crusty", "round", "grassy"
    ]
}

# Convert lists to sets for faster lookups and processing
ATTRIBUTE_SETS = {
    "colors": set(ATTRIBUTE_LISTS["colors"]),
    "materials": set(s.strip() for s in ATTRIBUTE_LISTS["materials"]), # Strip potential whitespace
    "sizes": set(ATTRIBUTE_LISTS["sizes"]),
    "actions": set(ATTRIBUTE_LISTS["actions"]),
    "other_attributes": set(ATTRIBUTE_LISTS["other_attributes"]),
    "all": set() # Initialize combined set
}

# Create the 'all' set by combining all categories
ATTRIBUTE_SETS["all"] = (
    ATTRIBUTE_SETS["colors"] |
    ATTRIBUTE_SETS["materials"] |
    ATTRIBUTE_SETS["sizes"] |
    ATTRIBUTE_SETS["actions"] |
    ATTRIBUTE_SETS["other_attributes"]
)


# Define known inverse relations (derived from user context)
INVERSE_RELATIONS = {
    "to the left of": "to the right of", "to the right of": "to the left of", "in front of": "behind", "behind": "in front of",
    "above": "below", "below": "above", "on top of": "under", "under": "on top of", "underneath": "on top of", "beneath": "on top of",
    "taller than": "shorter than", "shorter than": "taller than", "larger than": "smaller than", "smaller than": "larger than", "bigger than": "smaller than",
    "contain": "inside", "inside": "contain", "sitting under": "sitting on top of", "sitting on top of": "sitting under",
    "standing under": "standing on top of", "standing on top of": "standing under", "standing behind": "standing in front of", "standing in front of": "standing behind",
    "sitting behind": "sitting in front of", "sitting in front of": "sitting behind", "parked behind": "parked in front of", "parked in front of": "parked behind",
    "followed by": "following", "following": "followed by", "pushed by": "pushing", "pushing": "pushed by", "decorated by": "decorating", "decorating": "decorated by",
    "pulled by": "pulling", "pulling": "pulled by", "surrounded by": "surrounding", "surrounding": "surrounded by", "holding": "held by", "held by": "holding",
    "wearing": "worn on", "worn on": "wearing", "reflecting in": "reflected on", "reflected on": "reflecting in",
}


# --- Helper Functions ---

def load_scene_graphs(path):
    """Loads scene graphs from a JSON file."""
    try:
        print(f"Attempting to load JSON from: {path}")
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded JSON data.")
        return data
    except FileNotFoundError:
        print(f"FATAL ERROR: Scene graph file not found at {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"FATAL ERROR: Could not decode JSON from {path}. Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during loading: {e}")
        return None


def get_bbox(obj):
    """Extracts bounding box [x_min, y_min, x_max, y_max] from object data."""
    if not isinstance(obj, dict): return None
    if all(k in obj for k in ['x', 'y', 'w', 'h']):
        try:
            x, y, w, h = float(obj['x']), float(obj['y']), float(obj['w']), float(obj['h'])
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                return [x, y, x + w, y + h]
            else: return None
        except (ValueError, TypeError): return None
    else: return None


def collect_relations_and_classes(scene_graphs):
    """Collects all unique object classes and relation names from scene graphs."""
    if not scene_graphs: return [], []
    classes, relations = set(), set()
    if not isinstance(scene_graphs, dict):
        print("Error: scene_graphs is not a dictionary.")
        return [], []
    for img_id, sg in scene_graphs.items():
        if not isinstance(sg, dict) or 'objects' not in sg or not isinstance(sg['objects'], dict):
            print(f"Warning: Skipping invalid scene graph structure for image ID: {img_id}")
            continue
        for obj_id, od in sg['objects'].items():
            if not isinstance(od, dict): continue
            obj_name = od.get('name')
            if obj_name and isinstance(obj_name, str) and obj_name.strip():
                classes.add(obj_name.strip())
            obj_relations = od.get('relations', [])
            if isinstance(obj_relations, list):
                for r in obj_relations:
                    if isinstance(r, dict):
                        rel_name = r.get('name')
                        if rel_name and isinstance(rel_name, str) and rel_name.strip():
                            relations.add(rel_name.strip())
    return list(classes), list(relations)


def is_relation_true(sg, src_id, rel_name, tgt_id, inverse_relations_map):
    """Checks if a relationship holds between src_id and tgt_id, considering inverses."""
    if not isinstance(sg, dict) or 'objects' not in sg or not isinstance(sg['objects'], dict): return False
    src_id_str, tgt_id_str = str(src_id), str(tgt_id)
    src_obj = sg['objects'].get(src_id_str)
    if isinstance(src_obj, dict):
        for r in src_obj.get('relations', []):
            if isinstance(r, dict) and r.get('name') == rel_name and 'object' in r and str(r.get('object')) == tgt_id_str:
                return True
    inverse_rel_name = inverse_relations_map.get(rel_name)
    if inverse_rel_name:
        tgt_obj = sg['objects'].get(tgt_id_str)
        if isinstance(tgt_obj, dict):
            for r in tgt_obj.get('relations', []):
                if isinstance(r, dict) and r.get('name') == inverse_rel_name and 'object' in r and str(r.get('object')) == src_id_str:
                    return True
    return False

# --- Question Generation Functions (WITH TEMPLATES) ---

def make_concept_questions(sgs, all_classes, neg_ratio=0.5):
    """Generates Yes/No questions about object concepts (classes) using templates."""
    out = []
    if not sgs or not isinstance(sgs, dict): return out

    POSITIVE_CONCEPT_TEMPLATES = [
        lambda article, cls: f"Is the object in the red bounding box {article}{cls}?",
        lambda article, cls: f"Is the object inside of the red bounding box {article}{cls}?",
    ]
    NEGATIVE_CONCEPT_TEMPLATES = [
        lambda article, cls: f"Is the object inside of the red bounding box {article}{cls}?",
        lambda article, cls: f"Is the object in of the red bounding box {article}{cls}?",
    ]

    for img_id, sg in sgs.items():
        if not isinstance(sg, dict) or 'objects' not in sg or not isinstance(sg['objects'], dict): continue
        objects_in_scene = sg['objects']
        scene_classes = {od.get('name').strip() for od in objects_in_scene.values() if isinstance(od, dict) and od.get('name') and isinstance(od.get('name'), str) and od.get('name').strip()}
        scene_classes_list = list(scene_classes)

        for obj_id, od in objects_in_scene.items():
            if not isinstance(od, dict): continue
            cls = od.get('name')
            if cls and isinstance(cls, str): cls = cls.strip()
            bbox = get_bbox(od)
            if not cls or bbox is None: continue
            article = "an " if cls.lower().startswith(('a', 'e', 'i', 'o', 'u')) else "a "
            if cls.endswith("s") and not cls.endswith("ss"): article = ""

            template = random.choice(POSITIVE_CONCEPT_TEMPLATES)
            question_str = template(article, cls)
            if random.random() < 0.5:
                answer = f"Yes, the object in the red bounding box is {article}{cls}."
            else:
                answer = f"Yes."
             
            out.append({
                "imageId": img_id, "objectId": str(obj_id), "type": "concept", "subjectClassName": cls,
                "question": question_str, "bboxes": [bbox],
                "answer": answer, "type2": "class",
                "short_answer": "Yes", "concepts": [cls]
            })

            if random.random() < neg_ratio:
                possible_neg_classes = [c for c in scene_classes_list if c != cls] or [c for c in all_classes if c != cls]
                if possible_neg_classes:
                    neg_cls = random.choice(possible_neg_classes)
                    neg_article = "an " if neg_cls.lower().startswith(('a', 'e', 'i', 'o', 'u')) else "a "
                    if neg_cls.endswith("s") and not neg_cls.endswith("ss"): neg_article = ""
                    template = random.choice(NEGATIVE_CONCEPT_TEMPLATES)
                    question_str = template(neg_article, neg_cls)
                    if random.random() < 0.33:
                        answer = f"No, the object in the red bounding box is {article}{cls}."
                    elif random.random() < 0.66:
                        answer = f"No, the object in the red bounding box is not {neg_article}{neg_cls}."
                    else:
                        answer = f"No."
                    out.append({
                        "imageId": img_id, "objectId": str(obj_id), "type": "concept", "subjectClassName": cls,
                        "question": question_str, "bboxes": [bbox],
                        "answer": answer,
                        "short_answer": "No", "concepts": [neg_cls],
                        "type2": "class"
                    })
    return out

def make_relation_questions(sgs, all_relations, neg_ratio=0.5):
    """Generates Yes/No questions about relationships using templates."""
    out = []
    if not sgs or not isinstance(sgs, dict): return out

    POSITIVE_RELATION_TEMPLATES = [
        lambda sn, rn, tn: f"Is the {sn} in the red bounding box {rn} the {tn} in the green bounding box?",
    ]
    NEGATIVE_RELATION_TEMPLATES = [
        lambda sn, rn, tn: f"Is the {sn} in the red bounding box {rn} the {tn} in the green bounding box?",
    ]

    for img_id, sg in sgs.items():
        if not isinstance(sg, dict) or 'objects' not in sg or not isinstance(sg['objects'], dict): continue
        objs = sg['objects']
        if len(objs) < 2: continue
        valid_objs = {}
        for oid, od in objs.items():
             if isinstance(od, dict):
                 name = od.get('name')
                 if name and isinstance(name, str): name = name.strip()
                 bbox = get_bbox(od)
                 if name and bbox: valid_objs[str(oid)] = od
        if len(valid_objs) < 2: continue
        valid_obj_ids = list(valid_objs.keys())
        scene_relations = set()
        for oid, obj_data in valid_objs.items():
             obj_relations = obj_data.get('relations', [])
             if isinstance(obj_relations, list):
                 for r in obj_relations:
                     if isinstance(r, dict):
                         r_name = r.get('name')
                         if r_name and isinstance(r_name, str): r_name = r_name.strip()
                         tgt_id = str(r.get('object'))
                         if r_name and tgt_id in valid_objs: scene_relations.add(r_name)
        scene_relations_list = list(scene_relations)

        for src_id_str, od in valid_objs.items():
            src_bbox = get_bbox(od)
            src_name = od.get('name').strip()
            actual_relations_targets = defaultdict(list)
            actual_relation_names = set()
            src_relations = od.get('relations', [])
            if isinstance(src_relations, list):
                for rel in src_relations:
                    if isinstance(rel, dict):
                        tgt_obj_id = str(rel.get('object'))
                        r_name = rel.get('name')
                        if r_name and isinstance(r_name, str): r_name = r_name.strip()
                        if r_name and tgt_obj_id in valid_objs:
                            actual_relations_targets[r_name].append(tgt_obj_id)
                            actual_relation_names.add(r_name)

            for r_name, target_ids in actual_relations_targets.items():
                for tgt_id_str in target_ids:
                    tgt = valid_objs[tgt_id_str]
                    tgt_bbox = get_bbox(tgt)
                    tgt_name = tgt.get('name').strip()
                    template = random.choice(POSITIVE_RELATION_TEMPLATES)
                    question_str = template(src_name, r_name, tgt_name)
                    if random.random() < 0.5:
                        answer_str = f"Yes, the {src_name} in the red box is {r_name} the {tgt_name} in the green box."
                    else:
                        answer_str = f"Yes."

                    out.append({
                        "imageId": img_id, "type": "relation", "subjectId": src_id_str, "objectId": tgt_id_str,
                        "subjectClassName": src_name, "objectClassName": tgt_name, "question": question_str,
                        "bboxes": [src_bbox, tgt_bbox], "answer": answer_str,
                        "short_answer": "Yes", "concepts": [r_name], "type2": "relation"
                    })

            if random.random() < neg_ratio:
                possible_neg_relations = [r for r in scene_relations_list if r not in actual_relation_names] or [r for r in all_relations if r not in actual_relation_names]
                if not possible_neg_relations: continue
                random.shuffle(possible_neg_relations)
                found_true_negative = False
                for potential_neg_r in possible_neg_relations:
                    other_valid_obj_ids = [oid for oid in valid_obj_ids if oid != src_id_str]
                    if not other_valid_obj_ids: break
                    random.shuffle(other_valid_obj_ids)
                    attempt_count = 0
                    for potential_tgt_id in other_valid_obj_ids:
                        if attempt_count >= MAX_NEGATIVE_TARGET_ATTEMPTS: break
                        attempt_count += 1
                        if not is_relation_true(sg, src_id_str, potential_neg_r, potential_tgt_id, INVERSE_RELATIONS):
                            tgt_neg = valid_objs[potential_tgt_id]
                            chosen_neg_tgt_bbox = get_bbox(tgt_neg)
                            chosen_neg_tgt_name = tgt_neg.get('name').strip()
                            template = random.choice(NEGATIVE_RELATION_TEMPLATES)
                            question_str = template(src_name, potential_neg_r, chosen_neg_tgt_name)
                            if random.random() < 0.5:
                                answer_str = f"No, the {src_name} in the red box is not {potential_neg_r} the {chosen_neg_tgt_name} in the green box."
                            else:
                                answer_str = f"No."
                            out.append({
                                "imageId": img_id, "type": "relation", "subjectId": src_id_str, "objectId": potential_tgt_id,
                                "subjectClassName": src_name, "objectClassName": chosen_neg_tgt_name, "question": question_str,
                                "bboxes": [src_bbox, chosen_neg_tgt_bbox], "answer": answer_str,
                                "short_answer": "No", "concepts": [potential_neg_r], "type2": "relation"
                            })
                            found_true_negative = True
                            break
                    if found_true_negative: break
    return out

def make_attribute_yesno_questions(sgs, neg_ratio=0.5):
    """Generates Yes/No questions about object attributes using templates."""
    out = []

    if not sgs or not isinstance(sgs, dict): return out
    POSITIVE_ATTRIBUTE_YN_TEMPLATES = [
        lambda cn, attr: f"Is the {cn} in the red bounding box {attr}?",
    ]
    NEGATIVE_ATTRIBUTE_YN_TEMPLATES = [
        lambda cn, attr: f"Is the {cn} in the red bounding box {attr}?",
    ]
    all_attributes_list = list(ATTRIBUTE_SETS["all"])
    all_colors_set = ATTRIBUTE_SETS["colors"]
    all_materials_set = ATTRIBUTE_SETS["materials"]
    all_sizes_set = ATTRIBUTE_SETS["sizes"]
    all_actions_set = ATTRIBUTE_SETS["actions"]
    all_other_attrs_set = ATTRIBUTE_SETS["other_attributes"]

    for img_id, sg in sgs.items():
        if not isinstance(sg, dict) or 'objects' not in sg or not isinstance(sg['objects'], dict): continue
        for obj_id, od in sg['objects'].items():
            if not isinstance(od, dict): continue
            cls_name = od.get('name')
            if cls_name and isinstance(cls_name, str): cls_name = cls_name.strip()
            bbox = get_bbox(od)
            obj_attrs_raw = od.get('attributes', [])
            if not cls_name or bbox is None or not isinstance(obj_attrs_raw, list): continue
            obj_attrs = {str(attr).strip() for attr in obj_attrs_raw if isinstance(attr, str) and str(attr).strip()}
            if not obj_attrs: continue
            for attr in obj_attrs:
                if random.random() < 0.5:
                    answer_str = "Yes."
                else:
                    answer_str = f"Yes, the {cls_name} in the red bounding box is {attr}."
            
                template = random.choice(POSITIVE_ATTRIBUTE_YN_TEMPLATES)
                question_str = template(cls_name, attr)
                out.append({
                    "imageId": img_id, "objectId": str(obj_id), "type": "attribute_yesno", "subjectClassName": cls_name,
                    "question": question_str, "bboxes": [bbox],
                    "answer": answer_str,
                    "short_answer": "Yes", "concepts": [attr], "type2": "property"
                })

            if random.random() < neg_ratio and all_attributes_list:
                if not obj_attrs: continue
                attr_present = random.choice(list(obj_attrs))
                possible_neg_pool_set = None
                if attr_present in all_colors_set: possible_neg_pool_set = all_colors_set
                elif attr_present in all_materials_set: possible_neg_pool_set = all_materials_set
                elif attr_present in all_sizes_set: possible_neg_pool_set = all_sizes_set
                elif attr_present in all_actions_set: possible_neg_pool_set = all_actions_set
                elif attr_present in all_other_attrs_set: possible_neg_pool_set = all_other_attrs_set
                valid_negatives = []
                if possible_neg_pool_set: valid_negatives = [neg for neg in possible_neg_pool_set if neg not in obj_attrs]
                if not valid_negatives: valid_negatives = [neg for neg in all_attributes_list if neg not in obj_attrs]
                if valid_negatives:
                    neg_attr = random.choice(valid_negatives)
                    answer_str = f"No, the {cls_name} in the red bounding box is not {neg_attr}. For example, it is {attr_present}."
                    template = random.choice(NEGATIVE_ATTRIBUTE_YN_TEMPLATES)
                    question_str = template(cls_name, neg_attr)
                    
                    out.append({
                        "imageId": img_id, "objectId": str(obj_id), "type": "attribute", "subjectClassName": cls_name,
                        "question": question_str, "bboxes": [bbox], "answer": answer_str,
                        "short_answer": "No", "concepts": [neg_attr], "type2": "property"
                    })
    return out

# --- Data Handling Functions ---

def split_by_image(entries, fracs):
    """Splits entries into train/val/test sets based on image IDs."""
    if not entries: return {'train': [], 'val': [], 'test': []}
    if not isinstance(entries, list):
         print("Warning: Entries provided to split_by_image is not a list.")
         return {'train': [], 'val': [], 'test': []}
    if not all(0 <= f <= 1 for f in fracs) or sum(fracs) > 1.0001:
        raise ValueError("Split fractions must be between 0 and 1 and sum to at most 1.")
    imgs = list({e['imageId'] for e in entries if isinstance(e, dict) and e.get('imageId')})
    if not imgs:
        print("Warning: No valid image IDs found in entries for splitting.")
        return {'train': [], 'val': [], 'test': []}
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(fracs[0] * n)
    n_val = int(fracs[1] * n)
    n_test = n - n_train - n_val
    if n_test < 0:
      n_val = n - n_train
      n_test = 0
      print(f"Warning: Adjusted validation split size due to rounding. Test size is 0.")
    train_ids = set(imgs[:n_train])
    val_ids = set(imgs[n_train:n_train+n_val])
    test_ids = set(imgs[n_train+n_val:])
    splits = {'train': [], 'val': [], 'test': []}
    skipped_entries = 0
    for e in entries:
        if not isinstance(e, dict): skipped_entries += 1; continue
        iid = e.get('imageId')
        if not iid: skipped_entries += 1; continue
        if iid in train_ids: splits['train'].append(e)
        elif iid in val_ids: splits['val'].append(e)
        elif iid in test_ids: splits['test'].append(e)
    if skipped_entries > 0: print(f"Note: Skipped {skipped_entries} entries during split due to missing imageId or invalid format.")
    print(f"Splitting {n} images into: Train({len(train_ids)}), Val({len(val_ids)}), Test({len(test_ids)})")
    print(f"Resulting entries: Train({len(splits['train'])}), Val({len(splits['val'])}), Test({len(splits['test'])})")
    return splits

# --- REVISED compute_stats Function (Corrected Version) ---
def compute_stats(entries):
    """Computes comprehensive statistics for a list of question entries."""
    stats = {
        "global_counts": {"total_entries": 0, "skipped_entries": 0, "by_type": defaultdict(lambda: defaultdict(int))},
        "concept_breakdown": defaultdict(lambda: defaultdict(int)),
        "relation_breakdown": defaultdict(lambda: defaultdict(int)),
        "attribute_breakdown": defaultdict(lambda: defaultdict(int)),
    }
    if not entries or not isinstance(entries, list):
        print("Warning: compute_stats received empty or invalid input.")
        return stats
    for e in entries:
        stats["global_counts"]["total_entries"] += 1
        if not isinstance(e, dict):
            stats["global_counts"]["skipped_entries"] += 1; continue
        entry_type = e.get('type')
        concepts = e.get('concepts')
        short_ans = e.get("short_answer")
        question = e.get("question", "")
        if not entry_type or not isinstance(entry_type, str) or not concepts or not isinstance(concepts, list) or not concepts or not all(isinstance(c, str) and c.strip() for c in concepts):
            stats["global_counts"]["skipped_entries"] += 1; continue
        primary_concept = concepts[0].strip()
        subj_class = e.get("subjectClassName")
        obj_class_rel = e.get("objectClassName")
        stats["global_counts"]["by_type"][entry_type]["total"] += 1
        try:
            if entry_type == "concept":
                actual_class = subj_class
                if actual_class:
                    stats["concept_breakdown"][actual_class]["concept_q_total"] += 1
                    if short_ans == "Yes": stats["global_counts"]["by_type"][entry_type]["yes"] += 1; stats["concept_breakdown"][actual_class]["concept_q_yes"] += 1
                    elif short_ans == "No": stats["global_counts"]["by_type"][entry_type]["no"] += 1; stats["concept_breakdown"][actual_class]["concept_q_no"] += 1
            elif entry_type == "relation":
                rel_name = primary_concept
                stats["relation_breakdown"][rel_name]["total"] += 1
                if short_ans == "Yes": stats["global_counts"]["by_type"][entry_type]["yes"] += 1; stats["relation_breakdown"][rel_name]["yes"] += 1
                elif short_ans == "No": stats["global_counts"]["by_type"][entry_type]["no"] += 1; stats["relation_breakdown"][rel_name]["no"] += 1
                if subj_class: stats["concept_breakdown"][subj_class]["relation_q_subj"] += 1
                if obj_class_rel: stats["concept_breakdown"][obj_class_rel]["relation_q_obj"] += 1
            elif entry_type == "property":
                actual_class = primary_concept
                if actual_class: stats["concept_breakdown"][actual_class]["property_q"] += 1
            elif entry_type == "attribute_yesno":
                attr_name = primary_concept
                stats["attribute_breakdown"][attr_name]["yesno_q_total"] += 1
                if short_ans == "Yes": stats["global_counts"]["by_type"][entry_type]["yes"] += 1; stats["attribute_breakdown"][attr_name]["yesno_q_yes"] += 1
                elif short_ans == "No": stats["global_counts"]["by_type"][entry_type]["no"] += 1; stats["attribute_breakdown"][attr_name]["yesno_q_no"] += 1
                if subj_class: stats["concept_breakdown"][subj_class]["attribute_yesno_q"] += 1
            elif entry_type == "attribute_query":
                category = "unknown"
                q_lower = question.lower()
                if "what color" in q_lower: category = "color"
                elif "what material" in q_lower: category = "material"
                elif "size of the object" in q_lower or "how is the size" in q_lower: category = "size"
                elif "what is the" in q_lower and "doing" in q_lower: category = "action"
                category_key = f"category_{category}"
                stats["global_counts"]["by_type"][entry_type][category_key] += 1
                answered_attributes = concepts
                for attr_ans in answered_attributes:
                    if isinstance(attr_ans, str) and attr_ans.strip():
                       stats["attribute_breakdown"][attr_ans.strip()]["query_q_answered"] += 1
                if subj_class: stats["concept_breakdown"][subj_class]["attribute_query_q"] += 1
        except Exception as ex:
             print(f"Error processing entry during stats computation: {e.get('imageId')}/{e.get('type')}")
             print(f"  Entry details (first 200 chars): {str(e)[:200]}...")
             print(f"  Error details: {ex}")
             stats["global_counts"]["skipped_entries"] += 1
             continue
    try:
        stats["global_counts"]["by_type"] = dict(stats["global_counts"]["by_type"])
        for type_key in stats["global_counts"]["by_type"]: stats["global_counts"]["by_type"][type_key] = dict(stats["global_counts"]["by_type"][type_key])
        stats["concept_breakdown"] = dict(sorted(stats["concept_breakdown"].items()))
        for concept_key in stats["concept_breakdown"]: stats["concept_breakdown"][concept_key] = dict(stats["concept_breakdown"][concept_key])
        stats["relation_breakdown"] = dict(sorted(stats["relation_breakdown"].items()))
        for rel_key in stats["relation_breakdown"]: stats["relation_breakdown"][rel_key] = dict(stats["relation_breakdown"][rel_key])
        stats["attribute_breakdown"] = dict(sorted(stats["attribute_breakdown"].items()))
        for attr_key in stats["attribute_breakdown"]: stats["attribute_breakdown"][attr_key] = dict(stats["attribute_breakdown"][attr_key])
    except Exception as final_conversion_error:
        print(f"CRITICAL ERROR during final conversion of stats dictionaries: {final_conversion_error}")
        print("Returning partially converted stats structure.")
    return stats

# --- Sampling Function ---
def sample_per_class(entries, key, cap):
    """Samples entries, capping the number per *primary* concept specified by the key."""
    return entries
    if cap is None or cap <= 0: return entries
    if not entries or not isinstance(entries, list): return []
    by_primary_concept = defaultdict(list)
    skipped_sampling_entries = 0
    for e in entries:
        if not isinstance(e, dict): skipped_sampling_entries += 1; continue
        key_value_list = e.get(key)
        if key_value_list and isinstance(key_value_list, list) and key_value_list:
            primary_concept_raw = key_value_list[0]
            if isinstance(primary_concept_raw, str):
                primary_concept = primary_concept_raw.strip()
                if primary_concept: by_primary_concept[primary_concept].append(e)
                else: skipped_sampling_entries += 1
            else: skipped_sampling_entries += 1
        else: skipped_sampling_entries += 1
    if skipped_sampling_entries > 0: print(f"Note: Skipped {skipped_sampling_entries} entries during sampling prep due to issues with key '{key}'.")
    sampled = []
    over_cap_count = 0
    for concept_key, items_list in by_primary_concept.items():
        if len(items_list) > cap:
            sampled.extend(random.sample(items_list, cap))
            over_cap_count += 1
        else: sampled.extend(items_list)
    if over_cap_count > 0: print(f"Note: Capped entries for {over_cap_count} primary concepts based on key '{key}'.")
    return sampled

# --- Main Benchmark Building Function ---
def build_abstract_benchmark(
    scene_graph_path, out_prefix, stats_prefix, neg_concept=0.3, neg_relation=0.3, neg_attribute=0.3,
    max_yesno_per_class=None, max_query_per_class=None, split_fracs=(0.8,0.1,0.1), shuffle=True
):
    """Loads data, generates all question types using templates, samples, splits, and saves."""
    print(f"Loading scene graphs from: {scene_graph_path}")
    sgs = load_scene_graphs(scene_graph_path)
    if sgs is None: print("Failed to load scene graphs. Aborting."); return
    print("Collecting classes and relations...")
    all_classes, all_relations = collect_relations_and_classes(sgs)
    print(f"Found {len(all_classes)} unique classes and {len(all_relations)} unique relations.")
    print(f"Using {len(ATTRIBUTE_SETS['all'])} unique known attributes...") # (print breakdown)

    q_generators = {
        "Relation": (make_relation_questions, [sgs, all_relations, neg_relation]),
        "Attribute Yes/No": (make_attribute_yesno_questions, [sgs, neg_attribute]),
        "Concept": (make_concept_questions, [sgs, all_classes, neg_concept]),
    }
    results = {}
    total_generated = 0
    print("Generating questions using templates...")
    for name, (func, args) in q_generators.items():
         print(f"  Generating {name.lower()} questions...")
         start_gen_time = time.time()
         try:
             results[name] = func(*args)
             end_gen_time = time.time()
             count = len(results[name])
             total_generated += count
             print(f"    Generated {count} {name.lower()} questions in {end_gen_time - start_gen_time:.2f}s.")
         except Exception as e:
             print(f"    ERROR during {name.lower()} question generation: {e}")
             import traceback; traceback.print_exc()
             results[name] = []

    concepts = results.get("Concept", []); relations = results.get("Relation", [])
    properties = results.get("Property", []); attribute_yesno = results.get("Attribute Yes/No", [])
    attribute_query = results.get("Attribute Query", [])
    print("-" * 20)
    print(f"Total generated questions: {total_generated}")
    print(f" Breakdown: Concept={len(concepts)}, Relation={len(relations)}, Property={len(properties)}, Attr Y/N={len(attribute_yesno)}, Attr Query={len(attribute_query)}")
    print("-" * 20)
    if total_generated == 0: print("CRITICAL WARNING: No questions generated."); return

    print("Combining questions...")
    yes_no_questions = concepts + relations + attribute_yesno
    print(f"Sampling Yes/No questions (max {max_yesno_per_class} per primary concept)...")
    yes_no_sampled = sample_per_class(yes_no_questions, "concepts", max_yesno_per_class)
    print(f"  Sampled {len(yes_no_sampled)} Yes/No questions (from {len(yes_no_questions)}).")
    all_entries = yes_no_sampled
    ## sample 5000 from all_entries
    all_entries = random.sample(all_entries, min(10_000, len(all_entries))) if len(all_entries) > 5000 else all_entries
    print(f"Sampling Relation questions (max {max_yesno_per_class} per primary concept)...")
    if not all_entries: print("Warning: No questions remaining after sampling."); # Maybe return?
    print(f"Total entries after sampling: {len(all_entries)}")
    if shuffle: print("Shuffling all entries..."); random.shuffle(all_entries)

    print("Splitting entries by image...")
    splits = split_by_image(all_entries, split_fracs)
    out_dir = os.path.dirname(out_prefix); stats_dir = os.path.dirname(stats_prefix)
    try:
        if out_dir: os.makedirs(out_dir, exist_ok=True)
        if stats_dir: os.makedirs(stats_dir, exist_ok=True)
    except OSError as e: print(f"Error creating output directories: {e}. Aborting."); return

    print("Processing splits...")
    for split_name, items in splits.items():
        bench_path = f"{out_prefix}_{split_name}.json"; stats_path = f"{stats_prefix}_{split_name}.json"
        print(f"-- Processing Split: {split_name} --")
        if not items:
            print(f"  Skipping saving benchmark for empty split.")
            try:
                with open(stats_path, 'w') as f: json.dump({}, f)
                print(f"  Saved empty stats file to {stats_path}")
            except IOError as e: print(f"  Error saving empty stats file {stats_path}: {e}")
            continue
        print(f"  Saving {len(items)} entries to {bench_path}...")
        try:
            with open(bench_path, 'w') as f: json.dump(items, f, indent=2)
        except IOError as e: print(f"  Error saving benchmark file {bench_path}: {e}. Skipping stats."); continue
        print(f"  Computing and saving detailed stats...")
        stats_start_time = time.time()
        stats = compute_stats(items)
        stats_end_time = time.time()
        print(f"    Stats computed in {stats_end_time - stats_start_time:.2f}s")
        if not stats or not stats.get("global_counts", {}).get("total_entries"):
             print(f"  Warning: No valid stats computed for {split_name} split.")
             try:
                 with open(stats_path, 'w') as f: json.dump({}, f)
             except IOError as e: print(f"  Error saving empty stats file {stats_path}: {e}")
             continue
        try:
            with open(stats_path, 'w') as f: json.dump(stats, f, indent=2)
            print(f"  Saved detailed stats to {stats_path}")
        except IOError as e: print(f"  Error saving detailed stats file {stats_path}: {e}")
    print("-" * 30); print("Benchmark generation complete."); print("-" * 30)

# --- Main Execution Block ---
if __name__=="__main__":
    # --- USER CONFIGURATION ---
    
    SCENE_GRAPH_FILE = os.path.join(os.environ.get("GQA_DIR", "data/gqa"), "scene", "train_sceneGraphs.json")
    OUTPUT_DIR = "gqa_atomic_benchmark/questions/" # Updated output dir name

    NEG_CONCEPT_RATIO = 0.5; NEG_RELATION_RATIO = 0.5; NEG_ATTRIBUTE_RATIO = 0.5
    MAX_YESNO_PER_CLASS = 50; MAX_QUERY_PER_CLASS = 50
    SPLIT_FRACTIONS = (0.8, 0.1, 0.1); SHUFFLE_ENTRIES = True
    # --- END USER CONFIGURATION ---

    abs_scene_graph_path = os.path.abspath(SCENE_GRAPH_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_prefix = os.path.join(OUTPUT_DIR, "gqa_benchmark_templated")
    stats_prefix = os.path.join(OUTPUT_DIR, "gqa_benchmark_templated_stats")
    print("-" * 60); print("Starting Abstract Benchmark Generation with TEMPLATED Questions")
    print(f"Input Scene Graphs: {abs_scene_graph_path}"); print(f"Output Prefix: {output_prefix}"); print(f"Stats Prefix: {stats_prefix}")
    print(f"Config: Neg(C/R/A)=({NEG_CONCEPT_RATIO:.2f}/{NEG_RELATION_RATIO:.2f}/{NEG_ATTRIBUTE_RATIO:.2f}), Caps(Y/N/Q)=({MAX_YESNO_PER_CLASS}/{MAX_QUERY_PER_CLASS}), Split={SPLIT_FRACTIONS}, Shuffle={SHUFFLE_ENTRIES}")
    print("-" * 60)

    if not os.path.isfile(abs_scene_graph_path):
         print(f"FATAL ERROR: Input scene graph file not found: '{abs_scene_graph_path}'")
    else:
        start_time = time.time()
        try:
            build_abstract_benchmark(
                scene_graph_path=abs_scene_graph_path, out_prefix=output_prefix, stats_prefix=stats_prefix,
                neg_concept=NEG_CONCEPT_RATIO, neg_relation=NEG_RELATION_RATIO, neg_attribute=NEG_ATTRIBUTE_RATIO,
                max_yesno_per_class=MAX_YESNO_PER_CLASS, max_query_per_class=MAX_QUERY_PER_CLASS,
                split_fracs=SPLIT_FRACTIONS, shuffle=SHUFFLE_ENTRIES
            )
        except Exception as main_exec_err:
             print(f"\n{'!'*10} An error occurred during benchmark generation: {'!'*10}")
             import traceback; traceback.print_exc()
             print(f"{'!'*10} Benchmark generation failed. {'!'*10}")
        end_time = time.time()
        print("-" * 60); print(f"Total execution time: {end_time - start_time:.2f} seconds."); print("-" * 60)