import PIL
import torch
import math
import itertools
import numpy as np
import time
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
try:
    import spacy
except ImportError:
    print("spacy not installed. Skipping.")
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
   
from collections import defaultdict
from .agent_tools import Assets

class Agent(Assets):

    def __init__(self, wrapper=None):
        super().__init__()
        self.wrapper = wrapper
        self.cache = {
            "images": defaultdict(lambda: list()),
            "scores": defaultdict(lambda: list()),
        }


    def process_images(self, all_combinations, image, bboxes, mask=False, draw_bbox=False, pass_depth=False, contrastive_mask=False, use_mask=False, overlay_masks=False):
        if self.cache["images"][len(all_combinations[0])]:
            return self.cache["images"][len(all_combinations[0])]
        
        marked_images = []
        for combination in all_combinations:
            items = tuple([bboxes[x] for x in combination])  
            marked_image = self.marker.mark_objects(image, items, mask_background=mask, draw_bbox=draw_bbox, masks=use_mask, draw_text=draw_bbox, overlay_masks=overlay_masks)
            marked_images.append(marked_image)
        self.cache["images"][len(all_combinations[0])] = marked_images
        return self.cache["images"][len(all_combinations[0])]
    
    def process_images_for_with_mask(self, all_combinations, image, bboxes):
        if self.cache["images"][len(all_combinations[0])]:
            return self.cache["images"][len(all_combinations[0])]
        
        marking_start = time.time()
        
        # Pre-compute masks for all bboxes
        masks = self.marker.masker.mask_image(image, bboxes)
        
        def extract_and_crop(i):
            return self.extract_object_with_mask(image, masks[i]).crop((bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]))

        # Pre-compute all crops
        with ThreadPoolExecutor() as executor:
            all_crops = list(executor.map(extract_and_crop, range(len(bboxes))))

        def process_combination(combination):
            items = [bboxes[x] for x in combination]
            # Mark only the objects in this combination (Red, Green, etc.)
            marked_image = self.marker.mark_objects(image, items, draw_bbox=True, draw_text=False, overlay_masks=False, masks=False)
            current_crops = [all_crops[x] for x in combination]
            return [marked_image, *current_crops]

        with ThreadPoolExecutor() as executor:
            marked_images = list(executor.map(process_combination, all_combinations))
            
        marking_end = time.time()
        print(f"Total processing time for with_mask: {marking_end - marking_start:.2f} seconds")
        
        self.cache["images"][len(all_combinations[0])] = marked_images
        return self.cache["images"][len(all_combinations[0])]

    def _crop_marked_bundle_for_relation(self, marked_bundle, combination, bboxes, image_size):
        selected_bboxes = [bboxes[idx] for idx in combination]
        min_x = min(b[0] for b in selected_bboxes)
        min_y = min(b[1] for b in selected_bboxes)
        max_x = max(b[2] for b in selected_bboxes)
        max_y = max(b[3] for b in selected_bboxes)

        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image_size[0], max_x + padding)
        max_y = min(image_size[1], max_y + padding)

        cropped_marked_image = marked_bundle[0].crop((min_x, min_y, max_x, max_y))
        return [cropped_marked_image, *marked_bundle[1:]]

    def _process_images_for_combinations(self, all_combinations, image, bboxes, mask, draw_bbox, use_mask, overlay_masks):
        marked_images = [None] * len(all_combinations)
        
        def mark_image(i, combination):
            items = tuple([bboxes[x] for x in combination])
            marked_images[i] = self.marker.mark_objects(image, items, mask_background=mask, draw_bbox=draw_bbox, masks=use_mask, draw_text=draw_bbox, overlay_masks=overlay_masks)

        with ThreadPoolExecutor() as executor:
            for i, combination in enumerate(all_combinations):
                executor.submit(mark_image, i, combination)
        
        return marked_images

    def cache_scores(self, image: 'PIL.Image', bboxes:np.array, questions: List[List[Union[str, int, str]]], mask = False, draw_bbox=True, pass_depth=False, contrastive_mask=False, use_mask=False, overlay_masks=False, with_mask=False, use_batch=False):    
        all_marked_images = []
        all_questions = []
        all_question_types = []
        tasks = []
        question_metadata = defaultdict(list)
        print("Caching scores...")
        image_pre_processing_start_time = time.time()

        def make_zero_probability(reference_tensor=None):
            if isinstance(reference_tensor, torch.Tensor):
                zero = torch.zeros(1, device=reference_tensor.device, dtype=reference_tensor.dtype)
            elif hasattr(self, "dtype") and self.dtype is not None:
                zero = torch.zeros(1, device=self.device, dtype=self.dtype)
            else:
                zero = torch.zeros(1, device=self.device)
            zero = zero.squeeze(0)
            zero.requires_grad = True
            return zero

        # Group questions by number of objects (and question type for with_mask to allow custom preprocessing)
        questions_by_group = defaultdict(list)
        for idx, q in enumerate(questions):
            num_objects = q[1]
            question_type = q[2]
            group_key = (num_objects, question_type) if with_mask else (num_objects, None)
            questions_by_group[group_key].append((idx, q[0], question_type))


        processed_images_cache = {}
        relation_bundle_cache = {}

        for group_key, q_group in questions_by_group.items():
            num_objects = group_key[0]
            all_combinations = list(itertools.product(*[range(len(bboxes))]*num_objects))

            if num_objects not in processed_images_cache:
                if with_mask:
                    marked_images = self.process_images_for_with_mask(all_combinations, image, bboxes)
                else:
                    marked_images = self.process_images(all_combinations, image, bboxes, mask=mask, draw_bbox=draw_bbox, pass_depth=pass_depth, contrastive_mask=contrastive_mask, use_mask=use_mask, overlay_masks=overlay_masks)
                processed_images_cache[num_objects] = marked_images
            else:
                marked_images = processed_images_cache[num_objects]

            for question_idx, question, question_type in q_group:
                for comb_index, combination in enumerate(all_combinations):
                    if with_mask and num_objects > 1 and len(set(combination)) != num_objects:
                        question_metadata[question_idx].append(False)
                        continue

                    marked_bundle = marked_images[comb_index]
                    if with_mask and question_type == "relation":
                        relation_key = (num_objects, combination)
                        if relation_key not in relation_bundle_cache:
                            relation_bundle_cache[relation_key] = self._crop_marked_bundle_for_relation(marked_bundle, combination, bboxes, image.size)
                        processed_bundle = relation_bundle_cache[relation_key]
                    else:
                        processed_bundle = marked_bundle
                            ## if class has preload_images method, use it
                    if hasattr(self, "preload_images"):
                        self.preload_images(processed_bundle)
                    # all_marked_images.append(processed_bundle)
                    # all_questions.append(question + "\nAnswer the question using a single word or phrase.")
                    # all_question_types.append(question_type)
                    tasks.append({
                        "image": processed_bundle,
                        "question": question + "\nAnswer the question using a single word or phrase.",
                        "type": question_type,
                        "question_idx": question_idx,
                        "comb_index": comb_index,
                        "sort_key": (num_objects, comb_index, question_type == "relation"),
                    })
                    if with_mask:
                        question_metadata[question_idx].append(True)

        image_pre_processing_end_time = time.time()
        print(f"Image pre-processing time: {image_pre_processing_end_time - image_pre_processing_start_time:.2f} seconds")
        score_start_time = time.time()                                    

        # Sort tasks to maximize prefix caching
        sorted_tasks = sorted(tasks, key=lambda x: x['sort_key'])
        all_marked_images = [t['image'] for t in sorted_tasks]
        all_questions = [t['question'] for t in sorted_tasks]
        all_question_types = [t['type'] for t in sorted_tasks]

        raw_probabilities = []
        probability_items = []
        if sorted_tasks:
            if with_mask:
                if use_batch:
                    print("Using batch with_mask scoring...")
                    raw_probabilities = self._score_batch_with_mask(all_marked_images, all_questions, types=all_question_types)
                else:
                    print("Using non-batch with_mask scoring...")
                    raw_probabilities = [self._score_with_mask(all_marked_images[0], all_questions[0], all_question_types[0]) for i in range(len(all_marked_images))]
                    raw_probabilities = torch.stack(raw_probabilities, dim=0)
            else:
                if use_batch:
                    print("Using batch scoring...")
                    raw_probabilities = self._score_batch(all_marked_images, all_questions)
                else:
                    print("Using non-batch scoring...")
                    raw_probabilities = [self._score(all_marked_images[0], all_questions[0]) for i in range(len(all_marked_images))]
                    raw_probabilities = torch.stack(raw_probabilities, dim=0)

            if isinstance(raw_probabilities, torch.Tensor):
                probability_items = list(raw_probabilities.unbind(0))
            else:
                probability_items = list(raw_probabilities)

        score_end_time = time.time()
        print(f"Scoring time: {score_end_time - score_start_time:.2f} seconds")
        per_question_probabilities = defaultdict(list)
        for prob_item, task in zip(probability_items, sorted_tasks):
            per_question_probabilities[task["question_idx"]].append((task["comb_index"], prob_item))

        for question_idx_key in list(per_question_probabilities.keys()):
            entries = per_question_probabilities[question_idx_key]
            entries.sort(key=lambda x: x[0])
            per_question_probabilities[question_idx_key] = [prob for _, prob in entries]
        
        for question_idx, (question, num_objects, question_type) in enumerate(questions):
            num_combinations = len(bboxes)**num_objects
            validity_flags = question_metadata.get(question_idx) if with_mask else None
            use_metadata = with_mask and validity_flags is not None and len(validity_flags) == num_combinations

            if use_metadata:
                valid_probabilities = per_question_probabilities.get(question_idx, [])
                reference_tensor = valid_probabilities[0] if valid_probabilities else None
                expanded_probabilities = []
                valid_pointer = 0
                for is_valid in validity_flags:
                    if is_valid:
                        if valid_pointer < len(valid_probabilities):
                            expanded_probabilities.append(valid_probabilities[valid_pointer])
                            valid_pointer += 1
                        else:
                            expanded_probabilities.append(make_zero_probability(reference_tensor))
                    else:
                        expanded_probabilities.append(make_zero_probability(reference_tensor))
                current_probabilities = expanded_probabilities
            else:
                current_probabilities = per_question_probabilities.get(question_idx, [])
                if current_probabilities:
                    current_probabilities = list(current_probabilities)
                else:
                    reference_tensor = None
                    current_probabilities = [make_zero_probability(reference_tensor) for _ in range(num_combinations)]
                if len(current_probabilities) < num_combinations:
                    reference_tensor = current_probabilities[0] if current_probabilities else None
                    missing = num_combinations - len(current_probabilities)
                    current_probabilities.extend(make_zero_probability(reference_tensor) for _ in range(missing))
                elif len(current_probabilities) > num_combinations:
                    current_probabilities = current_probabilities[:num_combinations]
            
            if num_objects > 1:
                all_combinations = list(itertools.product(*[range(len(bboxes))]*num_objects))
                for comb_index, combination in enumerate(all_combinations):
                    if len(set(combination)) != num_objects:
                        probability = torch.zeros(1).to(self.device).squeeze(0)
                        probability.requires_grad = True
                        current_probabilities[comb_index] = probability
            
            if isinstance(current_probabilities, list):
                ## if any is on cuda, move all to cuda
                current_probabilities_new = []
                for prob in current_probabilities:
                    if not isinstance(prob, torch.Tensor):
                        prob = torch.tensor(prob).to(self.device)
                    current_probabilities_new.append(prob)
                current_probabilities = current_probabilities_new

                target_device = torch.device("cpu")
                for prob in current_probabilities:
                    if not isinstance(prob, torch.Tensor):
                        prob = torch.tensor(prob).to(self.device)
                    if prob.is_cuda:
                        target_device = prob.device
                        break

                current_probabilities = torch.stack([x.to(target_device) for x in current_probabilities]).squeeze(-1)
            if num_objects > 0:
                shape = int(math.ceil(current_probabilities.shape[0]**(1/num_objects)))
                current_probabilities = current_probabilities.reshape(*([shape]*num_objects))
            else:
                current_probabilities = current_probabilities.squeeze(0)

            self.cache["scores"][(question, num_objects, question_type)] = current_probabilities

                
    def score(self, image, bboxes, question, num_objects=1, mask=False, target_token="Yes", contrastive_mask=False, additional_context=False, draw_bbox=False, pass_bbox=False, pass_depth=False, use_mask=False, add_object_desc=False, pass_cot=False, just_use_cot=False, merge_description=False, use_batch=False, pass_og_image=False, overlay_masks=False, with_mask=False, candidates=None, type=None, history=None):
        # if (question) in self.cache["scores"]:
        if (question, num_objects, type) in self.cache["scores"]:
            probabilities = self.cache["scores"][(question, num_objects, type)]
            # probabilities = self.cache["scores"][question]
            if self.wrapper is not None:
                if num_objects == 0:
                    return self.wrapper(probabilities.to(self.device), extra_info=question, vars=[])
                else:
                    return self.wrapper(probabilities.to(self.device), extra_info=question)
            return probabilities
        question += "\nAnswer the question using a single word or phrase."        
        if pass_og_image:
            question = "These are two similar images, " + question
        original_question = question

        self.final_question_cache = []
        final_questions = []
        # context = "In this image the available shapes are cubes, spheres, and cylinders. The possible colors include gray, red, blue, green, brown, purple, cyan, and yellow. The relationships between objects can be to the left, right, behind, or in front. The sizes are either small or large. Finally, the objects can be made of either rubber or metal."
        context = ""
        questions = []
        if pass_depth:
            question = f"Given image is a depth estimation of a scene where brighter means it is closer to the camera. {question}"
        marked_images = []
        masked_images = []
        all_combinations = list(itertools.product(*[range(len(bboxes))]*num_objects))
        if add_object_desc:
            all_descriptions = self.get_description(image, bboxes)      
        for combination in all_combinations:
            items = tuple([bboxes[x] for x in combination])  
            if not with_mask:
                marked_image = self.marker.mark_objects(image, items, mask_background=mask, draw_bbox=draw_bbox, masks=use_mask, draw_text=draw_bbox, overlay_masks=overlay_masks)
                if pass_depth:
                    marked_image = self.depth_estimator.estimate_depth(image)
                marked_images.append(marked_image)
                if contrastive_mask:
                    masked_image = self.marker.mark_objects(image, items, mask_background=False, draw_bbox=False, masks=True, draw_text=False, mask_regions=True)
                    masked_images.append(masked_image)


        for combination in all_combinations:
            new_question = question
            
            if pass_bbox:
                normalized_bboxes = self._normalize_coordinates(items, image.width, image.height)
                if len(items) > 2:
                    new_question = f"Position of the blue bounding box: [{str(normalized_bboxes[2])}]\n"+ new_question 
                if len(items) > 1:
                    new_question = f"Position of the green bounding box: [{str(normalized_bboxes[1])}]\n"+ new_question 
                new_question = f"Position of the red bounding box: [{str(normalized_bboxes[0])}]\n" + new_question 
                
            if add_object_desc:
                descriptions = [all_descriptions[x] for x in combination]
                print("Descriptions:", descriptions)
                if len(descriptions) > 2:
                    new_question =  "\n" + f"Object in the blue bounding box: {descriptions[2]}\n" + new_question
                if len(descriptions) > 1:
                    new_question = "\n" + f"Object in the green bounding box: {descriptions[1]}\n" + new_question 
                new_question =  f"Object in the red bounding box: {descriptions[0]}\n" + new_question
            if additional_context:
                new_question = context + "\n" + new_question
            # print(new_question)
            questions.append(new_question)
        if pass_cot:
            cots = self.get_cot(marked_images, questions)
        if add_object_desc and merge_description:
            questions = self.translator.merge_description_and_description(questions)
        final_questions = []
        for comb_index, combination in enumerate(all_combinations):
            final_question = questions[comb_index]
            if pass_cot:
                final_question = cots[comb_index] + " " + final_question
            final_questions.append(final_question)
        probabilities = []
        if use_batch:
            probabilities = self._score_batch(marked_images, final_questions, target_token)
            if num_objects > 1:
                for comb_index, combination in enumerate(all_combinations):
                    if len(set(combination)) != num_objects:
                        # probability = 0
                        probability = torch.zeros(1).to(self.device).squeeze(0)
                        probability.requires_grad = True
                        probabilities[comb_index] = probability
        else:
            if num_objects == 0:
                probability = self._score(image, questions, target_token, candidates=candidates)
                probabilities.append(probability)
            else:                    
                for comb_index, combination in enumerate(all_combinations):
                    items = tuple([bboxes[x] for x in combination])   
                    if len(set(combination)) != num_objects:
                        probability = torch.zeros(1).to(self.device).squeeze(0)
                        probability.requires_grad = True
                    else:
                        if just_use_cot:
                            probability = self.get_prob_from_cot(cots[comb_index])
                        else:
                            final_question = final_questions[comb_index]
                            if marked_images:
                                self.final_question_cache.append((comb_index, marked_images[comb_index], final_question))
                            if with_mask:
                                probability = self._score_with_mask(image, final_question, items, candidates=candidates, type=type, history=history)
                            else:
                                probability = self._score(marked_images[comb_index], final_question, target_token, candidates=candidates)
                    probabilities.append(probability)
        if isinstance(probabilities, list):
            # probabilities = torch.tensor(probabilities)
            ## stack the tensors
            probabilities = torch.stack(probabilities)
            # train needs gradient
            # probabilities.requires_grad = True
        ### reshape
        if num_objects > 0:
            shape = int(math.ceil(probabilities.shape[0]**(1/num_objects)))
            ## 100 -> (10,10) for 2 objects
            ## 1000 -> (10,10,10) for 3 objects
            # probabilities = probabilities.reshape(shape,-1)
            probabilities = probabilities.reshape(*([shape]*num_objects))
        else:
            probabilities = probabilities.squeeze(0)
        # if num_objects == 1:
        #     probabilities = probabilities.squeeze(1)
        # print(question, probabilities)
        if self.wrapper is not None:
            if num_objects == 0:
                return self.wrapper(probabilities, extra_info=original_question, vars=[])
            else:
                return self.wrapper(probabilities, extra_info=original_question)
        return probabilities
    def clean_cache(self):
        self.cache = {
            "images": defaultdict(lambda: list()),
            "scores": defaultdict(lambda: list()),
        }
        if hasattr(self, "clear_image_cache"):
            self.clear_image_cache()
            
    def query(self, image, boxes, bbox_id, text, mask=False, contrastive_mask=False, draw_bbox=False, pass_og_image=False, overlay_masks=False):
        # new_query = "The available shapes are cubes, spheres, and cylinders. The possible colors include gray, red, blue, green, brown, purple, cyan, and yellow. The relationships between objects can be to the left, right, behind, or in front. The sizes are either small or large. Finally, the objects material is either rubber or metal.\n\n" + text + " You must give a short answer in English. "
        # new_query = text + " You must give a short answer in English. "
        # new_query = text + "(You must give a detailed answer in English and mentioned all names of the objects like if you see a sofa mention sofa or couch) "
        new_query = text + "\nAnswer the question using a single word or phrase."
        if getattr(bbox_id, "tensor", None) is not None:
            ## convert the probability tensor to a single bbox_id by 
            bbox_id = bbox_id.tensor.argmax()
        # new_query = text
        if bbox_id is not None:
            if contrastive_mask:
                box = [boxes[bbox_id]]
                marked_image_not_masked, chosen_colors = self.marker.mark_objects(image, box,  mask_background=mask, draw_bbox=draw_bbox)
                marked_image_masked, chosen_colors = self.marker.mark_objects(image, box,  mask_background=mask, draw_bbox=draw_bbox, mask_regions=True)
                answer = self._query_contrastive(marked_image_masked, marked_image_not_masked, new_query)
            else:
                box = [boxes[bbox_id]]
                # if draw_bbox:
                #     bbox_ratio = 0.01
                #     min_margin = 7
                #     exapsion_ratio = 0.1
                # else:
                #     bbox_ratio = 0.0001
                #     min_margin = 5
                #     exapsion_ratio = 0.0
                # marked_image, chosen_colors = self.marker.mark_objects(image, box,  mask_background=mask, box_width_ratio=bbox_ratio, min_margin=min_margin, expansion_ratio=exapsion_ratio)
                marked_image = self.marker.mark_objects(image, box, draw_bbox=draw_bbox, overlay_masks=overlay_masks, masks=mask, draw_text=draw_bbox)
                if pass_og_image:
                    new_query = "These are two similar images, second one is marked with bounding boxes" + new_query
                    marked_image = [image, marked_image]
                else:
                    marked_image = [marked_image]
                                
                answer = self._query(marked_image, new_query)
        else:
            answer = self._query(image, text)
        return answer

    def _score(self, image, text, target_token="Yes", candidates=None):
        raise NotImplementedError
    
    def _score_contrastive(self, image_masked, image_not_masked, text, target_token="Yes", alpha=1):
        raise NotImplementedError
    
    def _query(self, image, text, max_new_tokens=1):
        raise NotImplementedError
    
    def _query_contrastive(self, image_masked, image_not_masked, text, alpha=1):
        raise NotImplementedError

    def _normalize_coordinates(self, box, image_width, image_height):
        return box

    def _score_with_mask(self, image: 'PIL.Image', question: str, bboxes: list):
        raise NotImplementedError

    def ground(self, image, phrase):
        raise NotImplementedError

    def suggest_objects(self, image, query, objects):
        raise NotImplementedError

    def calculate_all_spatial_metrics(self, image, bboxes, wrapper=None):
        """
        Calculates comprehensive spatial metrics between bounding boxes.

        This includes:
        1. Six directional spatial relationship scores (left_of, right_of, above,
        below, front, behind) using an auto-scaled linear ramp. Scores
        range from 0.5 to 1.0 if the relation holds, with 1.0 for the
        maximum separation for that relation.
        2. A normalized 2D Euclidean distance matrix, where distances are
        normalized by the maximum observed Euclidean distance (range 0 to 1).

        Args:
            bboxes (list or np.ndarray): A list or array of k bounding boxes.
                Each bounding box should be in the format (xmin, ymin, xmax, ymax).
                Shape should be (k, 4).
            depths (list or np.ndarray): A list or array of k mean depth values,
                corresponding to each bounding box. Assumes larger depth means further away.
                Shape should be (k,).

        Returns:
            dict: A dictionary where keys are metric names (str) and values are
                k x k numpy arrays of scores/distances.
                Keys: 'left_of', 'right_of', 'above', 'below', 'fron',
                        'behind', 'normalized_euclidean_distance'.
                Returns a dictionary of zero matrices if k < 2.

        Raises:
            ValueError: If the number of bounding boxes and depth values do not match,
                        or if input shapes are incorrect.
        """
        masks = self.marker.masker.mask_image(image, bboxes)
        depths = self.depth_estimator.get_depth_for_masks(image, masks)
        # depths = self.depth_estimator.get_depth_for_ms(image, bboxes)
        bboxes = np.array(bboxes)
        depths = np.array(depths)

        # --- Input Validation ---
        if bboxes.shape[0] != depths.shape[0]:
            raise ValueError(f"Number of bounding boxes ({bboxes.shape[0]}) must match number of depth values ({depths.shape[0]}).")
        if bboxes.ndim != 2 or bboxes.shape[1] != 4:
            raise ValueError("Bounding boxes should be a kx4 array/list of (xmin, ymin, xmax, ymax).")
        if depths.ndim != 1:
            raise ValueError("Depths should be a 1D array/list of k values.")

        k = bboxes.shape[0]

        # Initialize results dictionary
        metric_names = ['left', 'right', 'above', 'below', 'front', 'behind', 'distance']
        all_metrics = {name: np.zeros((k, k), dtype=float) for name in metric_names}

        if k < 2:
            print("Warning: Need at least 2 bounding boxes to calculate spatial metrics. Returning zero matrices.")
            return all_metrics

        # Calculate center points (cx, cy) for all boxes
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2.0

        # --- Part 1: Directional Spatial Relations (Auto-scaled Linear) ---
        # dx_rel[i, j] = centers_x[j] - centers_x[i] (Object j relative to i)
        # dy_rel[i, j] = centers_y[j] - centers_y[i]
        # dz_rel[i, j] = depths[j] - depths[i]
        dx_rel = centers_x[np.newaxis, :] - centers_x[:, np.newaxis]
        dy_rel = centers_y[np.newaxis, :] - centers_y[:, np.newaxis]
        dz_rel = depths[np.newaxis, :] - depths[:, np.newaxis]

        relation_definitions = [
            ('left', dx_rel, lambda d: d > 0, lambda d: d),          # dist_val = x_j - x_i
            ('right', dx_rel, lambda d: d < 0, lambda d: -d),         # dist_val = x_i - x_j
            ('above', dy_rel, lambda d: d > 0, lambda d: d),          # dist_val = y_j - y_i (y increases down)
            ('below', dy_rel, lambda d: d < 0, lambda d: -d),         # dist_val = y_i - y_j
            ('front', dz_rel, lambda d: d < 0, lambda d: -d),          # dist_val = depth_j - depth_i
            ('behind', dz_rel, lambda d: d > 0, lambda d: d),         # dist_val = depth_i - depth_j
        ]

        for name, diff_matrix, condition_fn, distance_fn in relation_definitions:
            mask = condition_fn(diff_matrix)
            if np.any(mask):
                # Get the actual positive distances for the pairs satisfying the condition
                positive_distances_for_relation = distance_fn(diff_matrix[mask])
                max_dist_for_relation = np.max(positive_distances_for_relation)

                if max_dist_for_relation > 1e-6: # Effectively positive max distance
                    # scores = 0.8 + 0.2 * (positive_distances_for_relation / max_dist_for_relation)
                    normalized_score = 0.8 + 0.2 * (positive_distances_for_relation / max_dist_for_relation)
                    scores = 1 ## Disable the scaling
                else: # Max distance is ~0, so all distances for this mask are ~0.
                    # If relation holds with 0 distance, score is 0.5.
                    normalized_score = np.full_like(positive_distances_for_relation, 0.5)
                    scores = np.full_like(positive_distances_for_relation, 0.5)
                all_metrics[name][mask] = scores
                all_metrics[name+"_normalized"] = all_metrics[name]
                all_metrics[name+"_normalized"][mask] = normalized_score
                if max_dist_for_relation > 1e-6: # Effectively positive max distance
                    # scores = 0.8 + 0.2 * (positive_distances_for_relation / max_dist_for_relation)
                    scores = 1 ## Disable the scaling
                else: # Max distance is ~0, so all distances for this mask are ~0.
                    # If relation holds with 0 distance, score is 0.5.
                    scores = np.full_like(positive_distances_for_relation, 0.5)
                all_metrics[name][mask] = scores
                # all_metrics[name+"_normalized"][mask] = normalized_score
                
        # --- Part 2: Relation Matrices ---

        # --- Part 2: Normalized Euclidean Distance Matrix ---
        # dx_dist[i, j] = centers_x[i] - centers_x[j]
        # dy_dist[i, j] = centers_y[i] - centers_y[j]
        # Note: (centers_x[:, np.newaxis] - centers_x[np.newaxis, :]) is -dx_rel
        # Squaring removes the sign, so we can use dx_rel, dy_rel
        euclidean_distances = np.sqrt(dx_rel**2 + dy_rel**2 + dz_rel**2)
        max_overall_euclidean_dist = np.max(euclidean_distances)
        min_overall_euclidean_dist = np.min(euclidean_distances)

        if max_overall_euclidean_dist > 1e-6:
            all_metrics['distance'] = euclidean_distances / max_overall_euclidean_dist
        else:
            # All distances are ~0, so normalized distances are 0
            all_metrics['distance'] = np.zeros((k, k), dtype=float)

        all_metrics['closeness'] = 1 - all_metrics['distance']
        ## if all metrics are 0, set to 1
        np.fill_diagonal(all_metrics['closeness'], 0.0)
        if np.all(all_metrics['closeness'] == 0):
            all_metrics['closeness'] = np.ones((k, k), dtype=float)
        
        
        # Ensure diagonal is exactly 0 for distance matrix
        np.fill_diagonal(all_metrics['distance'], 0.0)
        np.fill_diagonal(all_metrics['closeness'], 0.0)

        # --- Part 3: Ternary Between Relation ---
        # between[i, j, k] means i is between j and k
        # We use the triangle inequality: dist(j, k) <= dist(j, i) + dist(i, k)
        # The closer dist(j, i) + dist(i, k) is to dist(j, k), the more "between" i is.
        
        D_ji = euclidean_distances[:, :, np.newaxis] # (i, j, 1) -> (i, j, k)
        D_ik = euclidean_distances[:, np.newaxis, :] # (i, 1, k) -> (i, j, k)
        D_jk = euclidean_distances[np.newaxis, :, :] # (1, j, k) -> (i, j, k)
        
        # Deviation from straight line
        deviation = (D_ji + D_ik) - D_jk
        
        # Normalize by the length of the segment jk
        # Add epsilon to avoid division by zero
        normalized_deviation = deviation / (D_jk + 1e-6)
        
        # Score: 1 when deviation is 0, decreases as deviation increases.
        # Using a linear ramp. If deviation is > 100% of the segment length, score is 0.
        between_scores = np.clip(1 - normalized_deviation, 0, 1)
        
        # Zero out cases where i=j or i=k or j=k to enforce distinct objects
        idx = np.arange(k)
        I, J, K = np.meshgrid(idx, idx, idx, indexing='ij')
        mask_distinct = (I != J) & (I != K) & (J != K)
        
        between_scores = between_scores * mask_distinct
        
        all_metrics['between'] = between_scores
        all_metrics["closeness"] = all_metrics["closeness"]
        metric_names = list(all_metrics.keys())
        for name in metric_names:
            if 'normalized' not in name:
                all_metrics[name+"_of"] = all_metrics[name]
            else:
                all_metrics[name.replace("_normalized", "_of_normalized")] = all_metrics[name]
        # Diagonals for relation matrices should already be 0 due to mask conditions (e.g. dx > 0)
        if wrapper is not None:
            for name in all_metrics.keys():
                all_metrics[name] = wrapper(torch.tensor(all_metrics[name]).to(self.device), extra_info=name)
        # Ensure all matrices are ProbabilisticTensor

        return all_metrics
