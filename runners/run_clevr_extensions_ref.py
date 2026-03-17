import os
import json
import re
import argparse
from pprint import pprint
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as coco_mask
from dotenv import load_dotenv

from code_generator import CodeGenerator
from probabilistic_tensor import ProbabilisticTensor, and_op, or_op
from vision_agents import get_model
from code_base_template import build_logic_executor_code, build_score_fn, build_query_fn
ProbabilisticTensor.and_op = "mul"

def get_gt_bbox(objects):
    gt_bboxes = []
    for item in objects:
        encoded_mask = item['mask']
        bbox = coco_mask.toBbox(encoded_mask)
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        gt_bboxes.append(bbox)
    return gt_bboxes

def load_config(config_file):
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            return json.load(f)
    return {}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate logic programs on CLEVR puzzles or RPM with VLMs.")
    parser.add_argument('--config_file', type=str, default=None, help="Path to JSON config file.")
    parser.add_argument('--model_name', type=str, help="Model name to use (e.g., ovis, intern25_8b)", default='intern25_8b')
    parser.add_argument('--use_gt', type=bool, help="Whether to use GT bounding boxes", default=False)
    parser.add_argument('--data_dir', type=str, default=None, help="Path to clevr-mini data directory.")
    parser.add_argument('--data_question_path', type=str, default=None, help="Path to refexps question JSON.")
    parser.add_argument('--prompt_file_address', type=str, default=None, help="Path to code generation prompt template.")
    parser.add_argument('--code_gen_model_name', type=str, default='deepseek-chat', help="Code generation model name.")
    parser.add_argument('--update_cache', type=bool, default=True, help="Update program cache.")

    args = parser.parse_args()

    cfg = load_config(args.config_file)
    for key, value in cfg.items():
        if getattr(args, key, None) is None or (parser.get_default(key) == getattr(args, key)):
            setattr(args, key, value)

    return args

def main():
    args = parse_args()

    MODEL_NAME = args.model_name
    DATA_DIR = args.data_dir or "data/clevr-mini"
    SCENE_PATH = os.path.join(DATA_DIR, "scenes.json")

    data_questions_json = args.data_question_path or "data/clevr-transfer/refexps-20230513.json"
    result_file = os.path.join("experiments", "clevr_refexps", MODEL_NAME, "results_new.json")

    if args.use_gt:
        PROGRAM_CACHE_ADDRESS = "programs/clevr_extension_refexps_gt.json"
        result_file = result_file.replace(".json", "_gt.json")
    else:
        PROGRAM_CACHE_ADDRESS = "programs/clevr_extension_refexps.json"

    PROMPT_FILE_ADDRESS = args.prompt_file_address or "prompts/clevr-extension.txt"

    os.makedirs(os.path.dirname(result_file), exist_ok=True)

    print(f"Loading model: {MODEL_NAME}")
    vl_model = get_model(MODEL_NAME, wrapper=ProbabilisticTensor)

    with open(data_questions_json, "r") as f:
        data_questions = json.load(f)
    questions_key = "refexps"
    questions = data_questions[questions_key]

    with open(SCENE_PATH, "r") as f:
        scenes = json.load(f)['scenes']

    load_dotenv()
    code_generator = CodeGenerator(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model_name=args.code_gen_model_name,
        program_cache_address=PROGRAM_CACHE_ADDRESS,
        prompt_file_address=PROMPT_FILE_ADDRESS,
        update_cache=args.update_cache
    )

    # Load cached results
    if os.path.exists(result_file):
        with open(result_file, "r") as f:
            results = json.load(f)
    else:
        results = {}

    corrects = []

    for i, item in enumerate(tqdm(questions)):
        qid = str(item.get("id", i))  # use id if exists, otherwise index
        if qid in results:
            corrects.append(results[qid]["correct"])
            print(f"[{i+1}/{len(questions)}] Skipping {qid}: already processed")
            continue

        question = item['question']
        answer:int = item['answer']

        file_name = f"CLEVR_mini_{item['scene_index']:06d}.png"
        image_path = os.path.join(DATA_DIR, "images", file_name)

        idx_match = re.search(r"_(\d+)\.png$", image_path)
        if not idx_match:
            print(f"Skipping invalid image path: {image_path}")
            continue
        idx = int(idx_match.group(1))

        image = Image.open(image_path).convert("RGB")
        scene = scenes[idx]
        bboxes = get_gt_bbox(scene['objects'])
        spatial_relations = vl_model.calculate_all_spatial_metrics(image, bboxes, wrapper=ProbabilisticTensor)
        code_snippet, _ = code_generator.generate_code(question, force_generate=False)
        question = "Here is a referral expression question:\n" + question

        code = build_logic_executor_code(code_snippet)
        print(f"Executing code for question {qid}: {question}")
        print(f"Code: {code}")
        exec_context = {"ProbabilisticTensor": ProbabilisticTensor, "and_op": and_op, "or_op": or_op, **spatial_relations}
        # try:
        exec(code, exec_context)
        logic_executor = exec_context["logic_executor"]

        score_fn = build_score_fn(vl_model, image, bboxes, history=None)
        query_fn = build_query_fn(vl_model, image, bboxes)
        ProbabilisticTensor.start_cache()
        res = logic_executor(question=question, score=score_fn, query=query_fn,
                            all_bounding_boxes=bboxes, image=image, history=None)
        _ = ProbabilisticTensor.end_cache()
        print(f"Execution result: {res}")
        if isinstance(res, list):
            final_bbox = res[0].argmax().item()
        elif isinstance(res, int):
            final_bbox = res
        else:
            final_bbox = res.argmax()
        
        response = final_bbox
        
        correct = response == answer
        print("Correct" if correct else "Incorrect")
        # except Exception as e:
        #     print(f"Error in question {qid}: {e}")
        #     response = None
        #     correct = False
        #     res = None

        # Save intermediate result
        results[qid] = {
            "question": question,
            "answer": answer,
            "response": response,
            "correct": correct,
            "execution_result": str(res),
        }

        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        corrects.append(correct)
        print(f"[{i+1}/{len(questions)}] Accuracy: {sum(corrects)/len(corrects):.3f}")

    print(f"\n✅ Final accuracy: {sum(corrects)}/{len(corrects)} = {sum(corrects)/len(corrects):.3f}")
    print(f"📝 Results saved to: {result_file}")

if __name__ == "__main__":
    main()
