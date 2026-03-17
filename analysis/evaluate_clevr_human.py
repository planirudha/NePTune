"""Evaluate VLM predictions against ground truth answers using an LLM judge."""

import json
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None

import requests
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


# Constants
VLLM_SERVER_URL = "http://127.0.0.1:8000/v1/completions"
MAX_TOKENS_CHECKER = 1024
MAX_TOKENS_SERVER = 1024
TEMPERATURE = 0.0


def check_answer_and_gt(llm, question, response, gt, think=False):
    """Check if a response is correct given the ground truth using an LLM judge.

    llm can be:
      - a vLLM LLM instance       (local GPU inference)
      - a plain string model ID   (vLLM server at VLLM_SERVER_URL)
      - an OpenAI client instance (DeepSeek or other OpenAI-compatible API)
    """
    prompt = (
        "You are an automatic answer checker! I will give you a question and answer and a generated response, "
        "and you will tell me if the response is correct given the ground truth answer. "
        "If the response is correct, you will say <Yes>, otherwise you will say <No>. "
        "Rubber and Matte are the same. Shiny and Metal or Metallic are the same. Square and Cube are the same. "
        "Yellow colors and Golden color are similar in the images too. Balls and Sphere are similar.\n\n"
        "Check if the response is correct given the question and the ground truth answer, thinking deeply about the context. "
        "Answer with <Yes> or <No> and don't mention them other than for the final answer. "
        "Important! Ignore the red bounding box reference in the generated response!\n"
        f'Question: "{question}"\n'
        f'Ground Truth Answer: "{gt}"\n'
        f'Generated Response: "{response}"\n'
    )

    if think:
        prompt += " <think>"

    if isinstance(llm, OpenAI):
        try:
            completion = llm.chat.completions.create(
                model=llm._model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS_SERVER,
                temperature=TEMPERATURE,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error querying API: {e}")
            return None

    if isinstance(llm, str):
        payload = {
            "model": llm,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS_SERVER,
            "temperature": TEMPERATURE,
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(VLLM_SERVER_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            print(f"Error querying vLLM API: {e}")
            return None
    else:
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS_CHECKER, temperature=TEMPERATURE)
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


def normalize_boolean_answer(answer):
    if str(answer).lower() in ["true", "false"]:
        return "yes" if str(answer).lower() == "true" else "no"
    try:
        numeric_value = float(answer)
        if 0 < numeric_value < 1:
            return "yes" if numeric_value > 0.5 else "no"
    except (ValueError, TypeError):
        pass
    return answer


def check_numeric_match(predicted, gt):
    if str(gt).isnumeric():
        gt_float = float(gt)
        try:
            predicted_float = float(predicted)
            return gt_float == predicted_float
        except (ValueError, TypeError):
            pass
    return False


def check_string_match(predicted, gt):
    return str(predicted).lower().strip() == str(gt).lower().strip()


def check_prefix_match(predicted, gt):
    if gt in ["yes", "no"] and isinstance(predicted, str):
        return predicted.lower().startswith(gt)
    return False


def check_substring_match(predicted, gt):
    if isinstance(predicted, str) and isinstance(gt, str):
        predicted_cleaned = predicted.replace("main object in the red bounding box", "object").strip()
        return gt.lower() in predicted_cleaned.lower()
    return False


def calculate_metrics(correct_count, execution_errors, total):
    accuracy = 100.0 * correct_count / total if total > 0 else 0.0
    exec_error_rate = 100.0 * execution_errors / total if total > 0 else 0.0
    semantic_error_rate = 100.0 - exec_error_rate - accuracy
    effective_accuracy = 100.0 * correct_count / (total - execution_errors) if (total - execution_errors) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "exec_error_rate": exec_error_rate,
        "semantic_error_rate": semantic_error_rate,
        "effective_accuracy": effective_accuracy,
    }


def evaluate_item(index, item, llm):
    """Evaluate a single item. Returns (index, correct, judge_answer, is_exec_error, is_substring).

    Thread-safe: only reads llm, no shared mutable state.
    """
    predicted_answer = item.get("code_generated_answer")
    gt_answer = str(item["answer"]).lower()
    query = item.get("query", "")

    # Fall back to VLM direct answer on execution error
    if predicted_answer is None:
        predicted_answer = item.get("vlm_direct_answer")

    if predicted_answer is None:
        return index, None, None, True, False  # execution error — no answer available

    correct = False
    is_substring = False
    judge_answer = None

    if check_numeric_match(predicted_answer, gt_answer):
        correct = True

    if not correct:
        predicted_answer = normalize_boolean_answer(predicted_answer)
        if str(gt_answer).lower() in ["yes", "no"]:
            predicted_answer = normalize_boolean_answer(predicted_answer)

    if not correct and check_string_match(predicted_answer, gt_answer):
        correct = True

    if not correct and check_prefix_match(predicted_answer, gt_answer):
        correct = True

    if not correct and check_substring_match(predicted_answer, gt_answer):
        correct = True
        is_substring = True

    if not correct:
        judge_answer = check_answer_and_gt(llm, query, predicted_answer, gt_answer)
        if judge_answer and judge_answer.lower().endswith(("yes", "<yes>", "yes.")):
            correct = True

    return index, correct, judge_answer, False, is_substring


def main(model_id, results_data_path, provider="vllm", workers=256):
    """Main evaluation function."""
    with open(results_data_path, "r") as f:
        vlm_data = json.load(f)
    if isinstance(vlm_data, dict):
        vlm_data = vlm_data.get("results", [])
    print(f"Loaded {len(vlm_data)} VLM results")
    if not vlm_data:
        print("No results to evaluate. Exiting.")
        return

    # Initialize LLM judge
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        llm = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        llm._model_id = model_id
        print(f"Using DeepSeek API with model: {model_id}")
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        llm = OpenAI(api_key=api_key)
        llm._model_id = model_id
        print(f"Using OpenAI API with model: {model_id}")
    else:
        # Local vLLM — not parallelized (GPU inference is not thread-safe here)
        workers = 1
        llm = LLM(
            model=model_id,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=4096,
            enable_prefix_caching=True,
        )

    correct_count = 0
    execution_errors = 0
    total = len(vlm_data)
    completed = 0
    save_lock = threading.Lock()

    print(f"Evaluating {total} items with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_item, i, item, llm): i
            for i, item in enumerate(vlm_data)
        }
        with tqdm(total=total, desc="Evaluating") as bar:
            for future in as_completed(futures):
                idx, correct, judge_answer, is_exec_error, is_substring = future.result()
                item = vlm_data[idx]
                query = item.get("query", "")
                predicted_answer = item.get("code_generated_answer") or item.get("vlm_direct_answer")
                gt_answer = str(item["answer"]).lower()

                if is_exec_error:
                    execution_errors += 1
                    vlm_data[idx]["correct"] = None
                else:
                    if is_substring:
                        vlm_data[idx]["correct_in"] = True
                    if judge_answer is not None:
                        vlm_data[idx]["judge_answer"] = judge_answer
                        if correct:
                            print(f"\n[{idx}] Correct via LLM judge:")
                        else:
                            print(f"\n[{idx}] Incorrect:")
                        print(f"  Q: {query}")
                        print(f"  GT: {gt_answer}")
                        print(f"  Pred: {predicted_answer}")
                        print(f"  Judge: {judge_answer}")
                    if correct:
                        correct_count += 1
                    vlm_data[idx]["correct"] = correct

                completed += 1
                metrics = calculate_metrics(correct_count, execution_errors, completed)
                bar.update(1)
                bar.set_postfix(
                    acc=f"{metrics['accuracy']:.1f}%",
                    exec_err=f"{metrics['exec_error_rate']:.1f}%",
                    eff=f"{metrics['effective_accuracy']:.1f}%",
                )

                # Periodic save every 100 completions
                if completed % 100 == 0:
                    with save_lock:
                        with open(results_data_path, "w") as f_out:
                            json.dump(vlm_data, f_out, indent=4)

    final_metrics = calculate_metrics(correct_count, execution_errors, total)

    print(f"\n{'='*60}")
    print("Final Results:")
    print(f"{'='*60}")
    print(f"Accuracy          : {final_metrics['accuracy']:.2f}%")
    print(f"Execution Error   : {final_metrics['exec_error_rate']:.2f}%")
    print(f"Semantic Error    : {final_metrics['semantic_error_rate']:.2f}%")
    print(f"Effective Accuracy: {final_metrics['effective_accuracy']:.2f}%")
    print(f"{'='*60}\n")

    eval_results_path = results_data_path.replace(".json", "-evaluation.json")
    output_data = {
        "all_results": [vlm_data[i].get("correct") for i in range(len(vlm_data))],
        "final_accuracy": final_metrics["accuracy"],
        "final_execution_error": final_metrics["exec_error_rate"],
        "final_semantic_error": final_metrics["semantic_error_rate"],
        "final_effective_accuracy": final_metrics["effective_accuracy"],
    }

    with open(eval_results_path, "w") as f_out:
        json.dump(output_data, f_out, indent=4)

    with open(results_data_path, "w") as f_out:
        json.dump(vlm_data, f_out, indent=4)

    print(f"Results saved to: {eval_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VLM predictions against ground truth using an LLM judge."
    )
    parser.add_argument("--model_id", type=str, default="deepseek-chat",
                        help="Model ID of the LLM judge.")
    parser.add_argument("--provider", type=str, default="deepseek",
                        choices=["deepseek", "openai", "vllm"],
                        help="Inference provider: 'deepseek' (API), 'openai' (API), or 'vllm' (local).")
    parser.add_argument("--vlm_output_data_path", type=str, default="results.json",
                        help="Path to the VLM evaluation output file.")
    parser.add_argument("--workers", type=int, default=256,
                        help="Parallel threads for API judge calls (default: 256, ignored for local vLLM).")

    args = parser.parse_args()
    main(args.model_id, args.vlm_output_data_path, provider=args.provider, workers=args.workers)
