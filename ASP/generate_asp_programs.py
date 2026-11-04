"""Generate ASP programs for CLEVR questions with an LLM.

Questions are read from the CLEVR dataset only (no NePTune program files are used).
Each question is sent to the LLM with ASP/clevr_asp_prompt.txt. The returned program
reuses the KDASP knowledge base in ASP/kb/. It is grounded with that knowledge base,
checked with check_chain, and saved as {question: program} in ASP/programs/clevr_asp.json.
No gold program or answer is used during generation.

Usage (from NePTune/):
    .venv/bin/python ASP/generate_asp_programs.py --num_questions 100
"""

import argparse
import json
import os
import random
import re

from dotenv import load_dotenv
from openai import OpenAI

from asp_runner import check_asp_syntax, check_chain

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

CLEVR_QUESTIONS = "/storage-01/datasets/CLEVR/CLEVR_v1.0/questions/CLEVR_val_questions.json"
ASP_PATTERN = re.compile(r"<asp>(.*?)</asp>", re.DOTALL)

PROVIDERS = {
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com"),
    # OpenAI-compatible endpoint; base URL from URL in .env (e.g. https://llm-service.ai.tu-darmstadt.de)
    "openai": ("OPENAI_API_KEY", os.getenv("URL", "").strip().rstrip("/") + "/v1" if os.getenv("URL", "").strip() else None),
}


def extract_asp(text: str) -> str:
    matches = ASP_PATTERN.findall(text or "")
    return matches[-1].strip() if matches else (text or "").strip()


def validate(program: str):
    """Return None if the program is usable, else an error message for the LLM."""
    return check_asp_syntax(program) or check_chain(program)


def generate(client, model, prompt_template, question, max_attempts=3):
    prompt = prompt_template.replace("{query}", question)
    program, error = None, None
    for _ in range(max_attempts):
        messages = [{"role": "system", "content": "You are an expert in Answer Set Programming."},
                    {"role": "user", "content": prompt}]
        if error:
            messages.append({"role": "user", "content": f"Your previous program was:\n{program}\nIt has this error: {error} "
                                                        f"Return a corrected program inside <asp></asp> tags."})
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=4096, temperature=0.0)
        program = extract_asp(response.choices[0].message.content)
        error = validate(program)
        if error is None:
            return program, None
    return program, error


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions_file", default=CLEVR_QUESTIONS)
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--seed", type=int, help="Shuffle before taking --num_questions.")
    parser.add_argument("--output", default=os.path.join(HERE, "programs", "clevr_asp.json"))
    parser.add_argument("--prompt_file", default=os.path.join(HERE, "clevr_asp_prompt.txt"))
    parser.add_argument("--provider", default="openai", choices=sorted(PROVIDERS))
    parser.add_argument("--model", default="Mistral-Medium-3.5-128B")
    args = parser.parse_args()

    key_name, base_url = PROVIDERS[args.provider]
    api_key = os.getenv(key_name)
    if not api_key:
        raise SystemExit(f"{key_name} is not set in {os.path.join(HERE, '.env')}")
    client = OpenAI(api_key=api_key, base_url=base_url)

    with open(args.prompt_file) as f:
        prompt_template = f.read()
    with open(args.questions_file) as f:
        questions = [q["question"] for q in json.load(f)["questions"]]
    if args.seed is not None:
        random.Random(args.seed).shuffle(questions)
    questions = questions[:args.num_questions]

    programs = {}
    if os.path.exists(args.output):  # resume
        with open(args.output) as f:
            programs = json.load(f)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    errors_path = args.output.replace(".json", "_errors.json")
    errors = {}

    for i, question in enumerate(questions):
        if question in programs:
            continue
        try:
            program, error = generate(client, args.model, prompt_template, question)
        except Exception as e:
            program, error = None, f"{type(e).__name__}: {e}"
        if error:
            errors[question] = {"program": program, "error": error}
            print(f"[{i}] FAILED: {question}\n    {error}")
            continue
        programs[question] = program
        print(f"[{i}] {question}\n{program}\n")
        with open(args.output + ".tmp", "w") as f:
            json.dump(programs, f, indent=2)
        os.replace(args.output + ".tmp", args.output)

    with open(errors_path, "w") as f:
        json.dump(errors, f, indent=2)
    print(f"Saved {len(programs)} programs to {args.output}; {len(errors)} failures in {errors_path}")


if __name__ == "__main__":
    main()
