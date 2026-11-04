"""Measure answer accuracy on CLEVR with ground-truth scenes.

    --gold            encode CLEVR's functional programs (upper bound of the KB)
    --programs FILE   evaluate generated {question: program} programs

Gold programs and answers are used here for scoring only, never for generation.

encode_scene follows KDASP's encode_CLEVR_scene. encode_question differs from KDASP's
encode_CLEVR_question in two ways (see README): each step directly follows its input,
and less_than/greater_than compute their second input first.

Usage (from NePTune/):
    .venv/bin/python ASP/eval_clevr.py --gold --num_questions 6000
    .venv/bin/python ASP/eval_clevr.py --programs ASP/programs/clevr_asp.json
"""

import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from asp_runner import answer_is_correct, solve

CLEVR = "/storage-01/datasets/CLEVR/CLEVR_v1.0"
RENAMED = {"union": "or", "intersect": "and"}


def encode_scene(scene):
    return "\n".join(f"obj({i},{o['pixel_coords'][0]},{o['pixel_coords'][1]},{o['material']},{o['color']},"
                     f"{o['shape']},{o['size']})." for i, o in enumerate(scene["objects"]))


def encode_question(program):
    atoms, step = [], [0]

    def emit(index):
        """Write the steps for a node; return the step holding its output."""
        node = program[index]
        name = node["function"]
        if name == "scene":
            t = step[0]
            atoms.append(f"scene({t}).")
            step[0] = t + 1
            return t + 1
        inputs = node["inputs"]
        if len(inputs) == 2:
            first, second = (inputs[1], inputs[0]) if name in ("less_than", "greater_than") else inputs
            t2 = emit(first)
            t = emit(second)
            atoms.append(f"{RENAMED.get(name, name)}({t},{t2}).")
        else:
            t = emit(inputs[0])
            if name == "relate" or name.startswith("filter_"):
                name = f"{name.split('_')[0]}_{node['value_inputs'][0]}"
            atoms.append(f"{name}({t}).")
        step[0] = t + 1
        return t + 1

    atoms.append(f"end({emit(len(program) - 1)}).")
    return "\n".join(atoms)


def load_split(split="val"):
    with open(f"{CLEVR}/questions/CLEVR_{split}_questions.json") as f:
        questions = json.load(f)["questions"]
    with open(f"{CLEVR}/scenes/CLEVR_{split}_scenes.json") as f:
        scenes = {s["image_index"]: s for s in json.load(f)["scenes"]}
    return questions, scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--programs")
    parser.add_argument("--num_questions", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Write per-question failures to this JSON file.")
    args = parser.parse_args()

    questions, scenes = load_split()
    if args.programs:
        with open(args.programs) as f:
            programs = json.load(f)
        questions = [q for q in questions if q["question"] in programs]
    else:
        questions = random.Random(args.seed).sample(questions, args.num_questions)

    stats = collections.defaultdict(collections.Counter)
    failures = []
    for q in questions:
        family = q["program"][-1]["function"]
        program = encode_question(q["program"]) if args.gold else programs[q["question"]]
        result = solve(program, encode_scene(scenes[q["image_index"]]))
        outcome = "correct" if answer_is_correct(result, q["answer"]) else result.status
        stats[family][outcome] += 1
        if outcome != "correct":
            failures.append({"question": q["question"], "image_index": q["image_index"], "expected": q["answer"],
                             "status": result.status, "possible": result.possible, "program": program})

    total = collections.Counter()
    for family, counts in sorted(stats.items()):
        n = sum(counts.values())
        print(f"{family:16s} {counts['correct']:5d}/{n:<5d} {counts['correct'] / n:7.2%}  {dict(counts)}")
        total.update(counts)
    n = sum(total.values())
    print(f"{'TOTAL':16s} {total['correct']:5d}/{n:<5d} {total['correct'] / n:7.2%}  {dict(total)}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(failures, f, indent=2)


if __name__ == "__main__":
    main()
