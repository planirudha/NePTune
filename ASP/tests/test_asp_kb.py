"""Tests for the KDASP knowledge base, evidence adapter, validation and execution.

Usage (from NePTune/):
    .venv/bin/python -m pytest ASP/tests -q
"""

import collections
import json
import os
import random
import re
import sys

import pytest

ASP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ASP)
from asp_runner import answer_is_correct, check_asp_syntax, check_chain, load_kb, solve
from eval_clevr import encode_question, encode_scene, load_split
from evidence_adapter import GROUPS, Thresholds, adapt, candidates, collect_evidence, to_list

# 0 large red metal cube (left, back); 1 small blue rubber sphere; 2 large blue rubber cylinder (right, front)
SCENE = """
obj(0,100,80,metal,red,cube,large).
obj(1,200,120,rubber,blue,sphere,small).
obj(2,300,160,rubber,blue,cylinder,large).
"""


def rules(text):
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("%")]


def answer(program, facts=SCENE):
    result = solve(program, facts)
    assert result.status == "determined", result
    return result.answer


# --- Knowledge base -----------------------------------------------------------

def test_modules_equal_kdasp_theory():
    with open(os.path.join(ASP, "kdasp", "perfect_theory.lp")) as f:
        theory = rules(f.read())
    assert collections.Counter(rules(load_kb())) == collections.Counter(theory)


@pytest.mark.parametrize("suite", ["train_suite.json", "test_suite.json"])
def test_kdasp_regression_suites(suite):
    with open(os.path.join(ASP, "kdasp", suite)) as f:
        examples = json.load(f)
    wrong = [e for e in examples if not answer_is_correct(solve(e["question"], e["scene"]), e["answer"])]
    assert not wrong


# --- Representative reasoning -------------------------------------------------

CASES = {
    "filter": ("scene(0). filter_large(1). filter_metal(2). unique(3). query_shape(4). end(5).", "cube"),
    "exists yes": ("scene(0). filter_large(1). filter_metal(2). exist(3). end(4).", "yes"),
    "exists no": ("scene(0). filter_small(1). filter_metal(2). exist(3). end(4).", "no"),
    "count": ("scene(0). filter_blue(1). count(2). end(3).", "2"),
    "count zero": ("scene(0). filter_green(1). count(2). end(3).", "0"),
    "retrieval": ("scene(0). filter_sphere(1). unique(2). query_material(3). end(4).", "rubber"),
    "intersection": ("scene(0). filter_blue(1). scene(2). filter_large(3). and(4,2). unique(5). query_shape(6). end(7).",
                     "cylinder"),
    "union": ("scene(0). filter_red(1). scene(2). filter_sphere(3). or(4,2). count(5). end(6).", "2"),
    "spatial left": ("scene(0). filter_sphere(1). unique(2). relate_left(3). unique(4). query_color(5). end(6).", "red"),
    "spatial front": ("scene(0). filter_sphere(1). unique(2). relate_front(3). unique(4). query_shape(5). end(6).",
                      "cylinder"),
    "same": ("scene(0). filter_sphere(1). unique(2). same_color(3). unique(4). query_shape(5). end(6).", "cylinder"),
    "equal attribute": ("scene(0). filter_sphere(1). unique(2). query_color(3). scene(4). filter_cube(5). unique(6). "
                        "query_color(7). equal_color(8,4). end(9).", "no"),
    "different": ("scene(0). filter_sphere(1). unique(2). query_color(3). scene(4). filter_cube(5). unique(6). "
                  "query_color(7). equal_color(8,4). boolean_negation(9). end(10).", "yes"),
    "universal yes": ("scene(0). filter_rubber(1). count(2). scene(3). filter_rubber(4). filter_blue(5). count(6). "
                      "equal_integer(7,3). end(8).", "yes"),
    "universal no": ("scene(0). filter_large(1). count(2). scene(3). filter_large(4). filter_blue(5). count(6). "
                     "equal_integer(7,3). end(8).", "no"),
    "fewer": ("scene(0). filter_blue(1). count(2). scene(3). filter_red(4). count(5). less_than(6,3). end(7).", "yes"),
    "more": ("scene(0). filter_blue(1). count(2). scene(3). filter_red(4). count(5). greater_than(6,3). end(7).", "no"),
    "multi-hop": ("scene(0). filter_cube(1). unique(2). relate_right(3). filter_sphere(4). unique(5). relate_right(6). "
                  "unique(7). query_shape(8). end(9).", "cylinder"),
}


@pytest.mark.parametrize("name", CASES)
def test_reasoning(name):
    program, expected = CASES[name]
    program = program.replace(". ", ".\n")
    assert check_asp_syntax(program) is None
    assert check_chain(program) is None
    assert answer(program) == expected


def test_rule_style_program():
    program = """
    qualifies(X) :- object(X), has_size(X,large), has_material(X,metal).
    qualifies_exists :- qualifies(X).
    ans(yes) :- qualifies_exists.
    ans(no) :- not qualifies_exists.
    """
    assert check_chain(program) is None
    assert answer(program) == "yes"


def test_unique_violation_is_unsat():
    assert solve("scene(0). filter_blue(1). unique(2). query_shape(3). end(4).", SCENE).status == "unsat"


# --- Validation ---------------------------------------------------------------

@pytest.mark.parametrize("program, fragment", [
    ("scene(0). filter_red(2). exist(3). end(4).", "step 2 holds nothing"),
    ("scene(0). count(1). filter_red(2). exist(3). end(4).", "reads a state"),
    ("scene(0). filter_red(1). end(2).", "must point at"),
    ("scene(0). filter_pink(1). exist(2). end(3).", "Unknown operation"),
    ("scene(0). filter_red(1). filter_cube(1). exist(2). end(3).", "unique"),
    ("scene(0). exist(1). end(1).", "must point at"),
    ("scene(0). filter_red(1). same_size(2). exist(3). end(4).", "needs a single object"),
    ("scene(0). filter_red(1). query_size(2). end(3).", "needs a single object"),
    ("scene(0). filter_red(1). scene(2). filter_cube(3). or(4,0). count(5). end(6).", "reads step 0"),
    ("has_color(1,red). ans(yes).", "has_color/2"),
    ("color(X,red) :- object(X). ans(yes).", "color/2"),
    ("qualifies(X) :- object(X).", "ans/1"),
])
def test_check_chain_rejects(program, fragment):
    assert fragment in check_chain(program)


def test_check_chain_accepts_gold_programs():
    questions, _ = load_split("val")
    errors = [check_chain(encode_question(q["program"])) for q in random.Random(2).sample(questions, 5000)]
    assert not [e for e in errors if e]


def test_syntax_errors_are_reported():
    assert check_asp_syntax("ans(X) :- not object(X).") is not None  # unsafe variable
    assert check_asp_syntax("scene(0") is not None


def prompt_examples():
    with open(os.path.join(ASP, "clevr_asp_prompt.txt")) as f:
        return re.findall(r"Question: (.+?)\n<asp>\n(.*?)</asp>", f.read(), re.DOTALL)


def test_prompt_examples_are_valid():
    examples = prompt_examples()
    assert len(examples) == 11
    for question, program in examples:
        assert check_asp_syntax(program) is None, question
        assert check_chain(program) is None, question


# --- Evidence adapter ---------------------------------------------------------

def one_hot(objects):
    """objects: list of value tuples -> attribute score vectors."""
    values = [v for group in GROUPS.values() for v in group]
    return {v: [1.0 if v in o else 0.0 for o in objects] for v in values}


OBJECTS = [("metal", "red", "cube", "large"), ("rubber", "blue", "sphere", "small"),
           ("rubber", "blue", "cylinder", "large")]
BOXES = [[90, 70, 110, 90], [190, 110, 210, 130], [290, 150, 310, 170]]


def test_confident_evidence_gives_kdasp_facts():
    scene = adapt({"boxes": BOXES, "attributes": one_hot(OBJECTS)})
    assert scene.spatial == "geometric"
    assert rules(scene.facts) == rules(SCENE)
    assert scene.objects[1]["scores"]["blue"] == 1.0  # raw scores are kept


def test_uncertain_attribute_is_not_negative():
    attributes = one_hot(OBJECTS)
    attributes["metal"][1], attributes["rubber"][1] = 0.55, 0.45  # object 1: unclear material
    scene = adapt({"boxes": BOXES, "attributes": attributes})
    assert scene.objects[1]["candidates"]["material"] == ["metal", "rubber"]
    # "Is there a metal sphere?" stays open instead of becoming no.
    result = solve("scene(0). filter_metal(1). filter_sphere(2). exist(3). end(4).", scene.facts)
    assert (result.status, result.possible) == ("ambiguous", ["no", "yes"])
    # A question that does not depend on it is still answered.
    assert answer("scene(0). filter_red(1). count(2). end(3).", scene.facts) == "1"


def test_missing_evidence_is_not_negative():
    attributes = one_hot(OBJECTS)
    del attributes["metal"], attributes["rubber"]
    scene = adapt({"boxes": BOXES, "attributes": attributes})
    assert scene.objects[0]["candidates"]["material"] == ["metal", "rubber"]
    result = solve("scene(0). filter_cube(1). filter_metal(2). exist(3). end(4).", scene.facts)
    assert result.possible == ["no", "yes"]


def test_all_low_scores_leave_attribute_open():
    assert candidates({"metal": 0.02, "rubber": 0.03}, GROUPS["material"], Thresholds()) == ["metal", "rubber"]


def test_thresholds_are_configurable():
    raw = {"metal": 0.6, "rubber": 0.3}
    assert candidates(raw, GROUPS["material"], Thresholds()) == ["metal"]  # 0.3 margin passes the default 0.2
    assert candidates(raw, GROUPS["material"], Thresholds(margin=0.1)) == ["metal"]
    assert candidates(raw, GROUPS["material"], Thresholds(margin=0.5)) == ["metal", "rubber"]


def test_relation_matrices_keep_ids_and_direction():
    n = len(OBJECTS)
    # Depth order disagrees with image y: object 0 is in front of object 2.
    relations = {"left": [[float(i < j) for j in range(n)] for i in range(n)],
                 "right": [[float(i > j) for j in range(n)] for i in range(n)],
                 "front": [[float(i < j) for j in range(n)] for i in range(n)],
                 "behind": [[float(i > j) for j in range(n)] for i in range(n)]}
    scene = adapt({"boxes": BOXES, "attributes": one_hot(OBJECTS), "relations": relations})
    assert scene.spatial == "evidence"
    program = "scene(0). filter_cylinder(1). unique(2). relate_front(3). count(4). end(5)."
    result = solve(program, scene.facts, scene.spatial)
    assert result.answer == "2"
    assert solve(program, adapt({"boxes": BOXES, "attributes": one_hot(OBJECTS)}).facts).answer == "0"


def test_uncertain_relation_stays_open():
    n = len(OBJECTS)
    left = [[1.0 if i < j else 0.0 for j in range(n)] for i in range(n)]
    left[0][2] = None
    relations = {"left": left, "right": [[0.0] * n] * n, "front": [[0.0] * n] * n, "behind": [[0.0] * n] * n}
    scene = adapt({"boxes": BOXES, "attributes": one_hot(OBJECTS), "relations": relations})
    result = solve("scene(0). filter_cylinder(1). unique(2). relate_left(3). count(4). end(5).",
                   scene.facts, scene.spatial)
    assert result.possible == ["1", "2"]


def test_collect_evidence_with_fake_score():
    torch = pytest.importorskip("torch")

    class Wrapped:  # same shape as ProbabilisticTensor: scores on .tensor
        def __init__(self, tensor):
            self.tensor = tensor

    calls = []

    def score(question, num_objects):
        calls.append((question, num_objects))
        return Wrapped(torch.full((3,), 0.25))

    evidence = collect_evidence(score, BOXES, {"left": Wrapped(torch.eye(3))})
    assert len(calls) == 15 and all(n == 1 for _, n in calls)
    assert evidence["attributes"]["red"] == [0.25, 0.25, 0.25]
    assert to_list(Wrapped(torch.eye(2))) == [[1.0, 0.0], [0.0, 1.0]]
    assert json.dumps(evidence)


# --- End to end on CLEVR ------------------------------------------------------

@pytest.fixture(scope="module")
def val():
    return load_split("val")


def perfect_evidence(scene, with_relations):
    objects = scene["objects"]
    boxes = [[o["pixel_coords"][0] - 5, o["pixel_coords"][1] - 5, o["pixel_coords"][0] + 5, o["pixel_coords"][1] + 5]
             for o in objects]
    evidence = {"boxes": boxes,
                "attributes": one_hot([(o["material"], o["color"], o["shape"], o["size"]) for o in objects])}
    if with_relations:
        evidence["relations"] = {}
        for name in ["left", "right", "front", "behind"]:
            matrix = [[0.0] * len(objects) for _ in objects]
            for j, holders in enumerate(scene["relationships"][name]):
                for i in holders:
                    matrix[i][j] = 1.0
            evidence["relations"][name] = matrix
    return evidence


def accuracy(val, with_relations, n=1500):
    questions, scenes = val
    sample = random.Random(1).sample(questions, n)
    correct = 0
    for q in sample:
        scene = adapt(perfect_evidence(scenes[q["image_index"]], with_relations))
        correct += answer_is_correct(solve(encode_question(q["program"]), scene.facts, scene.spatial), q["answer"])
    return correct / n


def test_adapter_end_to_end_with_true_relations(val):
    # With CLEVR's own relations the only remaining error source would be the KB itself.
    assert accuracy(val, True) > 0.999


def test_adapter_end_to_end_with_box_centres(val):
    assert accuracy(val, False) > 0.96


def test_encode_scene_matches_kdasp():
    with open(os.path.join(ASP, "kdasp", "test_suite.json")) as f:
        example = json.load(f)[0]
    questions, scenes = load_split("val")
    assert encode_scene(scenes[0]) == example["scene"]
