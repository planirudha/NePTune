"""Convert NePTune visual evidence into KDASP obj/7 facts.

Evidence holds the raw outputs of NePTune's pipeline, indexed by the bounding-box
order so that object ID i is NePTune's tensor index i:

    {"boxes": [[xmin, ymin, xmax, ymax], ...],
     "attributes": {"large": [p0, p1, ...], "metal": [...], ...},   # score(question, 1)
     "relations": {"left": [[...], ...], ...}}                      # optional, [i][j]: i left of j

A confident object becomes one obj/7 fact. An uncertain one becomes a choice over the
candidate obj/7 facts, so the theory is unchanged and an uncertain attribute is left
open rather than false. A None score or an absent attribute counts as unknown.
"""

import itertools
from dataclasses import dataclass, field

GROUPS = {
    "material": ["metal", "rubber"],
    "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"],
    "shape": ["cube", "sphere", "cylinder"],
    "size": ["large", "small"],
}
RELATIONS = ["left", "right", "front", "behind"]
# Wording of the score() questions; {value} forms are NOT taken from NePTune's programs.
QUESTIONS = {
    **{v: f"Is the object in the red box {v}?" for v in ["large", "small", "gray", "red", "blue", "green",
                                                         "brown", "purple", "cyan", "yellow"]},
    "metal": "Is the object in the red box made of metal?",
    "rubber": "Is the object in the red box made of rubber?",
    "cube": "Is the object in the red box a cube?",
    "sphere": "Is the object in the red box a sphere?",
    "cylinder": "Is the object in the red box a cylinder?",
}


@dataclass
class Thresholds:
    accept: float = 0.5        # top value must reach this to be taken as certain
    margin: float = 0.2        # and beat the runner-up by this much
    candidate: float = 0.1     # other values above this stay possible
    relation_accept: float = 0.5  # relation score at or above this holds
    relation_reject: float = 0.0  # at or below this does not; between (or None) stays open


@dataclass
class Scene:
    facts: str
    spatial: str  # 'geometric' or 'evidence', the mode to pass to asp_runner.solve
    objects: list = field(default_factory=list)  # per object: id, box, raw scores, candidates


def to_list(value):
    """ProbabilisticTensor, torch tensor or numpy array -> nested list."""
    value = getattr(value, "tensor", value)
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return value.tolist() if hasattr(value, "tolist") else value


def collect_evidence(score, boxes, spatial_relations=None, questions=QUESTIONS):
    """Call NePTune's score(question, num_objects) once per attribute value."""
    evidence = {"boxes": [list(map(float, b)) for b in to_list(boxes)],
                "attributes": {value: to_list(score(q, 1)) for value, q in questions.items()}}
    if spatial_relations:
        evidence["relations"] = {r: to_list(spatial_relations[r]) for r in RELATIONS if r in spatial_relations}
    return evidence


def candidates(raw, values, t):
    """Values of one attribute that remain possible, given raw {value: score or None}."""
    known = sorted(((raw[v], v) for v in values if raw.get(v) is not None), reverse=True)
    unknown = [v for v in values if raw.get(v) is None]
    if not known:
        return list(values)
    top, best = known[0]
    runner_up = max([s for s, _ in known[1:]] + ([1.0] if unknown else [0.0]))
    if top >= t.accept and top - runner_up >= t.margin:
        return [best]
    kept = [v for s, v in known if s >= t.candidate] + unknown
    return kept if kept else list(values)


def adapt(evidence, thresholds=None):
    t = thresholds or Thresholds()
    attributes = evidence.get("attributes", {})
    lines, objects = [], []
    for i, box in enumerate(evidence["boxes"]):
        x, y = round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)
        raw = {}
        for values in GROUPS.values():
            for v in values:
                scores = attributes.get(v)
                raw[v] = scores[i] if scores is not None else None
        options = {group: candidates(raw, values, t) for group, values in GROUPS.items()}
        facts = [f"obj({i},{x},{y},{m},{c},{f},{s})" for m, c, f, s in itertools.product(
            options["material"], options["color"], options["shape"], options["size"])]
        lines.append(f"{facts[0]}." if len(facts) == 1 else "1 { " + "; ".join(facts) + " } 1.")
        objects.append({"id": i, "box": box, "scores": raw, "candidates": options})

    relations = evidence.get("relations")
    if relations:
        for name in RELATIONS:
            matrix = relations.get(name)
            for i, j in itertools.permutations(range(len(evidence["boxes"])), 2):
                score = matrix[i][j] if matrix is not None else None
                atom = f"spatial_evidence({name},{i},{j})"
                if score is None or t.relation_reject < score < t.relation_accept:
                    lines.append("{ " + atom + " }.")
                elif score >= t.relation_accept:
                    lines.append(atom + ".")
    return Scene("\n".join(lines), "evidence" if relations else "geometric", objects)
