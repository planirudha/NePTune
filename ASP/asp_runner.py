"""Validate and execute question programs together with the KDASP knowledge base.

check_asp_syntax and answer_is_correct are adapted from KDASP utils.py. check_chain
is new: it checks step numbering without needing a gold answer.
"""

import os
from dataclasses import dataclass, field

import clingo
import clingo.ast

HERE = os.path.dirname(os.path.abspath(__file__))
KB_MODULES = ["scene", "relate", "selection", "quantification", "attributes", "comparison", "answer"]

# Value type each step reads and writes; used by check_chain.
ATTRS = ["size", "color", "material", "shape"]
VALUES = ["large", "small", "gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow",
          "metal", "rubber", "sphere", "cylinder", "cube"]
UNARY_OPS = {"unique": ("state", "state"), "count": ("state", "int"), "exist": ("state", "bool"),
             "boolean_negation": ("bool", "bool")}
UNARY_OPS.update({f"filter_{v}": ("state", "state") for v in VALUES})
UNARY_OPS.update({f"relate_{d}": ("state", "state") for d in ["left", "right", "front", "behind"]})
UNARY_OPS.update({f"query_{a}": ("state", a) for a in ATTRS})
UNARY_OPS.update({f"same_{a}": ("state", "state") for a in ATTRS})
BINARY_OPS = {"and": ("state", "state"), "or": ("state", "state"), "equal_integer": ("int", "bool"),
              "less_than": ("int", "bool"), "greater_than": ("int", "bool")}
BINARY_OPS.update({f"equal_{a}": (a, "bool") for a in ATTRS})

# Predicates the KB derives; a question program must not define them. ans/1 is allowed.
RESERVED = {("object", 1), ("position", 3), ("state", 2), ("bool", 2), ("int", 2), ("obj", 7),
            ("left_of", 2), ("right_of", 2), ("in_front_of", 2), ("behind_of", 2)}
RESERVED |= {(f"has_{a}", 2) for a in ATTRS} | {(a, 2) for a in ATTRS}


@dataclass
class Result:
    status: str  # determined | ambiguous | unsat | error
    answer: str = None
    possible: list = field(default_factory=list)
    error: str = None


def load_kb(spatial="geometric"):
    """spatial: 'geometric' (KDASP, from position/3) or 'evidence' (NePTune relation matrices)."""
    modules = KB_MODULES + ["spatial" if spatial == "geometric" else "spatial_evidence"]
    parts = []
    for module in modules:
        with open(os.path.join(HERE, "kb", f"{module}.lp")) as f:
            parts.append(f.read())
    return "\n".join(parts)


def check_asp_syntax(program, facts="", spatial="geometric"):
    """Return None if the program grounds with the KB, else clingo's error message."""
    messages = []
    try:
        ctl = clingo.Control(["--warn=none"], logger=lambda c, m: messages.append(m.strip()))
        ctl.add("base", [], load_kb(spatial) + "\n" + facts + "\n" + program)
        ctl.ground([("base", [])])
        return None
    except RuntimeError as e:
        return "; ".join(messages) or str(e)


def check_chain(program):
    """Return None if the program is well formed, else a message the LLM can act on.

    Facts are read as a step chain: scene(T) writes objects to T+1, and each op reads
    step T and writes T+1 with the types in UNARY_OPS/BINARY_OPS. Programs without
    end/1 are rule style and must derive ans/1 themselves.
    """
    heads, steps = _parse(program)
    clash = sorted(f"{n}/{a}" for n, a in heads if (n, a) in RESERVED)
    if clash:
        return f"Do not define knowledge base predicates: {', '.join(clash)}."
    if ("#show" in program) or ("#const" in program):
        return "Do not use #show or #const."

    ends = [s for s in steps if s[0] == "end"]
    if not ends:
        return None if ("ans", 1) in heads else "The program needs end(T) or a rule deriving ans/1."
    if len(ends) > 1:
        return "Use exactly one end(T)."

    written = {0: "state"}  # state(0,ID) holds all objects
    used = set()
    single = set()  # steps written by unique/1
    for name, t, t2 in sorted((s for s in steps if s[0] != "end"), key=lambda s: s[1]):
        if t in used:
            return f"Step {t} is used by two operations; each step number must be unique."
        used.add(t)
        if name == "scene":
            written[t + 1] = "state"
            continue
        ops = BINARY_OPS if t2 is not None else UNARY_OPS
        if name not in ops:
            return f"Unknown operation {name} with {'2' if t2 is not None else '1'} argument(s)."
        if name.startswith(("relate_", "same_", "query_")) and t not in single:
            return f"{name}({t}) needs a single object: put unique directly before it."
        if t2 == 0:
            return f"{name}({t},0) reads step 0, which is all objects; use the step that ends the first chain."
        if name == "unique":
            single.add(t + 1)
        read, write = ops[name]
        for step in [t] if t2 is None else [t, t2]:
            if written.get(step) != read:
                return (f"{name}({t}{'' if t2 is None else f',{t2}'}) reads a {read} at step {step}, "
                        f"but step {step} holds {written.get(step, 'nothing')}.")
        written[t + 1] = write
    final = written.get(ends[0][1])
    if final is None or final == "state":
        return f"end({ends[0][1]}) must point at the step holding the answer value, not {final or 'nothing'}."
    return None


def _parse(program):
    """Return (signatures of rule heads, integer facts as (name, T, U or None))."""
    heads, steps = set(), []

    def visit(node):
        if node.ast_type != clingo.ast.ASTType.Rule or node.head.ast_type != clingo.ast.ASTType.Literal:
            return
        atom = node.head.atom
        if atom.ast_type != clingo.ast.ASTType.SymbolicAtom or atom.symbol.ast_type != clingo.ast.ASTType.Function:
            return
        name, args = atom.symbol.name, atom.symbol.arguments
        heads.add((name, len(args)))
        numbers = [a.symbol.number for a in args if a.ast_type == clingo.ast.ASTType.SymbolicTerm
                   and a.symbol.type == clingo.SymbolType.Number]
        if not node.body and len(args) in (1, 2) and len(numbers) == len(args):
            steps.append((name, numbers[0], numbers[1] if len(numbers) == 2 else None))

    clingo.ast.parse_string(program, visit, logger=lambda c, m: None)
    return heads, steps


def solve(program, facts, spatial="geometric"):
    """Run program + KB + facts and read ans/1 over all answer sets.

    Uses brave consequences so uncertain evidence (choice rules in the facts) is not
    enumerated. 'determined' means every answer set gives the same single answer.
    """
    try:
        ctl = clingo.Control(["--warn=none", "--models=0", "--enum-mode=brave"], logger=lambda c, m: None)
        ctl.add("base", [], load_kb(spatial) + "\n" + facts + "\n" + program)
        ctl.ground([("base", [])])
        possible = []

        def on_model(model):
            possible[:] = [str(s.arguments[0]) for s in model.symbols(shown=True) if s.name == "ans"]

        if not ctl.solve(on_model=on_model).satisfiable:
            return Result("unsat")
    except RuntimeError as e:
        return Result("error", error=str(e))
    possible.sort()
    if len(possible) == 1:
        return Result("determined", answer=possible[0], possible=possible)
    return Result("ambiguous", possible=possible)


def answer_is_correct(result, expected):
    return result.status == "determined" and result.answer == str(expected).lower()
