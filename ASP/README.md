# ASP reasoning for NePTune, built on KDASP's CLEVR theory

An LLM turns each CLEVR question into a short ASP program. The program runs with KDASP's
CLEVR rules and scene facts built from NePTune's visual evidence, and clingo derives `ans/1`.

```
CLEVR question --LLM + clevr_asp_prompt.txt--> program --check_asp_syntax, check_chain--> programs/clevr_asp.json
                                                  |
NePTune score() + relation matrices --evidence_adapter.py--> obj/7 facts
                                                  |
                                      kb/*.lp + program + facts --clingo--> ans/1
```

The NePTune Python pipeline and KDASP are unchanged. Everything here lives in `ASP/`.

| Path | Purpose |
|---|---|
| `kdasp/` | Verbatim KDASP copies: `perfect_theory.lp`, `train_suite.json`, `test_suite.json`, `SOURCE_COMMIT` |
| `kb/` | KDASP's theory split into modules, plus one new module |
| `clevr_asp_prompt.txt` | LLM prompt; examples come from the CLEVR train split, plus two written by hand |
| `generate_asp_programs.py` | Question to program, with validation and retry |
| `asp_runner.py` | `load_kb`, `check_asp_syntax`, `check_chain`, `solve` |
| `evidence_adapter.py` | NePTune evidence to KDASP facts |
| `eval_clevr.py` | Accuracy on CLEVR val with ground-truth scenes |
| `tests/test_asp_kb.py` | Regression and behaviour tests |

```
.venv/bin/python ASP/generate_asp_programs.py --num_questions 200 --seed 0
.venv/bin/python ASP/eval_clevr.py --programs ASP/programs/clevr_asp.json
.venv/bin/python ASP/eval_clevr.py --gold --num_questions 6000
.venv/bin/python -m pytest ASP/tests -q
```

To run one question from NePTune evidence:

```python
from evidence_adapter import adapt, collect_evidence
from asp_runner import solve

evidence = collect_evidence(score, boxes, spatial_relations)  # raw scores, JSON-serialisable
scene = adapt(evidence)
result = solve(program, scene.facts, scene.spatial)  # result.status, .answer, .possible
```

## KDASP component inventory

The KDASP working tree holds only the data, logs and `.pyc` files. Its source is deleted from
the working tree but still tracked, so the paths marked `git` were read from `HEAD`
(see `kdasp/SOURCE_COMMIT`) without restoring them.

| Component | KDASP path | Use here |
|---|---|---|
| CLEVR theory, 72 statements | `git: preprompt/theory/CLEVR/perfect_theory.lp` | Reused verbatim in `kb/` |
| Reduced theories | `git: preprompt/theory/CLEVR/{light,medium,heavy}.lp` | Not used; ablation inputs for rule learning |
| Scene encoding | `git: prompt/translate_CLEVR.py` `encode_CLEVR_scene` | Reused as `eval_clevr.encode_scene` |
| Question encoding | `git: prompt/translate_CLEVR.py` `encode_CLEVR_question` | Adapted as `eval_clevr.encode_question` (two fixes below) |
| Operation and scene docs | `git: preprompt/question/flat/CLEVR.txt`, `preprompt/scene/CLEVR.txt` | Basis for the prompt |
| Answer representation | `git: preprompt/answer.txt` | `ans/1` is the answer predicate |
| Syntax check | `git: utils.py` `check_asp_syntax` | Adapted in `asp_runner.py` |
| Semantic check | `git: utils.py` `run_asp_code`, `answer_is_correct` | Adapted as `solve`, `answer_is_correct`, `eval_clevr.py` |
| Syntax repair | `git: utils.py` `mend_syntax` | Same idea: error fed back to the LLM, up to 3 attempts |
| Semantic repair | `git: utils.py` `mend_semantics` | Not used; it needs the gold answer (see differences) |
| Rule-learning loop | `git: main.py` | Not used; the theory already covers CLEVR |
| Regression suites | `dataset/CLEVR/{train,test}_suite.json` | Regression tests; 918 examples each |
| Per-example programs | `dataset/CLEVR/asp/{train,test}/*.lp` | Not copied: the theory plus one suite entry each |

KDASP has no question-to-ASP step of its own: its questions are translated from CLEVR's gold
functional programs. The LLM generator here fills that gap.

## Knowledge base

`step` is an integer. An operation at step `T` reads step `T` and writes step `T+1`.
Two-argument operations read `T` and `U`. All rows are KDASP rules unless marked **new**.

| Category | Module | Predicate | Meaning | Input | Output |
|---|---|---|---|---|---|
| Scene | `scene` | `obj(ID,X,Y,M,C,F,S)` | Object fact: position, material, color, shape, size | adapter | `object/1`, `position/3`, `has_*/2` |
| Scene | `scene` | `object(ID)`, `position(ID,X,Y)` | Object domain and coordinates | `obj/7` | used by all rules |
| Attributes | `scene` | `has_size/2`, `has_color/2`, `has_material/2`, `has_shape/2` | `has_color(ID,red)` | `obj/7` | used by filters, query, same |
| Scene | `scene` | `scene(T)`, `state(T,ID)` | Start a chain; objects held at step `T`. `state(0,ID)` holds all | `object/1` | `state/2` |
| Filtering | `selection` | `filter_V(T)` | Keep objects with value `V` (15 values) | `state/2`, `has_*/2` | `state/2` |
| Uniqueness | `selection` | `unique(T)` | Constraint: at most one object at `T` | `state/2` | `state/2`; prunes the model otherwise |
| Intersection | `selection` | `and(T,U)` | Objects at both steps | `state/2` | `state/2` |
| Union | `selection` | `or(T,U)` | Objects at either step | `state/2` | `state/2` |
| Spatial | `spatial` | `left_of/2`, `right_of/2`, `in_front_of/2`, `behind_of/2` | From `position/3` | `position/3`, `state/2` | used by `relate_*` |
| Spatial | `spatial_evidence` | same heads, from `spatial_evidence(R,I,J)` | **new**: from NePTune's matrices; replaces `spatial` | adapter | used by `relate_*` |
| Spatial | `relate` | `relate_left/right/front/behind(T)` | Objects in that relation to the object at `T` | `state/2`, relation | `state/2` |
| Existence | `quantification` | `exist(T)` | yes if step `T` holds an object | `state/2` | `bool/2` |
| Counting | `quantification` | `count(T)` | Number of objects | `state/2` | `int/2` |
| Negation | `quantification` | `boolean_negation(T)` | Opposite yes/no | `bool/2` | `bool/2` |
| Retrieval | `attributes` | `query_size/color/material/shape(T)` | Attribute of the object at `T` | `state/2`, `has_*/2` | `size/2`, `color/2`, `material/2`, `shape/2`, as `(T,V)` |
| Same | `attributes` | `same_size/color/material/shape(T)` | Other objects sharing the attribute | `state/2`, `has_*/2` | `state/2` |
| Comparison | `attributes` | `equal_size/color/material/shape(T,U)` | Are the values equal? | `size/2` etc. | `bool/2` |
| Comparison | `comparison` | `equal_integer(T,U)`, `less_than(T,U)`, `greater_than(T,U)` | Compare `int(T)` with `int(U)` | `int/2` | `bool/2` |
| Answer | `answer` | `end(T)`, `ans(V)` | Value at step `T` is the answer; `:- not ans(_).` | `bool/2`, `int/2`, `size/2` etc. | `ans/1` |

Requested categories with no dedicated rule, and how they are expressed:

| Category | Expression | Test |
|---|---|---|
| Universal ("are all A B?") | `count(A)` and `count(A and B)` with `equal_integer` | `universal yes`, `universal no` |
| "Different" | `equal_A` then `boolean_negation` | `different` |
| Multi-hop | Chains of `unique` and `relate_*` | `multi-hop` |

No new reasoning rule was needed. `kb/spatial_evidence.lp` is the only addition.
`test_modules_equal_kdasp_theory` checks that the modules are exactly KDASP's statements.

Example, "How many other things are the same shape as the tiny cyan matte object?":

```prolog
scene(0). filter_small(1). filter_cyan(2). filter_rubber(3). unique(4). same_shape(5). count(6). end(7).
```

## Generated programs

The LLM writes operation chains with `%` comments. When the operations cannot express a
question it may write plain rules over `object/1`, `has_*/2` and the relations, deriving `ans/1`:

```prolog
qualifies(X) :- object(X), has_size(X,large), has_material(X,metal).
qualifies_exists :- qualifies(X).
ans(yes) :- qualifies_exists.
ans(no) :- not qualifies_exists.
```

KDASP's vocabulary replaces the earlier prompt's: `metal` not `metallic`, `behind_of` not
`behind`, `ans/1` not `answer/1`, `has_color/2` not `color/2`. The last is required, because
`color/2` in KDASP means "color at step T".

Generation uses no gold data. A program is accepted when it grounds with the knowledge base
and passes `check_chain`, which rejects steps that read nothing or the wrong type, reused step
numbers, unknown operations, an `end` not pointing at a value, and redefined KB predicates.
These checks cannot tell whether a well-formed program means what the question means; only
`eval_clevr.py` measures that.

## Visual evidence adapter

| | KDASP | NePTune | Adapter |
|---|---|---|---|
| Object ID | Index in the scene | Index in `boxes` and on every tensor axis | `obj` ID = NePTune index |
| Attributes | One value each in `obj/7` | `P(Yes)` per object per question | Confident: one `obj/7`. Uncertain: choice over candidates |
| Position | `pixel_coords` | Box `(xmin,ymin,xmax,ymax)` | Box centre, rounded |
| left/right | From X | From box centres | Same relation; matrices also accepted |
| front/behind | From Y | From depth | Needs the matrices; box centres alone only approximate it |

`Thresholds` holds every cut-off, and `Scene.objects` keeps the raw scores and candidates.
A value is certain when it reaches `accept` and beats the runner-up by `margin`. A `None` or
absent score is unknown, and all-low scores leave the attribute open, so neither becomes false.

An uncertain object is written as `1 { obj(...); obj(...) } 1.`, so the theory sees ordinary
`obj/7` facts. `solve` uses brave consequences, which avoids enumerating the combinations:

| `status` | Meaning |
|---|---|
| `determined` | Every answer set gives this answer |
| `ambiguous` | Evidence allows several answers; see `possible` |
| `unsat` | No consistent reading, usually a failed `unique` |
| `error` | Grounding failed |

## Differences from KDASP

- **Theory:** unchanged, split into modules.
- **`encode_question`:** `encode_CLEVR_question` has two faults. It swaps `less_than` and
  `greater_than` arguments, so "A < B" tests `int(B) < int(A)`, and when both `query_*` steps
  come last in a CLEVR program it points `equal_*` at the wrong steps. The fix emits each step
  right after its input and computes a comparison's second input first.
- **KDASP suites are filtered:** `generate_CLEVR.py` keeps only examples the theory answers
  correctly, so their 100% is not an accuracy estimate and they hide both faults. They remain
  useful as regression tests.
- **`solve`:** reads brave consequences instead of KDASP's first-model check, and adds `ambiguous`.
- **Semantic repair is not used:** `mend_semantics` needs the expected answer, which is not
  available at inference and would contaminate evaluation.
- **clingo:** 5.8.2 here; KDASP pinned 5.7.1.

## Results

All results use ground-truth CLEVR val scenes, not NePTune's visual evidence.
`collect_evidence` has only been tested with a fake `score()`.

| Configuration | Questions | Correct |
|---|---|---|
| KDASP suites (filtered, regression only) | 918 + 918 | 100% |
| KDASP theory, KDASP encoding, unfiltered | 6000 | 90.23% |
| KDASP theory, `encode_question` (ceiling with pixel relations) | 6000 | 97.35% |
| Through the adapter, box centres | 6000 | 97.17% |
| Through the adapter, CLEVR's true relations | 6000 | 100% |
| LLM programs, first prompt (`clevr_asp.json`, seed 0) | 244 | 80.33% |
| LLM programs, current prompt (`clevr_asp_seed1.json`, seed 1) | 241 | 91.29% |

LLM model: `Mistral-Medium-3.5-128B`. Each run sampled 200 questions; a question text can occur
on several images, so more instances are scored. Questions with no valid program are excluded
from the rows above: 4 of 200 for seed 0 and 3 of 200 for seed 1. Counting those 3 questions
as wrong gives at most 220/244 = 90.2% for seed 1. With about 240 instances, the
uncertainty is roughly 3.5 points either way.

The prompt was changed after reading seed 0 failures, so seed 0 is not a clean score for the
current prompt. Seed 1 shares 1 question with seed 0.

Of the 21 seed 1 failures, 5 also fail with the gold program (pixel relations) and 16 are
translation errors. The two main patterns:

- "How many X are the same A as Y": the LLM still writes `query_A`, a new chain and `unique`
  instead of `same_A` followed by filters.
- `less_than` and `greater_than`: the LLM often keeps the question's order, which inverts the
  comparison. `check_chain` cannot detect this.
