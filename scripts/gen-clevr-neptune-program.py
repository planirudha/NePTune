#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert CLEVR structured programs to NePTune score/query programs.

Takes programs from CLEVR questions.json and converts each functional program
into NePTune-style code using score(), query(), and_op, .exists(), .iota(), etc.

Usage:
    python gen-clevr-neptune-program.py \
        --input  /path/to/questions.json \
        --output /path/to/output.json \
        [--limit N]
"""

import json
import argparse
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# Score question templates
# ═══════════════════════════════════════════════════════════════════════════

FILTER_SCORE_QUESTIONS = {
    # Colors
    'gray':     'Is the main object in the red bounding box gray?',
    'red':      'Is the main object in the red bounding box red?',
    'blue':     'Is the main object in the red bounding box blue?',
    'green':    'Is the main object in the red bounding box green?',
    'brown':    'Is the main object in the red bounding box brown?',
    'purple':   'Is the main object in the red bounding box purple?',
    'cyan':     'Is the main object in the red bounding box cyan?',
    'yellow':   'Is the main object in the red bounding box yellow?',
    # Sizes
    'large':    'Is the main object in the red bounding box large?',
    'small':    'Is the main object in the red bounding box small?',
    # Materials
    'metal':    'Is the main object in the red bounding box made of metal?',
    'rubber':   'Is the main object in the red bounding box made of rubber?',
    # Shapes
    'cube':     'Is the main object in the red bounding box a cube?',
    'sphere':   'Is the main object in the red bounding box a sphere?',
    'cylinder': 'Is the main object in the red bounding box a cylinder?',
}

FILTER_PRED_NAMES = {
    'gray': 'is_gray', 'red': 'is_red', 'blue': 'is_blue', 'green': 'is_green',
    'brown': 'is_brown', 'purple': 'is_purple', 'cyan': 'is_cyan', 'yellow': 'is_yellow',
    'large': 'is_large', 'small': 'is_small',
    'metal': 'is_metal', 'rubber': 'is_rubber',
    'cube': 'is_cube', 'sphere': 'is_sphere', 'cylinder': 'is_cylinder',
}

# Spatial relations are pre-computed by the spatial engine and injected
# into the execution context as 2D ProbabilisticTensor matrices.
# No score() declaration is needed — they are used as bare variable names.
RELATION_PRED_NAMES = {
    'left':   'left',
    'right':  'right',
    'behind': 'behind',
    'front':  'front',
}

# Keys whose scores come from the spatial engine, not the VLM.
SPATIAL_RELATIONS = frozenset({'left', 'right', 'behind', 'front'})

SAME_SCORE_QUESTIONS = {
    'same_color':    'Is the main object in the red bounding box the same color as the main object in the green bounding box?',
    'same_shape':    'Is the main object in the red bounding box the same shape as the main object in the green bounding box?',
    'same_size':     'Is the main object in the red bounding box the same size as the main object in the green bounding box?',
    'same_material': 'Is the main object in the red bounding box made of the same material as the main object in the green bounding box?',
}

QUERY_QUESTIONS = {
    'color':    'What is the color of the main object in the red bounding box?',
    'shape':    'What is the shape of the main object in the red bounding box?',
    'size':     'What is the size of the main object in the red bounding box?',
    'material': 'What is the material of the main object in the red bounding box?',
}


# ═══════════════════════════════════════════════════════════════════════════
# concept2query utilities
# ═══════════════════════════════════════════════════════════════════════════

def build_score_questions_from_concept2query(concept2query):
    """Extract score/query question strings from an agent's concept2query dict.

    Builds three lookup tables used by clevr_to_neptune():
      - filter_questions : {value → score question}   (1-object attributes)
      - same_questions   : {same_X → score question}  (2-object relations, VLM only)
      - query_questions  : {metaconcept → query str}  (open-ended descriptions)

    Spatial relations (left / right / behind / front) are skipped because they
    are provided by the spatial engine at runtime, not via score().

    Args:
        concept2query: dict in the InternVL / NePTune agent format, with keys
                       'attribute', 'relation', 'description'.

    Returns:
        tuple: (filter_questions, same_questions, query_questions)
    """
    filter_questions = {}
    for value, cfg in concept2query.get('attribute', {}).items():
        filter_questions[value] = cfg['query']

    same_questions = {}
    for rel_key, cfg in concept2query.get('relation', {}).items():
        # Skip spatial relations — they come from the spatial engine.
        if rel_key in SPATIAL_RELATIONS:
            continue
        # Map e.g. 'same_color' → question string
        if rel_key.startswith('same_'):
            same_questions[rel_key] = cfg['query']

    query_questions = {}
    for metaconcept, cfg in concept2query.get('description', {}).items():
        query_questions[metaconcept] = cfg['query']

    return filter_questions, same_questions, query_questions


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_op_type(op):
    """Get the operation type from a CLEVR program op (handles both 'type' and 'function' keys)."""
    return op.get('type', op.get('function'))


def compose_terms(terms):
    """Compose a list of predicate term strings into a single expression.

    - 0 terms  → ''
    - 1 term   → the term itself
    - 2 terms  → t1 & t2
    - 3+ terms → and_op(t1, t2, t3, ...)
    """
    if len(terms) == 0:
        return ''
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f'{terms[0]} & {terms[1]}'
    return f'and_op({", ".join(terms)})'


def replace_var_in_terms(terms, old_var, new_var):
    """Replace every occurrence of a quoted variable name in term strings.

    Replaces '"old_var"' with '"new_var"' — the enclosing quotes prevent
    partial matches (e.g. "x1" won't match "x10").
    """
    old = f'"{old_var}"'
    new = f'"{new_var}"'
    return [t.replace(old, new) for t in terms]


def make_term(pred_name, *variables):
    """Build a term string like  pred_name("x1", "x2")."""
    args = ', '.join(f'"{v}"' for v in variables)
    return f'{pred_name}({args})'


# ═══════════════════════════════════════════════════════════════════════════
# Core conversion function
# ═══════════════════════════════════════════════════════════════════════════

def clevr_to_neptune(program, concept2query=None):
    """Convert a single CLEVR structured program to a NePTune code string.

    Each intermediate result in the CLEVR program is stored as a dict with:
      kind        – 'object_set' | 'reference' | 'scalar' | 'query'
      var         – free variable name           (object_set / reference / query)
      terms       – list of predicate terms       (object_set / reference / query)
      bound_vars  – variables from reference objs  (object_set only, legacy path)
      ref_id_var  – Python variable holding .iota() result  (reference / query)
      expr        – scalar expression string      (scalar)
      metaconcept – 'color'|'shape'|'size'|'material' (query)

    The key design choice: when processing a `unique` op we immediately emit an
    `.iota()` assignment and record the resulting Python variable name as
    `ref_id_var`.  Subsequent `relate` / `same_*` / `query_*` ops that consume
    a reference use `ref_id_var` as a direct integer index into the spatial or
    same-* 2-D tensors, keeping the downstream object_set 1-D and loop-free.

    Args:
        program:       list of CLEVR op dicts, each with 'function', 'inputs', 'value_inputs'
        concept2query: optional agent concept2query dict (e.g. InternVL.concept2query).
                       When provided, score/query question strings are taken from it so
                       each model can use its own phrasing. Spatial relations are always
                       skipped regardless of this argument.

    Returns:
        str: multi-line NePTune program code (score declarations + iota assignments +
             any union assignments + final return statement)
    """
    # Resolve question lookup tables — prefer per-model concept2query over defaults.
    if concept2query is not None:
        filter_q, same_q, query_q = build_score_questions_from_concept2query(concept2query)
        # Fall back to built-in defaults for any key not present in concept2query.
        _filter_questions = {**FILTER_SCORE_QUESTIONS, **filter_q}
        _same_questions   = {**SAME_SCORE_QUESTIONS,   **same_q}
        _query_questions  = {**QUERY_QUESTIONS,         **query_q}
    else:
        _filter_questions = FILTER_SCORE_QUESTIONS
        _same_questions   = SAME_SCORE_QUESTIONS
        _query_questions  = QUERY_QUESTIONS

    index_to_result = {}
    var_counter   = [0]
    ref_counter   = [0]   # unique reference objects resolved via .iota()
    union_counter = [0]
    declared_scores   = {}   # pred_name → True  (dedup)
    declaration_lines = []   # score(...) declarations  (emitted first)
    iota_lines        = []   # ref_N = (...).iota(...)  (emitted after declarations)
    extra_lines       = []   # union intermediate assignments

    # --- local helpers ------------------------------------------------

    def new_var():
        var_counter[0] += 1
        return f'x{var_counter[0]}'

    def new_ref():
        ref_counter[0] += 1
        return f'ref_{ref_counter[0]}'

    def declare_score(pred_name, question, num_objects):
        """Register a score(...) declaration, deduplicating by pred name."""
        if pred_name not in declared_scores:
            declared_scores[pred_name] = True
            declaration_lines.append(
                f'{pred_name} = score("{question}", num_objects={num_objects})'
            )

    def expr_with_exists(expr, bound_vars):
        """Existentially quantify bound variables out of an expression."""
        for bv in bound_vars:
            expr = f'({expr}).exists("{bv}")'
        return expr

    # --- process ops --------------------------------------------------

    for i, op in enumerate(program):
        op_type = get_op_type(op)

        # ── scene ─────────────────────────────────────────────────────
        if op_type == 'scene':
            v = new_var()
            index_to_result[i] = {
                'kind': 'object_set', 'var': v, 'terms': [], 'bound_vars': [],
            }

        # ── filter_* ──────────────────────────────────────────────────
        elif op_type in ('filter_size', 'filter_color', 'filter_material', 'filter_shape'):
            value = op['value_inputs'][0]
            parent = index_to_result[op['inputs'][0]]
            pred_name = FILTER_PRED_NAMES[value]
            declare_score(pred_name, _filter_questions[value], 1)
            index_to_result[i] = {
                'kind': parent['kind'],
                'var': parent['var'],
                'terms': parent['terms'] + [make_term(pred_name, parent['var'])],
                'bound_vars': list(parent['bound_vars']),
            }

        # ── unique ────────────────────────────────────────────────────
        # Emit .iota() to resolve an object set to a single-peaked 1-D tensor
        # (ref_N).  ref_N is callable — ref_N("xN") acts as a 1-D mask — so
        # downstream relate/same_* ops can compose it with string variables
        # instead of hard integer indexing.  We do NOT .exists() bound vars
        # before .iota(); the framework handles multi-dim iota natively.
        elif op_type == 'unique':
            parent = index_to_result[op['inputs'][0]]
            expr = compose_terms(parent['terms'])
            ref_var = new_ref()
            if expr:
                iota_lines.append(f'{ref_var} = ({expr}).iota("{parent["var"]}")')
            else:
                # Bare scene → first object (edge case, should not normally occur)
                iota_lines.append(f'{ref_var} = objects_count - 1  # fallback')
            index_to_result[i] = {
                'kind': 'reference',
                'ref_id_var': ref_var,   # Python variable name, e.g. "ref_1"
                'var': parent['var'],    # string variable name, e.g. "x1"
                'terms': [],
                'bound_vars': [],
            }

        # ── relate ────────────────────────────────────────────────────
        # Use soft composition: direction("x_new", "x_ref") & ref_name("x_ref")
        # This keeps everything as string-variable tensor composition — the ref
        # tensor (peaked at one index) constrains x_ref rather than hard-indexing.
        elif op_type == 'relate':
            parent = index_to_result[op['inputs'][0]]  # must be a reference
            direction = op['value_inputs'][0]
            pred_name = RELATION_PRED_NAMES[direction]
            # Spatial relations are pre-computed; no score() declaration needed.
            v = new_var()
            ref_callable = parent['ref_id_var']   # e.g. "ref_1"
            ref_var_str  = parent['var']           # e.g. "x1"
            # e.g.  left("x2", "x1") & ref_1("x1")
            terms = [
                f'{pred_name}("{v}", "{ref_var_str}")',
                f'{ref_callable}("{ref_var_str}")',
            ]
            index_to_result[i] = {
                'kind': 'object_set',
                'var': v,
                'terms': terms,
                'bound_vars': [ref_var_str],   # needs exist-marginalisation for count/exist
            }

        # ── same_* ───────────────────────────────────────────────────
        # Same soft-composition pattern: same_prop("x_new","x_ref") & ref("x_ref")
        elif op_type in ('same_size', 'same_color', 'same_material', 'same_shape'):
            parent = index_to_result[op['inputs'][0]]  # must be a reference
            pred_name = op_type
            declare_score(pred_name, _same_questions[op_type], 2)
            v = new_var()
            ref_callable = parent['ref_id_var']
            ref_var_str  = parent['var']
            # e.g.  same_color("x2", "x1") & ref_1("x1")
            terms = [
                f'{pred_name}("{v}", "{ref_var_str}")',
                f'{ref_callable}("{ref_var_str}")',
            ]
            index_to_result[i] = {
                'kind': 'object_set',
                'var': v,
                'terms': terms,
                'bound_vars': [ref_var_str],
            }

        # ── intersect ────────────────────────────────────────────────
        elif op_type == 'intersect':
            p1 = index_to_result[op['inputs'][0]]
            p2 = index_to_result[op['inputs'][1]]
            v = new_var()
            terms1 = replace_var_in_terms(p1['terms'], p1['var'], v)
            terms2 = replace_var_in_terms(p2['terms'], p2['var'], v)
            merged_bound = list(dict.fromkeys(p1['bound_vars'] + p2['bound_vars']))
            index_to_result[i] = {
                'kind': 'object_set',
                'var': v,
                'terms': terms1 + terms2,
                'bound_vars': merged_bound,
            }

        # ── union ────────────────────────────────────────────────────
        elif op_type == 'union':
            p1 = index_to_result[op['inputs'][0]]
            p2 = index_to_result[op['inputs'][1]]
            v = new_var()
            terms1 = replace_var_in_terms(p1['terms'], p1['var'], v)
            terms2 = replace_var_in_terms(p2['terms'], p2['var'], v)

            # Compose each branch; existentially quantify bound variables
            # so both branches reduce to 1-D tensors over the unified free var.
            expr1 = expr_with_exists(compose_terms(terms1), p1['bound_vars'])
            expr2 = expr_with_exists(compose_terms(terms2), p2['bound_vars'])

            uid = union_counter[0]
            union_counter[0] += 1
            a_name = f'union_a{uid}'
            b_name = f'union_b{uid}'
            u_name = f'union_set{uid}'

            extra_lines.append(f'{a_name} = {expr1}')
            extra_lines.append(f'{b_name} = {expr2}')
            extra_lines.append(f'{u_name} = {a_name} | {b_name}')

            index_to_result[i] = {
                'kind': 'object_set',
                'var': v,
                'terms': [f'{u_name}("{v}")'],
                'bound_vars': [],           # bound vars already quantified above
            }

        # ── count ────────────────────────────────────────────────────
        elif op_type == 'count':
            parent = index_to_result[op['inputs'][0]]
            expr = compose_terms(parent['terms'])
            expr = expr_with_exists(expr, parent['bound_vars'])
            if not expr:
                scalar = 'objects_count'    # bare scene → total object count
            else:
                scalar = f'({expr}).count()'
            index_to_result[i] = {'kind': 'scalar', 'expr': scalar}

        # ── exist ────────────────────────────────────────────────────
        elif op_type == 'exist':
            parent = index_to_result[op['inputs'][0]]
            expr = compose_terms(parent['terms'])
            expr = expr_with_exists(expr, parent['bound_vars'])
            index_to_result[i] = {'kind': 'scalar', 'expr': f'({expr}).exists()'}

        # ── query_* ──────────────────────────────────────────────────
        # The parent is always a `reference`; ref_id_var is the resolved index.
        elif op_type in ('query_shape', 'query_color', 'query_material', 'query_size'):
            parent = index_to_result[op['inputs'][0]]  # must be a reference
            metaconcept = op_type.split('_')[1]
            index_to_result[i] = {
                'kind': 'query',
                'ref_id_var': parent['ref_id_var'],
                'var': parent['var'],
                'terms': [],
                'bound_vars': [],
                'metaconcept': metaconcept,
            }

        # ── equal_integer ────────────────────────────────────────────
        elif op_type == 'equal_integer':
            s1 = index_to_result[op['inputs'][0]]
            s2 = index_to_result[op['inputs'][1]]
            index_to_result[i] = {
                'kind': 'scalar',
                'expr': f'{s1["expr"]} == {s2["expr"]}',
            }

        # ── greater_than / less_than ─────────────────────────────────
        elif op_type in ('greater_than', 'less_than'):
            s1 = index_to_result[op['inputs'][0]]
            s2 = index_to_result[op['inputs'][1]]
            sym = '>' if op_type == 'greater_than' else '<'
            index_to_result[i] = {
                'kind': 'scalar',
                'expr': f'{s1["expr"]} {sym} {s2["expr"]}',
            }

        # ── equal_color / equal_shape / equal_material / equal_size ──
        # Both inputs are `query` results, each carrying a ref_id_var (callable
        # 1-D tensor) and var (string variable name).
        # Use soft composition: same_prop("v1","v2") & ref_1("v1") & ref_2("v2")
        # then .exists() to get a scalar.
        elif op_type in ('equal_color', 'equal_shape', 'equal_material', 'equal_size'):
            q1 = index_to_result[op['inputs'][0]]  # query result
            q2 = index_to_result[op['inputs'][1]]  # query result

            prop = op_type.replace('equal_', '')
            pred_name = f'same_{prop}'
            declare_score(pred_name, _same_questions[pred_name], 2)

            v1, v2 = q1['var'], q2['var']
            r1, r2 = q1['ref_id_var'], q2['ref_id_var']
            # e.g.  (same_color("x1", "x2") & ref_1("x1") & ref_2("x2")).exists()
            all_terms = [
                f'{pred_name}("{v1}", "{v2}")',
                f'{r1}("{v1}")',
                f'{r2}("{v2}")',
            ]
            expr = compose_terms(all_terms)
            index_to_result[i] = {
                'kind': 'scalar',
                'expr': f'({expr}).exists()',
            }

        else:
            raise ValueError(f'Unknown op type: {op_type}, op={op}')

    # ── assemble output code ─────────────────────────────────────────
    last_op_type = get_op_type(program[-1])
    last = index_to_result[len(program) - 1]
    lines = list(declaration_lines) + list(iota_lines) + list(extra_lines)

    if last['kind'] == 'scalar':
        # Return final answers as strings based on the original terminal op.
        # - count          -> numeric string (e.g., "2")
        # - boolean ops    -> "yes" / "no"
        if last_op_type == 'count':
            lines.append(f'return str(int(round(float({last["expr"]}))))')
        elif last_op_type in {
            'exist', 'greater_than', 'less_than', 'equal_integer',
            'equal_color', 'equal_shape', 'equal_material', 'equal_size',
        }:
            lines.append(f'answer_value = {last["expr"]}')
            lines.append('if isinstance(answer_value, ProbabilisticTensor):')
            lines.append('    answer_value = answer_value.tensor.item()')
            lines.append('return "yes" if float(answer_value) > 0.5 else "no"')
        else:
            lines.append(f'return {last["expr"]}')

    elif last['kind'] == 'query':
        # ref_id_var is a callable 1-D tensor (peaked at one object).
        # .argmax() gives the integer index required by query().
        q = _query_questions[last['metaconcept']]
        lines.append(f'return query("{q}", object_id={last["ref_id_var"]}.argmax())')

    elif last['kind'] == 'reference':
        # A bare `unique` at the root — return the tensor; callers use .argmax().
        lines.append(f'return {last["ref_id_var"]}')

    elif last['kind'] == 'object_set':
        # An unresolved object set — call iota directly (no prior exists; the
        # framework handles multi-dim iota natively, as shown by the user).
        expr = compose_terms(last['terms'])
        lines.append(f'return ({expr}).iota("{last["var"]}")')

    return '\n'.join(lines)


def load_concept2query_from_model(vlm_model_name):
    """Load concept2query from the chosen VLM agent, if available.

    Returns:
        dict | None: The concept2query mapping when present; otherwise None.
    """
    if not vlm_model_name:
        return None

    print(f"Loading VLM agent for concept2query: {vlm_model_name}")
    try:
        from vision_agents import get_model
        from probabilistic_tensor import ProbabilisticTensor
        model = get_model(vlm_model_name, wrapper=ProbabilisticTensor)
    except Exception as e:
        print(
            f"Warning: failed to initialize model '{vlm_model_name}' "
            f"for concept2query ({type(e).__name__}: {e}). Falling back to defaults."
        )
        return None

    concept2query = getattr(model, 'concept2query', None)
    if concept2query is None:
        print(
            f"Warning: model '{vlm_model_name}' has no concept2query. "
            "Falling back to built-in templates."
        )
        return None

    print(f"Using concept2query from model '{vlm_model_name}'.")
    return concept2query


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Convert CLEVR structured programs to NePTune score/query programs.',
    )
    parser.add_argument('--input', required=True, help='Path to CLEVR questions.json')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument(
        '--vlm_model_name',
        type=str,
        default=None,
        help='VLM agent name to source question templates from model.concept2query.',
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process first N questions (for testing)')
    args = parser.parse_args()

    concept2query = load_concept2query_from_model(args.vlm_model_name)

    with open(args.input) as f:
        data = json.load(f)

    questions = data['questions']
    if args.limit:
        questions = questions[:args.limit]

    output = {}
    errors = []
    for q in tqdm(questions, desc='Converting'):
        question_str = q['question']
        try:
            neptune_code = clevr_to_neptune(q['program'], concept2query=concept2query)
            output[question_str] = neptune_code
        except Exception as e:
            errors.append({'question': question_str, 'error': str(e)})

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nConverted {len(output)} questions, {len(errors)} errors.')
    if errors:
        print(f'First {min(5, len(errors))} errors:')
        for err in errors[:5]:
            print(f'  Q: {err["question"][:100]}')
            print(f'  E: {err["error"]}')


if __name__ == '__main__':
    main()
