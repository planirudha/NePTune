#!/usr/bin/env python3
import ast
import argparse
import json
from typing import Dict, List, Set, Tuple

from tqdm import tqdm


FILTER_SCORE_QUESTIONS = {
    'gray':     'Is the main object in the red bounding box gray?',
    'red':      'Is the main object in the red bounding box red?',
    'blue':     'Is the main object in the red bounding box blue?',
    'green':    'Is the main object in the red bounding box green?',
    'brown':    'Is the main object in the red bounding box brown?',
    'purple':   'Is the main object in the red bounding box purple?',
    'cyan':     'Is the main object in the red bounding box cyan?',
    'yellow':   'Is the main object in the red bounding box yellow?',
    'large':    'Is the main object in the red bounding box large?',
    'small':    'Is the main object in the red bounding box small?',
    'metal':    'Is the main object in the red bounding box made of metal?',
    'rubber':   'Is the main object in the red bounding box made of rubber?',
    'cube':     'Is the main object in the red bounding box a cube?',
    'sphere':   'Is the main object in the red bounding box a sphere?',
    'cylinder': 'Is the main object in the red bounding box a cylinder?',
}

SAME_SCORE_QUESTIONS = {
    'same_color':    'Is the main object in the red bounding box the same color as the main object in the green bounding box?',
    'same_shape':    'Is the main object in the red bounding box the same shape as the main object in the green bounding box?',
    'same_size':     'Is the main object in the red bounding box the same size as the main object in the green bounding box?',
    'same_material': 'Is the main object in the red bounding box made of the same material as the main object in the green bounding box?',
}

SPATIAL_RELATIONS = frozenset({'left', 'right', 'behind', 'front'})


def compose_terms(terms: List[str]) -> str:
    if len(terms) == 0:
        return ''
    if len(terms) == 1:
        return terms[0]
    if len(terms) == 2:
        return f'{terms[0]} & {terms[1]}'
    return f'and_op({", ".join(terms)})'


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


class LambdaToNeptuneConverter:
    def __init__(self):
        self.declared_scores: Dict[str, bool] = {}
        self.declaration_lines: List[str] = []
        self.iota_lines: List[str] = []
        self.ref_counter = 0

    def new_ref(self) -> str:
        self.ref_counter += 1
        return f'ref_{self.ref_counter}'

    def declare_score(self, pred_name: str) -> None:
        if pred_name in self.declared_scores:
            return
        if pred_name in FILTER_SCORE_QUESTIONS:
            q = FILTER_SCORE_QUESTIONS[pred_name]
            n = 1
        elif pred_name in SAME_SCORE_QUESTIONS:
            q = SAME_SCORE_QUESTIONS[pred_name]
            n = 2
        else:
            raise ValueError(f'Unsupported predicate for score(): {pred_name}')
        self.declared_scores[pred_name] = True
        self.declaration_lines.append(f'{pred_name} = score("{q}", num_objects={n})')

    @staticmethod
    def _name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        raise ValueError(f'Expected variable name, got: {ast.dump(node)}')

    @staticmethod
    def _func_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Name):
            return node.func.id
        raise ValueError(f'Unsupported call target: {ast.dump(node.func)}')

    def _parse_iota_arg(self, node: ast.AST) -> Tuple[str, str]:
        if not isinstance(node, ast.Call):
            raise ValueError('Expected iota(...) call in relation argument.')
        if self._func_name(node) != 'iota':
            raise ValueError(f'Expected iota call, got: {self._func_name(node)}')
        if len(node.args) != 2:
            raise ValueError('iota() must have 2 args: Object, lambda ...')
        lam = node.args[1]
        if not isinstance(lam, ast.Lambda) or len(lam.args.args) != 1:
            raise ValueError('iota second arg must be lambda with one variable.')

        iota_var = lam.args.args[0].arg
        terms, free_vars = self.parse_conjunction(lam.body)
        terms = dedup_keep_order(terms)
        expr = compose_terms(terms)
        if not expr:
            raise ValueError('Empty iota body is not supported.')

        other_vars = sorted(v for v in free_vars if v != iota_var)
        for var in other_vars:
            expr = f'({expr}).exists("{var}")'

        ref_name = self.new_ref()
        self.iota_lines.append(f'{ref_name} = ({expr}).iota("{iota_var}")')
        return ref_name, iota_var

    def _parse_arg(self, node: ast.AST) -> Tuple[List[str], Set[str], str]:
        if isinstance(node, ast.Name):
            v = node.id
            return [], {v}, v
        if isinstance(node, ast.Call) and self._func_name(node) == 'iota':
            ref_name, ref_var = self._parse_iota_arg(node)
            return [f'{ref_name}("{ref_var}")'], {ref_var}, ref_var
        raise ValueError(f'Unsupported relation argument: {ast.dump(node)}')

    def parse_conjunction(self, node: ast.AST) -> Tuple[List[str], Set[str]]:
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, ast.And):
                raise ValueError('Only conjunctions (and) are supported in lambda programs.')
            all_terms, all_vars = [], set()
            for value in node.values:
                terms, vars_set = self.parse_conjunction(value)
                all_terms.extend(terms)
                all_vars |= vars_set
            return all_terms, all_vars

        if isinstance(node, ast.Call):
            fname = self._func_name(node)

            if fname in FILTER_SCORE_QUESTIONS:
                if len(node.args) != 1:
                    raise ValueError(f'Unary predicate {fname} must have one arg.')
                var = self._name(node.args[0])
                self.declare_score(fname)
                return [f'{fname}("{var}")'], {var}

            if fname in SPATIAL_RELATIONS or fname in SAME_SCORE_QUESTIONS:
                if len(node.args) != 2:
                    raise ValueError(f'Binary predicate {fname} must have two args.')
                if fname in SAME_SCORE_QUESTIONS:
                    self.declare_score(fname)
                pre_a, vars_a, a_var = self._parse_arg(node.args[0])
                pre_b, vars_b, b_var = self._parse_arg(node.args[1])
                main = f'{fname}("{a_var}", "{b_var}")'
                return pre_a + pre_b + [main], vars_a | vars_b

            if fname == 'exists':
                # Rare inside bodies; support directly to keep parser robust.
                qvars, body = self.collect_exists(node)
                terms, free_vars = self.parse_conjunction(body)
                expr = compose_terms(dedup_keep_order(terms))
                for qv in reversed(qvars):
                    expr = f'({expr}).exists("{qv}")'
                return [expr], free_vars - set(qvars)

            raise ValueError(f'Unsupported call in lambda body: {fname}')

        raise ValueError(f'Unsupported expression in lambda body: {ast.dump(node)}')

    def collect_exists(self, call_node: ast.Call) -> Tuple[List[str], ast.AST]:
        qvars = []
        current = call_node
        while isinstance(current, ast.Call) and self._func_name(current) == 'exists':
            if len(current.args) != 2:
                raise ValueError('exists() must have 2 args: Object, lambda ...')
            lam = current.args[1]
            if not isinstance(lam, ast.Lambda) or len(lam.args.args) != 1:
                raise ValueError('exists second arg must be lambda with one variable.')
            qvar = lam.args.args[0].arg
            qvars.append(qvar)
            body = lam.body
            if isinstance(body, ast.Call) and self._func_name(body) == 'exists':
                current = body
            else:
                return qvars, body
        raise ValueError('Malformed exists chain.')

    def convert_program(self, program_str: str) -> str:
        parsed = ast.parse(program_str, mode='eval').body
        if not isinstance(parsed, ast.Call):
            raise ValueError('Top-level program must be a function call.')

        root = self._func_name(parsed)
        lines: List[str] = []

        if root == 'exists':
            qvars, body = self.collect_exists(parsed)
            terms, _ = self.parse_conjunction(body)
            terms = dedup_keep_order(terms)
            expr = compose_terms(terms)
            if not expr:
                raise ValueError('Empty exists body.')
            for qv in reversed(qvars):
                expr = f'({expr}).exists("{qv}")'
            lines.extend(self.declaration_lines)
            lines.extend(self.iota_lines)
            lines.append(f'answer_value = {expr}')
            lines.append('if isinstance(answer_value, ProbabilisticTensor):')
            lines.append('    answer_value = answer_value.tensor.item()')
            lines.append('return "yes" if float(answer_value) > 0.5 else "no"')
            return '\n'.join(lines)

        if root == 'point':
            if len(parsed.args) != 2:
                raise ValueError('point() must have 2 args: Object, lambda ...')
            lam = parsed.args[1]
            if not isinstance(lam, ast.Lambda) or len(lam.args.args) != 1:
                raise ValueError('point second arg must be lambda with one variable.')
            target_var = lam.args.args[0].arg
            terms, free_vars = self.parse_conjunction(lam.body)
            terms = dedup_keep_order(terms)
            expr = compose_terms(terms)
            if not expr:
                raise ValueError('Empty point body.')

            other_vars = sorted(v for v in free_vars if v != target_var)
            for var in other_vars:
                expr = f'({expr}).exists("{var}")'

            lines.extend(self.declaration_lines)
            lines.extend(self.iota_lines)
            lines.append(f'return ({expr}).iota("{target_var}")')
            return '\n'.join(lines)

        raise ValueError(f'Unsupported top-level op: {root}')


def detect_questions_key(data: dict) -> str:
    for key in ('puzzles', 'rpms', 'refexps'):
        if key in data:
            return key
    raise ValueError('Could not detect question list key (puzzles/rpms/refexps).')


def main():
    parser = argparse.ArgumentParser(
        description='Deterministically convert CLEVR-extension lambda GT programs to NePTune code.',
    )
    parser.add_argument('--input', required=True, help='Input JSON path (puzzle/rpm/refexps).')
    parser.add_argument('--output', required=True, help='Output JSON mapping question->NePTune code.')
    parser.add_argument('--limit', type=int, default=None, help='Only process first N examples.')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        data = json.load(f)

    key = detect_questions_key(data)
    items = data[key]
    if args.limit is not None:
        items = items[:args.limit]

    output = {}
    errors = []
    duplicate_same_program = 0
    duplicate_conflict = 0

    for item in tqdm(items, desc='Converting lambda->NePTune'):
        question = item['question']
        program = item['program']
        try:
            converter = LambdaToNeptuneConverter()
            code = converter.convert_program(program)
            if question in output:
                if output[question] == code:
                    duplicate_same_program += 1
                else:
                    duplicate_conflict += 1
                    errors.append({
                        'question': question,
                        'error': 'Duplicate question has conflicting converted programs.',
                    })
            else:
                output[question] = code
        except Exception as e:
            errors.append({'question': question, 'error': f'{type(e).__name__}: {e}'})

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(
        f'Processed {len(items)} items. '
        f'Unique questions written: {len(output)}. '
        f'Duplicate same-program: {duplicate_same_program}. '
        f'Duplicate conflicts: {duplicate_conflict}. '
        f'Errors: {len(errors)}'
    )
    if errors:
        print('First errors:')
        for err in errors[:10]:
            print(f'- Q: {err["question"][:120]}')
            print(f'  E: {err["error"]}')


if __name__ == '__main__':
    main()
