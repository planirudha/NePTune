import ast
from typing import Optional

class ScoreBBoxFixer(ast.NodeTransformer):
    """
    Node transformer that:
      1. For score(...) calls whose question mentions both "red bounding box"
         and "green bounding box":
         - Ensures num_objects == 2.
         - If "green bounding box" appears before "red bounding box" in the text,
           swaps the color phrases so the logical mapping is:
             red -> first object, green -> second object.
      2. For score(...) calls whose question mentions ONLY "green bounding box"
         and is effectively a one object predicate (num_objects == 1 or missing):
         - Replaces "green bounding box" with "red bounding box".
      3. For cases where both red and green are mentioned but num_objects is not 2:
         - Forces num_objects = 2.
    """

    RED_PHRASE = "red bounding box"
    GREEN_PHRASE = "green bounding box"

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # First, visit children so other transforms propagate
        self.generic_visit(node)

        # We only care about bare score(...) calls
        func = node.func
        if isinstance(func, ast.Name) and func.id in ["score", "query"]:
            node = self._fix_score_call(node)

        return node

    def _get_question_str(self, node: ast.Call) -> Optional[ast.Constant]:
        """Return the question string Constant node if the first arg is a string."""
        if not node.args:
            return None
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return first
        return None

    def _get_num_objects_info(self, node: ast.Call):
        """
        Return (index_in_args, keyword_index, value) for num_objects.
        Only one of index_in_args or keyword_index will be not None.
        """
        arg_index = None
        kw_index = None
        value = None

        # Keyword form: score("...", num_objects=1, ...)
        for i, kw in enumerate(node.keywords):
            if kw.arg == "num_objects":
                kw_index = i
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                    value = kw.value.value
                return arg_index, kw_index, value

        # Positional form: score("...", 1, "class")
        if len(node.args) >= 2:
            second = node.args[1]
            if isinstance(second, ast.Constant) and isinstance(second.value, int):
                arg_index = 1
                value = second.value

        return arg_index, kw_index, value

    def _set_num_objects(self, node: ast.Call, new_val: int):
        """
        Set num_objects to new_val in either positional or keyword form,
        or add it as a keyword if missing.
        """
        arg_index, kw_index, value = self._get_num_objects_info(node)

        if kw_index is not None:
            node.keywords[kw_index].value = ast.Constant(value=new_val)
        elif arg_index is not None:
            node.args[arg_index] = ast.Constant(value=new_val)
        else:
            # Append as keyword if not present
            node.keywords.append(ast.keyword(arg="num_objects",
                                             value=ast.Constant(value=new_val)))

    def _swap_red_green_in_text_order_sensitive(self, text: str) -> str:
        """
        If green appears before red in the string, swap the phrases
        "red bounding box" and "green bounding box".
        This effectively corrects which color is tied to which position.
        """
        if (self.RED_PHRASE in text and self.GREEN_PHRASE in text and
                text.index(self.GREEN_PHRASE) < text.index(self.RED_PHRASE)):
            # Swap the phrases using a temporary marker
            tmp = "__TMP_RED_BOUNDING_BOX__"
            text = text.replace(self.RED_PHRASE, tmp)
            text = text.replace(self.GREEN_PHRASE, self.RED_PHRASE)
            text = text.replace(tmp, self.GREEN_PHRASE)
        return text

    def _fix_score_call(self, node: ast.Call) -> ast.Call:
        q_node = self._get_question_str(node)
        if q_node is None:
            return node

        q_text = q_node.value
        has_red = self.RED_PHRASE in q_text
        has_green = self.GREEN_PHRASE in q_text

        arg_index, kw_index, num_val = self._get_num_objects_info(node)

        # Case 1: both red and green mentioned
        if has_red and has_green:
            # Fix color assignment if green appears first
            q_text_fixed = self._swap_red_green_in_text_order_sensitive(q_text)

            # Ensure num_objects = 2
            self._set_num_objects(node, 2)

            # Update the question literal
            q_node.value = q_text_fixed
            return node

        # Case 2: only green mentioned, treat as a misplaced single object prompt
        if has_green and not has_red:
            # If num_objects is explicitly 1 or missing, we assume a mistake
            if num_val == 1 or num_val is None:
                q_text_fixed = q_text.replace(self.GREEN_PHRASE, self.RED_PHRASE)
                q_node.value = q_text_fixed
                # Make sure num_objects = 1 if not set
                self._set_num_objects(node, 1)
                return node

        # Case 3: both red and green are expected for num_objects=2 but num_objects wrong
        # (This is partially handled in Case 1; we only need extra logic if someone
        # later edits the question string without num_objects.)
        if has_red and has_green:
            # Already handled above, but kept for clarity
            self._set_num_objects(node, 2)

        return node

def fix_framework_code(source: str) -> tuple[str, bool]:
    """
    Return (new_source, changed).

    `changed` is True if the AST after running ScoreBBoxFixer
    is different from the original AST.
    """
    try:
        tree = ast.parse(source)
        old_dump = ast.dump(tree, include_attributes=False)

        fixer = ScoreBBoxFixer()
        new_tree = fixer.visit(tree)
        ast.fix_missing_locations(new_tree)

        new_dump = ast.dump(new_tree, include_attributes=False)
        changed = (new_dump != old_dump)

        new_source = ast.unparse(new_tree)
        return new_source, changed
    except Exception as e:
        print(f"Error processing source code: {e}")
        return source, False
