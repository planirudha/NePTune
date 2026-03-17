import textwrap
from typing import Any, List, Optional

BOUND_CODE_TEMPLATE = """
def logic_executor(question, score, query, all_bounding_boxes, image, history):
    objects_count = len(all_bounding_boxes)
{code}
"""

def build_logic_executor_code(code_snippet: str, template: Optional[str] = None) -> str:
    active_template = template if template is not None else BOUND_CODE_TEMPLATE
    return active_template.format(code=textwrap.indent(code_snippet, "    "))
    # return active_template.format(code=textwrap.indent(code_snippet))

def build_score_fn(
    vl_model_inst: Any,
    image: Any,
    proposals: Optional[List[List[float]]],
    history: Optional[List[List[str]]],
    with_mask: bool = False,
    draw_bbox: bool = True,
):
    def score(question, num_objects, type=None):
        return vl_model_inst.score(
            image,
            proposals,
            question,
            num_objects,
            draw_bbox=draw_bbox,
            use_batch=False,
            pass_og_image=False,
            overlay_masks=False,
            mask=False,
            contrastive_mask=False,
            with_mask=with_mask,
            type=type,
            history=history,
        )
    return score

def build_query_fn(vl_model_inst: Any, image: Any, proposals: Optional[List[List[float]]]):
    def query(question, object_id=None, mask=False, mask_outside=False, draw_bbox=True, extract_object=False):
        return vl_model_inst.query(
            image,
            proposals,
            object_id,
            question,
            draw_bbox=draw_bbox,
            pass_og_image=False,
            overlay_masks=False,
        )
    return query