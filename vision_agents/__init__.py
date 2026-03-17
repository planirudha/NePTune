from .ovis import OvisGemma
from .ovis_2 import Ovis2
from .qwen_vl_2 import QwenVL2
from .intern_vl import InternVL
from .dummy import Dummy
from .agent import Agent
from .marker import Marker, MarkerV2
from .depth_anything_v2.dpt import DepthAnythingV2
from .naver import NaverBaseModel


__all__ = ['OvisGemma', 'Ovis2', 'InternVL', 'QwenVL2', 'Agent', 'mark_objects', 'get_model', 'Marker', 'MarkerV2', 'Dummy']

def get_model(model_id, device="cuda", wrapper=None):
    if model_id == "dummy":
        return Dummy(device=device, wrapper=wrapper)
    elif model_id == "ovis2":
        return Ovis2(device=device, wrapper=wrapper)
    elif model_id == "ovis2_1b":
        return Ovis2(model_name="AIDC-AI/Ovis2-1B",device=device, wrapper=wrapper)
    elif model_id == "ovis2_2b":
        return Ovis2(model_name="AIDC-AI/Ovis2-2B",device=device, wrapper=wrapper)
    elif model_id == "ovis2_4b":
        return Ovis2(model_name="AIDC-AI/Ovis2-4B", device=device, wrapper=wrapper)
    elif model_id == "ovis2_8b":
        return Ovis2(model_name="AIDC-AI/Ovis2-8B", device=device, wrapper=wrapper)
    elif model_id == "ovis2_34b":
        return Ovis2(model_name="AIDC-AI/Ovis2-34B", device=device, wrapper=wrapper)
    elif model_id == "ovis":
        return OvisGemma(model_name="AIDC-AI/Ovis1.6-Gemma2-9B", device=device, wrapper=wrapper)
    elif model_id == "ovis_3b":
        return OvisGemma(model_name="AIDC-AI/Ovis1.6-Llama3.2-3B", device=device, wrapper=wrapper)
    elif model_id == "ovis_9b":
        return OvisGemma(model_name="AIDC-AI/Ovis1.6-Gemma2-9B", device=device, wrapper=wrapper)
    elif "intern_tuned" in model_id:
        from .intern_vl_tuned import InternVLTuned
        return InternVLTuned(model_path=model_id,device=device, wrapper=wrapper)
    elif model_id == "naver":
        return NaverBaseModel(device=device, wrapper=wrapper)
    elif model_id == "intern_vllm":
        from .intern_vl_vllm import InternVL as InternVLLM
        import warnings
        warnings.filterwarnings('ignore', message="The following intended overrides are not keyword args and will be dropped: {'truncation'}")
        return InternVLLM(device=device, wrapper=wrapper)
        # return InternVLLM(model_path="OpenGVLab/InternVL3_5-1B", device=device, wrapper=wrapper)
    elif model_id == "intern_vllm1":
        from .intern_vl_vllm import InternVL as InternVLLM
        import warnings
        warnings.filterwarnings('ignore', message="The following intended overrides are not keyword args and will be dropped: {'truncation'}")
        return InternVLLM(device=device, wrapper=wrapper)
        # return InternVLLM(model_path="OpenGVLab/InternVL3_5-1B", device=device, wrapper=wrapper)
    elif model_id == "intern":
        return InternVL(device=device, wrapper=wrapper)
    elif model_id == "intern_25_8b_awq":
        return InternVL(model_path="OpenGVLab/InternVL2_5-8B-MPO-AWQ", device=device, wrapper=wrapper)
    elif model_id == "intern2_8b":
        return InternVL(model_path="OpenGVLab/InternVL2-8B", load_4bit=False, device=device, wrapper=wrapper)
    elif model_id == "intern25_8b":
        return InternVL(model_path="OpenGVLab/InternVL2_5-8B-MPO", load_4bit=False, device=device, wrapper=wrapper)
    elif model_id == "intern25_8b_4b":
        return InternVL(model_path="OpenGVLab/InternVL2_5-8B-MPO", load_4bit=True, device=device, wrapper=wrapper)
    elif model_id == "intern25_4b":
        return InternVL(model_path="OpenGVLab/InternVL2_5-4B-MPO", load_4bit=False, device=device, wrapper=wrapper)
    elif model_id == "intern25_2b":
        return InternVL(model_path="OpenGVLab/InternVL2_5-2B-MPO", load_4bit=False, device=device, wrapper=wrapper)
    elif model_id == "intern25_1b":
        return InternVL(model_path="OpenGVLab/InternVL2_5-1B-MPO", load_4bit=False, device=device, wrapper=wrapper)
    elif model_id == "intern25_26b":
        return InternVL(model_path="OpenGVLab/InternVL2_5-26B-MPO", load_4bit=False, device=device, wrapper=wrapper)
    elif model_id == "intern25_26b_4b":
        return InternVL(model_path="OpenGVLab/InternVL2_5-26B-MPO", load_4bit=True, device=device, wrapper=wrapper)
    elif model_id == "qwen2":
        return QwenVL2(device=device, wrapper=wrapper)
    elif model_id == "qwen2_2b":
        return QwenVL2(model_name="Qwen/Qwen2-VL-2B-Instruct",device=device, wrapper=wrapper)
    elif model_id == "qwen2_7b":
        return QwenVL2(model_name="Qwen/Qwen2-VL-7B-Instruct",device=device, wrapper=wrapper)
    else:
        raise ValueError(f"Model {model_id} not found.")