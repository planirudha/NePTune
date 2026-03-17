from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
from openai import OpenAI
import fcntl
import time
from typing import Optional
from .atomic_query_utils import fix_framework_code


BBOX_PATTERN_CODEGEN = re.compile(r"<code>(.*?)</code>", re.DOTALL)
BBOX_PATTERN_CODEGEN_MD = re.compile(r"```python(.*?)```", re.DOTALL)
OBJECT_PATTERN = re.compile(r"<objects>(.*?)</objects>", re.DOTALL)


def extract_code(text_response: str) -> str:
    """Extract Python code from HTML tags or markdown code blocks."""
    code_matches = BBOX_PATTERN_CODEGEN.findall(text_response)
    if code_matches:
        return code_matches[-1].strip()
    
    code_matches = BBOX_PATTERN_CODEGEN_MD.findall(text_response)
    if code_matches:
        return code_matches[-1].strip()
    
    return text_response.strip()


def extract_objects(text_response: str) -> Optional[str]:
    """Extract objects from XML tags."""
    objects_matches = OBJECT_PATTERN.findall(text_response)
    return objects_matches[-1].strip() if objects_matches else None

class CodeGenerator:
    def __init__(self,
                api_key: str,
                model_name: str,
                program_cache_address: str,
                prompt_file_address: str,
                update_cache: bool = False,
                use_cache: bool = True,
                provider: str = "deepseek",
                enable_reasoning: bool = False,
                ):
        
        self.api_key = api_key
        self.update_cache = update_cache
        self.model_name = model_name
        self.provider = provider.lower()
        self.cache_file = program_cache_address
        self.objects_file = program_cache_address.replace(".json", "_objects.json")
        self.prompt_file_address = prompt_file_address
        if provider.lower() == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        elif provider.lower() == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "") if api_key is None else api_key
            self.client = OpenAI(api_key=api_key)
        else:  # deepseek (default)
            api_key = os.getenv("DEEPSEEK_API_KEY", "") if api_key is None else api_key
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.enable_reasoning = enable_reasoning
        self.use_cache = use_cache
        if use_cache:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r") as f: self.program_cache = json.load(f)
            else: self.program_cache = {}
            if os.path.exists(self.objects_file):
                with open(self.objects_file, "r") as f: self.objects_cache = json.load(f)
            else: self.objects_cache = {}
        else:
            self.program_cache = {}
            self.objects_cache = {}

    def generate_code(self, query, reasoning=None, caption=None, prompt_format=None, force_generate=False, prev_code=None, exec_error=None):
        if self.use_cache:
            if os.path.exists(self.cache_file): # Reload cache
                with open(self.cache_file, "r") as f: self.program_cache = json.load(f)
            if os.path.exists(self.objects_file): # Reload cache
                with open(self.objects_file, "r") as f: self.objects_cache = json.load(f)

        if query in self.program_cache and query in self.objects_cache and not force_generate:
            extracted_code, _ = fix_framework_code(self.program_cache[query])
            return extracted_code, self.objects_cache[query]

        if prompt_format is None:
            with open(self.prompt_file_address, "r", encoding="utf-8") as f: prompt_format = f.read()

        prompt = prompt_format.replace("{query}", query)
        if reasoning: prompt += f"\nUse this reasoning chain: {json.dumps(reasoning)}"
        if caption: prompt += f"\nHere is the image caption: {json.dumps(caption)}"
        if prev_code and exec_error:
            prompt += (
                f"\n\nYour previous code attempt failed with an execution error. "
                f"Fix the bug and return a corrected version.\n"
                f"Previous code:\n```python\n{prev_code}\n```\n"
                f"Execution error:\n{exec_error}"
            )
        messages = [{"role": "system", "content": "You are an expert Python programmer."}, {"role": "user", "content": prompt}]

        try:
            is_thinking_model = any(k in self.model_name for k in ("reasoner", "thinking", "deepseek-v3.2"))
            kwargs = dict(model=self.model_name, messages=messages, max_tokens=2**13)
            if not is_thinking_model:
                kwargs["temperature"] = 0.0
            if self.provider == "openrouter":
                kwargs["extra_body"] = {"reasoning": {"enabled": self.enable_reasoning}}
            response = self.client.chat.completions.create(**kwargs)
            text_response = response.choices[0].message.content
            print(text_response)
            extracted_code = extract_code(text_response)
            objects = extract_objects(text_response)

            # Always update in-memory caches
            self.program_cache[query] = extracted_code
            self.objects_cache[query] = objects if objects else ""

            # Persist with atomic, locked write to avoid race conditions
            if self.update_cache:
                self._persist_caches_atomic(query)

            return extracted_code, objects
        except Exception as e:
            print(f"Error generating code for query: {query}\nError: {e}")
            return None, None
    
    def generate_code_json(self, query, reasoning=None, caption=None, prompt_format=None, force_generate=False, return_json_response=False):
        if os.path.exists(self.cache_file): # Reload cache
            with open(self.cache_file, "r") as f: self.program_cache = json.load(f)
        if os.path.exists(self.objects_file): # Reload cache
            with open(self.objects_file, "r") as f: self.objects_cache = json.load(f)

        if query in self.program_cache and query in self.objects_cache and not force_generate:
            extracted_code, _ = fix_framework_code(self.program_cache[query])
            return extracted_code, self.objects_cache[query]

        if prompt_format is None:
            with open(self.prompt_file_address, "r", encoding="utf-8") as f: prompt_format = f.read()

        prompt = prompt_format.replace("{query}", query)
        if reasoning: prompt += f"\nUse this reasoning chain: {json.dumps(reasoning)}"
        if caption: prompt += f"\nHere is the image caption: {json.dumps(caption)}"
        json_structure = "Your output should be a json object with three fields: 'thinking' which describes your thinking about the problem 'code' containing the Python code, and 'objects' containing the identified objects. Ensure the JSON is properly formatted. Example: {\"thinking\": \"I will first identify the largest object...\",  \"logical_representation_of_steps\": \"1. score: is_largest, 2. score: is_red, 3.Composition largest & red\", \"code\": \"def logic_executor(...): ...\", \"objects\": \"sphere, cube\",}"
        messages = [{"role": "system", "content": f"You are an expert Python programmer. {json_structure}"}, {"role": "user", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=8192,
                temperature=0.0,
                n=1,
                response_format={
                    'type': 'json_object'
            })
            text_response = response.choices[0].message.content
            # print(f"LLM Raw Response for {query[:30]}...:\n{text_response}") # Debug
            text_response = json.loads(text_response)
            if return_json_response:
                return text_response
            if 'code' not in text_response or 'objects' not in text_response:
                print(f"Invalid JSON structure in response for query: {query}\nResponse: {text_response}")
                return None, None
            
            extracted_code = text_response['code']
            objects = text_response['objects']
            extracted_code, _ = fix_framework_code(extracted_code)
            # Update in-memory caches
            self.program_cache[query] = extracted_code
            self.objects_cache[query] = objects if objects else ""

            # Atomic persist
            if self.update_cache:
                self._persist_caches_atomic(query)

            return extracted_code, objects
        except Exception as e:
            print(f"Error generating code for query: {query}\nError: {e}")
            return None, None

    # --------------------
    # Persistence helpers
    # --------------------
    def _persist_caches_atomic(self, query_key: str):
        """Atomically merge-update and persist both caches under an exclusive file lock.
        Ensures concurrent writers don't clobber each other's updates and reads see complete files.
        """
        lock_path = f"{self.cache_file}.lock"
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.objects_file) or ".", exist_ok=True)

        with open(lock_path, "w") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                # Re-read latest on-disk caches while holding the lock to merge safely
                disk_prog = {}
                disk_objs = {}
                if os.path.exists(self.cache_file):
                    try:
                        with open(self.cache_file, "r") as f:
                            disk_prog = json.load(f)
                    except Exception:
                        # If file is corrupt, keep going with empty dict
                        disk_prog = {}
                if os.path.exists(self.objects_file):
                    try:
                        with open(self.objects_file, "r") as f:
                            disk_objs = json.load(f)
                    except Exception:
                        disk_objs = {}

                # If another process already populated this query, prefer on-disk values
                if query_key in disk_prog and query_key in disk_objs:
                    self.program_cache[query_key] = disk_prog[query_key]
                    self.objects_cache[query_key] = disk_objs.get(query_key, "")
                else:
                    # Merge our in-memory caches into on-disk dicts
                    disk_prog.update(self.program_cache)
                    disk_objs.update(self.objects_cache)

                # Write both files atomically
                self._atomic_write_json(self.cache_file, disk_prog)
                self._atomic_write_json(self.objects_file, disk_objs)

                # Update in-memory copies to reflect disk
                self.program_cache = disk_prog
                self.objects_cache = disk_objs
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    @staticmethod
    def _atomic_write_json(path: str, data: dict):
        """Write JSON to a temp file and atomically replace the target file."""
        dir_name = os.path.dirname(path) or "."
        base = os.path.basename(path)
        tmp_name = f".{base}.tmp.{os.getpid()}.{int(time.time()*1e6)}"
        tmp_path = os.path.join(dir_name, tmp_name)
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
