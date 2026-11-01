# NePTune CLEVR pipeline: code map

This file traces one run of `python run.py --config configs/clevr.json` from the command line to the saved results. Each stage links to the exact code involved. Object detection is done by **Grounding DINO** (Stage 4).

```
run.py ──► run_vqa_parallel.py ──► run_vqa.py (one worker per GPU)
                                      │
   ┌──────────────────────────────────┼──────────────────────────────────────┐
   │ per sample:                      │                                      │
   │  1. load image + question        (local_datasets/clevr.py)             │
   │  2. get program                  (code_generator/code_with_obj.py)     │
   │  3. build detector prompt        (run_vqa.py: nouns → "object. thing.")│
   │  4. GROUNDING DINO → boxes       (vision_utils.py: Dino.predict)       │
   │  5. expand boxes ×1.1            (vision_utils.py: expand_bboxes)      │
   │  6. run program:                                                        │
   │       score(q, n) → InternVL P("Yes") per box / box pair  (agent.py)   │
   │       combine with ProbabilisticTensor: & | exists iota               │
   │  7. answer string → results JSON                                       │
   └─────────────────────────────────────────────────────────────────────────┘
```

---

## How to run

```bash
cd /storage-01/apoovalakunta/dev_kit/NePTune
CLEVR_DIR=$PWD/data/clevr .venv/bin/python run.py --config configs/clevr.json --gpus 0
```

- `data/clevr/val/` holds symlinks into `/storage-01/datasets/CLEVR/CLEVR_v1.0`:
  - `questions.json` → `questions/CLEVR_val_questions.json`
  - `scenes.json` → `scenes/CLEVR_val_scenes.json`
  - `images/` → `images/val/`
- Per-GPU log: `experiments/clevr/intern25_8b/results_clevr_split0.attempt0.log`
- Per-GPU results: `experiments/clevr/intern25_8b/results_clevr_split0.json`, merged at the end into `results_clevr.json`
- The config is [configs/clevr.json](configs/clevr.json). The fields that matter most here are `od_model: "dino"`, `od_box_threshold: 0.25`, `vlm_model_name: "intern25_8b"`, `num_samples: 2000` and `program_cache_address`.

---

## Stage 0: Entrypoint and GPU fan-out

| What | Where |
|---|---|
| Choose the runner from `dataset` (`clevr` → VQA) | [run.py:43](run.py#L43), [run.py:135-142](run.py#L135-L142) |
| Start one `run_vqa.py` worker per GPU, pinned with `CUDA_VISIBLE_DEVICES` | [runners/run_vqa_parallel.py:135-160](runners/run_vqa_parallel.py#L135-L160) |
| Worker command line (`--num_splits`, `--split_index`) | [runners/run_vqa_parallel.py:136-142](runners/run_vqa_parallel.py#L136-L142) |
| Merge the split result files at the end | [runners/run_vqa_parallel.py:42](runners/run_vqa_parallel.py#L42), [runners/run_vqa_parallel.py:224](runners/run_vqa_parallel.py#L224) |
| Worker entry: config overrides the argparse defaults | [runners/run_vqa.py:545-551](runners/run_vqa.py#L545-L551) |

## Stage 1: Dataset loading

| What | Where |
|---|---|
| `get_dataset("clevr")` reads `CLEVR_DIR/val/{questions.json, scenes.json, images/}` | [local_datasets/__init__.py:128-134](local_datasets/__init__.py#L128-L134) |
| `CLEVRDataset.__getitem__` returns image, question, answer and `bboxes` | [local_datasets/clevr.py:42-82](local_datasets/clevr.py#L42-L82) |
| **Detector switch:** `bboxes=None` when the scene has no `objects_detection` field, which is the case for the official CLEVR scenes. `None` makes Stage 4 run Grounding DINO. | [local_datasets/clevr.py:80](local_datasets/clevr.py#L80) |
| Split the dataset across GPUs | [runners/run_vqa.py:570-588](runners/run_vqa.py#L570-L588) |

> Previously this returned `[]` for official scenes. An empty list is not `None`, so the runner skipped DINO, every program received zero boxes, and `score()` failed with `RuntimeError: stack expects a non-empty TensorList`.

## Stage 2: Model initialisation

| What | Where |
|---|---|
| `initialize_models` builds the code generator, detector and VLM | [runners/run_vqa.py:243-273](runners/run_vqa.py#L243-L273) |
| Detector built from `od_model` and `od_box_threshold` | [runners/run_vqa.py:258-259](runners/run_vqa.py#L258-L259) |
| `object_detector("dino")` → `Dino(threshold)` | [vision_utils.py:1022-1026](vision_utils.py#L1022-L1026) |
| `Dino.__init__` loads `IDEA-Research/grounding-dino-base` via `transformers` onto `cuda:0` | [vision_utils.py:411-417](vision_utils.py#L411-L417) |
| VLM registry: `intern25_8b` → `OpenGVLab/InternVL2_5-8B-MPO` | [vision_agents/\_\_init\_\_.py:58-59](vision_agents/__init__.py#L58-L59) |
| `InternVL` class | [vision_agents/intern_vl.py:527](vision_agents/intern_vl.py#L527) |

## Stage 3: Program generation (LLM, cached)

| What | Where |
|---|---|
| Per-sample loop | [runners/run_vqa.py:646](runners/run_vqa.py#L646) |
| Retry loop (up to 2 regenerations after an execution error) | [runners/run_vqa.py:721-735](runners/run_vqa.py#L721-L735) |
| `generate_code_and_object_queries` | [runners/run_vqa.py:275-304](runners/run_vqa.py#L275-L304) |
| `CodeGenerator.generate_code`: returns the program from the cache on a hit, otherwise calls the LLM | [code_generator/code_with_obj.py:78-127](code_generator/code_with_obj.py#L78-L127) |
| Cache lookup: `programs/clevr.json` and `programs/clevr_objects.json` | [code_generator/code_with_obj.py:84-86](code_generator/code_with_obj.py#L84-L86) |
| Few-shot prompt sent to the LLM on a cache miss | [prompts/clevr-declarative.txt](prompts/clevr-declarative.txt) |
| Post-processing of generated code | [code_generator/atomic_query_utils.py:142](code_generator/atomic_query_utils.py#L142) |
| Wrap the program into `def logic_executor(...)` | [runners/code_base_template.py:4-12](runners/code_base_template.py#L4-L12), called at [runners/run_vqa.py:748](runners/run_vqa.py#L748) |

The cache covers all 2000 configured questions, so no API key is needed. Regenerating after an execution error does need `OPENROUTER_API_KEY` in `.env`. Without it the retry fails and the same cached program runs again.

Example cached program (Q: *"Are there any other things that are the same shape as the big metallic object?"*):

```python
is_large   = score("Is the main object in the red bounding box large?", num_objects=1)
is_metal   = score("Is the main object in the red bounding box made of metal?", num_objects=1)
same_shape = score("Is the main object in the red bounding box the same shape as the main object in the green bounding box?", num_objects=2)
ref_1 = (is_large("x1") & is_metal("x1")).iota("x1")
answer_value = ((same_shape("x2", "x1") & ref_1("x1")).exists("x1")).exists()
...
return "yes" if float(answer_value) > 0.5 else "no"
```

## Stage 4: Object detection with Grounding DINO ⭐

**4a. Build the text prompt** in [runners/run_vqa.py:281-303](runners/run_vqa.py#L281-L303):
- Take the object names from the objects cache. These are all empty for CLEVR.
- Add the spaCy noun lemmas from the question ([runners/run_vqa.py:288-292](runners/run_vqa.py#L288-L292)).
- Join them, longest first, into `"object. thing. shape."`. If there are none, fall back to `"object. entity."` ([runners/run_vqa.py:301-302](runners/run_vqa.py#L301-L302)).

**4b. Call the detector** in [runners/run_vqa.py:751-761](runners/run_vqa.py#L751-L761):

```python
if item_data.get("bboxes", None) is None:                      # true now for official CLEVR scenes
    proposal_tensor = od_predictor.predict(image_input, text=od_prompt, apply_custom_nms=True)
    all_proposals_from_od = proposal_tensor.cpu().numpy().tolist()
    print(f"[{item_id}] OD generated {len(...)} proposals using prompt: ...")
else:
    all_proposals_from_od = item_data["bboxes"]                # precomputed Mask R-CNN boxes
```

**4c. Inside `Dino.predict`** in [vision_utils.py:419-456](vision_utils.py#L419-L456):

| Step | Line |
|---|---|
| Processor: image and text prompt → tensors | [vision_utils.py:420](vision_utils.py#L420) |
| Grounding DINO forward pass (`torch.no_grad`) | [vision_utils.py:421-422](vision_utils.py#L421-L422) |
| `post_process_grounded_object_detection`, box threshold `0.25` (config), text threshold `0.25` | [vision_utils.py:423-430](vision_utils.py#L423-L430) |
| Custom NMS (IoU 0.5), sorted by score | [vision_utils.py:444-449](vision_utils.py#L444-L449), `apply_nms` at [vision_utils.py:52](vision_utils.py#L52) |
| Returns an `(N, 4)` xyxy tensor | [vision_utils.py:456](vision_utils.py#L456) |

`Dino.predict_v2` ([vision_utils.py:458](vision_utils.py#L458)) runs one query per `.`-separated phrase. The runner does not use it.

**4d. Expand the boxes** by 1.1× around their centres and clamp them to the image: [runners/run_vqa.py:769](runners/run_vqa.py#L769) → [vision_utils.py:169](vision_utils.py#L169).

Sanity check on `CLEVR_val_000000.png`: prompt `"object. thing. shape."` gives 5 boxes, and the scene has 5 ground-truth objects.

**Where the DINO output goes:** the full pipeline does not save the boxes. It only records the box count (`num_proposals`) and the prompt (`od_prompt_used`) in the results JSON, and prints an `OD generated N proposals` log line. To get the boxes themselves, run DINO on its own with [scripts/run_dino_only.py](scripts/run_dino_only.py):

```bash
.venv/bin/python scripts/run_dino_only.py --num_images 20 --save_vis   # -1 = all images
```

It writes to `experiments/clevr/dino_only/`:
- `detections.json`: per image, `boxes_xyxy` (raw DINO output), `boxes_expanded_xyxy` (the 1.1× boxes the VLM sees), `num_boxes` and `gt_num_objects`
- `vis/<image>.png` (with `--save_vis`): the image with the boxes drawn and numbered

On the first 20 validation images, the box count matched the ground-truth object count on 20/20.

## Stage 5: Program execution (VLM + probabilistic tensors)

| What | Where |
|---|---|
| `execute_generated_code`: `exec` the program and call `logic_executor` | [runners/run_vqa.py:328-489](runners/run_vqa.py#L328-L489), call at [runners/run_vqa.py:461-468](runners/run_vqa.py#L461-L468) |
| Optional two-pass "fast-cache" (off in this config) | [runners/run_vqa.py:375-452](runners/run_vqa.py#L375-L452) |
| `score` / `query` closures bound to the image and the DINO boxes | [runners/code_base_template.py:15-51](runners/code_base_template.py#L15-L51) |
| `Agent.score`: every box combination (n boxes ^ `num_objects`) | [vision_agents/agent.py:303-430](vision_agents/agent.py#L303-L430), combinations at [vision_agents/agent.py:328](vision_agents/agent.py#L328) |
| Draw red/green boxes on the image for each combination | [vision_agents/agent.py:99-110](vision_agents/agent.py#L99-L110), [vision_agents/marker.py:445](vision_agents/marker.py#L445) |
| Per-image VLM call and stacking into a tensor (the old crash site) | [vision_agents/agent.py:376-411](vision_agents/agent.py#L376-L411) |
| `InternVL._score`: P("Yes") from the answer-token logits | [vision_agents/intern_vl.py:1097](vision_agents/intern_vl.py#L1097) |
| `Agent.query`: open-ended question about one box | [vision_agents/agent.py:440-476](vision_agents/agent.py#L440-L476), `InternVL._query` at [vision_agents/intern_vl.py:1472](vision_agents/intern_vl.py#L1472) |
| `ProbabilisticTensor` algebra (`&`, `\|`, `~`) | [probabilistic_tensor/probabilistic_tensor.py:293](probabilistic_tensor/probabilistic_tensor.py#L293), `and_op` at [:72](probabilistic_tensor/probabilistic_tensor.py#L72), `or_op` at [:194](probabilistic_tensor/probabilistic_tensor.py#L194) |
| `.exists()`, `.argmax()`, `.iota()` | [probabilistic_tensor/probabilistic_tensor.py:537](probabilistic_tensor/probabilistic_tensor.py#L537), [:641](probabilistic_tensor/probabilistic_tensor.py#L641), [:964](probabilistic_tensor/probabilistic_tensor.py#L964) |

## Stage 6: Results and evaluation

| What | Where |
|---|---|
| Record the answer, errors, `od_prompt_used`, `num_proposals` and the program | [runners/run_vqa.py:789-807](runners/run_vqa.py#L789-L807) |
| Periodic save (every `save_interval=50`) | [runners/run_vqa.py:836-838](runners/run_vqa.py#L836-L838), `save_results_to_file` at [runners/run_vqa.py:522](runners/run_vqa.py#L522) |
| Resume: already-processed IDs are skipped | [runners/run_vqa.py:199](runners/run_vqa.py#L199), [runners/run_vqa.py:657-659](runners/run_vqa.py#L657-L659) |
| Answer normalisation | [utils.py:48](utils.py#L48) |
| LLM-judge evaluation (`--evaluate`, needs `DEEPSEEK_API_KEY`) | [run.py:115-130](run.py#L115-L130) → [analysis/evaluate_clevr_human.py](analysis/evaluate_clevr_human.py) |

---

## Useful log lines

Search the per-GPU `.attempt0.log` for these:

- `OD generated N proposals using prompt: '...'`: Grounding DINO ran (Stage 4).
- `Using provided OD proposals`: precomputed boxes were used instead of DINO.
- `Executing code for VQA question`: start of Stage 5.
- `Code Generated Answer: '...'`: final answer for the sample.
- `Code execution failed`: the program raised an error. The traceback follows it.
