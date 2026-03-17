# NePTune

[![Paper](https://img.shields.io/badge/arXiv-2509.25757-b31b1b.svg)](https://arxiv.org/pdf/2509.25757)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **NePTune: A Neuro-Pythonic Framework for Tunable Compositional Reasoning on Vision-Language**
> Danial Kamali, Parisa Kordjamshidi — Michigan State University, HLR Lab — Accepted at **ICLR 2026**

**NePTune** is a neurosymbolic visual reasoning framework that combines LLM-generated logic programs, Vision-Language Model (VLM) oracles, and a probabilistic tensor algebra to answer questions about images and ground referring expressions.

---

## How It Works

For each image–question pair, NePTune runs a three-stage pipeline:

1. **Object Detection** — Grounding DINO or Florence-2 proposes bounding boxes over the scene.
2. **Code Generation** — A DeepSeek (or OpenAI-compatible) LLM translates the natural language question into a short Python logic program, using a few-shot prompt template. Generated programs are cached for reuse.
3. **Program Execution** — The program calls two primitives:
   - `score(question, num_objects)` — marks each bounding box on the image and asks a VLM to score it (logit-based soft probability).
   - `query(question, object_id)` — asks the VLM an open-ended question about a specific object.

   Results are composed using the **ProbabilisticTensor** algebra (`and_op`, `or_op`, `.exists()`, `.iota()`) to produce a final bounding box (referring expression) or answer string (VQA).

```
Image + Question
      │
      ├─── Object Detector (DINO / Florence-2)
      │         └─► bounding box proposals
      │
      ├─── Code Generator (DeepSeek LLM)
      │         └─► Python logic program
      │
      └─── Program Execution
                ├─► VLM Oracle (InternVL / Ovis2 / Qwen2-VL)
                │       └─► per-object soft scores [0, 1]
                └─► ProbabilisticTensor Engine
                        └─► final answer / bounding box
```

---

## Supported Tasks & Datasets

| Task | Datasets |
|------|----------|
| Visual Question Answering | CLEVR, CLEVR-Humans |
| Referring Expression Comprehension | RefCOCO-Adv, RefGTA |
| CLEVR-Transfer | RPM puzzles, Raven's Progressive Matrices, Referring Expressions |


## Installation

### 1. Clone the repository

```bash
git clone --recurse-submodules git@github.com:iamdanialkamali/NePTune.git
cd NePTune
```

> `Concepts` and `Jacinle` are included as git submodules. If you forgot `--recurse-submodules`, run:
> ```bash
> git submodule update --init --recursive
> ```

### 2. Install dependencies

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e Concepts/ -e Jacinle/
python -m spacy download en_core_web_sm
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e Concepts/ -e Jacinle/
python -m spacy download en_core_web_sm
```

### 3. Install SAM / SAM2 (optional — used for segmentation/masking)

```bash
# SAM (Segment Anything v1)
pip install 'git+https://github.com/facebookresearch/segment-anything.git'
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
# Place sam_vit_h_4b8939.pth at the repo root

# SAM2 (Segment Anything v2.1)
git clone https://github.com/facebookresearch/sam2.git
cd sam2 && pip install -e . && cd ..
mkdir -p sam_res/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
     -O sam_res/checkpoints/sam2.1_hiera_large.pt
# Note: the checkpoint must be at sam_res/checkpoints/sam2.1_hiera_large.pt specifically
```

### 4. Install Depth Anything V2 (optional — used for depth estimation)

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2 && pip install -r requirements.txt && cd ..
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth \
     -O depth_anything_v2/depth_anything_v2_vits.pth
```

### 5. Decompress program cache

```bash
gunzip -k programs/clevr.json.gz
```

### 6. Set up API keys

Create a `.env` file at the repo root:

```
OPENROUTER_API_KEY=your_openrouter_key   # recommended (deepseek/deepseek-v3.2)
DEEPSEEK_API_KEY=your_deepseek_key       # alternative for code generation
OPENAI_API_KEY=your_openai_key           # optional
```

### 7. VLM weights

The following VLMs are supported and loaded automatically via HuggingFace `from_pretrained`:

| Model family | HuggingFace IDs |
|---|---|
| InternVL2.5 | `OpenGVLab/InternVL2_5-1B-MPO`, `…-2B`, `…-4B`, `…-8B`, `…-26B` |
| Ovis2 | `AIDC-AI/Ovis2-1B`, `…-2B`, `…-4B`, `…-8B` |
| Qwen2-VL | `Qwen/Qwen2-VL-2B-Instruct`, `Qwen/Qwen2-VL-7B-Instruct` |

Grounding DINO and Florence-2 (object detectors) are also loaded automatically via `transformers`.

---

## Usage

All experiments are launched through the unified entrypoint `run.py`. It reads a config file, infers the task automatically, and dispatches to the appropriate runner. Any extra flags are forwarded transparently to the underlying script.

```
python run.py --config <config_file> [--gpus <ids>] [--evaluate] [options]
```

### Step 1 — Run the experiment

#### Visual Question Answering (CLEVR, CLEVR-Humans)

```bash
# All available GPUs (default)
python run.py --config configs/clevr-humans.json

# Specific GPUs
python run.py --config configs/clevr-humans.json --gpus 0 1 2 3

# CLEVR
python run.py --config configs/clevr.json --gpus 0 1 2 3
```

#### Referring Expression Comprehension (RefCOCO-Adv, RefGTA)

```bash
python run.py --config configs/refadv.json
python run.py --config configs/refgta.json
```

#### CLEVR-Transfer (RPM, Puzzle, Referring Expressions)

```bash
python run.py --config configs/clevr-rpm.json
python run.py --config configs/clevr-puzzle.json
python run.py --config configs/clevr-refexps.json
```

### Step 2 — Evaluate with LLM judge (VQA only)

```bash
python run.py --config configs/clevr-humans.json --evaluate
python run.py --config configs/clevr-humans.json --evaluate --judge_model deepseek-chat --judge_provider deepseek
```

### Key `run.py` flags

| Flag | Description |
|------|-------------|
| `--config` | Path to config file (required) |
| `--gpus` | GPU indices for parallel VQA, e.g. `--gpus 0 1 2 3` (default: all CUDA devices) |
| `--evaluate` | Run LLM-judge evaluation on an existing results file |
| `--workers` | Parallel threads for `--evaluate` (default: 256) |
| `--judge_model` | Model ID for the LLM judge (default: `deepseek-chat`) |
| `--judge_provider` | Provider for the judge: `deepseek`, `openai`, or `vllm` (default: `deepseek`) |

### Configuration

Each experiment is controlled by a JSON config file in `configs/`. Key fields:

| Field | Description |
|-------|-------------|
| `dataset` | Dataset name — determines which runner is used |
| `vlm_model_name` | VLM oracle: `intern25_8b`, `intern_vllm`, `ovis2_4b`, `qwen2`, … |
| `code_gen_model_name` | LLM for code generation, e.g. `deepseek/deepseek-v3.2` |
| `code_gen_provider` | API provider: `openrouter`, `deepseek`, or `openai` |
| `od_model` | Object detector: `dino` or `florence` |
| `program_cache_address` | Path to program cache JSON (shared across runs) |
| `num_samples` | Number of samples to evaluate (`-1` = all) |

All config fields can be overridden from the command line by passing them as extra flags — they are forwarded to the underlying runner.

### API keys

Add your keys to `.env`:

```
OPENROUTER_API_KEY=your_openrouter_key   # recommended (deepseek/deepseek-v3.2)
DEEPSEEK_API_KEY=your_deepseek_key       # alternative
OPENAI_API_KEY=your_openai_key           # optional
```

---

## Project Structure

```
NePTune-Final/
├── run.py                          # Unified entrypoint — start here
├── runners/                        # Per-task runner scripts
│   ├── run_vqa.py                  #   VQA single-GPU worker
│   ├── run_vqa_parallel.py         #   VQA multi-GPU launcher
│   ├── run_ref_exp_fast.py         #   Referring expression grounding
│   ├── run_clevr_extensions_rpm_puzzle.py  # CLEVR-Transfer RPM / Puzzle
│   ├── run_clevr_extensions_ref.py         # CLEVR-Transfer RefExps
│   └── run_clevr_symbolic.py       #   CLEVR with pre-generated programs
├── configs/                        # Experiment config files
│   ├── clevr-humans.json
│   ├── clevr.json
│   ├── clevr-rpm.json
│   ├── clevr-puzzle.json
│   ├── clevr-refexps.json
│   ├── refadv.json
│   └── refgta.json
├── scripts/                        # Utility scripts
├── analysis/                       # Evaluation & analysis scripts
│   └── evaluate_clevr_human.py     #   Parallel LLM-judge evaluation
├── code_generator/                 # LLM code generation + caching
├── vision_agents/                  # VLM wrappers (InternVL, Ovis2, Qwen2-VL, …)
├── probabilistic_tensor/           # Neurosymbolic tensor algebra
├── local_datasets/                 # Dataset loaders
├── prompts/                        # Few-shot prompt templates
├── programs/                       # Generated program caches (JSON)
├── experiments/                    # Results output directory
├── Concepts/                       # Concepts library (vendored)
├── Jacinle/                        # Jacinle ML utilities (vendored)
└── env.yml                         # Conda environment
```

---

## Citation

If you use NePTune in your research, please cite:

```bibtex
@article{kamali2025neptune,
  title   = {NePTune: A Neuro-Pythonic Framework for Tunable Compositional Reasoning on Vision-Language},
  author  = {Kamali, Danial and Kordjamshidi, Parisa},
  journal = {arXiv preprint arXiv:2509.25757},
  year    = {2025}
}
```

---

## License

See [LICENSE](LICENSE) for details.
