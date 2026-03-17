# Atomic Concept Benchmark

This module evaluates the **atomic concept understanding** of Vision-Language Models (VLMs) — their ability to recognize fundamental visual properties and relations in isolation, independent of compositional reasoning.

Atomic concepts include:
- **Attributes** — color, material, size, shape (e.g., *"Is the object red?"*)
- **Relations** — spatial relationships between objects (e.g., *"Is the cube to the left of the sphere?"*)
- **Classes** — object category identity (e.g., *"Is this object a car?"*)

Two benchmarks are provided: one built from **GQA scene graphs** (real-world images) and one from **CLEVR** (synthetic images with ground-truth scenes).

---

## Structure

```
atomic_benchmark/
├── src/
│   ├── gqa_based_benchark_data_generator.py   # Generate benchmark questions from GQA scene graphs
│   ├── run_gqa_benchmark.py                   # Run a VLM on the generated GQA benchmark
│   ├── atomic_concept_evaluation_val_clevr.py # Evaluate a VLM on CLEVR validation scenes
│   ├── atomic_concept_metrics.py              # Compute precision/recall/F1 from CLEVR results
│   └── gqa_based_bench_results.ipynb          # Analysis notebook
├── questions/                                  # Output directory for generated GQA benchmark files
└── results/                                    # Output directory for VLM evaluation results
```

---

## GQA Benchmark

### Step 1 — Generate benchmark questions

Generates yes/no questions about object classes, attributes, and relations from GQA scene graph annotations.

```bash
python src/gqa_based_benchark_data_generator.py
```

**Requires:**
- GQA `train_sceneGraphs.json` — set path in `SCENE_GRAPH_FILE` at the bottom of the script or via the `GQA_DIR` environment variable.

**Outputs** (written to `questions/`):
- `gqa_benchmark_templated_{split}.json` — benchmark questions for train/val/test splits
- `gqa_benchmark_templated_stats_{split}.json` — per-concept statistics

Each question is a yes/no item with the format:
```json
{
  "image_id": "...",
  "question": "Is the object in the red bounding box a car?",
  "answer": "Yes",
  "type": "concept",
  "bbox": [x, y, w, h]
}
```

### Step 2 — Run a VLM on the benchmark

```bash
python src/run_gqa_benchmark.py \
    --model intern25_8b \
    --bench questions/gqa_benchmark_templated_test.json \
    --image-root data/gqa/images \
    --out results/gqa_intern25_8b.json
```

| Flag | Description |
|------|-------------|
| `--model` | VLM model name (passed to `get_model()`) |
| `--bench` | Path to benchmark JSON |
| `--image-root` | Directory containing GQA images |
| `--out` | Output results file |
| `--relation_only` | Evaluate only relation questions |
| `--with_mask` | Use the VLM's with_mask scoring mode |

---

## CLEVR Benchmark

CLEVR provides ground-truth scene graphs, enabling exact evaluation of atomic concept scoring without annotation noise.

### Step 1 — Score concepts on CLEVR validation images

Runs the VLM over all CLEVR validation images, scoring each object against every atomic concept (color, shape, size, material, and pairwise spatial relations).

```bash
python src/atomic_concept_evaluation_val_clevr.py \
    --vlm_model_name intern25_8b \
    --output vlm_atomic_results.jsonl
```

Set `CLEVR_DIR` in your environment (or `.env`) to point to your CLEVR dataset root. The script reads `$CLEVR_DIR/val/scenes.json` for ground-truth.

**Output:** A JSONL file where each line is one image's per-concept VLM scores and ground-truth labels:
```json
{
  "index": 0,
  "image_filename": "CLEVR_val_000000.png",
  "concepts": {
    "red":  {"scores": [0.92, 0.03, 0.41], "gt": [1, 0, 0]},
    "left": {"scores": [[0.1, 0.8], [0.7, 0.1]], "gt": [[0, 1], [1, 0]]}
  }
}
```

### Step 2 — Compute metrics

```bash
python src/atomic_concept_metrics.py \
    --input vlm_atomic_results.jsonl \
    --threshold 0.5
```

Outputs per-concept and aggregated precision, recall, and F1:

```
Concept         Precision   Recall      F1    Support
red               0.850      0.920   0.885       100
cube              0.910      0.880   0.895       120
left              0.780      0.840   0.809       200
...
Macro avg         0.875      0.890   0.882      5200
Micro avg         0.862      0.901   0.881      5200
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GQA_DIR` | GQA dataset root (contains `scene/`, `images/`) |
| `CLEVR_DIR` | CLEVR dataset root (contains `val/scenes.json`, `val/images/`) |
