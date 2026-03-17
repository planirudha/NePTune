# Fine-tuning

This module fine-tunes a VLM (InternVL2.5) with LoRA adapters using the NePTune neurosymbolic supervision signal. Instead of using human-annotated answer labels directly, the model is trained by executing LLM-generated logic programs that score candidate bounding boxes — the gradient flows back through the VLM's scoring function.

The pipeline has two stages:

1. **Generate training metadata** (`referral-trainer-data.py`) — run the dataset through OD + code generator to produce a JSON file with candidate boxes and GT indices.
2. **Train the model** (`refgta-trainer.py`) — fine-tune InternVL2.5 with LoRA using the metadata.

Currently supports **RefGTA** (referring expression grounding on GTA images).

---

## Requirements

Install fine-tuning dependencies in addition to the main `requirements.txt`:

```bash
uv pip install peft tensorboard bitsandbytes
```

Download the required model weights:

```bash
# SAM2 checkpoint
mkdir -p sam_res/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt \
     -O sam_res/checkpoints/sam2.1_hiera_large.pt

# Depth Anything V2
mkdir -p depth_anything_v2
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth \
     -O depth_anything_v2/depth_anything_v2_vits.pth
```

---

## Stage 1 — Generate training metadata

```bash
python fine_tuning/referral-trainer-data.py \
    --dataset refgta \
    --num_samples 1000 \
    --prompt_file prompts/refexp.txt
```

This runs Grounding DINO on each image to get candidate boxes, calls the LLM code generator to produce a NePTune program per expression, and matches the ground-truth box by IoU. Results are saved to `data/training_refgta.json`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `refgta` | Dataset name (passed to `get_dataset`) |
| `--num_samples` | `1000` | Number of samples to process |
| `--code_gen_model` | `deepseek/deepseek-v3.2` | LLM model for code generation |
| `--code_gen_provider` | `openrouter` | API provider |
| `--od_model` | `dino` | Object detector |
| `--od_threshold` | `0.25` | OD confidence threshold |
| `--iou_threshold` | `0.5` | Minimum IoU to accept a GT match |
| `--force_regen` | `False` | Regenerate code even if cached |
| `--output` | `data/training_{dataset}.json` | Output metadata file |
| `--prompt_file` | `prompts/refexp.txt` | Code generation prompt template |

---

## Stage 2 — Fine-tune with LoRA

```bash
python fine_tuning/refgta-trainer.py \
    --image_dir /path/to/refgta/images \
    --epochs 3 \
    --lr 4e-5 \
    --lora_r 16
```

Alternatively, set `GTA_IMAGE_DIR` in your `.env` file and omit `--image_dir`.

The trainer:
- Loads InternVL2.5 from HuggingFace and applies LoRA to language + vision encoders
- Executes the NePTune logic program per sample (using `exec()`) to get soft box scores
- Trains with BCE loss against a one-hot GT target
- Validates every `--save_interval_steps` steps with early stopping
- Saves the best checkpoint to `model_weights/refcocog_lora_adapters_refgta/best_model/`
- Logs to TensorBoard under `model_weights/refcocog_lora_adapters_refgta/runs/`

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | `$GTA_IMAGE_DIR` | Path to RefGTA images |
| `--output_dir` | `model_weights/refcocog_lora_adapters_refgta` | Checkpoint output directory |
| `--metadata_file` | `data/training_refgta.json` | Metadata from Stage 1 |
| `--model_path` | `OpenGVLab/InternVL2_5-1B-MPO` | Base VLM model |
| `--epochs` | `1` | Number of training epochs |
| `--lr` | `4e-5` | Learning rate |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--save_interval_steps` | `200` | Validation + checkpoint interval |

**Note:** Training requires a single GPU. Use `CUDA_VISIBLE_DEVICES=0` if you have multiple GPUs to avoid device mismatch errors.

### View training logs

```bash
tensorboard --logdir model_weights/refcocog_lora_adapters_refgta/runs/
```

---

## Stage 3 — Merge LoRA weights and evaluate

After training, merge the LoRA adapters into the base model following the [InternVL fine-tuning instructions](https://internvl.readthedocs.io/en/latest/internvl2.5/finetune.html).

Once merged, use `intern_tuned` as part of the VLM model name for evaluation. The model name must contain `intern_tuned` and be a valid path to the merged model directory:

```bash
python run.py --evaluate \
    --config configs/your_config.json \
    --vlm_model_name /path/to/intern_tuned_merged_model
```

For example, if you merged the weights into `model_weights/intern_tuned_refgta/`, use that path as `--vlm_model_name`.
