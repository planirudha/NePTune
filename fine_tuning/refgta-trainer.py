from __future__ import annotations

import os
import json
import time
import math
import logging
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, Union
for noisy_module in [
    "transformers_modules",
    "transformers",
    "transformers.configuration_utils",
    "transformers.modeling_utils",
    "transformers.tokenization_utils",
]:
    logging.getLogger(noisy_module).setLevel(logging.WARNING)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from tqdm import tqdm

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_scheduler
from peft import LoraConfig, get_peft_model

# --- Custom libraries (must be on PYTHONPATH) ---
from probabilistic_tensor import ProbabilisticTensor, and_op, or_op
from vision_agents import Agent
from vision_agents.intern_vl import get_conv_template, load_image, InternVL
# =============================================================
# 1. Configuration
# =============================================================

@dataclass
class PathConfig:
    image_dir: str = os.getenv("GTA_IMAGE_DIR", "data/refcoco/refgta")
    output_dir: str = "model_weights/refcocog_lora_adapters_refgta"
    metadata_file: str = "data/training_refgta.json"

@dataclass
class ModelConfig:
    model_path: str = "OpenGVLab/InternVL2_5-1B-MPO"
    load_4bit: bool = False
    device: str = "cuda"

@dataclass
class LoraConfigWrap:
    r: int = 16
    alpha: int = 32 # A common practice is r * 2
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None

@dataclass
class TrainConfig:
    epochs: int = 1
    lr: float = 4e-5
    batch_size: int = 1  # MUST be 1 due to exec() per-sample logic.
    
    grad_accum_steps: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_interval_steps: int = 200
    report_every_steps: int = 1 # Changed from 10 to match `logging_steps 1`
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    validation_split: float = 0.1 # Using a 10% split for validation
    early_stopping_patience: int = 5

@dataclass
class ExecConfig:
    code_template: str = (
        """
def logic_executor(query, score_fn, query_fn ,all_bounding_boxes, image, history):

    score = lambda x, num_objects, type=None: score_fn(image, all_bounding_boxes, x, num_objects, draw_bbox=True, use_batch=False, pass_og_image=False, overlay_masks=False, mask=False, contrastive_mask=False, with_mask=False, type=type, history=history)
    query = lambda x, object_id=None, mask=False, mask_outside=False, draw_bbox=False, extract_object=False : query_fn(image, all_bounding_boxes, object_id, x, draw_bbox=True, pass_og_image=False, overlay_masks=False)
    objects_count = len(all_bounding_boxes)
    {code}
        """
    )

# =============================================================
# 2. Utilities
# =============================================================

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )

def freeze(module: torch.nn.Module, requires_grad: bool = False) -> None:
    for p in module.parameters():
        p.requires_grad = requires_grad

def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# =============================================================
# 3. InternVL Agent with (optional) LoRA
# =============================================================

class InternVLWithLoRA(Agent):
    def __init__(
        self,
        model_cfg: ModelConfig,
        lora_cfg: LoraConfigWrap,
        use_lora: bool = True,
        max_num_patches: int = 6,
        trust_remote_code: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        device = torch.device(model_cfg.device if torch.cuda.is_available() else "cpu")
        self.device = device
        self.max_num_patches = max_num_patches

        if device.type == "cuda":
            self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.compute_dtype = torch.float32
        logging.info(f"Initializing InternVL on {device} (dtype={self.compute_dtype}).")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.model_path,
            model_max_length=8192,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<PAD>"

        load_kwargs = dict(
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        base_model = AutoModel.from_pretrained(model_cfg.model_path, **load_kwargs)
        base_model.gradient_checkpointing_enable()

        for block in [base_model.language_model, base_model.mlp1, base_model.vision_model]:
            freeze(block, True)

        base_model.language_model.enable_input_require_grads()
        base_model.language_model.config.use_cache = False
        base_model.vision_model.gradient_checkpointing = True
        base_model.vision_model.encoder.gradient_checkpointing = True
        if use_lora:
            self._apply_lora(base_model, lora_cfg)

        self.model = base_model.to(dtype=self.compute_dtype, device=self.device)

        self.IMG_START_TOKEN = "<img>"
        self.IMG_END_TOKEN = "</img>"
        self.IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(self.IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id

        self.template, self.eos_token_id = self._init_template()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id

    def _apply_lora(self, base_model: torch.nn.Module, cfg: LoraConfigWrap) -> None:
        if cfg.target_modules is None:
            cfg.target_modules = [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "mlp.gate_proj", "mlp.down_proj", "mlp.up_proj",
            ]
        lcfg = LoraConfig(
            r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
            target_modules=cfg.target_modules, task_type="CAUSAL_LM",
        )
        base_model.language_model = get_peft_model(base_model.language_model, lcfg)
        base_model.language_model.enable_input_require_grads()
        base_model.language_model.print_trainable_parameters()
        logging.info("LoRA applied to language_model.")

        vision_lora_config = LoraConfig(
            r=cfg.r // 2, target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=cfg.alpha // 2, lora_dropout=cfg.dropout,
        )
        base_model.vision_model = get_peft_model(base_model.vision_model, vision_lora_config)
        base_model.vision_model.print_trainable_parameters()
        logging.info("LoRA applied to vision_model.")

    def _init_template(self):
        config_to_check = getattr(self.model, "config", None)
        template_name = "internvl2_5" # Set from shell script's --conv_style
        system_message = getattr(config_to_check, "system_message", None) if config_to_check else None

        if template_name:
            try:
                template = get_conv_template(template_name)
                if system_message:
                    template.system_message = system_message
                sep = getattr(template, "sep", None)
                eos_id = self.tokenizer.convert_tokens_to_ids(sep.strip()) if sep else self.tokenizer.eos_token_id
                logging.info(f"Using conversation template: {template_name}")
            except Exception as e:
                logging.warning(f"Failed to get template '{template_name}': {e}. Using tokenizer EOS.")
                template = None; eos_id = self.tokenizer.eos_token_id
        else:
            template = None; eos_id = self.tokenizer.eos_token_id

        if eos_id is None: eos_id = self.tokenizer.pad_token_id
        if eos_id is None: raise ValueError("Cannot resolve EOS token id.")
        return template, eos_id

    def _build_prompt(
        self, question: str, history: Optional[List[Tuple[str, str]]] = None,
        pixel_values: Optional[torch.Tensor] = None, num_patches_list: Optional[List[int]] = None,
    ) -> str:
        if self.template is None: raise ValueError("Conversation template not found.")
        self.template.messages = []
        if history:
            for q, a in history:
                self.template.append_message(self.template.roles[0], q)
                self.template.append_message(self.template.roles[1], a)
        if pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question
        self.template.append_message(self.template.roles[0], question)
        self.template.append_message(self.template.roles[1], None)
        prompt = self.template.get_prompt()
        if pixel_values is not None:
            if num_patches_list is None: num_patches_list = [pixel_values.shape[0]]
            for num_patches in num_patches_list:
                image_tokens = (
                    self.IMG_START_TOKEN + self.IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + self.IMG_END_TOKEN
                )
                prompt = prompt.replace("<image>", image_tokens, 1)
        return prompt

    def _score(
        self, image: Union[str, Image.Image], question: str, target_token: str = "Yes",
        temperature: float = 1.0, type: str = "all", candidates: Optional[list] = None,
    ) -> torch.Tensor:
        question = question + "\nAnswer the question using a single word or phrase."
        try:
            pixel_values = load_image(image, max_num=self.max_num_patches)
            pixel_values = pixel_values.to(self.model.dtype).to(self.device)
            num_patches_list = [pixel_values.shape[0]]
        except Exception as e:
            logging.error(f"Image load/processing error in _score: {e}")
            return torch.tensor(0.5, device=self.device, dtype=self.model.dtype, requires_grad=True)

        prompt = self._build_prompt("<image>\n" + question, pixel_values=pixel_values, num_patches_list=num_patches_list)
        model_inputs = self.tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs["attention_mask"].to(self.device)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.enable_grad():
            outputs = self.model(
                pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                image_flags=torch.tensor([1] * num_patches_list[0], dtype=torch.long, device=self.device),
            )
        logits = outputs.logits[:, -1, :]
        scaled = logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        yes_id = self.tokenizer.convert_tokens_to_ids(target_token)
        return probs[:, yes_id][0]

# =============================================================
# 4. Dataset
# =============================================================

class ReferralTrainingDataset(Dataset):
    def __init__(self, metadata_file: str, image_base_dir: str):
        self.metadata_file = metadata_file
        self.image_base_dir = image_base_dir
        logging.info(f"Loading metadata: {metadata_file}")
        try:
            with open(metadata_file, "r") as f:
                data = json.load(f)
            self.metadata = list(data.values()) if isinstance(data, dict) else data
        except Exception as e:
            logging.error(f"Failed to load metadata: {e}")
            self.metadata = []
        logging.info(f"Loaded {len(self.metadata)} samples.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.metadata[idx]
        image_rel = item.get("image_path")
        if not image_rel:
            raise ValueError(f"Missing image_path at index {idx}")
        image_path = os.path.join(self.image_base_dir, image_rel)
        expression = item.get("expression")
        boxes = item.get("candidate_bboxes")
        gt_index = item.get("gt_index")
        code_snippet = item.get("code_snippet")

        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        if not expression:
            raise ValueError(f"Missing expression at {idx}")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError(f"Invalid candidate_bboxes at {idx}")
        if not isinstance(gt_index, int) or not (0 <= gt_index < len(boxes)):
            raise ValueError(f"Invalid gt_index at {idx}")
        if not code_snippet:
            raise ValueError(f"Missing code_snippet at {idx}")

        return {
            "image_path": image_path,
            "expression": expression,
            "candidate_bboxes": boxes,
            "gt_index": gt_index,
            "code_snippet": code_snippet,
            "metadata_idx": idx,
        }

# =============================================================
# 5a. Sanity / Evaluation Helpers
# =============================================================

def forward_once(vl_model, batch, exec_cfg, criterion=None, grad: bool = False):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        image_path = batch["image_path"][0]
        expression = batch["expression"][0]
        candidate_bboxes = batch["candidate_bboxes"]
        candidate_bboxes = [list(map(float, box)) for box in candidate_bboxes]
        gt_index = batch["gt_index"][0].item()
        code_snippet = batch["code_snippet"][0]
        if isinstance(code_snippet, tuple):
            code_snippet = code_snippet[0]
        image = Image.open(image_path).convert("RGB")
        code_to_exec = exec_cfg.code_template.format(code=code_snippet.replace("\n","\n    "))
        code_to_exec = code_to_exec.encode("utf-8").decode("utf-8")
        spatial_relations = vl_model.calculate_all_spatial_metrics(image, candidate_bboxes, wrapper=ProbabilisticTensor)
        spatial_relations = {k.encode("utf-8").decode("utf-8"): v for k, v in spatial_relations.items()}
        ns = {"ProbabilisticTensor": ProbabilisticTensor, "and_op": and_op, "or_op": or_op, **spatial_relations}
        exec(code_to_exec, ns, ns)
        logic_executor = ns["logic_executor"]
        ProbabilisticTensor.start_cache()
        out = logic_executor(query=expression, score_fn=vl_model.score, query_fn=vl_model.query,
            all_bounding_boxes=candidate_bboxes, image=image, history=None,)
        ProbabilisticTensor.end_cache()
        probs = out.tensor if isinstance(out, ProbabilisticTensor) else out
        loss = None
        if criterion is not None:
            target = torch.zeros_like(probs)
            target[gt_index] = 1.0
            loss = criterion(probs, target)
    return probs, gt_index, loss

def evaluate(vl_model, loader, criterion, exec_cfg):
    ProbabilisticTensor.and_op = "mul"
    vl_model.model.eval()
    total = correct = 0
    cum_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            try:
                probs, gt_idx, loss = forward_once(vl_model, batch, exec_cfg, criterion, grad=False)
            except Exception as e:
                logging.error(f"Error during evaluation: {e}\n{traceback.format_exc()}")
                continue
            pred = probs.argmax().item()
            correct += int(pred == gt_idx)
            cum_loss += float(loss.item()) if loss is not None else 0.0
            total += 1
    acc = correct / max(1, total)
    avg_loss = cum_loss / max(1, total)
    return acc, avg_loss

def has_any_grad(model: torch.nn.Module) -> bool:
    for p in model.parameters():
        if p.requires_grad and p.grad is not None and torch.any(p.grad != 0):
            return True
    return False

# =============================================================
# 5. Training Loop
# =============================================================

def run_training(
    paths: PathConfig, model_cfg: ModelConfig, lora_cfg: LoraConfigWrap,
    train_cfg: TrainConfig, exec_cfg: ExecConfig,
) -> None:
    ProbabilisticTensor.and_op = "min"
    setup_logging()
    logging.info("=== Starting VLM LoRA fine-tuning with new hyperparameters ===")
    os.makedirs(paths.output_dir, exist_ok=True) # Ensure output dir exists

    run_name = time.strftime("%Y-%m-%d_%H-%M-%S-refgta")
    log_dir = os.path.join(paths.output_dir, "runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    logging.info(f"TensorBoard logs will be saved to: {log_dir}")

    vl_model = InternVLWithLoRA(model_cfg=model_cfg, lora_cfg=lora_cfg, use_lora=True, wrapper=ProbabilisticTensor)
    logging.info(f"Trainable params: {count_trainable_params(vl_model.model):,}")

    full_dataset = ReferralTrainingDataset(paths.metadata_file, paths.image_dir)
    valid_idx: List[int] = []
    for i in tqdm(range(len(full_dataset)), desc="Validating data"):
        try:
            _ = full_dataset[i]
            valid_idx.append(i)
        except Exception as e:
            logging.warning(f"Skipping idx {i}: {e}")
    dataset = Subset(full_dataset, valid_idx)
    logging.info(f"Using {len(dataset)} valid items.")

    if not (0 < train_cfg.validation_split < 1):
        raise ValueError("validation_split must be between 0 and 1")
    val_size = min(max(int(len(dataset) * train_cfg.validation_split), 1), len(dataset) - 1)
    train_size = len(dataset) - val_size
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    logging.info(f"Split data into {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(vl_model.model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    num_training_steps = len(train_loader) // train_cfg.grad_accum_steps * train_cfg.epochs
    num_warmup_steps = int(num_training_steps * train_cfg.warmup_ratio)
    scheduler = get_scheduler(
        name=train_cfg.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps,
    )
    criterion = nn.BCELoss()

    best_val_accuracy = -1.0
    patience_counter = 0
    total_steps = (len(train_loader) // train_cfg.grad_accum_steps) * train_cfg.epochs
    step_bar = tqdm(total=total_steps, desc="Global steps")
    global_step = 0

    for epoch in range(train_cfg.epochs):
        logging.info(f"Epoch {epoch + 1}/{train_cfg.epochs}")
        vl_model.model.train()
        epoch_loss = 0.0
        acc_counter, acc_correct = 0, 0.0
        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        optimizer.zero_grad()

        for i, batch in enumerate(batch_bar):
            try:
                try:
                    probs, gt_index, loss = forward_once(vl_model, batch, exec_cfg, criterion, grad=True)
                except Exception as e:
                    logging.error(f"Error during forward pass: {e}\n{traceback.format_exc()}")
                    continue
                if loss is None: continue
                loss = loss / train_cfg.grad_accum_steps
                loss.backward()
                pred = torch.argmax(probs, dim=0)
                acc_correct += float(pred == gt_index)
                acc_counter += 1
                epoch_loss += loss.item() * train_cfg.grad_accum_steps
            except Exception as e:
                logging.error(f"Error during training step: {e}\n{traceback.format_exc()}")
                optimizer.zero_grad()
                continue

            if ((i + 1) % train_cfg.grad_accum_steps == 0) or (i + 1 == len(train_loader)):
                ProbabilisticTensor.and_op = "min"
                torch.nn.utils.clip_grad_norm_([p for p in vl_model.model.parameters() if p.requires_grad], train_cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                step_bar.update(1)
                writer.add_scalar("Loss/train", loss.item() * train_cfg.grad_accum_steps, global_step)
                writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], global_step)

                if global_step % train_cfg.report_every_steps == 0 and acc_counter > 0:
                    current_accuracy = acc_correct / acc_counter
                    writer.add_scalar("Accuracy/train_running", current_accuracy, global_step)
                    acc_correct, acc_counter = 0.0, 0

                if paths.output_dir and global_step > 0 and global_step % train_cfg.save_interval_steps == 0:
                    logging.info(f"\n--- Running Validation at Step {global_step} ---")
                    val_acc, val_loss = evaluate(vl_model, val_loader, criterion, exec_cfg)
                    writer.add_scalar("Loss/validation", val_loss, global_step)
                    writer.add_scalar("Accuracy/validation", val_acc, global_step)
                    logging.info(f"Validation Results: Loss={val_loss:.4f}, Accuracy={val_acc:.4f}")
                    vl_model.model.train()

                    if val_acc > best_val_accuracy:
                        logging.info(f"🚀 New best validation accuracy! {val_acc:.4f} (previously {best_val_accuracy:.4f})")
                        best_val_accuracy = val_acc
                        patience_counter = 0
                        best_dir = os.path.join(paths.output_dir, "best_model")
                        os.makedirs(best_dir, exist_ok=True)
                        vl_model.model.save_pretrained(best_dir)
                        vl_model.tokenizer.save_pretrained(best_dir)
                    else:
                        patience_counter += 1
                        logging.info(f"Validation accuracy did not improve. Patience: {patience_counter}/{train_cfg.early_stopping_patience}")

                    if patience_counter >= train_cfg.early_stopping_patience:
                        logging.warning(f"🚨 Early stopping triggered after {patience_counter} checks with no improvement.")
                        writer.close()
                        return

        avg_epoch_loss = epoch_loss / max(1, len(train_dataset))
        logging.info(f"Epoch {epoch + 1} average training loss: {avg_epoch_loss:.6f}")

    writer.close()
    logging.info(f"✅ Training finished. Best model saved to {os.path.join(paths.output_dir, 'best_model')}")

# =============================================================
# 6. CLI Entrypoint
# =============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune InternVL2.5 with LoRA on RefGTA referring expressions.")
    parser.add_argument("--image_dir", type=str, default=None, help="Path to RefGTA images (overrides GTA_IMAGE_DIR env var).")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--metadata_file", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="HuggingFace model ID or local path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--save_interval_steps", type=int, default=None)
    cli = parser.parse_args()

    paths = PathConfig()
    model_cfg = ModelConfig()
    lora_cfg = LoraConfigWrap()
    train_cfg = TrainConfig()
    exec_cfg = ExecConfig()

    if cli.image_dir:        paths.image_dir = cli.image_dir
    if cli.output_dir:       paths.output_dir = cli.output_dir
    if cli.metadata_file:    paths.metadata_file = cli.metadata_file
    if cli.model_path:       model_cfg.model_path = cli.model_path
    if cli.epochs:           train_cfg.epochs = cli.epochs
    if cli.lr:               train_cfg.lr = cli.lr
    if cli.lora_r:           lora_cfg.r = cli.lora_r
    if cli.lora_alpha:       lora_cfg.alpha = cli.lora_alpha
    if cli.save_interval_steps: train_cfg.save_interval_steps = cli.save_interval_steps

    setup_logging()
    logging.info("--- CONFIGURATIONS ---")
    logging.info(json.dumps({
        "paths": asdict(paths), "model": asdict(model_cfg),
        "lora": asdict(lora_cfg), "train": asdict(train_cfg),
    }, indent=2))

    run_training(paths, model_cfg, lora_cfg, train_cfg, exec_cfg)