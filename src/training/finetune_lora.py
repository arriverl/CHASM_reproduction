"""LoRA / QLoRA 微调 (论文 Appendix B, Table 3)。"""

from __future__ import annotations

import argparse
import gc
import io
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.dataset import decode_image, iter_split, load_chasm_dataset, print_dataset_stats
from src.metrics.evaluation import compute_metrics, parse_binary_output, save_metrics
from src.prompts.templates import get_prompt


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_records(dataset, split: str, max_samples: int | None = None):
    records = []
    for sample in iter_split(dataset, split, max_samples=max_samples, shuffle=True, seed=42):
        records.append({
            "id": sample.id,
            "text": sample.text,
            "label": sample.label,
            "images": sample.images,
        })
    return records


class CHASMSFTDataset(Dataset):
    """将 CHASM 样本转为 SFT 格式：user=帖子+提示词，assistant=0/1。"""

    def __init__(
        self,
        records: list[dict],
        processor,
        prompt_header: str,
        max_length: int = 2048,
        max_images: int = 0,
    ):
        self.records = records
        self.processor = processor
        self.prompt_header = prompt_header
        self.max_length = max_length
        self.max_images = max_images
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_data):
        raw = decode_image(image_data)
        if not raw:
            return None
        from PIL import Image
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        user_text = f"{self.prompt_header}\n\n[Post]\n{rec['text']}"
        answer = str(rec["label"])

        pil_images = []
        if self.max_images > 0 and rec.get("images"):
            for img_data in rec["images"][: self.max_images]:
                img = self._load_image(img_data)
                if img is not None:
                    pil_images.append(img)

        if pil_images:
            content = [{"type": "image", "image": img} for img in pil_images]
            content.append({"type": "text", "text": user_text})
            messages = [{"role": "user", "content": content}]
        else:
            messages = [{"role": "user", "content": user_text}]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt + answer

        if pil_images:
            model_inputs = self.processor(
                text=[full_text],
                images=pil_images,
                padding=False,
                return_tensors="pt",
            )
            prompt_inputs = self.processor(
                text=[prompt],
                images=pil_images,
                padding=False,
                return_tensors="pt",
            )
        else:
            model_inputs = self.tokenizer(full_text, return_tensors="pt", truncation=False)
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False)

        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs.get("attention_mask", torch.ones_like(input_ids))[0]
        prompt_len = prompt_inputs["input_ids"].shape[1]

        if input_ids.shape[0] > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            prompt_len = min(prompt_len, self.max_length - 1)

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key in ("pixel_values", "image_grid_thw"):
            if key in model_inputs:
                item[key] = model_inputs[key][0] if model_inputs[key].dim() > 1 else model_inputs[key]
        return item


def collate_fn(batch: list[dict]) -> dict:
    """batch_size=1 为主；此处仍支持 pad。"""
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids, attention_mask, labels = [], [], []
    out = {"input_ids": None, "attention_mask": None, "labels": None}

    pad_id = 0
    for x in batch:
        seq_len = x["input_ids"].shape[0]
        pad_len = max_len - seq_len
        input_ids.append(
            torch.cat([x["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([x["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels.append(
            torch.cat([x["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    out["input_ids"] = torch.stack(input_ids)
    out["attention_mask"] = torch.stack(attention_mask)
    out["labels"] = torch.stack(labels)

    if "pixel_values" in batch[0]:
        out["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    if "image_grid_thw" in batch[0]:
        out["image_grid_thw"] = torch.stack([x["image_grid_thw"] for x in batch])
    return out


def load_train_model(model_id: str, ft_cfg: dict, load_4bit: bool = True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as VLModel

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28,
    )

    model_kwargs = {"device_map": "auto", "low_cpu_mem_usage": True}
    if load_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = VLModel.from_pretrained(model_id, **model_kwargs)
    if load_4bit:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=ft_cfg["lora_rank"],
        lora_alpha=ft_cfg["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


@torch.inference_mode()
def evaluate_adapter(
    model,
    processor,
    records: list[dict],
    prompt_header: str,
    max_images: int = 0,
) -> dict:
    model.eval()
    y_true, y_pred = [], []
    tokenizer = getattr(processor, "tokenizer", processor)

    for rec in tqdm(records, desc="eval", leave=False):
        user_text = f"{prompt_header}\n\n[Post]\n{rec['text']}"
        messages = [{"role": "user", "content": user_text}]
        pil_images = []
        if max_images > 0 and rec.get("images"):
            for img_data in rec["images"][:max_images]:
                raw = decode_image(img_data)
                if raw:
                    from PIL import Image
                    pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        if pil_images:
            content = [{"type": "image", "image": img} for img in pil_images]
            content.append({"type": "text", "text": user_text})
            messages = [{"role": "user", "content": content}]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if pil_images:
            inputs = processor(text=[prompt], images=pil_images, return_tensors="pt")
        else:
            inputs = tokenizer([prompt], return_tensors="pt")

        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        resp = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True).strip()
        y_true.append(rec["label"])
        y_pred.append(parse_binary_output(resp))

    metrics = compute_metrics(y_true, y_pred)
    model.train()
    return metrics.to_dict()


def run_finetune(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    config_path: Path = Path("configs/default.yaml"),
    output_dir: Path = Path("outputs/finetune"),
    max_samples: int | None = None,
    max_images: int = 0,
    load_4bit: bool = True,
    eval_after_train: bool = True,
    dry_run: bool = False,
):
    cfg = load_config(config_path)
    ft_cfg = cfg["finetune"]
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "lora_adapter"

    dataset = load_chasm_dataset()
    print_dataset_stats(dataset)

    train_records = prepare_records(dataset, "train", max_samples)
    val_records = prepare_records(dataset, "validation", max_samples)
    test_records = prepare_records(dataset, "test", max_samples)

    print(f"Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    print(f"LoRA r={ft_cfg['lora_rank']} alpha={ft_cfg['lora_alpha']}, "
          f"bs={ft_cfg['per_device_train_batch_size']}x{ft_cfg['gradient_accumulation_steps']}, "
          f"epochs={ft_cfg['num_epochs']}, max_images={max_images}, 4bit={load_4bit}")

    if dry_run:
        summary = {"status": "dry_run_ok", "train": len(train_records), "max_images": max_images}
        with (output_dir / "finetune_plan.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    prompt_header = get_prompt("zero_shot")
    model, processor = load_train_model(model_id, ft_cfg, load_4bit=load_4bit)

    train_ds = CHASMSFTDataset(
        train_records, processor, prompt_header,
        max_length=min(ft_cfg["max_seq_length"], 2048),
        max_images=max_images,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=ft_cfg["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ft_cfg["learning_rate"],
        betas=(ft_cfg["adam_beta1"], ft_cfg["adam_beta2"]),
        eps=ft_cfg["adam_epsilon"],
        weight_decay=ft_cfg["weight_decay"],
    )
    grad_accum = ft_cfg["gradient_accumulation_steps"]
    total_steps = (len(train_loader) // grad_accum + 1) * ft_cfg["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    model.train()
    global_step = 0
    for epoch in range(ft_cfg["num_epochs"]):
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{ft_cfg['num_epochs']}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1} avg_loss={avg_loss:.4f}")

        if val_records:
            val_metrics = evaluate_adapter(model, processor, val_records, prompt_header, max_images)
            print(f"  Val F1={val_metrics['f1']:.4f} P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f}")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    results = {"adapter_dir": str(adapter_dir), "epochs": ft_cfg["num_epochs"]}
    if eval_after_train and test_records:
        print("Evaluating on test split...")
        from src.metrics.evaluation import EvalMetrics
        test_metrics = evaluate_adapter(model, processor, test_records, prompt_header, max_images)
        results["test"] = test_metrics
        save_metrics(EvalMetrics(**test_metrics), output_dir / "test_metrics.json")
        print(f"Test F1={test_metrics['f1']:.4f} (paper FT target≈0.756)")

    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune on CHASM (Table 3)")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument("--max-samples", type=int, default=None, help="调试用，限制训练样本数")
    parser.add_argument("--max-images", type=int, default=0, help="0=纯文本微调(省显存); 1=多模态")
    parser.add_argument("--no-4bit", action="store_true", help="全精度 bf16（需更大显存）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    run_finetune(
        model_id=args.model_id,
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        max_images=args.max_images,
        load_4bit=not args.no_4bit,
        eval_after_train=not args.no_eval,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
