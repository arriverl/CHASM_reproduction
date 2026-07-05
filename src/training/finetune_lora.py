"""LoRA 微调脚本 (论文 Appendix B, Table 3)。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data.dataset import iter_split, load_chasm_dataset
from src.metrics.evaluation import compute_metrics, save_metrics


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_training_data(dataset, split: str = "train", max_samples: int | None = None):
    records = []
    for sample in iter_split(dataset, split, max_samples=max_samples):
        records.append({
            "id": sample.id,
            "text": sample.text,
            "label": sample.label,
            "images": sample.images,
        })
    return records


def run_finetune(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    config_path: Path = Path("configs/default.yaml"),
    output_dir: Path = Path("outputs/finetune"),
    max_samples: int | None = None,
    folds: int = 5,
    dry_run: bool = False,
    use_demo: bool = False,
):
    """
    使用 ms-swift / transformers + peft 进行 LoRA 微调。
    dry_run=True 时仅验证数据与配置，不加载大模型。
    """
    cfg = load_config(config_path)
    ft_cfg = cfg["finetune"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_demo:
        from src.data.dataset import create_demo_dataset
        dataset = create_demo_dataset()
    else:
        try:
            dataset = load_chasm_dataset()
        except Exception as e:
            print(f"Warning: failed to load HF dataset ({e}). Falling back to demo data.")
            from src.data.dataset import create_demo_dataset
            dataset = create_demo_dataset()
    records = prepare_training_data(dataset, "train", max_samples)
    labels = [r["label"] for r in records]

    print(f"Prepared {len(records)} training samples for {model_id}")
    print(f"LoRA: rank={ft_cfg['lora_rank']}, alpha={ft_cfg['lora_alpha']}")
    print(f"Batch: {ft_cfg['per_device_train_batch_size']} x accum {ft_cfg['gradient_accumulation_steps']}")
    print(f"LR={ft_cfg['learning_rate']}, epochs={ft_cfg['num_epochs']}, max_len={ft_cfg['max_seq_length']}")

    if dry_run:
        skf = StratifiedKFold(n_splits=min(folds, len(set(labels))), shuffle=True, random_state=42)
        fold_info = []
        for i, (train_idx, val_idx) in enumerate(skf.split(records, labels)):
            fold_info.append({"fold": i, "train": len(train_idx), "val": len(val_idx)})
        summary = {"model_id": model_id, "folds": fold_info, "status": "dry_run_ok"}
        with (output_dir / "finetune_plan.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    try:
        import torch
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import AutoProcessor, AutoModelForVision2Seq, TrainingArguments, Trainer
    except ImportError as e:
        raise ImportError(
            "Fine-tuning requires: pip install transformers peft accelerate bitsandbytes"
        ) from e

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if ft_cfg["bf16"] else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=ft_cfg["lora_rank"],
        lora_alpha=ft_cfg["lora_alpha"],
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # 实际训练循环需结合多模态 collator；此处导出训练参数供 ms-swift CLI 使用
    train_args = {
        "model_id": model_id,
        "per_device_train_batch_size": ft_cfg["per_device_train_batch_size"],
        "gradient_accumulation_steps": ft_cfg["gradient_accumulation_steps"],
        "learning_rate": ft_cfg["learning_rate"],
        "num_train_epochs": ft_cfg["num_epochs"],
        "weight_decay": ft_cfg["weight_decay"],
        "lr_scheduler_type": ft_cfg["lr_scheduler"],
        "warmup_ratio": ft_cfg["warmup_ratio"],
        "bf16": ft_cfg["bf16"],
        "max_length": ft_cfg["max_seq_length"],
        "lora_rank": ft_cfg["lora_rank"],
        "lora_alpha": ft_cfg["lora_alpha"],
        "cross_validation_folds": folds,
        "num_samples": len(records),
    }
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as f:
        json.dump(train_args, f, ensure_ascii=False, indent=2)

    print("Model loaded. Use scripts/run_finetune.sh with ms-swift for full 5-fold CV training.")
    return train_args


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune MLLM on CHASM (Table 3)")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    run_finetune(
        model_id=args.model_id,
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
        folds=args.folds,
        dry_run=args.dry_run,
        use_demo=args.demo,
    )


if __name__ == "__main__":
    main()
