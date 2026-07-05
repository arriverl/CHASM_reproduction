"""本地开源 MLLM 推理 (InternVL, Qwen2.5-VL, LLaVA 等)。"""

from __future__ import annotations

import argparse
import csv
import gc
import io
from pathlib import Path

from PIL import Image

from src.data.dataset import decode_image, iter_split, load_chasm_dataset, print_dataset_stats
from src.metrics.evaluation import compute_metrics, parse_binary_output, save_metrics
from src.prompts.templates import format_post_text, get_prompt


def print_gpu_status(min_free_gb: float = 8.0) -> None:
    import os
    try:
        import torch
    except ImportError:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    print(f"CUDA_VISIBLE_DEVICES={visible}  →  进程内 GPU 编号从 0 重新映射")
    if not torch.cuda.is_available():
        print("CUDA 不可用，将使用 CPU（极慢）")
        return
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        free_gb, total_gb = free / 1e9, total / 1e9
        name = torch.cuda.get_device_name(i)
        print(f"  进程内 GPU {i}: {name}, 空闲 {free_gb:.2f}GB / 总计 {total_gb:.2f}GB")
        if free_gb < min_free_gb:
            print(
                f"  ⚠ GPU {i} 空闲不足 {min_free_gb}GB！"
                " 请先 `nvidia-smi` 结束占用显存的进程，或使用 --load-4bit --no-images"
            )


def load_model_and_processor(
    model_id: str,
    load_4bit: bool = False,
    load_8bit: bool = False,
):
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except ImportError:
        from transformers import AutoModelForVision2Seq as VLModel

    processor_kwargs = {
        # 限制视觉 token 数量，显著降低显存（Qwen2.5-VL 官方推荐）
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 512 * 28 * 28,
    }
    processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)

    model_kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }

    if load_4bit or load_8bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except ImportError as e:
            raise ImportError("量化加载需要: pip install bitsandbytes") from e
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    print(f"Loading {model_id} (4bit={load_4bit}, 8bit={load_8bit})...")
    model = VLModel.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return model, processor


def load_pil_images(sample, max_images: int, max_side: int = 768) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    for img_data in (sample.images or [])[:max_images]:
        raw = decode_image(img_data)
        if not raw:
            continue
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        imgs.append(img)
    return imgs


def build_messages(user_prompt: str, pil_images: list[Image.Image]) -> list[dict]:
    if pil_images:
        content = [{"type": "image", "image": img} for img in pil_images]
        content.append({"type": "text", "text": user_prompt})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": user_prompt}]


def predict_one(model, processor, user_prompt: str, pil_images: list[Image.Image]) -> tuple[int, str]:
    import torch

    messages = build_messages(user_prompt, pil_images)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    if pil_images:
        inputs = processor(text=[text], images=pil_images, padding=True, return_tensors="pt")
    else:
        inputs = processor(text=[text], return_tensors="pt")

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
        )

    # 只解码新生成的 token（逐条 slice，避免 padding 导致空输出）
    new_ids = output_ids[0, input_len:]
    tokenizer = getattr(processor, "tokenizer", processor)
    resp = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    pred = parse_binary_output(resp)
    return pred, resp


def run_local_inference(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    mode: str = "zero_shot",
    split: str = "test",
    max_samples: int | None = None,
    output_dir: Path = Path("outputs/local"),
    include_images: bool = True,
    include_comments: bool = True,
    max_images: int = 1,
    load_4bit: bool = False,
    load_8bit: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print_gpu_status()
    model, processor = load_model_and_processor(model_id, load_4bit=load_4bit, load_8bit=load_8bit)

    dataset = load_chasm_dataset()
    print_dataset_stats(dataset)

    prompt_header = get_prompt(mode)
    y_true, y_pred = [], []
    records = []

    csv_path = output_dir / f"predictions_{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["id", "label", "pred", "raw_response"])

        for i, sample in enumerate(iter_split(dataset, split, max_samples=max_samples)):
            text = format_post_text(
                sample.title,
                sample.description,
                sample.comments if include_comments else [],
                include_comments=include_comments,
            )
            user_prompt = f"{prompt_header}\n\n[Post]\n{text}"

            pil_images = load_pil_images(sample, max_images=max_images) if include_images else []

            try:
                pred, raw_resp = predict_one(model, processor, user_prompt, pil_images)
            except Exception as e:
                if "out of memory" in str(e).lower() and pil_images:
                    import torch
                    print(f"  [{i}] OOM with images, retry text-only...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    pred, raw_resp = predict_one(model, processor, user_prompt, [])
                else:
                    raise

            y_true.append(sample.label)
            y_pred.append(pred)
            writer.writerow([sample.id, sample.label, pred, raw_resp])
            records.append({"id": sample.id, "label": sample.label, "pred": pred, "raw": raw_resp})

            if i < 3:
                print(f"  [debug] id={sample.id} label={sample.label} pred={pred} raw={raw_resp!r}")

            if (i + 1) % 5 == 0:
                pos_pred = sum(y_pred)
                print(f"  {i + 1} samples | pred_1={pos_pred} pred_0={len(y_pred)-pos_pred}")
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    pred_pos = sum(y_pred)
    pred_neg = len(y_pred) - pred_pos
    true_pos = sum(y_true)
    print(f"Predictions: pred_1={pred_pos}, pred_0={pred_neg} | gold_1={true_pos}, gold_0={len(y_true)-true_pos}")

    metrics = compute_metrics(y_true, y_pred)
    tag = model_id.split("/")[-1]
    save_metrics(metrics, output_dir / f"{tag}_{mode}_{split}_metrics.json")
    print(f"F1={metrics.f1:.4f} P={metrics.precision:.4f} R={metrics.recall:.4f} (paper ZS target≈0.421)")
    print(f"Saved predictions: {csv_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Local MLLM inference on CHASM")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--mode", default="zero_shot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/local")
    parser.add_argument("--no-images", action="store_true", help="纯文本推理，省显存")
    parser.add_argument("--no-comments", action="store_true")
    parser.add_argument("--max-images", type=int, default=1, help="每条样本最多使用几张图（默认1）")
    parser.add_argument("--load-4bit", action="store_true", help="4bit 量化加载，约需 6-8GB 显存")
    parser.add_argument("--load-8bit", action="store_true", help="8bit 量化加载")
    args = parser.parse_args()

    if args.load_4bit and args.load_8bit:
        parser.error("不能同时指定 --load-4bit 和 --load-8bit")

    run_local_inference(
        model_id=args.model_id,
        mode=args.mode,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir),
        include_images=not args.no_images,
        include_comments=not args.no_comments,
        max_images=args.max_images,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
    )


if __name__ == "__main__":
    main()
