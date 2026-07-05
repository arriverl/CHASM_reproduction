"""本地开源 MLLM 推理 (InternVL, Qwen2.5-VL, LLaVA 等)。"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.dataset import iter_split, load_chasm_dataset
from src.metrics.evaluation import compute_metrics, parse_binary_output, save_metrics
from src.prompts.templates import get_prompt, format_post_text


def run_local_inference(
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    mode: str = "zero_shot",
    split: str = "test",
    max_samples: int | None = None,
    output_dir: Path = Path("outputs/local"),
    include_images: bool = True,
    include_comments: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
    except ImportError as e:
        raise ImportError("pip install transformers torch accelerate") from e

    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    dataset = load_chasm_dataset()
    prompt_header = get_prompt(mode)
    y_true, y_pred = [], []

    for i, sample in enumerate(iter_split(dataset, split, max_samples=max_samples)):
        text = format_post_text(
            sample.title,
            sample.description,
            sample.comments if include_comments else [],
            include_comments=include_comments,
        )
        user_prompt = f"{prompt_header}\n\n{text}"

        inputs = processor(text=user_prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        if include_images and sample.images:
            from src.data.dataset import decode_image
            from PIL import Image
            import io

            imgs = []
            for img_data in sample.images[:3]:
                raw = decode_image(img_data)
                if raw:
                    imgs.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            if imgs:
                inputs = processor(text=user_prompt, images=imgs, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        resp = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        pred = parse_binary_output(resp.split("\n")[-1])

        y_true.append(sample.label)
        y_pred.append(pred)
        if (i + 1) % 5 == 0:
            print(f"  {i + 1} samples processed")

    metrics = compute_metrics(y_true, y_pred)
    tag = model_id.split("/")[-1]
    save_metrics(metrics, output_dir / f"{tag}_{mode}_{split}_metrics.json")
    print(f"F1={metrics.f1:.4f} P={metrics.precision:.4f} R={metrics.recall:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Local MLLM inference on CHASM")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--mode", default="zero_shot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/local")
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--no-comments", action="store_true")
    args = parser.parse_args()
    run_local_inference(
        model_id=args.model_id,
        mode=args.mode,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir),
        include_images=not args.no_images,
        include_comments=not args.no_comments,
    )


if __name__ == "__main__":
    main()
