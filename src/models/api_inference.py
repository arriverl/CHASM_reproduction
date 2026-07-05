"""API 推理：Zero-shot / In-Context Learning (论文 Table 2)。"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
from pathlib import Path

from openai import OpenAI

from src.data.dataset import iter_split, load_chasm_dataset, select_few_shot_examples
from src.metrics.evaluation import compute_metrics, parse_binary_output, save_metrics
from src.prompts.templates import (
    SYSTEM_GUIDELINES_EN,
    build_few_shot_prompt,
    format_post_text,
    get_prompt,
)


def image_to_data_url(image_bytes: bytes, mime: str = "image/jpeg") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def call_openai(messages, model: str, api_key: str | None = None, base_url: str | None = None) -> str:
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=16,
    )
    return response.choices[0].message.content.strip()


def build_messages(
    post_text: str,
    mode: str,
    pos_example: str | None = None,
    neg_example: str | None = None,
    image_data_urls: list[str] | None = None,
) -> list[dict]:
    if mode in {"in_context", "detailed_in_context"}:
        prompt = build_few_shot_prompt(
            get_prompt(mode),
            pos_example or "",
            neg_example or "",
            post_text,
        )
    else:
        prompt = f"{get_prompt(mode)}\n\n{post_text}"

    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_data_urls:
        for url in image_data_urls[:5]:
            content.insert(0, {"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": SYSTEM_GUIDELINES_EN},
        {"role": "user", "content": content},
    ]


def run_api_inference(
    model: str = "gpt-4o-2024-08-06",
    mode: str = "zero_shot",
    split: str = "test",
    sample_ratio: float = 1.0,
    max_samples: int | None = None,
    output_dir: Path = Path("outputs/api"),
    seed: int = 42,
    include_images: bool = True,
    include_comments: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_chasm_dataset()

    pos_ex, neg_ex = None, None
    if mode in {"in_context", "detailed_in_context"}:
        pos_ex, neg_ex = select_few_shot_examples(dataset, split="train", seed=seed)

    samples = list(iter_split(dataset, split, max_samples=max_samples, shuffle=True, seed=seed))
    if sample_ratio < 1.0:
        k = max(1, int(len(samples) * sample_ratio))
        samples = random.Random(seed).sample(samples, k)

    y_true, y_pred = [], []
    csv_path = output_dir / f"{model.replace('/', '_')}_{mode}_{split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label", "pred", "text_preview"])

        for i, sample in enumerate(samples):
            text = format_post_text(
                sample.title,
                sample.description,
                sample.comments if include_comments else [],
                include_comments=include_comments,
            )
            image_urls = []
            if include_images and sample.images:
                from src.data.dataset import decode_image

                for img in sample.images[:3]:
                    raw = decode_image(img)
                    if raw:
                        image_urls.append(image_to_data_url(raw))

            messages = build_messages(
                text,
                mode=mode,
                pos_example=pos_ex.text if pos_ex else None,
                neg_example=neg_ex.text if neg_ex else None,
                image_data_urls=image_urls or None,
            )
            try:
                resp = call_openai(messages, model=model)
                pred = parse_binary_output(resp)
            except Exception as e:
                print(f"[{i}] API error: {e}")
                pred = 0

            y_true.append(sample.label)
            y_pred.append(pred)
            writer.writerow([sample.id, sample.label, pred, text[:80]])
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(samples)}")

    metrics = compute_metrics(y_true, y_pred)
    save_metrics(metrics, output_dir / f"{model.replace('/', '_')}_{mode}_{split}_metrics.json")
    print(f"F1={metrics.f1:.4f} P={metrics.precision:.4f} R={metrics.recall:.4f} AUC={metrics.auc:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="API inference for CHASM (Table 2)")
    parser.add_argument("--model", default="gpt-4o-2024-08-06")
    parser.add_argument("--mode", default="zero_shot", choices=[
        "zero_shot", "in_context", "detailed_zero_shot", "detailed_in_context"
    ])
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-ratio", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="outputs/api")
    args = parser.parse_args()
    run_api_inference(
        model=args.model,
        mode=args.mode,
        split=args.split,
        sample_ratio=args.sample_ratio,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
