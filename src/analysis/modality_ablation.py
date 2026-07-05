"""模态消融实验 (论文 Section 4.3, Figure 3)。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def run_modality_ablation(
    config_path: Path = Path("configs/default.yaml"),
    output_dir: Path = Path("outputs/ablation"),
    max_samples: int | None = None,
):
    """
    复现 Figure 3：在微调 Qwen2.5-7B 上去除 comments 或 images。
    需先完成微调，或使用已发布 checkpoint。
    """
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [
        ("baseline", {"include_images": True, "include_comments": True}),
        ("w_o_comments", {"include_images": True, "include_comments": False}),
        ("w_o_images", {"include_images": False, "include_comments": True}),
    ]

    from src.models.local_inference import run_local_inference

    results = {}
    for name, kwargs in conditions:
        print(f"\n=== Ablation: {name} ===")
        metrics = run_local_inference(
            model_id="Qwen/Qwen2.5-VL-7B-Instruct",
            mode="zero_shot",
            split="test",
            max_samples=max_samples,
            output_dir=output_dir / name,
            **kwargs,
        )
        results[name] = metrics.to_dict()

    paper = cfg.get("paper_targets", {}).get("modality_ablation_qwen_ft", {})
    comparison = {}
    for name, ours in results.items():
        if name in paper:
            comparison[name] = {
                "ours_f1": ours["f1"],
                "paper_f1": paper[name]["f1"],
                "delta": ours["f1"] - paper[name]["f1"],
            }

    summary = {"results": results, "paper_comparison": comparison}
    with (output_dir / "ablation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Modality ablation (Figure 3)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    run_modality_ablation(
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
