"""错误分析 (论文 Section 5.1, Table 4)。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ERROR_TYPES = [
    "insufficient_evidence",
    "missing_clue_comment",
    "missing_clue_image",
    "language_style",
    "post_structure",
    "other",
]


def categorize_error_manual() -> dict:
    """论文 Table 4 中的参考分布（用于对照）。"""
    return {
        "GPT4o(ZS)": {
            "insufficient_evidence": 22,
            "missing_clue": 10,
            "language_style": 8,
            "post_structure": 3,
            "other": 3,
        },
        "DeepSeek-V3(ICL)": {
            "insufficient_evidence": 16,
            "missing_clue": 15,
            "language_style": 9,
            "post_structure": 2,
            "other": 2,
        },
        "Qwen2.5(ZS)": {
            "insufficient_evidence": 38,
            "missing_clue": 32,
            "language_style": 16,
            "post_structure": 6,
            "other": 6,
        },
        "Qwen2.5(FT)": {
            "insufficient_evidence": 6,
            "missing_clue": 14,
            "language_style": 5,
            "post_structure": 6,
            "other": 3,
        },
    }


def analyze_predictions_csv(csv_path: Path) -> dict:
    """从预测 CSV 统计错误样本（需后续人工标注错误类型）。"""
    errors = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row["label"])
            pred = int(row["pred"])
            if label != pred:
                errors.append({
                    "id": row["id"],
                    "label": label,
                    "pred": pred,
                    "error_type": "pending_manual_review",
                })
    return {
        "total_errors": len(errors),
        "errors": errors[:100],
    }


def compare_error_distribution(
    manual_counts: dict[str, int],
    paper_counts: dict[str, int],
) -> dict:
    total_manual = sum(manual_counts.values()) or 1
    total_paper = sum(paper_counts.values()) or 1
    comparison = {}
    for et in set(manual_counts) | set(paper_counts):
        m = manual_counts.get(et, 0)
        p = paper_counts.get(et, 0)
        comparison[et] = {
            "ours_count": m,
            "ours_pct": m / total_manual,
            "paper_count": p,
            "paper_pct": p / total_paper,
        }
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Error analysis (Table 4)")
    parser.add_argument("--predictions", type=str, help="Path to predictions CSV")
    parser.add_argument("--output-dir", default="outputs/error_analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_ref = categorize_error_manual()
    with (output_dir / "paper_reference_table4.json").open("w", encoding="utf-8") as f:
        json.dump(paper_ref, f, ensure_ascii=False, indent=2)

    if args.predictions:
        result = analyze_predictions_csv(Path(args.predictions))
        with (output_dir / "prediction_errors.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Found {result['total_errors']} misclassified samples for manual review.")
    else:
        print("Paper reference error distribution saved. Provide --predictions for auto extraction.")


if __name__ == "__main__":
    main()
