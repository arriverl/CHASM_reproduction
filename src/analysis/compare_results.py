"""将实验结果与论文目标指标对比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.metrics.evaluation import EvalMetrics, compare_with_paper


def load_json_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compare_all_results(
    results_dir: Path = Path("outputs"),
    config_path: Path = Path("configs/default.yaml"),
) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    targets = cfg.get("paper_targets", {})

    report = {"comparisons": [], "missing": []}

    mapping = {
        "zero_shot": ("api", "zero_shot"),
        "in_context": ("api", "in_context"),
        "finetune": ("finetune", None),
        "baselines": ("baselines", None),
    }

    for result_file in results_dir.rglob("*_metrics.json"):
        data = load_json_metrics(result_file)
        metrics = EvalMetrics(**{k: data[k] for k in EvalMetrics.__dataclass_fields__ if k in data})

        matched = False
        for section, (_, _) in mapping.items():
            if section not in targets:
                continue
            for model_name, paper_vals in targets[section].items():
                if model_name.lower().replace("-", "").replace("_", "") in str(result_file).lower().replace("-", "").replace("_", ""):
                    diff = compare_with_paper(metrics, paper_vals)
                    report["comparisons"].append({
                        "file": str(result_file),
                        "model": model_name,
                        "section": section,
                        "diff": diff,
                    })
                    matched = True
        if not matched:
            report["missing"].append(str(result_file))

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare results with paper targets")
    parser.add_argument("--results-dir", default="outputs")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="outputs/paper_comparison.json")
    args = parser.parse_args()

    report = compare_all_results(Path(args.results_dir), Path(args.config))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Compared {len(report['comparisons'])} result files.")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
