"""轻量级基线模型 (论文 Appendix C, Table 7)。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.dataset import iter_split, load_chasm_dataset
from src.metrics.evaluation import compute_metrics, save_metrics


def collect_texts_labels(dataset, split: str, max_samples: int | None = None):
    texts, labels = [], []
    for sample in iter_split(dataset, split, max_samples=max_samples):
        texts.append(sample.text)
        labels.append(sample.label)
    return texts, labels


def train_tfidf_lr(train_texts, train_labels):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(train_texts, train_labels)
    return pipe


def train_tfidf_svm(train_texts, train_labels):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)),
        ("clf", LinearSVC(class_weight="balanced", random_state=42)),
    ])
    pipe.fit(train_texts, train_labels)
    return pipe


def evaluate_pipeline(model, test_texts, test_labels):
    preds = model.predict(test_texts)
    try:
        scores = model.decision_function(test_texts)
        if scores.ndim > 1:
            scores = scores[:, 1]
    except Exception:
        scores = preds.astype(float)
    return compute_metrics(test_labels, preds, scores)


def run_baselines(
    max_samples: int | None = None,
    output_dir: Path = Path("outputs/baselines"),
    use_demo: bool = False,
):
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

    train_texts, train_labels = collect_texts_labels(dataset, "train", max_samples)
    test_texts, test_labels = collect_texts_labels(dataset, "test", max_samples)

    results = {}
    for name, trainer in [("TF-IDF_LR", train_tfidf_lr), ("TF-IDF_SVM", train_tfidf_svm)]:
        print(f"Training {name}...")
        model = trainer(train_texts, train_labels)
        metrics = evaluate_pipeline(model, test_texts, test_labels)
        results[name] = metrics.to_dict()
        save_metrics(metrics, output_dir / f"{name.lower()}.json")
        joblib.dump(model, output_dir / f"{name.lower()}.joblib")
        print(f"  F1={metrics.f1:.4f} P={metrics.precision:.4f} R={metrics.recall:.4f}")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Run TF-IDF baselines (Table 7)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/baselines")
    parser.add_argument("--demo", action="store_true", help="Use offline demo dataset")
    args = parser.parse_args()
    run_baselines(max_samples=args.max_samples, output_dir=Path(args.output_dir), use_demo=args.demo)


if __name__ == "__main__":
    main()
