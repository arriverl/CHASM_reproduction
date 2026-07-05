"""评估指标：Precision, Recall, F1, AUC (论文 Section 4.1)。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[float] | None = None,
) -> EvalMetrics:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    if y_score is not None and len(set(y_true)) > 1:
        auc_val = float(roc_auc_score(y_true, y_score))
    else:
        auc_val = 0.5

    return EvalMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc=auc_val,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def save_metrics(metrics: EvalMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)


def compare_with_paper(
    metrics: EvalMetrics,
    paper_targets: dict,
    tolerance: float = 0.05,
) -> dict:
    """与论文目标指标对比，返回各维度偏差。"""
    diffs = {}
    for key in ("precision", "recall", "f1", "auc"):
        if key in paper_targets:
            diffs[key] = {
                "ours": getattr(metrics, key),
                "paper": paper_targets[key],
                "delta": getattr(metrics, key) - paper_targets[key],
                "within_tolerance": abs(getattr(metrics, key) - paper_targets[key]) <= tolerance,
            }
    return diffs


def parse_binary_output(text: str) -> int:
    text = (text or "").strip()
    if text in {"0", "1"}:
        return int(text)
    for ch in text:
        if ch in {"0", "1"}:
            return int(ch)
    return 0
