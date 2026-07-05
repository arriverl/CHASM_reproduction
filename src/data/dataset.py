"""CHASM 数据集加载与预处理。"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Any, Iterator

from datasets import Dataset, concatenate_datasets, load_dataset

from src.prompts.templates import format_post_text

DATASET_NAME = "Jingyi77/CHASM-Covert_Advertisement_on_RedNote"

# HuggingFace 实际 split 名称 (WebDataset 分片)
HF_SPLITS = {
    "example": "Example",
    "train_shards": ["Train_1", "Train_2", "Train_3", "Train_4"],
    "validation": "Validation",
    "test": "Test",
}

# 代码内部统一使用的 split 名
SPLIT_ALIASES = {
    "train": "train",
    "validation": "validation",
    "val": "validation",
    "test": "test",
    "example": "example",
}


@dataclass
class CHASMSample:
    id: str
    title: str
    description: str
    comments: list | str
    images: list
    label: int
    split: str
    date: str = ""

    @property
    def text(self) -> str:
        return format_post_text(self.title, self.description, self.comments, include_comments=True)

    @property
    def text_no_comments(self) -> str:
        return format_post_text(self.title, self.description, self.comments, include_comments=False)


def load_chasm_dataset(
    dataset_name: str = DATASET_NAME,
    streaming: bool = False,
):
    """加载 CHASM 数据集并合并 train 分片。"""
    raw = load_dataset(dataset_name, streaming=streaming)

    if streaming:
        return raw

    merged = {}
    train_parts = []
    for shard in HF_SPLITS["train_shards"]:
        if shard in raw:
            train_parts.append(raw[shard])

    if train_parts:
        merged["train"] = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]

    for internal, hf_name in [("validation", HF_SPLITS["validation"]), ("test", HF_SPLITS["test"]), ("example", HF_SPLITS["example"])]:
        if hf_name in raw:
            merged[internal] = raw[hf_name]

    if not merged:
        return raw

    from datasets import DatasetDict
    return DatasetDict(merged)


def _parse_label(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "ad", "covert", "covert_ad"}:
        return 1
    if text in {"0", "false", "no", "non-ad", "normal"}:
        return 0
    return int(float(text))


def extract_comment_texts(comments: Any) -> list[str]:
    """兼容 HF 上多种 comments 字段格式。"""
    if comments is None:
        return []
    if isinstance(comments, str):
        text = comments.strip()
        if not text:
            return []
        try:
            import json
            return extract_comment_texts(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            return [text]
    if isinstance(comments, dict):
        for key in ("content", "text", "comment", "body", "message"):
            if comments.get(key):
                return [str(comments[key])]
        return [str(v) for v in comments.values() if isinstance(v, str) and v.strip()]
    if isinstance(comments, list):
        out: list[str] = []
        for item in comments:
            out.extend(extract_comment_texts(item))
        return out
    return [str(comments)]


def normalize_sample(raw: dict[str, Any], split: str = "unknown") -> CHASMSample:
    comments = extract_comment_texts(raw.get("comments", []))
    images = raw.get("images") or raw.get("image") or []
    if images and not isinstance(images, list):
        images = [images]
    return CHASMSample(
        id=str(raw.get("id", "")),
        title=str(raw.get("title", "") or ""),
        description=str(raw.get("description", "") or ""),
        comments=comments,
        images=images,
        label=_parse_label(raw.get("label", 0)),
        split=str(raw.get("split", split)),
        date=str(raw.get("date", "") or ""),
    )


def print_dataset_stats(dataset) -> None:
    """打印各 split 样本量与标签分布，便于排查加载问题。"""
    print("=== CHASM Dataset Stats ===")
    for split in dataset.keys():
        labels = [_parse_label(dataset[split][i].get("label", 0)) for i in range(len(dataset[split]))]
        pos = sum(labels)
        neg = len(labels) - pos
        avg_len = 0.0
        if labels:
            sample = normalize_sample(dataset[split][0], split=split)
            avg_len = sum(len(normalize_sample(dataset[split][i], split=split).text) for i in range(min(100, len(labels)))) / min(100, len(labels))
        print(f"  {split}: n={len(labels)}, pos={pos}, neg={neg}, avg_text_len~{avg_len:.0f}")
    print("============================")


def resolve_split(dataset, split: str) -> str:
    split = SPLIT_ALIASES.get(split.lower(), split)
    if split in dataset:
        return split
    # 回退：快速测试用 example
    if "example" in dataset:
        return "example"
    available = list(dataset.keys())
    raise KeyError(f"Split '{split}' not found. Available: {available}")


def iter_split(
    dataset,
    split: str,
    max_samples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterator[CHASMSample]:
    split = resolve_split(dataset, split)
    data = dataset[split]
    indices = list(range(len(data)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    count = 0
    for idx in indices:
        if max_samples is not None and count >= max_samples:
            break
        yield normalize_sample(data[idx], split=split)
        count += 1


def samples_to_records(samples: list[CHASMSample]) -> list[dict]:
    return [
        {
            "id": s.id,
            "text": s.text,
            "text_no_comments": s.text_no_comments,
            "label": s.label,
            "split": s.split,
            "image_count": len(s.images),
        }
        for s in samples
    ]


def select_few_shot_examples(dataset, split: str = "train", seed: int = 42) -> tuple[CHASMSample, CHASMSample]:
    split = resolve_split(dataset, split)
    pos, neg = None, None
    rng = random.Random(seed)
    indices = list(range(len(dataset[split])))
    rng.shuffle(indices)
    for idx in indices:
        sample = normalize_sample(dataset[split][idx], split=split)
        if sample.label == 1 and pos is None:
            pos = sample
        elif sample.label == 0 and neg is None:
            neg = sample
        if pos and neg:
            break
    if not pos or not neg:
        raise RuntimeError("Could not find both positive and negative few-shot examples.")
    return pos, neg


def decode_image(image_data) -> bytes | None:
    """将 base64/PIL/bytes 统一为 bytes。"""
    if image_data is None:
        return None
    if isinstance(image_data, bytes):
        return image_data
    if hasattr(image_data, "save"):
        buf = io.BytesIO()
        image_data.save(buf, format="PNG")
        return buf.getvalue()
    if isinstance(image_data, str):
        import base64

        if image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[-1]
        try:
            return base64.b64decode(image_data)
        except Exception:
            return None
    if isinstance(image_data, dict) and "bytes" in image_data:
        return image_data["bytes"]
    return None


def create_demo_dataset() -> "DatasetDict":
    """离线演示数据集（网络不可用时用于验证流程）。"""
    from datasets import DatasetDict

    demo_posts = [
        {"id": "demo_1", "title": "今日穿搭分享", "description": "这套衣服来自A品牌和B品牌，各有优缺点。", "comments": [{"content": "好看！"}], "images": [], "label": 0, "split": "example", "date": "09-01"},
        {"id": "demo_2", "title": "必入！XX牌精华", "description": "买1送2，私信我发链接，全网最低价！", "comments": [{"content": "求链接"}], "images": [], "label": 1, "split": "example", "date": "09-02"},
        {"id": "demo_3", "title": "护肤心得", "description": "用了三款不同品牌面霜，这款偏油但保湿好。", "comments": [], "images": [], "label": 0, "split": "example", "date": "09-03"},
        {"id": "demo_4", "title": "Brand X  toner 测评", "description": "Brand X toner 真的绝了！加群领取优惠券购买。", "comments": [{"content": "加微信xxx"}], "images": [], "label": 1, "split": "example", "date": "09-04"},
    ]
    return DatasetDict({"example": Dataset.from_list(demo_posts), "train": Dataset.from_list(demo_posts * 3), "test": Dataset.from_list(demo_posts)})
