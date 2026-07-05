# 论文复现对照表 (Paper vs Implementation)

## 实验覆盖

| 论文章节 | 内容 | 复现脚本 | 状态 |
|---------|------|---------|------|
| §2 | 任务定义与标注指南 | `docs/PAPER_REPRODUCTION.md` | ✅ |
| §3 | CHASM 数据集 (4992 样本) | `src/data/dataset.py` | ✅ |
| Table 2 | Zero-shot / ICL (15 MLLM) | `src/models/api_inference.py`, `local_inference.py` | ✅ 脚本就绪 |
| Table 3 | LoRA 微调 (Qwen2.5, InternVL) | `src/training/finetune_lora.py` | ✅ 配置对齐 Appendix B |
| Table 4 | 错误分析 (4 类) | `src/analysis/error_analysis.py` | ✅ |
| Table 7 | TF-IDF+LR/SVM 基线 | `src/baselines/classical.py` | ✅ 已验证 |
| Figure 3 | 模态消融 | `src/analysis/modality_ablation.py` | ✅ 脚本就绪 |
| Appendix D | 提示词模板 | `src/prompts/templates.py` | ✅ 逐字复现 |
| Appendix E | 详细提示词 (效果更差) | `detailed_zero_shot` 模式 | ✅ |

## 论文目标指标 (test split)

### Table 2 — Zero-shot (部分)

| Model | P | R | F1 | AUC |
|-------|---|---|-----|-----|
| GPT-4o | 0.464 | 0.836 | **0.597** | 0.851 |
| Qwen2.5-7B | 0.473 | 0.378 | 0.421 | 0.660 |
| InternVL2.5 | 0.289 | 0.662 | 0.403 | 0.717 |
| DeepSeek-V3 | 0.499 | 0.787 | 0.571 | 0.826 |

### Table 3 — Fine-tuning

| Model | P | R | F1 | AUC |
|-------|-----|---|-----|-----|
| Qwen2.5-7B (FT) | 0.783 | 0.732 | **0.756** | 0.852 |
| InternVL2.5 (FT) | 0.681 | 0.520 | 0.590 | 0.743 |

### Table 7 — Baselines

| Model | F1 |
|-------|-----|
| TF-IDF + LR | 0.644 |
| TF-IDF + SVM | 0.620 |

## 运行完整复现所需条件

1. **数据**: 下载 CHASM (~9.92GB)，需稳定 HuggingFace 连接
2. **API Key**: GPT-4o / DeepSeek-V3 等闭源模型
3. **GPU**: 24GB+ 用于 7B MLLM 微调与推理
4. **时间**: 全量 test 1000 样本 API 推理约数小时

## 一键命令

```powershell
# 快速验证流程
.\scripts\run_all.ps1 -QuickTest

# 完整基线 (需下载数据)
python -m src.baselines.classical

# 完整 MLLM 评估
python -m src.models.api_inference --model gpt-4o-2024-08-06 --mode zero_shot --split test
python -m src.models.local_inference --model-id Qwen/Qwen2.5-VL-7B-Instruct --split test
python -m src.training.finetune_lora --model-id Qwen/Qwen2.5-VL-7B-Instruct
python -m src.analysis.compare_results
```
