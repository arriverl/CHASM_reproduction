# CHASM 论文完整复现指南

## 论文信息

- **标题**: CHASM: Unveiling Covert Advertisements on Chinese Social Media
- **会议**: NeurIPS 2025 (Datasets and Benchmarks Track)
- **arXiv**: 2604.20511
- **数据集**: [HuggingFace CHASM](https://huggingface.co/datasets/Jingyi77/CHASM-Covert_Advertisement_on_RedNote)
- **官方代码**: [GitHub Jingyi62/CHASM](https://github.com/Jingyi62/CHASM)

## 任务定义 (Section 2.1)

**隐性广告 (Covert Advertisement)** 需满足：
1. 作者有明确推广产品/付费服务以获取金钱收益的意图
2. 作者故意伪装成普通内容（平台或用户已标注的广告不算）

**标签**: `1` = 隐性广告, `0` = 非广告

## 数据集统计 (Table 1)

| 指标 | 数值 |
|------|------|
| 总样本 | 4,992 |
| 隐性广告 (正类) | 612 (12.3%) |
| 非广告 (负类) | 4,380 (87.7%) |
| 产品分享难例 | 1,127 (22.6%) |
| 平均图片数/帖 | 5.28 |
| 平均正文长度 | 196.63 |
| Train/Val/Test | 3493 / 499 / 1000 |

## 实验复现清单

### Table 2: Zero-shot & In-Context Learning (15 个 MLLM)

```powershell
# Zero-shot GPT-4o (论文 F1=0.597)
$env:OPENAI_API_KEY="your_key"
python -m src.models.api_inference --model gpt-4o-2024-08-06 --mode zero_shot --split test

# In-Context Learning
python -m src.models.api_inference --model gpt-4o-2024-08-06 --mode in_context --split test

# 本地开源模型 (需 GPU)
python -m src.models.local_inference --model-id Qwen/Qwen2.5-VL-7B-Instruct --max-samples 1000
python -m src.models.local_inference --model-id OpenGVLab/InternVL2_5-8B --max-samples 1000
```

### Table 3: LoRA 微调 (Appendix B)

超参数：
- LoRA rank=8, alpha=32, 所有 linear 层
- batch=1, grad_accum=16, AdamW(lr=1e-4, wd=0.1)
- 3 epochs, cosine scheduler, max_length=4096, bf16
- 5-fold cross-validation

```powershell
python -m src.training.finetune_lora --model-id Qwen/Qwen2.5-VL-7B-Instruct
# 目标: Qwen2.5 F1=0.756, InternVL F1=0.590
```

### Table 7: 轻量级基线

```powershell
python -m src.baselines.classical
# TF-IDF+LR F1=0.644, TF-IDF+SVM F1=0.620
```

### Figure 3: 模态消融 (微调 Qwen2.5)

```powershell
python -m src.analysis.modality_ablation
# baseline F1=0.756, w/o comments F1≈0.54, w/o images F1≈0.62
```

### Table 4: 错误分析

四类错误：Insufficient Evidence, Missing Clue, Language Style, Post Structure

```powershell
python -m src.analysis.error_analysis --predictions outputs/api/gpt-4o_zero_shot_test.csv
```

## 论文目标指标对照

所有目标值已写入 `configs/default.yaml` 的 `paper_targets` 节。
运行实验后执行：

```powershell
python -m src.analysis.compare_results
```

## 提示词模板 (Appendix D)

| 模式 | 文件 |
|------|------|
| Zero-shot | `src/prompts/templates.py` → `ZERO_SHOT_PROMPT` |
| Few-shot (ICL) | `FEW_SHOT_PROMPT_HEADER` + 正负例 |
| Detailed prompt | `DETAILED_ZERO_SHOT_PROMPT` (Appendix E, 效果更差) |

## 评估指标 (Section 4.1)

Precision, Recall, **F1** (主指标), AUC

## 硬件需求

| 实验 | 最低配置 |
|------|----------|
| TF-IDF 基线 | CPU, 16GB RAM |
| 本地 7B MLLM 推理 | GPU 16GB+ |
| LoRA 微调 | GPU 24GB+ (A100 推荐) |
| API 闭源模型 | OpenAI/DeepSeek/Qwen API Key |

## 数据下载

完整数据集约 **9.92 GB**，首次运行会自动从 SorahFace Hub 缓存。

```python
from datasets import load_dataset
ds = load_dataset("Jingyi77/CHASM-Covert_Advertisement_on_RedNote")
```

## 引用

```bibtex
@inproceedings{zheng2025chasm,
  title={CHASM: Unveiling Covert Advertisements on Chinese Social Media},
  author={Zheng, Jingyi and Hu, Tianyi and Liu, Yule and Sun, Zhen and Zhang, Zongmin and Peng, Zifan and Dong, Wenhan and He, Xinlei},
  booktitle={NeurIPS 2025 Datasets and Benchmarks Track},
  year={2025}
}
```
