# CHASM 论文完整复现项目

基于 NeurIPS 2025 论文 **「CHASM: Unveiling Covert Advertisements on Chinese Social Media」** 的系统性复现框架。

## 项目概述

本项目在小红书（RedNote）隐性广告检测数据集 **CHASM** 上，完整复现论文中的：

- **Table 2**: 15 个 MLLM 的 Zero-shot / In-Context Learning 评估
- **Table 3**: Qwen2.5-VL-7B / InternVL2.5 LoRA 微调
- **Table 4**: 四类错误分析
- **Table 7**: TF-IDF + LR/SVM 等轻量级基线
- **Figure 3**: 去除 comments / images 的模态消融

## 目录结构

```
CHASM_reproduction/
├── configs/default.yaml      # 论文超参数与目标指标
├── docs/PAPER_REPRODUCTION.md # 详细复现指南
├── src/
│   ├── data/dataset.py       # 数据集加载
│   ├── prompts/templates.py  # Appendix D 提示词
│   ├── metrics/evaluation.py # P/R/F1/AUC
│   ├── baselines/classical.py# Table 7 基线
│   ├── models/
│   │   ├── api_inference.py  # API 闭源 MLLM
│   │   └── local_inference.py# 本地开源 MLLM
│   ├── training/finetune_lora.py # Appendix B 微调
│   └── analysis/             # 消融 & 错误分析
├── scripts/run_all.ps1       # 一键复现 (Windows)
└── outputs/                  # 实验结果
```

## 快速开始

### 1. 环境准备

```powershell
cd d:\Code_development\gitproduct\CHASM_reproduction
pip install -r requirements.txt
```

### 2. 一键运行（快速验证）

```powershell
.\scripts\run_all.ps1 -QuickTest
```

### 3. 完整实验

```powershell
# 轻量级基线 (无需 GPU)
python -m src.baselines.classical

# API 模型 Zero-shot (需 OPENAI_API_KEY)
python -m src.models.api_inference --model gpt-4o-2024-08-06 --mode zero_shot

# 本地 MLLM (需 GPU 24GB+)
python -m src.models.local_inference --model-id Qwen/Qwen2.5-VL-7B-Instruct

# LoRA 微调
python -m src.training.finetune_lora --model-id Qwen/Qwen2.5-VL-7B-Instruct

# 与论文指标对比
python -m src.analysis.compare_results
```

## 论文核心结论

| 设置 | 最佳模型 | F1 |
|------|----------|-----|
| Zero-shot | GPT-4o | 0.597 |
| In-Context | DeepSeek-V3 | 0.592 |
| Fine-tune | Qwen2.5-7B | **0.756** |

> 当前 MLLM 在 Zero-shot/ICL 下均不足以可靠检测隐性广告；在 CHASM 上微调开源 MLLM 可显著提升性能。

## 数据与资源

- 数据集: https://huggingface.co/datasets/Jingyi77/CHASM-Covert_Advertisement_on_RedNote (~9.92GB)
- 官方代码: https://github.com/Jingyi62/CHASM
- 论文: arXiv:2604.20511

## 许可证

研究复现用途。数据集与模型使用请遵循原作者许可协议。
