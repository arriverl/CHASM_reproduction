# 一键复现脚本 (Windows PowerShell)
# 用法: .\scripts\run_all.ps1 [-QuickTest]

param(
    [switch]$QuickTest
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== CHASM 论文复现 ===" -ForegroundColor Cyan
Write-Host "项目根目录: $Root"

# 安装依赖
Write-Host "`n[1/5] 安装依赖..." -ForegroundColor Yellow
pip install -r requirements.txt -q

$MaxSamples = if ($QuickTest) { 200 } else { $null }
$MaxArg = if ($MaxSamples) { "--max-samples $MaxSamples" } else { "" }

# 基线实验 Table 7
Write-Host "`n[2/5] 运行轻量级基线 (TF-IDF+LR/SVM)..." -ForegroundColor Yellow
python -m src.baselines.classical --output-dir outputs/baselines --demo $MaxArg

# 微调配置验证 Appendix B
Write-Host "`n[3/5] 验证微调配置 (dry-run)..." -ForegroundColor Yellow
python -m src.training.finetune_lora --dry-run --demo --output-dir outputs/finetune

# 错误分析参考 Table 4
Write-Host "`n[4/5] 导出错误分析参考..." -ForegroundColor Yellow
python -m src.analysis.error_analysis --output-dir outputs/error_analysis

# 与论文对比
Write-Host "`n[5/5] 生成论文对比报告..." -ForegroundColor Yellow
python -m src.analysis.compare_results --output outputs/paper_comparison.json

Write-Host "`n=== 复现流程完成 ===" -ForegroundColor Green
Write-Host "完整 MLLM 实验需额外运行:"
Write-Host "  API Zero-shot:  python -m src.models.api_inference --model gpt-4o-2024-08-06 --mode zero_shot"
Write-Host "  API ICL:        python -m src.models.api_inference --model gpt-4o-2024-08-06 --mode in_context"
Write-Host "  本地 MLLM:      python -m src.models.local_inference --model-id Qwen/Qwen2.5-VL-7B-Instruct"
Write-Host "  模态消融:       python -m src.analysis.modality_ablation"
Write-Host "  LoRA 微调:      python -m src.training.finetune_lora --model-id Qwen/Qwen2.5-VL-7B-Instruct"
