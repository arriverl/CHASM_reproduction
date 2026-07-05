#!/usr/bin/env bash
# CHASM 论文复现一键脚本 (Linux/macOS)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

QUICK="${1:-}"
MAX_ARG=""
if [ "$QUICK" = "--quick" ]; then
  MAX_ARG="--max-samples 200"
fi

echo "=== CHASM Paper Reproduction ==="
pip install -r requirements.txt -q

python -m src.baselines.classical --output-dir outputs/baselines $MAX_ARG
python -m src.training.finetune_lora --dry-run --output-dir outputs/finetune
python -m src.analysis.error_analysis --output-dir outputs/error_analysis
python -m src.analysis.compare_results --output outputs/paper_comparison.json

echo "Done. See README.md for full MLLM experiments."
