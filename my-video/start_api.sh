#!/bin/bash
# DigiHuman API server startup script
# Usage: ./start_api.sh
# Environment variables:
#   CUDA_VISIBLE_DEVICES - set to control which GPU to use (default: 0)

set -e

CONDA_ENV="/home/claude/miniconda3/envs/digihuman"
PYTHON="$CONDA_ENV/bin/python"
FFMPEG="$CONDA_ENV/bin/ffmpeg"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU device selection (default to GPU 0)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Export ffmpeg path for all engines that need it
export DIGIHUMAN_DIGITAL_HUMAN_FFMPEG_BIN="$FFMPEG"
export DIGIHUMAN_SUBTITLE_FFMPEG_BIN="$FFMPEG"
export DIGIHUMAN_BGM_FFMPEG_BIN="$FFMPEG"

# Ensure conda bin is on PATH (for any subprocess that shells out to ffmpeg)
export PATH="$CONDA_ENV/bin:$PATH"

# GPU availability check
echo "=== GPU Environment Check ==="
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi failed"
else
    echo "WARNING: nvidia-smi not found"
fi

$PYTHON -c "
import torch
if torch.cuda.is_available():
    dev = torch.cuda.get_device_properties(0)
    print(f'PyTorch CUDA: {torch.version.cuda}, Device: {dev.name}, VRAM: {dev.total_mem / 1024**3:.1f}GB')
else:
    print('WARNING: torch.cuda.is_available() = False — all inference will run on CPU!')
" 2>/dev/null || echo "WARNING: PyTorch CUDA check failed"
echo "=============================="

cd "$PROJECT_DIR"
exec "$PYTHON" -m api.main "$@"
