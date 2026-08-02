#!/usr/bin/env bash
# =============================================================================
# setup_thor.sh — environment setup for TRUNet on NVIDIA IGX Thor
#   OS  : Ubuntu 24.04.4 LTS (aarch64 / Grace-class CPU)
#   GPU : RTX PRO 6000 Blackwell Max-Q  (sm_120, 96 GB)
#
# Installs a Blackwell-capable PyTorch (CUDA 12.8) + the rest of the deps, then
# verifies the GPU is visible and that this torch build can target sm_120.
#
# Usage:
#   bash setup_thor.sh
#   source .venv/bin/activate
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "==> [1/5] Checking NVIDIA driver / GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "    nvidia-smi not found. Install the NVIDIA driver first:"
  echo "      sudo apt-get update && sudo apt-get install -y nvidia-driver-570-open"
  echo "    (Blackwell / RTX PRO 6000 needs the 570+ 'open' driver branch and CUDA 12.8+.)"
  echo "    Re-run this script after a reboot."
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo "==> [2/5] Creating virtualenv (.venv) with system Python 3"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

echo "==> [3/5] Installing PyTorch for CUDA 12.8 (Blackwell / sm_120)"
# cu128 wheels include sm_120 kernels and have aarch64 builds for Grace/Thor.
# If a stable cu128 wheel is unavailable for your Python, use the nightly line below.
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision || {
  echo "    stable cu128 failed — falling back to nightly cu128"
  pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu128 torch torchvision
}

echo "==> [4/5] Installing the remaining Python dependencies"
pip install -r requirements_thor.txt

echo "==> [5/5] Verifying Blackwell support"
python - <<'PY'
import torch
print("torch:", torch.__version__, "| built CUDA:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not available — check driver install"
cap = torch.cuda.get_device_capability(0)
arches = torch.cuda.get_arch_list()
print("GPU:", torch.cuda.get_device_name(0), "| capability: sm_%d%d" % cap)
print("arch_list:", arches)
sm = "sm_%d%d" % cap
ok = any(a.startswith("sm_%d" % cap[0]) for a in arches)
# quick kernel test on-device
x = torch.randn(1024, 1024, device="cuda"); y = (x @ x).sum().item()
print("on-device matmul OK (sum=%.1f)" % y)
print("BLACKWELL SUPPORT:", "OK" if ok else "*** MISSING sm_120 — reinstall cu128/nightly ***")
PY

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then follow RUN_STEPS_THOR.md (preprocess -> train)."
