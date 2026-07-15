#!/bin/bash
# ============================================================================
# CÀI CUDA TOOLKIT 12.4 (khớp torch 2.4.1+cu124) — CẦN SUDO.
# Chạy TỪNG PHẦN, kiểm tra kết quả mỗi bước. KHÔNG chạy một mạch lần đầu.
# Máy: Ubuntu 22.04, driver CUDA 13.0 (forward-compat với toolkit 12.4 — OK).
# ============================================================================
set -e
# tự suy ra project root (chạy được ở mọi máy/path — đừng hardcode nữa)
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
echo "PROJ=$PROJ"

# ---------- PHẦN 1: cài CUDA toolkit 12.4 (NVIDIA official repo) ----------
echo "===== PHẦN 1: CUDA toolkit 12.4 ====="
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
# (chỉ toolkit — KHÔNG cài driver, driver 13.0 hiện tại giữ nguyên)

# ---------- PHẦN 2: biến môi trường (thêm vào ~/.bashrc để bền) ----------
echo "===== PHẦN 2: env ====="
if ! grep -q "cuda-12.4/bin" ~/.bashrc; then
cat >> ~/.bashrc <<'EOF'

# CUDA 12.4 toolkit (khớp torch cu124)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF
fi
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ---------- PHẦN 3: verify (PHẢI thấy "release 12.4") ----------
echo "===== PHẦN 3: verify ====="
nvcc --version
echo ">>> Nếu thấy 'release 12.4' ở trên là THÀNH CÔNG. Nếu không, DỪNG và báo mình."

# ---------- PHẦN 4: fused-ssim (CHỈ nếu .venv đã có) ----------
# Chạy SETUP_NVCC.sh TRƯỚC SETUP_MAIN_VENV.sh là đúng thứ tự → .venv chưa có ở đây
# là BÌNH THƯỜNG. SETUP_MAIN_VENV.sh bước 5 tự build fused-ssim khi thấy nvcc 12.x.
echo "===== PHẦN 4: fused-ssim ====="
cd "$PROJ"
if [ -x .venv/bin/pip ]; then
  .venv/bin/pip install --no-build-isolation \
    "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
    && .venv/bin/python -c "import fused_ssim; print('fused-ssim OK')" \
    || echo "⚠ fused-ssim fail → fallback torchmetrics, vẫn chạy được"
else
  echo "→ .venv chưa có: BỎ QUA (không phải lỗi)."
  echo "  SETUP_MAIN_VENV.sh sẽ tự build fused-ssim vì nvcc 12.4 giờ đã sẵn."
fi

echo ""
echo "========================================================================"
echo "XONG NVCC. Bước tiếp:"
echo "  1) bash tools/VERIFY_SERVER.sh   # 10s, chỉ đọc — BẮT python3.10 trước khi tải 2.5GB torch"
echo "  2) bash tools/SETUP_MAIN_VENV.sh # .venv + torch + gsplat wheel + fused-ssim"
echo "========================================================================"
