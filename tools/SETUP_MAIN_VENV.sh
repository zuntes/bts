#!/bin/bash
# ============================================================================
# TÁI TẠO .venv CHÍNH trên máy mới (pipeline standard gsplat + scoring + L16).
# Chạy TRONG thư mục dự án (path KHÔNG dấu cách càng tốt, nhưng .venv chính chịu
# được space). Cần: python3.10, và (cho fused-ssim) nvcc 12.4 — chạy SETUP_NVCC.sh trước.
# Chạy: bash tools/SETUP_MAIN_VENV.sh
# ============================================================================
set -e
cd "$(dirname "$0")/.."
PROJ="$(pwd)"

echo "===== 1: tạo .venv (BẮT BUỘC python3.10) ====="
# wheel gsplat==1.5.3+pt24cu124 pin cp310 và torch 2.4.1+cu124 không có cp313/cp314.
# KHÔNG dùng `python3` trần: trên server có conda base, `python3` có thể là 3.14 → hỏng.
PY=""
for c in python3.10 python3; do
  command -v "$c" &>/dev/null || continue
  "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null \
    && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  echo "❌ KHÔNG tìm thấy python3.10 (wheel gsplat/torch pin cp310)."
  echo "   python3 hiện tại: $(python3 -V 2>&1)"
  echo "   Cài: sudo apt install -y python3.10 python3.10-venv   rồi chạy lại."
  exit 1
fi
echo "dùng $PY → $("$PY" -V 2>&1)"
"$PY" -m venv .venv
.venv/bin/python -V
.venv/bin/pip install -q --upgrade pip wheel setuptools

echo "===== 2: torch 2.4.1 + cu124 ====="
.venv/bin/pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

echo "===== 3: gsplat 1.5.3 wheel dựng sẵn (KHÔNG build) ====="
.venv/bin/pip install -q ninja jaxtyping rich packaging
.venv/bin/pip install --no-deps gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124

echo "===== 4: deps pipeline ====="
.venv/bin/pip install -q "numpy<2.0.0" viser "imageio[ffmpeg]" scikit-learn tqdm \
  "torchmetrics[image]" opencv-python "tyro>=0.8.8" Pillow tensorboard tensorly \
  pyyaml matplotlib lpips pandas splines \
  "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787" \
  "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"

echo "===== 5: fused-ssim (cần nvcc 12.4 — bỏ qua nếu chưa cài, có fallback) ====="
if command -v nvcc &>/dev/null && nvcc --version | grep -q "release 12"; then
  export CUDA_HOME=$(dirname $(dirname $(command -v nvcc)))
  .venv/bin/pip install --no-build-isolation \
    "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5" \
    && echo "fused-ssim OK" || echo "⚠ fused-ssim fail (dùng fallback torchmetrics, vẫn chạy)"
else
  echo "⚠ chưa có nvcc → fused-ssim skip (trainer patch có fallback torchmetrics)"
fi

echo "===== 6: verify ====="
.venv/bin/python -c "import torch,gsplat,lpips,cv2,tyro; print('torch',torch.__version__,'cuda',torch.cuda.is_available()); print('gsplat',gsplat.__version__)"
echo ""
echo "XONG .venv chính."
echo "gsplat/ trong repo ĐÃ patch sẵn (không còn .git riêng — đừng 'git branch' trong đó,"
echo "và TUYỆT ĐỐI không 'pip install -e gsplat/': third_party/glm rỗng, dùng wheel là đủ)."
echo "Tiếp: bash tools/VERIFY_SERVER.sh   # xác nhận toàn bộ"
