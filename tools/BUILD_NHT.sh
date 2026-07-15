#!/bin/bash
# ============================================================================
# BUILD 3DGRUT + NHT tại ~/3dgrut — tự clone, build, vá header tcnn, bắc cầu render.
# Env riêng ~/3dgrut/.venv (py3.11, torch 2.6+cu124) — KHÔNG trộn với .venv gsplat (py3.10).
# ~30-60 phút. Chạy:  bash tools/BUILD_NHT.sh 2>&1 | tee /tmp/build_nht.txt
# ============================================================================
set -e
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.9"      # Ada sm_89 — đúng cho CẢ RTX 4060 lẫn L40S
export PATH="$HOME/.local/bin:$PATH"   # uv
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "########## BUILD NHT — $(date '+%F %T') ##########"
echo "PROJ=$PROJ  CUDA_HOME=$CUDA_HOME  ARCH=$TORCH_CUDA_ARCH_LIST"

echo; echo "===== 0: kiểm tiên quyết ====="
nvcc --version | grep -q "release 12.4" || { echo "❌ nvcc không phải 12.4 → source ~/.bashrc"; exit 1; }
echo "  ✅ nvcc 12.4"
# CHẶN CỨNG nếu đang trong conda env.
# install_env_uv.sh dòng ~150: `if [ -z "${CONDA_PREFIX:-}" ]` → chỉ tạo .venv py3.11 khi
# KHÔNG có conda. Có CONDA_PREFIX thì nó GIẢ ĐỊNH conda env do scripts/create_conda.sh tạo
# (py3.11) và cài thẳng vào đó. Conda base ở đây là py3.14 → build hỏng SAU 30-60 phút.
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "❌ ĐANG Ở TRONG CONDA ENV: '${CONDA_DEFAULT_ENV:-?}' (python3 = $(python3 -V 2>&1))"
  echo "   install_env_uv.sh sẽ BỎ QUA việc tạo .venv py3.11 và cài vào conda env này."
  echo "   → Thoát conda rồi chạy lại:"
  echo "        conda deactivate        # nếu prompt còn '(base)' thì chạy thêm lần nữa"
  echo "        bash tools/BUILD_NHT.sh 2>&1 | tee /tmp/build_nht.txt"
  exit 1
fi
echo "  ✅ không ở trong conda env"
command -v git >/dev/null || { echo "❌ thiếu git"; exit 1; }

# uv: install_env_uv.sh bắt buộc có. Máy local sẵn ở ~/.local/bin nên dễ tưởng là có.
if command -v uv &>/dev/null; then
  echo "  ✅ uv $(uv --version 2>/dev/null)"
else
  echo "  → uv chưa có, cài (installer chính thức astral.sh, không cần sudo)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  command -v uv &>/dev/null || { echo "❌ cài uv xong vẫn không thấy trong PATH"; exit 1; }
  echo "  ✅ uv $(uv --version)"
fi

echo; echo "===== 1: clone 3dgrut (nếu chưa có) ====="
if [ -d ~/3dgrut/.git ]; then
  echo "  ~/3dgrut đã có → chỉ cập nhật submodule"
else
  [ -d ~/3dgrut ] && { echo "  ~/3dgrut dở dang → xoá"; rm -rf ~/3dgrut; }
  git clone --recursive https://github.com/nv-tlabs/3dgrut.git ~/3dgrut
fi
cd ~/3dgrut
git submodule update --init --recursive
echo "  ✅ $(git log --oneline -1)"

echo; echo "===== 2: build env (install_env_uv.sh — lâu nhất, ~30-60ph) ====="
CUDA_HOME=/usr/local/cuda-12.4 bash install_env_uv.sh 3dgrut 2>&1 | tail -60

PY=~/3dgrut/.venv/bin/python
[ -x "$PY" ] || { echo "❌ không thấy ~/3dgrut/.venv/bin/python — build thất bại"; exit 1; }

echo; echo "===== 3: vá header cho tinycudann JIT (bẫy đã biết) ====="
# tcnn JIT-compile lúc chạy, cần vector_types.h... trong include dir của nó, nếu không
# sẽ lỗi 'vector_types.h not found' GIỮA lúc train (sau khi đã tốn giờ GPU).
TCNN_INC=$("$PY" -c "import tinycudann,os;print(os.path.join(os.path.dirname(tinycudann.__file__),'rtc','include'))" 2>/dev/null || true)
if [ -n "$TCNN_INC" ] && [ -d "$TCNN_INC" ]; then
  ln -sf $CUDA_HOME/include/vector_types.h     "$TCNN_INC/" 2>/dev/null || true
  ln -sf $CUDA_HOME/include/vector_functions.h "$TCNN_INC/" 2>/dev/null || true
  ln -sf $CUDA_HOME/include/device_types.h     "$TCNN_INC/" 2>/dev/null || true
  ln -sf $CUDA_HOME/include/crt                "$TCNN_INC/" 2>/dev/null || true
  echo "  ✅ symlink header → $TCNN_INC"
else
  echo "  ⚠ chưa xác định được thư mục include của tinycudann — bỏ qua (vá tay nếu lỗi lúc train)"
fi

echo; echo "===== 4: cầu render competition-pose ====="
cp "$PROJ/tools/render_comp.py" ~/3dgrut/ && echo "  ✅ render_comp.py → ~/3dgrut/"

echo; echo "===== 5: verify ====="
"$PY" -c "import torch; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
for m in kaolin tinycudann; do
  "$PY" -c "import $m; print('  ✅ $m OK')" 2>&1 | tail -1
done

echo
echo "########################################################################"
echo "XONG NHT. Gửi Claude 3 dòng verify (torch/kaolin/tinycudann)."
echo "Lưu ý: 2 env RIÊNG — gsplat dùng $PROJ/.venv (py3.10),"
echo "       NHT dùng ~/3dgrut/.venv (py3.11). KHÔNG trộn."
echo "########################################################################"
