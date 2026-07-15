#!/bin/bash
# ============================================================================
# VERIFY_SERVER.sh — chẩn đoán CHỈ ĐỌC, chạy NGAY SAU KHI CLONE trên server mới.
# Không cài gì, không sửa gì. In một báo cáo → copy toàn bộ output gửi lại Claude.
#
#   bash tools/VERIFY_SERVER.sh 2>&1 | tee /tmp/verify_server.txt
#
# Chạy được cả khi chưa có .venv / chưa có data / chưa cài nvcc 12.4.
# ============================================================================
cd "$(dirname "$0")/.." || exit 1
PROJ="$(pwd)"
ok(){   printf "  ✅ %s\n" "$1"; }
bad(){  printf "  ❌ %s\n" "$1"; }
warn(){ printf "  ⚠  %s\n" "$1"; }
inf(){  printf "     %s\n" "$1"; }
hdr(){  printf "\n══ %s ══\n" "$1"; }

echo "################ BTS VERIFY SERVER — $(date '+%F %T') ################"
echo "host=$(hostname)  user=$(whoami)"
echo "project=$PROJ"

# ─────────────────────────────────────────────────────────────────────────
hdr "A. PATH & REPO"
case "$PROJ" in
  *\ *) bad "PATH CÓ DẤU CÁCH → 3DGRUT sẽ VỠ. Đổi sang path kiểu ~/bts rồi clone lại." ;;
  *)    ok "path không dấu cách" ;;
esac
if [ -d .git ]; then
  ok "git repo — commit: $(git log --oneline -1 2>/dev/null)"
  DIRTY=$(git status --porcelain 2>/dev/null | wc -l)
  [ "$DIRTY" -eq 0 ] && ok "working tree sạch" || warn "$DIRTY file thay đổi so với commit"
else
  bad "KHÔNG phải git repo — clone hỏng?"
fi

# ─────────────────────────────────────────────────────────────────────────
hdr "B. TOOLS + PATCH GSPLAT (cốt lõi — thiếu là dừng)"
MISS=0
for f in tools/prepare_scene.py tools/render_test_poses.py tools/score_local.py \
         tools/normalize_compat.py tools/make_submission.py tools/add_sky_points.py \
         tools/ensemble.py tools/enhance_net.py tools/render_comp.py \
         tools/SETUP_MAIN_VENV.sh tools/SETUP_NVCC.sh tools/BUILD_NHT.sh \
         results/experiments.csv gsplat/examples/simple_trainer.py; do
  [ -f "$f" ] || { bad "THIẾU $f"; MISS=1; }
done
[ $MISS -eq 0 ] && ok "14/14 file cốt lõi có mặt"
[ -f results/experiments.csv ] && inf "experiments.csv: $(($(wc -l < results/experiments.csv)-1)) dòng kết quả"

# patch BTS phải nằm trong simple_trainer.py
ST=gsplat/examples/simple_trainer.py
if [ -f "$ST" ]; then
  for pat in raw_distortion global_seed fused_ssim; do
    grep -q "$pat" "$ST" && ok "patch '$pat' có trong simple_trainer.py" \
                         || bad "MẤT patch '$pat' → sai pipeline, DỪNG LẠI"
  done
else
  bad "không có $ST"
fi

# tàn dư git của upstream (phải sạch)
LEFT=$(ls -a gsplat/ 2>/dev/null | grep -E '^\.git' | tr '\n' ' ')
[ -z "$LEFT" ] && ok "gsplat sạch .git*/.gitmodules (patch nằm sẵn — KHÔNG cần git apply)" \
               || warn "gsplat còn: $LEFT"

# glm submodule rỗng — chỉ là bẫy nếu build from source
GLM=$(find gsplat/gsplat/cuda/csrc/third_party/glm -type f 2>/dev/null | wc -l)
if [ "$GLM" -eq 0 ]; then
  warn "third_party/glm RỖNG (submodule chưa init) → TUYỆT ĐỐI không 'pip install -e gsplat/'."
  inf  "Ta dùng wheel dựng sẵn nên KHÔNG cần glm. Clone chỉ để chạy examples/."
else
  ok "glm có $GLM file (build from source được)"
fi

# ─────────────────────────────────────────────────────────────────────────
hdr "C. GPU (kỳ vọng: 2× L40S, dùng GPU 0)"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader \
    | while IFS= read -r l; do inf "GPU $l"; done
  inf "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  FREE0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null)
  if [ -n "$FREE0" ]; then
    [ "$FREE0" -gt 40000 ] && ok "GPU0 trống ${FREE0}MiB → đủ cho cap 5-6M" \
                           || warn "GPU0 chỉ còn ${FREE0}MiB trống — có ai đang chiếm?"
  fi
  echo "  --- ai đang chiếm GPU ---"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | sed 's/^/     /' || inf "(không có process)"
  [ -n "$CUDA_VISIBLE_DEVICES" ] && ok "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
                                 || warn "CUDA_VISIBLE_DEVICES CHƯA đặt → phải 'export CUDA_VISIBLE_DEVICES=0' (GPU1 bận VLLM)"
else
  bad "không có nvidia-smi"
fi

# ─────────────────────────────────────────────────────────────────────────
hdr "D. TOOLCHAIN (quyết định cách cài)"
inf "OS: $( (lsb_release -ds 2>/dev/null || grep -oP '(?<=^PRETTY_NAME=").*(?=")' /etc/os-release) )"
inf "  → SETUP_NVCC.sh đang trỏ repo 'ubuntu2204'. Nếu OS trên KHÔNG phải 22.04 → phải sửa URL."
inf "kernel: $(uname -r)   RAM: $(free -g | awk '/^Mem:/{print $2}')GB"

# python 3.10 — wheel gsplat/torch pin cp310
inf "python3 mặc định: $(python3 -V 2>&1)"
[ -n "$CONDA_DEFAULT_ENV" ] && warn "conda env '$CONDA_DEFAULT_ENV' đang bật → 'python3' là của conda, có thể KHÔNG phải 3.10"
if command -v python3.10 &>/dev/null && python3.10 -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null; then
  ok "python3.10 có thật ($(python3.10 -V 2>&1)) → SETUP_MAIN_VENV.sh sẽ tự chọn nó"
  python3.10 -c "import venv" 2>/dev/null && ok "python3.10-venv OK" || bad "thiếu python3.10-venv → sudo apt install -y python3.10-venv"
else
  bad "KHÔNG có python3.10 → wheel gsplat 1.5.3+pt24cu124 (cp310) + torch 2.4.1 KHÔNG cài được"
  inf "sudo apt install -y python3.10 python3.10-venv"
fi

# nvcc: 11.8 vô hại cho standard; NHT cần 12.4
if command -v nvcc &>/dev/null; then
  NV=$(nvcc --version | grep -oP 'release \K[0-9.]+')
  inf "nvcc: $NV  (CUDA_HOME=${CUDA_HOME:-chưa đặt})"
  case "$NV" in
    12.4*) ok "nvcc 12.4 — khớp torch cu124 → NHT/3DGRUT + fused-ssim build được" ;;
    *)     warn "nvcc $NV ≠ 12.4 → standard pipeline VẪN CHẠY (dùng wheel, không compile)," ;
           inf  "nhưng NHT/3DGRUT + fused-ssim CẦN 12.4 → chạy tools/SETUP_NVCC.sh (cần sudo)" ;;
  esac
  inf "which nvcc: $(command -v nvcc)"
  # CUDA_HOME phải khớp nvcc, nếu không build fused-ssim/3DGRUT lỗi khó hiểu
  if [ -n "$CUDA_HOME" ]; then
    REAL=$(readlink -f "$CUDA_HOME" 2>/dev/null)
    inf "CUDA_HOME=$CUDA_HOME → thực tế: $REAL"
    case "$REAL" in
      *12.4*) ok "CUDA_HOME khớp nvcc 12.4" ;;
      *)      bad "CUDA_HOME trỏ '$REAL' NHƯNG nvcc là $NV → LỆCH, build sẽ lỗi"
              inf "sửa: export CUDA_HOME=/usr/local/cuda-12.4 (thêm vào ~/.bashrc)" ;;
    esac
  else
    warn "CUDA_HOME chưa đặt → export CUDA_HOME=/usr/local/cuda-12.4"
  fi
  ls -d /usr/local/cuda-12.4 &>/dev/null && ok "/usr/local/cuda-12.4 tồn tại" || true
else
  warn "không có nvcc → standard pipeline vẫn chạy; NHT thì cần SETUP_NVCC.sh"
fi
inf "gcc: $(gcc --version 2>/dev/null | head -1 || echo 'KHÔNG CÓ → apt install build-essential')"
sudo -n true 2>/dev/null && ok "sudo không cần mật khẩu" || warn "sudo cần mật khẩu (hoặc không có) — SETUP_NVCC.sh sẽ hỏi"

# ─────────────────────────────────────────────────────────────────────────
hdr "E. DATA (git KHÔNG mang — phải rsync riêng)"
if [ -d VAI_NVS_DATA/phase1 ]; then
  ok "VAI_NVS_DATA/phase1 có ($(du -sh VAI_NVS_DATA/phase1 2>/dev/null | cut -f1)) — kỳ vọng ~3.2G"
  for s in public_set private_set1; do
    N=$(ls VAI_NVS_DATA/phase1/$s 2>/dev/null | wc -l)
    [ "$N" -gt 0 ] && inf "$s: $N scene — $(ls VAI_NVS_DATA/phase1/$s | tr '\n' ' ')" || bad "$s TRỐNG"
  done
else
  bad "CHƯA CÓ VAI_NVS_DATA → rsync từ máy local:"
  inf 'rsync -avP "VAI_NVS_DATA/" <user>@<server>:'"$PROJ"'/VAI_NVS_DATA/'
fi
inf "đĩa trống: $(df -h "$PROJ" | awk 'NR==2{print $4}')  (cần ≥60G: data 3.2G + venv 6G + 3dgrut + ckpt)"

# ─────────────────────────────────────────────────────────────────────────
hdr "F. MÔI TRƯỜNG PYTHON (chưa có là bình thường nếu chưa chạy setup)"
if [ -x .venv/bin/python ]; then
  inf "$(.venv/bin/python -V 2>&1)"
  .venv/bin/python - <<'PY' 2>&1 | sed 's/^/     /'
try:
    import torch; print("torch", torch.__version__, "| cuda avail:", torch.cuda.is_available(),
                        "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
    print("torch built for CUDA", torch.version.cuda, "(cần 12.4)")
except Exception as e: print("❌ torch:", e)
for m in ("gsplat","lpips","cv2","tyro","torchmetrics"):
    try:
        mod=__import__(m); print("✅", m, getattr(mod,"__version__",""))
    except Exception as e: print("❌", m, e)
try:
    import fused_ssim; print("✅ fused-ssim (train nhanh hơn)")
except Exception: print("⚠  fused-ssim chưa có → dùng fallback torchmetrics (vẫn chạy, chậm hơn chút)")
PY
else
  warn ".venv chưa có → chạy: bash tools/SETUP_MAIN_VENV.sh"
fi
[ -d "$HOME/3dgrut" ] && ok "~/3dgrut có" || inf "~/3dgrut chưa có (cần cho canh bạc NHT cap cao)"

# ─────────────────────────────────────────────────────────────────────────
hdr "G. VIỆC TIẾP THEO"
[ ! -x .venv/bin/python ]   && echo "  1) bash tools/SETUP_MAIN_VENV.sh      # standard pipeline, KHÔNG cần nvcc"
[ -z "$CUDA_VISIBLE_DEVICES" ] && echo "  2) echo 'export CUDA_VISIBLE_DEVICES=0' >> ~/.bashrc && source ~/.bashrc"
[ ! -d VAI_NVS_DATA/phase1 ] && echo "  3) rsync data từ máy local (lệnh ở mục E)"
command -v nvcc &>/dev/null && nvcc --version 2>/dev/null | grep -q 'release 12.4' \
  || echo "  4) sửa PROJ trong tools/SETUP_NVCC.sh → '$PROJ', rồi bash tools/SETUP_NVCC.sh  # cho NHT"
[ ! -d "$HOME/3dgrut" ] && echo "  5) clone 3dgrut + bash tools/BUILD_NHT.sh  # sau khi có nvcc 12.4"
echo "  → Gửi TOÀN BỘ output này lại cho Claude."
echo "################ HẾT ################"
