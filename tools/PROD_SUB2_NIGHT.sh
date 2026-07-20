#!/bin/bash
# ============================================================================
# PROD_SUB2_NIGHT — production SUB2 chạy THẲNG trên 1 GPU (gán qua tmux ngoài).
# Tuần tự: OBJ (bonsai/chair classic SH4 3M — ngắn, fail sớm) → HCM (3DGUT 6M)
# → FINALIZE (mean seed+members → vgg → zip q96 + contact sheet). ~14-16h L40S.
#
# Chạy trên server (GPU1 prod, GPU0 để audit — 2 tmux độc lập, VRAM riêng từng GPU,
# tài nguyên chung duy nhất là ĐĨA — có disk_guard + PRUNE_CKPT lo):
#   tmux new -d -s prod "CUDA_VISIBLE_DEVICES=1 bash tools/PROD_SUB2_NIGHT.sh 2>&1 | tee /tmp/prod_night.txt"
#
# Env: SUBTAG=2 · SEEDS="42 7 123" · PRUNE_CKPT=1 (mặc định BẬT: xoá ckpt seed 7/123
#      SAU khi render verify đủ ảnh — seed 42 giữ cho enhancer; giữ hết: PRUNE_CKPT=0)
#      MIN_FREE_GB=25 · resume-safe: chết giữa chừng chạy lại là tự bỏ qua phần xong.
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=.venv/bin/python
SUBTAG=${SUBTAG:-2}
SEEDS=${SEEDS:-"42 7 123"}
PRUNE_CKPT=${PRUNE_CKPT:-1}
MIN_FREE_GB=${MIN_FREE_GB:-25}
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }

say "0. tiên quyết (fail sớm trước khi đụng GPU $CUDA_VISIBLE_DEVICES)"
[ -x "$PY" ] || die "thiếu .venv"
$PY -c "from pycolmap import SceneManager" 2>/dev/null || die ".venv hỏng pycolmap rmbrualla (DOC3 §3.7)"
echo "  git HEAD: $(git log --oneline -1 2>/dev/null || echo '?')"
for s in bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  [ -s "workspace_r2/$s/sparse/0/images.bin" ] || die "thiếu workspace_r2/$s → bash tools/prepare_r2.sh trước"
done
echo "  ✅ workspace_r2 đủ 7 scene"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
N_SEED=$(echo $SEEDS | wc -w)
if [ "$PRUNE_CKPT" = "1" ]; then NEED=$((5*2 + 2*1 + 10)); else NEED=$((5*2*N_SEED + 2*N_SEED + 10)); fi
echo "  đĩa trống ${FREE}GB · ước tính cần ~${NEED}GB (PRUNE_CKPT=$PRUNE_CKPT)"
[ "$FREE" -lt "$MIN_FREE_GB" ] && die "đĩa ${FREE}GB < MIN_FREE_GB=${MIN_FREE_GB} — dọn trước (pip cache purge / renders đã chốt verdict)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'

ENVSTR="SUBTAG=$SUBTAG SEEDS=$SEEDS PRUNE_CKPT=$PRUNE_CKPT OBJ_SH_DEGREE=4 CAP_HCM=6000000 CAP_OBJ=3000000"

say "1/3 OBJ (bonsai/chair — classic SH4 cap3M ×$N_SEED seed, ~4-5h)"
# tin EXIT-CODE thật (pipefail đã bật), KHÔNG grep ký tự ❌ trong log — đã dính 19/07:
# log chứa ❌ vô hại → die oan sau khi OBJ xong sạch, HCM không bao giờ chạy
SUBTAG=$SUBTAG SEEDS="$SEEDS" PRUNE_CKPT=$PRUNE_CKPT OBJ_SH_DEGREE=4 CAP_OBJ=3000000 \
  SCENES_HCM="" bash tools/run_sub_r2.sh 2>&1 | tee /tmp/prod_obj.txt \
  || die "OBJ exit lỗi — đọc /tmp/prod_obj.txt, sửa xong chạy lại (resume-safe)"

say "2/3 HCM (5 scene — 3DGUT raw-distortion cap6M ×$N_SEED seed, ~11-12h)"
SUBTAG=$SUBTAG SEEDS="$SEEDS" PRUNE_CKPT=$PRUNE_CKPT CAP_HCM=6000000 \
  SCENES_OBJ="" bash tools/run_sub_r2.sh 2>&1 | tee /tmp/prod_hcm.txt \
  || die "HCM exit lỗi — đọc /tmp/prod_hcm.txt, sửa xong chạy lại (resume-safe)"

say "3/3 FINALIZE — mean(seed+members) → vgg → zip q96 + contact sheet"
SUBTAG=$SUBTAG MIN_SEEDS=$N_SEED bash tools/FINALIZE_SUB2.sh 2>&1 | tee /tmp/finalize_sub${SUBTAG}.txt || die "finalize"

say "tổng kết"
ls -la --time-style=full-iso submission_R2_SUB${SUBTAG}.zip 2>/dev/null || echo "⚠ không thấy zip?"
df -h . | tail -1
echo
echo "########################################################################"
echo "#  PROD-NIGHT-DONE: soi contact sheet /tmp/sheets_sub${SUBTAG}/*.jpg BẰNG MẮT"
echo "#  (round 2 không có GT — mắt là lưới cuối) RỒI MỚI NỘP. scp zip về local."
echo "#  Nếu W1-sharp (audit GPU0) thắng ≥+0.003 → hỏi Claude cách thêm member"
echo "#  --test-weight cho 5 scene HCM TRƯỚC khi nộp (chỉ +1 lượt train/scene)."
echo "########################################################################"
