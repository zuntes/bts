#!/bin/bash
# ============================================================================
# PROD_SUB2_NIGHT — orchestrator PRODUCTION SUB2 trên server L40S (đêm 19/07).
# Tự chia 2 GPU nếu GPU1 rảnh (CUDA_VISIBLE_DEVICES tách luồng):
#   GPU_HCM: 5 scene HCM  (3DGUT raw-distortion, cap 6M, 3 seed)   ~11-12h
#   GPU_OBJ: bonsai/chair (classic SH4,          cap 3M, 3 seed)   ~4-5h
# GPU1 bận (VLLM) → tự chạy TUẦN TỰ hết trên GPU0 (~16h, vẫn xong trước trưa).
# Xong cả 2 → tự FINALIZE (mean seed+members → vgg → zip q96 + contact sheet).
#
# Chạy trên server:  bash tools/PROD_SUB2_NIGHT.sh 2>&1 | tee /tmp/prod_night.txt
# Env: SUBTAG=2 · SEEDS="42 7 123" · PRUNE_CKPT=1 (mặc định BẬT trên server —
#      xoá ckpt seed 7/123 sau khi render verify đủ ảnh; seed 42 giữ cho enhancer.
#      Muốn giữ hết: PRUNE_CKPT=0) · MIN_FREE_GB=25
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
PY=.venv/bin/python
SUBTAG=${SUBTAG:-2}
SEEDS=${SEEDS:-"42 7 123"}
PRUNE_CKPT=${PRUNE_CKPT:-1}
MIN_FREE_GB=${MIN_FREE_GB:-25}
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }

say "0. tiên quyết (fail sớm trước khi đụng GPU)"
[ -x "$PY" ] || die "thiếu .venv"
$PY -c "from pycolmap import SceneManager" 2>/dev/null || die ".venv hỏng pycolmap rmbrualla (DOC3 §3.7)"
command -v tmux >/dev/null || die "thiếu tmux"
echo "  git HEAD: $(git log --oneline -1 2>/dev/null || echo '?')"
git fetch --dry-run 2>/dev/null | grep -q . && echo "  ⚠ origin có commit mới — cân nhắc git pull trước"

# data + workspace đủ 7 scene (kiểm FILE thật)
for s in bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  [ -s "workspace_r2/$s/sparse/0/images.bin" ] || die "thiếu workspace_r2/$s → bash tools/prepare_r2.sh trước"
done
echo "  ✅ workspace_r2 đủ 7 scene"

# đĩa: ước tính nhu cầu — HCM 6M ckpt ~1.5GB, OBJ 3M ~1GB; renders ~0.2GB/dir
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
N_SEED=$(echo $SEEDS | wc -w)
if [ "$PRUNE_CKPT" = "1" ]; then NEED=$((5*2 + 2*1 + 10)); else NEED=$((5*2*N_SEED + 2*N_SEED + 10)); fi
echo "  đĩa trống ${FREE}GB · ước tính cần ~${NEED}GB (PRUNE_CKPT=$PRUNE_CKPT)"
[ "$FREE" -lt "$MIN_FREE_GB" ] && die "đĩa ${FREE}GB < MIN_FREE_GB=${MIN_FREE_GB} — dọn trước (pip cache purge / renders cũ đã chốt verdict)"
[ "$FREE" -lt "$NEED" ] && echo "  ⚠ trống ${FREE} < ước tính ${NEED} — PRUNE_CKPT=1 là bắt buộc, theo dõi sát"

# GPU: GPU1 rảnh nếu used < 2000MiB
mapfile -t GPUMEM < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "  GPU0 used=${GPUMEM[0]:-?}MiB · GPU1 used=${GPUMEM[1]:-N/A}MiB"
TWO_GPU=0
[ -n "${GPUMEM[1]:-}" ] && [ "${GPUMEM[1]}" -lt 2000 ] && TWO_GPU=1

say "1. khởi động production (SUBTAG=$SUBTAG · SEEDS=[$SEEDS] · PRUNE_CKPT=$PRUNE_CKPT)"
tmux kill-session -t r2hcm 2>/dev/null || true
tmux kill-session -t r2obj 2>/dev/null || true
ENVSTR="SUBTAG=$SUBTAG SEEDS='$SEEDS' PRUNE_CKPT=$PRUNE_CKPT OBJ_SH_DEGREE=4 CAP_HCM=6000000 CAP_OBJ=3000000"
if [ "$TWO_GPU" = "1" ]; then
  echo "  → 2 GPU song song: GPU0=HCM · GPU1=OBJ"
  tmux new -d -s r2hcm "CUDA_VISIBLE_DEVICES=0 $ENVSTR SCENES_OBJ='' bash tools/run_sub_r2.sh 2>&1 | tee /tmp/prod_hcm.txt"
  tmux new -d -s r2obj "CUDA_VISIBLE_DEVICES=1 $ENVSTR SCENES_HCM='' bash tools/run_sub_r2.sh 2>&1 | tee /tmp/prod_obj.txt"
else
  echo "  → GPU1 bận: chạy TUẦN TỰ toàn bộ trên GPU0 (OBJ trước — ngắn, fail sớm nếu có bug)"
  tmux new -d -s r2hcm "CUDA_VISIBLE_DEVICES=0 $ENVSTR SCENES_HCM='' bash tools/run_sub_r2.sh 2>&1 | tee /tmp/prod_obj.txt && CUDA_VISIBLE_DEVICES=0 $ENVSTR SCENES_OBJ='' bash tools/run_sub_r2.sh 2>&1 | tee /tmp/prod_hcm.txt"
fi
sleep 5; tmux ls
echo "  theo dõi: tmux attach -t r2hcm (Ctrl+B rồi D để thoát) · tail -f /tmp/prod_hcm.txt /tmp/prod_obj.txt"

say "2. chờ xong (poll 5ph/lần, in tiến độ + canh đĩa)"
while tmux has-session -t r2hcm 2>/dev/null || tmux has-session -t r2obj 2>/dev/null; do
  sleep 300
  FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
  P_H=$(grep -ac "SCENE-OK" /tmp/prod_hcm.txt 2>/dev/null || echo 0)
  P_O=$(grep -ac "SCENE-OK" /tmp/prod_obj.txt 2>/dev/null || echo 0)
  echo "[$(date +%H:%M)] scene xong: HCM $P_H/5 · OBJ $P_O/2 · đĩa ${FREE}GB"
  [ "$FREE" -lt 6 ] && echo "  🚨 ĐĨA SẮP ĐẦY (${FREE}GB) — job sẽ tự die ở disk_guard; dọn NGAY renders/ckpt đã chốt verdict"
done

# kiểm cả 2 log kết thúc SẠCH (không die giữa chừng)
for f in /tmp/prod_hcm.txt /tmp/prod_obj.txt; do
  [ -s "$f" ] || continue
  grep -aq "❌" "$f" && { echo "⚠ $f có lỗi — ĐỌC LOG, sửa xong chạy lại script này (resume-safe, tự bỏ qua phần xong)"; FAIL=1; }
done
[ "${FAIL:-0}" = "1" ] && die "production chưa sạch — chưa finalize"

say "3. FINALIZE — mean(seed+members) → vgg → zip q96 + contact sheet"
SUBTAG=$SUBTAG MIN_SEEDS=$N_SEED bash tools/FINALIZE_SUB2.sh 2>&1 | tee /tmp/finalize_sub${SUBTAG}.txt || die "finalize"

say "4. tổng kết"
ls -la --time-style=full-iso submission_R2_SUB${SUBTAG}.zip 2>/dev/null || echo "⚠ không thấy zip?"
df -h . | tail -1
echo
echo "########################################################################"
echo "#  PROD-NIGHT-DONE: soi contact sheet /tmp/sheets_sub${SUBTAG}/*.jpg BẰNG MẮT"
echo "#  (7 scene × thumbnail — round 2 không có GT, mắt là lưới cuối) RỒI MỚI NỘP."
echo "#  scp zip về local để nộp. Điểm dự phóng nội bộ ≈ 78-79 (xem experiments.csv)."
echo "########################################################################"
