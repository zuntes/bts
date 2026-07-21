#!/bin/bash
# ============================================================================
# GATE HCM-PUSH — các THAM SỐ CHƯA TUNE (trước đây bỏ oan) trên HCM0204 FULL-RES GT.
# HCM = 71% điểm, top-4=77.7 chỉ cách 1.2 → mọi +0.002 đáng. Bàn native GT thật.
#
# Mốc base full-res 6M = xem [base]. Test (mỗi cái so base, ≥+0.002 = port prod):
#   [P1] ssim_lambda 0.2→0.3   (loss align metric: metric 0.3·SSIM, train chỉ 0.2)
#   [P2] refine_stop 25k + 45k steps  (test STEP ĐÚNG — 60k cũ crippled refine dừng 25k)
#   [P3] cap 9M native  (knee full-res=12M, half=6M → native gốc/4 knee giữa, prod 6M under?)
#   [P4] ssim_lambda 0.5  (đẩy mạnh SSIM — metric-aware)
# Chạy server GPU1: tmux new -d -s push "CUDA_VISIBLE_DEVICES=1 bash tools/GATE_HCM_PUSH.sh 2>&1 | tee /tmp/gate_push.txt"
# Env: SEED=42
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
SEED=${SEED:-42}
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
gpu_busy(){ local g free tot thr; g=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  thr=$(( tot*55/100 )); [ "$thr" -gt 18000 ] && thr=18000; [ "${free:-0}" -lt "$thr" ]; }
for i in $(seq 1 300); do gpu_busy || break; [ "$i" = 1 ] && echo "chờ GPU..."; sleep 60; done
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S"
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>/dev/null | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

run(){  # $1=tag $2=cap $3=steps $4=extra_flags
  local tag=$1 cap=$2 steps=$3 extra=$4
  local res="results/push_${S}__${tag}" rend="renders_push/${tag}"
  local stop=$(( steps > 30000 ? steps*5/6 : 25000 ))   # refine_stop scale theo steps
  say "$tag — cap=$cap steps=$steps refine_stop=$stop $extra"
  FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 10 ] && die "đĩa ${FREE}GB"
  if ! [ -s "$res/ckpts/ckpt_$((steps-1))_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps "$steps" --test-every 999999 \
      --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$cap" --strategy.refine-stop-iter "$stop" \
      --eval-steps "$steps" --save-steps "$steps" --global-seed "$SEED" \
      $extra > "/tmp/push_${tag}.log" 2>&1
    [ -s "$res/ckpts/ckpt_$((steps-1))_rank0.pt" ] || { echo "  ⚠ fail:"; tail -3 "/tmp/push_${tag}.log"|sed 's/^/    /'; return 0; }
    rm -rf "$res/videos"
  else echo "  ⏩ ckpt có"; fi
  if [ "$(ls "$rend" 2>/dev/null | wc -l)" -lt 60 ]; then
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_$((steps-1))_rank0.pt" --csv "$CSV" \
      --out "$rend" --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 $K1 \
      2>&1 | grep -av "render " || die "render $tag"
  fi
  score "$rend" "$tag"
}

run base   6000000 30000 ""
run ssim03 6000000 30000 "--ssim-lambda 0.3"
run ssim05 6000000 30000 "--ssim-lambda 0.5"
run s45k   6000000 45000 ""
run cap9M  9000000 30000 ""

echo
echo "########################################################################"
echo "#  VERDICT HCM-PUSH (full-res GT, mốc [base]): ≥+0.002 → port prod."
echo "#  P1/P4 ssim_lambda: metric-align có ăn? · P2 s45k: STEP đúng cách có ăn?"
echo "#  P3 cap9M: native muốn thêm hạt? (nếu + → round-2 CAP_HCM nên 9M)"
echo "#  ⚠ HCM 71% → +0.002/scene = +0.7 tổng. top-4=77.7 chỉ cách 1.2."
echo "########################################################################"
echo "HCM-PUSH-GATE-DONE"
