#!/bin/bash
# ============================================================================
# B3 — TRANSIENT MASKING (ĐỘC LẬP với GATE_A — chạy được song song GPU khác)
# Phố HCM/HN đông: xe/người di chuyển giữa các frame (cách ~1s) → supervision
# mâu thuẫn → ghost/blur mặt đường. Mask khỏi loss (zero-gradient, đã patch trainer).
# ~1h40 (mask ~10ph + train 12M ~90ph). MỐC: standard@12M plain = 0.75587.
# Chạy:  export CUDA_VISIBLE_DEVICES=<gpu rảnh>
#        tmux new -s gb3 && bash tools/GATE_B3.sh 2>&1 | tee /tmp/gate_b3.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ===== $* ====="; }
die(){ echo "❌ $*"; exit 1; }

say "0. tiên quyết"
[ -x "$PY" ] || die "thiếu .venv"
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 15 ] && die "đĩa ${FREE}GB<15"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
echo "  ✅ ok · đĩa ${FREE}GB · dùng GPU $CUDA_VISIBLE_DEVICES"

say "1. sinh transient masks (deeplabv3, ~10ph — tải weights 160MB lần đầu)"
if [ "$(ls workspace_raw/$S/transient_masks 2>/dev/null | wc -l)" -lt 200 ]; then
  $PY tools/make_transient_masks.py --workspace "workspace_raw/$S" --vis 3 || die "make masks"
else echo "  ⏩ masks đã có"; fi
echo "  → soi mắt: workspace_raw/$S/transient_vis/*_vis.jpg (đỏ = bị mask)"

say "2. train 12M VỚI transient mask (~90ph)"
if ! [ -s "results/${S}__tmask12M/ckpts/ckpt_29999_rank0.pt" ]; then
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
    --result-dir "$PWD/results/${S}__tmask12M" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max 12000000 --eval-steps 30000 --save-steps 30000 \
    2>&1 | tee /tmp/b3_train.log || die "train tmask"
  rm -f results/${S}__tmask12M/ckpts/ckpt_14999_rank0.pt; rm -rf results/${S}__tmask12M/videos
fi

say "3. render + score"
$PY tools/render_test_poses.py --ckpt "results/${S}__tmask12M/ckpts/ckpt_29999_rank0.pt" \
  --csv "$CSV" --out "renders/${S}__tmask12M" --data_dir "workspace_raw/$S" \
  --antialiased --with_ut --radial_k1 $K1 2>&1 | tee /tmp/b3_render.log
V=$($PY tools/score_local.py --pred_dir "renders/${S}__tmask12M" --gt_dir "$GT" 2>&1 \
  | tee /tmp/b3_score.txt | grep -aE "n=|★")
echo "$V"

echo
echo "########################################################################"
echo "#  VERDICT B3 — so với plain12M = 0.75587"
echo "#  Δ ≥ +0.002 → BẬT transient-mask cho production (bắt buộc từ lúc train)"
echo "#  Δ ≤ 0       → bỏ, xe/người không phải nguồn lỗi đáng kể ở scene này"
echo "########################################################################"
echo "DÁN KHỐI NÀY CHO CLAUDE."
