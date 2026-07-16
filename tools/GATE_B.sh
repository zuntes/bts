#!/bin/bash
# ============================================================================
# PHASE B (2 cổng nhanh) — chạy SAU GATE_A (cần renders/HCM0204__ens12 + mốc A1).
#   B3 transient-mask : mask xe/người → train 12M masked → render → score (MỐC: plain12M 0.75587)
#   B1 restoration    : enhance-net encoder VGG16-pretrained trên ens12 → score (MỐC: A1)
# ~3h. Chạy: tmux new -s gb && bash tools/GATE_B.sh 2>&1 | tee /tmp/gate_b.txt
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
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

say "B3.1 sinh transient masks (deeplabv3, ~10ph — tải weights 160MB lần đầu)"
if [ "$(ls workspace_raw/$S/transient_masks 2>/dev/null | wc -l)" -lt 200 ]; then
  $PY tools/make_transient_masks.py --workspace "workspace_raw/$S" --vis 3 || die "make masks"
else echo "  ⏩ masks đã có"; fi
echo "  → soi mắt 3 ảnh workspace_raw/$S/transient_vis/*_vis.jpg (đỏ = bị mask)"

say "B3.2 train 12M VỚI transient mask (~90ph — dataset tự thấy transient_masks/)"
if ! [ -s "results/${S}__tmask12M/ckpts/ckpt_29999_rank0.pt" ]; then
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
    --result-dir "$PWD/results/${S}__tmask12M" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max 12000000 --eval-steps 30000 --save-steps 30000 \
    2>&1 | tee /tmp/b3_train.log || die "train tmask"
  rm -f results/${S}__tmask12M/ckpts/ckpt_14999_rank0.pt; rm -rf results/${S}__tmask12M/videos
fi
$PY tools/render_test_poses.py --ckpt "results/${S}__tmask12M/ckpts/ckpt_29999_rank0.pt" \
  --csv "$CSV" --out "renders/${S}__tmask12M" --data_dir "workspace_raw/$S" \
  --antialiased --with_ut --radial_k1 $K1 2>&1 | tee /tmp/b3_render.log
score renders/${S}__tmask12M "B3-tmask12M"
echo "  MỐC plain12M = 0.75587 → Δ ≥ +0.002 thì BẬT mask cho production"

say "B1 restoration-prior (encoder VGG16 pretrained) trên ens12 (~40ph)"
[ -d "renders/${S}__ens12" ] || die "thiếu renders/${S}__ens12 — chạy GATE_A trước"
if ! [ -s results/${S}__b1vgg/net.pt ]; then
  $PY tools/enhance_net.py train --workspace "workspace_raw/$S" \
    --ckpt "results/${S}__cap12M/ckpts/ckpt_29999_rank0.pt" \
    --out results/${S}__b1vgg/net.pt --with_ut --radial_k1 $K1 \
    --arch vgg --steps 8000 --patch 320 2>&1 | grep -aE "VAL-GAIN|val BASE|step [0-9]+000:" \
    || die "B1 train"
fi
$PY tools/enhance_net.py apply --net results/${S}__b1vgg/net.pt \
  --in_dir renders/${S}__ens12 --out_dir renders/${S}__b1 >/dev/null || die "B1 apply"
score renders/${S}__b1 "B1-vgg-prior"
echo "  MỐC A1 (ens12 + L16-XL) = xem /tmp/gate_a.txt → Δ ≥ +0.002 thì thay L16-XL bằng B1"

echo
echo "########################################################################"
echo "#  VERDICT PHASE B-nhanh:  B3 so 0.75587 · B1 so [A1] trong gate_a.txt"
echo "#  B2 (pose refine) = cổng riêng: bash tools/GATE_B2.sh"
echo "########################################################################"
echo "DÁN các dòng [B*] + [A1] CHO CLAUDE."
