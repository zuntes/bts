#!/bin/bash
# ============================================================================
# B1 — RESTORATION PRIOR (encoder VGG16 pretrained). PHỤ THUỘC GATE_A (cần A1 xong
# → renders/HCM0204__ens12) — KHÔNG chạy song song A được. Khác B2/B3 (độc lập, xem đó).
# ~40ph. Chạy: tmux new -s gb1 && bash tools/GATE_B1.sh 2>&1 | tee /tmp/gate_b1.txt
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
echo "#  VERDICT B1: so với [A1] trong gate_a.txt"
echo "#  B2 (pose refine, độc lập) + B3 (transient mask, độc lập) = cổng riêng,"
echo "#  chạy song song GATE_A trên GPU khác: GATE_B2.sh · GATE_B3.sh"
echo "########################################################################"
echo "DÁN các dòng [B*] + [A1] CHO CLAUDE."
