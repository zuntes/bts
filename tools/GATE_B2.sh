#!/bin/bash
# ============================================================================
# B2 — POSE/INTRINSICS REFINE GAUGE-LOCKED (đòn biên độ lớn nhất, rủi ro cao nhất)
# SIMPLE_RADIAL 1-tham-số → OPENCV 8-tham-số + BA, neo gauge Umeyama về frame gốc.
# ~2h (BA vài phút CPU + train 12M 90ph). MỐC: plain12M = 0.75587.
# Chạy: tmux new -s gb2 && bash tools/GATE_B2.sh 2>&1 | tee /tmp/gate_b2.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=HCM0204
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ===== $* ====="; }
die(){ echo "❌ $*"; exit 1; }

say "0. pycolmap"
$PY -c "import pycolmap" 2>/dev/null || { echo "  → cài pycolmap..."; $PY -m pip install -q pycolmap || die "pip pycolmap fail"; }
$PY -c "import pycolmap; print('  ✅ pycolmap', pycolmap.__version__)"

say "1. refine (BA + neo gauge) — ĐỌC KỸ các dòng in ra"
$PY tools/pose_refine.py --in_ws "workspace_raw/$S" --out_ws "workspace_ref/$S" 2>&1 \
  | tee /tmp/b2_refine.txt || die "pose_refine fail — dán /tmp/b2_refine.txt"
grep -q "⚠ drift lớn" /tmp/b2_refine.txt && die "gauge drift lớn — DỪNG, dán /tmp/b2_refine.txt cho Claude"
# lấy intrinsics refined cho render
K1=$(grep -oE "k1=[+-][0-9.]+" /tmp/b2_refine.txt | head -1 | cut -d= -f2)
K2=$(grep -oE "k2=[+-][0-9.]+" /tmp/b2_refine.txt | head -1 | cut -d= -f2)
P1=$(grep -oE "p1=[+-][0-9.]+" /tmp/b2_refine.txt | head -1 | cut -d= -f2)
P2=$(grep -oE "p2=[+-][0-9.]+" /tmp/b2_refine.txt | head -1 | cut -d= -f2)
echo "  render sẽ dùng: k1=$K1 k2=$K2 p1=$P1 p2=$P2"

say "2. train 12M trên workspace REFINED (~90ph)"
if ! [ -s "results/${S}__ref12M/ckpts/ckpt_29999_rank0.pt" ]; then
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_ref/$S" --data-factor 1 \
    --result-dir "$PWD/results/${S}__ref12M" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max 12000000 --eval-steps 30000 --save-steps 30000 \
    2>&1 | tee /tmp/b2_train.log || die "train ref"
  rm -f results/${S}__ref12M/ckpts/ckpt_14999_rank0.pt; rm -rf results/${S}__ref12M/videos
fi

say "3. render tại test poses GỐC (frame đã neo) + intrinsics refined + score"
$PY tools/render_test_poses.py --ckpt "results/${S}__ref12M/ckpts/ckpt_29999_rank0.pt" \
  --csv "$CSV" --out "renders/${S}__ref12M" --data_dir "workspace_ref/$S" \
  --antialiased --with_ut --radial_k1 "$K1" --radial_k2 "$K2" --tangential "$P1" "$P2" \
  2>&1 | tee /tmp/b2_render.log
$PY tools/score_local.py --pred_dir "renders/${S}__ref12M" --gt_dir "$GT" 2>&1 | grep -aE "n=|★" | sed 's/^/  [B2] /'

echo
echo "########################################################################"
echo "#  VERDICT B2:  so [B2] với plain12M = 0.75587"
echo "#   ≥ +0.004 → ĐÒN LỚN: refine TOÀN BỘ scene trước production (BA rẻ, chỉ +vài phút/scene)"
echo "#   +0.001..+0.004 → dùng, cộng dồn miễn phí"
echo "#   ≤ 0 → pose BTC vốn đã sạch/gauge lệch → bỏ, đã loại được nghi phạm gốc rễ"
echo "########################################################################"
echo "DÁN /tmp/b2_refine.txt (các dòng reprojection/gauge) + dòng [B2] CHO CLAUDE."
