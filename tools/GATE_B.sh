#!/bin/bash
# ============================================================================
# GATE_B — B1 (restoration prior) + B2 (pose-refine) + B3 (transient-mask), TUẦN TỰ.
# Chạy SAU GATE_A (B1 cần renders/HCM0204__ens12 + ckpt cap12M từ A). ~4h10.
# Chạy: tmux new -s gb && bash tools/GATE_B.sh 2>&1 | tee /tmp/gate_b.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

say "0. tiên quyết"
[ -x "$PY" ] || die "thiếu .venv"
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 30 ] && die "đĩa ${FREE}GB<30"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
echo "  ✅ ok · đĩa ${FREE}GB · GPU $CUDA_VISIBLE_DEVICES"

# ============================== B1 — RESTORATION PRIOR ==============================
say "B1. restoration-prior (encoder VGG16 pretrained) trên ens12 (~40ph)"
if [ -d "renders/${S}__ens12" ]; then
  if ! [ -s results/${S}__b1vgg/net.pt ]; then
    $PY tools/enhance_net.py train --workspace "workspace_raw/$S" \
      --ckpt "results/${S}__cap12M/ckpts/ckpt_29999_rank0.pt" \
      --out results/${S}__b1vgg/net.pt --with_ut --radial_k1 $K1 \
      --arch vgg --steps 8000 --patch 320 2>&1 | tee /tmp/b1_train.log \
      || die "B1 train"
  fi
  $PY tools/enhance_net.py apply --net results/${S}__b1vgg/net.pt \
    --in_dir renders/${S}__ens12 --out_dir renders/${S}__b1 >/dev/null || die "B1 apply"
  score renders/${S}__b1 "B1-vgg-prior"
  echo "  MỐC A1 (ens12+L16-XL, xem /tmp/gate_a.txt) → Δ≥+0.002 thì thay L16-XL bằng B1"
else
  echo "  ⏩ thiếu renders/${S}__ens12 (GATE_A chưa chạy/chưa tới A1) — BỎ QUA B1, làm B2/B3 trước"
fi

# ============================== B2 — POSE REFINE ==============================
say "B2.0 pycolmap"
$PY -c "import pycolmap" 2>/dev/null || { echo "  → cài pycolmap..."; $PY -m pip install -q pycolmap || die "pip pycolmap fail"; }
$PY -c "
import pycolmap
try: v = pycolmap.__version__
except AttributeError:
    import importlib.metadata
    try: v = importlib.metadata.version('pycolmap')
    except Exception: v = '?'
print('  ✅ pycolmap', v)
"

say "B2.1 refine (BA + neo gauge) — ĐỌC KỸ dòng in ra"
if [ -s "workspace_ref/$S/sparse/0/images.bin" ]; then
  echo "  ⏩ workspace_ref/$S đã có"
else
  $PY tools/pose_refine.py --in_ws "workspace_raw/$S" --out_ws "workspace_ref/$S" 2>&1 \
    | tee /tmp/b2_refine.txt || die "pose_refine fail — dán /tmp/b2_refine.txt"
  grep -q "⚠ drift lớn" /tmp/b2_refine.txt && die "gauge drift lớn — DỪNG, dán /tmp/b2_refine.txt"
fi
K1R=$(grep -oE "k1=[+-][0-9.]+" /tmp/b2_refine.txt 2>/dev/null | head -1 | cut -d= -f2)
K2R=$(grep -oE "k2=[+-][0-9.]+" /tmp/b2_refine.txt 2>/dev/null | head -1 | cut -d= -f2)
P1R=$(grep -oE "p1=[+-][0-9.]+" /tmp/b2_refine.txt 2>/dev/null | head -1 | cut -d= -f2)
P2R=$(grep -oE "p2=[+-][0-9.]+" /tmp/b2_refine.txt 2>/dev/null | head -1 | cut -d= -f2)
echo "  render B2 sẽ dùng: k1=$K1R k2=$K2R p1=$P1R p2=$P2R"

say "B2.2 train 12M trên workspace REFINED (~90ph)"
if ! [ -s "results/${S}__ref12M/ckpts/ckpt_29999_rank0.pt" ]; then
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_ref/$S" --data-factor 1 \
    --result-dir "$PWD/results/${S}__ref12M" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max 12000000 --eval-steps 30000 --save-steps 30000 \
    2>&1 | tee /tmp/b2_train.log || die "train ref"
  rm -f results/${S}__ref12M/ckpts/ckpt_14999_rank0.pt; rm -rf results/${S}__ref12M/videos
fi

say "B2.3 render (test poses gốc, intrinsics refined) + score"
$PY tools/render_test_poses.py --ckpt "results/${S}__ref12M/ckpts/ckpt_29999_rank0.pt" \
  --csv "$CSV" --out "renders/${S}__ref12M" --data_dir "workspace_ref/$S" \
  --antialiased --with_ut --radial_k1 "$K1R" --radial_k2 "$K2R" --tangential "$P1R" "$P2R" \
  2>&1 | tee /tmp/b2_render.log
score renders/${S}__ref12M "B2"

# ============================== B3 — TRANSIENT MASK ==============================
say "B3.1 sinh transient masks (deeplabv3, ~10ph — tải weights 160MB lần đầu)"
if [ "$(ls workspace_raw/$S/transient_masks 2>/dev/null | wc -l)" -lt 200 ]; then
  $PY tools/make_transient_masks.py --workspace "workspace_raw/$S" --vis 3 || die "make masks"
else echo "  ⏩ masks đã có"; fi
echo "  → soi mắt: workspace_raw/$S/transient_vis/*_vis.jpg (đỏ = bị mask)"

say "B3.2 train 12M VỚI transient mask (~90ph)"
if ! [ -s "results/${S}__tmask12M/ckpts/ckpt_29999_rank0.pt" ]; then
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
    --result-dir "$PWD/results/${S}__tmask12M" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max 12000000 --eval-steps 30000 --save-steps 30000 \
    2>&1 | tee /tmp/b3_train.log || die "train tmask"
  rm -f results/${S}__tmask12M/ckpts/ckpt_14999_rank0.pt; rm -rf results/${S}__tmask12M/videos
fi

say "B3.3 render + score"
$PY tools/render_test_poses.py --ckpt "results/${S}__tmask12M/ckpts/ckpt_29999_rank0.pt" \
  --csv "$CSV" --out "renders/${S}__tmask12M" --data_dir "workspace_raw/$S" \
  --antialiased --with_ut --radial_k1 $K1 2>&1 | tee /tmp/b3_render.log
score renders/${S}__tmask12M "B3"

echo
echo "########################################################################"
echo "#  VERDICT GATE_B — mốc plain12M = 0.75587 · mốc A1 = xem /tmp/gate_a.txt"
echo "#  [B1] ≥ [A1]+0.002 → thay L16-XL bằng B1 cho production"
echo "#  [B2] ≥ 0.75587+0.004 → ĐÒN LỚN: refine toàn scene trước production"
echo "#  [B2] 0.75587+0.001..+0.004 → dùng, cộng dồn miễn phí"
echo "#  [B3] ≥ 0.75587+0.002 → BẬT transient-mask cho production"
echo "#  B2+B3 cộng dồn được (sửa pose khác sửa loss, trực giao)"
echo "########################################################################"
echo "DÁN [B1] [B2] [B3] + khối này CHO CLAUDE."
