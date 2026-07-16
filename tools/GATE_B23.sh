#!/bin/bash
# ============================================================================
# GATE_B23 — B2 (pose-refine) + B3 (transient-mask) GỘP 1 SCRIPT, chạy TUẦN TỰ.
# Cả 2 độc lập với GATE_A (không cần renders/ens12) — dùng khi chỉ còn 1 GPU
# (chạy chung với GATE_A trên CÙNG GPU). Cảnh báo: 2 job cùng GPU chia sẻ compute
# → mỗi job chậm hơn ~1.5-2× so với chạy riêng, nhưng vẫn tiến được thay vì đợi không.
# ~4h (B2 ~2h + B3 ~1h40, cộng thêm chậm do share GPU với GATE_A).
# Chạy: tmux new -s gb23 && bash tools/GATE_B23.sh 2>&1 | tee /tmp/gate_b23.txt
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

say "0. tiên quyết"
[ -x "$PY" ] || die "thiếu .venv"
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 30 ] && die "đĩa ${FREE}GB<30 (2 job × ~2.6GB ckpt + data)"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
echo "  ✅ ok · đĩa ${FREE}GB · dùng GPU $CUDA_VISIBLE_DEVICES (CHUNG với GATE_A nếu đang chạy — sẽ chậm hơn bình thường)"

# ============================== PHẦN B2 — POSE REFINE ==============================
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

say "B2.2 train 12M trên workspace REFINED (~90ph, chậm hơn nếu share GPU)"
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
$PY tools/score_local.py --pred_dir "renders/${S}__ref12M" --gt_dir "$GT" 2>&1 \
  | tee /tmp/b2_score.txt | grep -aE "n=|★" | sed 's/^/  [B2] /'

# ============================== PHẦN B3 — TRANSIENT MASK ==============================
say "B3.1 sinh transient masks (deeplabv3, ~10ph — tải weights 160MB lần đầu)"
if [ "$(ls workspace_raw/$S/transient_masks 2>/dev/null | wc -l)" -lt 200 ]; then
  $PY tools/make_transient_masks.py --workspace "workspace_raw/$S" --vis 3 || die "make masks"
else echo "  ⏩ masks đã có"; fi
echo "  → soi mắt: workspace_raw/$S/transient_vis/*_vis.jpg (đỏ = bị mask)"

say "B3.2 train 12M VỚI transient mask (~90ph, chậm hơn nếu share GPU)"
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
$PY tools/score_local.py --pred_dir "renders/${S}__tmask12M" --gt_dir "$GT" 2>&1 \
  | tee /tmp/b3_score.txt | grep -aE "n=|★" | sed 's/^/  [B3] /'

echo
echo "########################################################################"
echo "#  VERDICT B2+B3 — so cả 2 với standard@12M plain = 0.75587"
echo "#  [B2] ≥ +0.004 → ĐÒN LỚN: refine toàn scene trước production"
echo "#  [B2] +0.001..+0.004 → dùng, cộng dồn miễn phí"
echo "#  [B3] ≥ +0.002 → BẬT transient-mask cho production"
echo "#  Cả 2 dương → CỘNG ĐƯỢC (B2 sửa pose, B3 sửa loss — trực giao)"
echo "########################################################################"
echo "DÁN [B2] + [B3] + khối này CHO CLAUDE."
