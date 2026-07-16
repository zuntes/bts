#!/bin/bash
# ============================================================================
# CỔNG P4 — BILATERAL GRID Ở TẦNG TRAIN (docs/12 §9-P4)
#
# Ý tưởng: mỗi ảnh drone lệch exposure/WB. 3DGS không biết → nhồi sai lệch đó vào
# MÀU gaussian → model bẩn. Bilateral grid học biến đổi màu RIÊNG cho từng ảnh train,
# hút hết biến thiên exposure vào grid → gaussian giữ màu sạch. Render test thì BỎ grid
# → ảnh canonical sạch hơn.
#
# Bằng chứng còn tín hiệu: L16-XL ăn +0.004 CHỈ trên scene lệch exposure.
# P4 trị ở gốc (train) thay vì vá ở hậu xử lý (L16 vẫn chồng lên được sau).
#
# A/B: standard@12M (mốc 0.75587 — cap production) vs standard@12M + bilateral grid.
# Chạy: bash tools/GATE_P4.sh 2>&1 | tee /tmp/gate_p4.txt      (~2-3h, tmux)
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCENE=HCM0204; CAP=12000000; BASE=0.75587   # cap production (GATE_CAP) — A/B phải ở ĐÚNG cap sẽ dùng thật
CSV="VAI_NVS_DATA/phase1/public_set/$SCENE/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$SCENE/test/images"
K1=$(.venv/bin/python - "$SCENE" <<'EOF'
import struct,sys
with open(f"workspace_raw/{sys.argv[1]}/sparse/0/cameras.bin","rb") as f:
    f.read(8); struct.unpack("<iiQQ",f.read(24)); print(repr(struct.unpack("<dddd",f.read(32))[3]))
EOF
)
die(){ echo; echo "❌ $*"; exit 1; }
echo "===== 0. tiên quyết ====="
[ -x .venv/bin/python ] || die "thiếu .venv"
[ -s "workspace_raw/$SCENE/sparse/0/images.bin" ] || die "thiếu workspace_raw/$SCENE"
[ "$(ls "$GT" | wc -l)" -ge 60 ] || die "thiếu GT"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 15 ] && die "đĩa ${FREE}GB<15"
echo "  ✅ ok · đĩa ${FREE}GB · k1=$K1"

TAG=bilagrid12M
echo; echo "===== 1. TRAIN standard@12M + BILATERAL GRID (~90 phút) ====="
if [ -s "results/${SCENE}__${TAG}/ckpts/ckpt_29999_rank0.pt" ]; then echo "  ⏩ đã có"; else
  .venv/bin/python gsplat/examples/simple_trainer.py mcmc \
    --data-dir "workspace_raw/$SCENE" --data-factor 1 \
    --result-dir "$PWD/results/${SCENE}__${TAG}" \
    --max-steps 30000 --test-every 999999 --disable-viewer --antialiased \
    --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max $CAP --eval-steps 30000 --save-steps 30000 \
    --use-bilateral-grid 2>&1 | tee /tmp/p4_train.log || die "train thất bại (flag --use-bilateral-grid sai tên? xem /tmp/p4_train.log)"
fi
[ -s "results/${SCENE}__${TAG}/ckpts/ckpt_29999_rank0.pt" ] || die "không thấy ckpt"

echo; echo "===== 2. RENDER (KHÔNG áp grid → canonical) + SCORE ====="
# render_test_poses.py không biết tới grid → đúng ý đồ: lấy màu canonical sạch
.venv/bin/python tools/render_test_poses.py \
  --ckpt "results/${SCENE}__${TAG}/ckpts/ckpt_29999_rank0.pt" --csv "$CSV" \
  --out "renders/${SCENE}__${TAG}" --data_dir "workspace_raw/$SCENE" \
  --antialiased --with_ut --radial_k1 "$K1" 2>&1 | tail -2
.venv/bin/python tools/score_local.py --pred_dir "renders/${SCENE}__${TAG}" --gt_dir "$GT" \
  2>&1 | tee /tmp/p4_score.txt | grep -aE "n=|★"
V=$(grep -aoE "Score_BTC\[[^]]*\] = [0-9.]+" /tmp/p4_score.txt | grep -oE "[0-9]+\.[0-9]+" | tail -1)

echo
echo "########################################################################"
echo "#  VERDICT — CỔNG P4 (bilateral grid)"
echo "########################################################################"
echo "  standard@12M + bilagrid : v50 = ${V:-ERR}"
echo "  standard@12M (mốc)      : v50 = $BASE"
[ -n "$V" ] && .venv/bin/python -c "
d=float('$V')-float('$BASE')
print(f'  Δ = {d:+.5f}  ({d*100:+.2f} điểm BTC)')
print()
print('  ✅ ĂN → dùng cho SUB7, và L16 vẫn chồng lên được (trực giao)' if d>=0.002 else
      ('  🟡 mỏng → chỉ dùng cho scene lệch exposure cao (HCM1439, HCM0249)' if d>0 else
       '  ❌ KHÔNG ăn → bỏ P4, exposure đã được L16 xử lý đủ'))"
echo "########################################################################"
echo "DÁN KHỐI NÀY CHO CLAUDE."
