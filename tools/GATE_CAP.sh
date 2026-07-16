#!/bin/bash
# ============================================================================
# CỔNG: standard cap CAO (8M/12M) — L40S mở khoá thứ 4060 chặn ở 5M.
# Trả lời: có cần NHT không? Mốc: NHT@5M=0.75102 · standard@5M=0.74727
#   standard@12M ≥ 0.751  → BỎ NHT (pipeline đơn giản thắng) → dồn giờ cho §9
#   standard@12M < 0.751  → NHT còn cửa, cân nhắc
# Chạy: bash tools/GATE_CAP.sh 2>&1 | tee /tmp/gate_cap.txt     (~2-4h, dùng tmux)
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SCENE=HCM0204
CSV="VAI_NVS_DATA/phase1/public_set/$SCENE/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$SCENE/test/images"
K1=$(.venv/bin/python - "$SCENE" <<'EOF'
import struct,sys
with open(f"workspace_raw/{sys.argv[1]}/sparse/0/cameras.bin","rb") as f:
    f.read(8); struct.unpack("<iiQQ",f.read(24)); print(repr(struct.unpack("<dddd",f.read(32))[3]))
EOF
)
say(){ echo; echo "===== $* ====="; }
die(){ echo; echo "❌ $*"; exit 1; }

say "0. tiên quyết"
[ -x .venv/bin/python ] || die "thiếu .venv"
[ -s "workspace_raw/$SCENE/sparse/0/images.bin" ] || die "thiếu workspace_raw/$SCENE (chạy GATE_NHT trước, nó tự prepare)"
[ "$(ls "$GT" | wc -l)" -ge 60 ] || die "thiếu GT"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 25 ] && die "đĩa ${FREE}GB < 25GB"
echo "  ✅ ok — đĩa ${FREE}GB · k1=$K1"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'

for CAP in 8000000 12000000; do
  TAG="cap$((CAP/1000000))M"
  say "TRAIN standard $TAG (30k steps)"
  if [ -s "results/${SCENE}__${TAG}/ckpts/ckpt_29999_rank0.pt" ]; then echo "  ⏩ đã có"; else
    .venv/bin/python gsplat/examples/simple_trainer.py mcmc \
      --data-dir "workspace_raw/$SCENE" --data-factor 1 \
      --result-dir "$PWD/results/${SCENE}__${TAG}" \
      --max-steps 30000 --test-every 999999 --disable-viewer --antialiased \
      --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max $CAP --eval-steps 30000 --save-steps 30000 2>&1 | tee /tmp/gc_train.log \
      || { echo "  ❌ train $TAG thất bại (OOM? xem trên)"; continue; }
  fi
  say "RENDER + SCORE $TAG"
  .venv/bin/python tools/render_test_poses.py \
    --ckpt "results/${SCENE}__${TAG}/ckpts/ckpt_29999_rank0.pt" --csv "$CSV" \
    --out "renders/${SCENE}__${TAG}" --data_dir "workspace_raw/$SCENE" \
    --antialiased --with_ut --radial_k1 "$K1" 2>&1 | tee /tmp/gc_render.log
  .venv/bin/python tools/score_local.py --pred_dir "renders/${SCENE}__${TAG}" --gt_dir "$GT" \
    2>&1 | grep -aE "n=|★" | sed "s/^/  [$TAG] /"
  rm -f "results/${SCENE}__${TAG}/ckpts/ckpt_14999_rank0.pt"
done

echo
echo "########################################################################"
echo "#  VERDICT — CỔNG CAP CAO (HCM0204 competition)"
echo "########################################################################"
echo "  MỐC:  standard@5M = 0.74727   |   NHT@5M = 0.75102"
echo "  → so 2 dòng ★ ở trên (cap8M / cap12M) với 0.75102:"
echo "     ≥ 0.751  → BỎ NHT, dùng standard cap cao (đơn giản hơn, rẻ hơn)"
echo "     < 0.751  → NHT còn giá trị, cân nhắc giữ"
echo "########################################################################"
echo "DÁN KHỐI NÀY + 2 DÒNG ★ CHO CLAUDE."
