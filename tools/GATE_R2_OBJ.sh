#!/bin/bash
# ============================================================================
# GATE R2-OBJ — chiến thuật cho 2 scene "lạ" bonsai/chair (pinhole, video, mờ).
# Bàn chấm: giữ lại 10% train làm val (make_holdout) — có GT thật, chấm v50 chuẩn BTC.
#
# Trả lời 3 câu hỏi:
#   [O1] cap 3M đủ chưa hay 6M hơn?     (scene nhỏ 54-80k điểm SfM, GT mờ → nghi 3M đủ)
#   [O2] L16-XL ăn bao nhiêu trên scene mờ? (giả thuyết: NHIỀU hơn scene nét — học blur GT)
#   [O3] (RUN_UT=1, tuỳ chọn) 3DGUT-không-méo có hơn classic trên video scene?
#
# Chạy được trên 4060 local (~6-8h) hoặc server: bash tools/GATE_R2_OBJ.sh 2>&1 | tee /tmp/gate_r2obj.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ [ "$(ls "$1" 2>/dev/null | wc -l)" -lt 5 ] && { echo "  [$3] ⏭ chưa có render (train bị bỏ/OOM)"; return 0; }
         $PY tools/score_local.py --pred_dir "$1" --gt_dir "$2" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$3] /"; }

say "0. tiên quyết"
[ -x "$PY" ] || die "thiếu .venv"
for s in bonsai chair; do
  [ -s "workspace_r2/$s/sparse/0/images.bin" ] || die "thiếu workspace_r2/$s → bash tools/prepare_r2.sh"
done
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 8 ] && die "đĩa ${FREE}GB<8"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'

# ngân sách VRAM: bỏ QUA (không chết) biến thể có cap×pixel vượt sức GPU hiện tại.
# 4060 8GB: bonsai(1920×1080) 3M fit ~7GB, 6M OOM → ngưỡng ~7e12. Server 46GB: đặt CEIL cực lớn.
TOTAL_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
OBJ_MAX_CAPPX=${OBJ_MAX_CAPPX:-$([ "$TOTAL_VRAM" -gt 20000 ] && echo 200e12 || echo 7.2e12)}
px_of(){ $PY - "$1" <<'EOF'
import struct, sys
with open(f"workspace_r2v/{sys.argv[1]}/sparse/0/cameras.bin","rb") as f:
    f.read(8); cid,model,w,h=struct.unpack("<iiQQ",f.read(24)); print(w*h)
EOF
}

train_one(){  # $1=scene $2=tag $3=cap $4=extra_train_flags
  local s=$1 tag=$2 cap=$3 extra=${4:-}
  local res="results/r2v_${s}__${tag}"
  local px; px=$(px_of "$s")
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    if $PY -c "import sys; sys.exit(0 if $cap*$px > $OBJ_MAX_CAPPX else 1)"; then
      echo "  ⏭ BỎ QUA $s $tag: cap×px=$(($cap*$px/1000000000000))e12 > ngưỡng VRAM máy này → ĐỂ SERVER CHẠY (OBJ_MAX_CAPPX)"
      return 2
    fi
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2v/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased $extra \
      --strategy.cap-max "$cap" --eval-steps 30000 --save-steps 30000 --global-seed 42 \
      2>&1 | tee "/tmp/r2v_${s}_${tag}.log" || { echo "  ⚠ train $s $tag lỗi (OOM?) — bỏ qua, làm tiếp"; return 2; }
    rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
  else echo "  ⏩ $s $tag ckpt có"; fi
  if [ "$(ls "renders_r2v/${s}__${tag}" 2>/dev/null | wc -l)" -lt 5 ]; then
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
      --csv "workspace_r2v/$s/val_poses.csv" --out "renders_r2v/${s}__${tag}" \
      --data_dir "workspace_r2v/$s" --antialiased ${5:-} 2>&1 | grep -av "render " || die "render $s $tag"
  fi
}

for s in bonsai chair; do
  say "$s — holdout 10% + O1 cap"
  if ! [ -s "workspace_r2v/$s/val_poses.csv" ]; then
    $PY tools/make_holdout.py --ws "workspace_r2/$s" --out_ws "workspace_r2v/$s" --every 10 || die "holdout $s"
  else echo "  ⏩ holdout có"; fi

  train_one "$s" cap3M 3000000
  score "renders_r2v/${s}__cap3M" "workspace_r2v/$s/val_gt" "O1-$s-3M"
  train_one "$s" cap6M 6000000
  score "renders_r2v/${s}__cap6M" "workspace_r2v/$s/val_gt" "O1-$s-6M"

  if [ "${RUN_UT:-0}" = "1" ]; then
    say "$s — O3 3DGUT không méo (tuỳ chọn)"
    train_one "$s" ut3M 3000000 "--with-ut --with-eval3d" "--with_ut"
    score "renders_r2v/${s}__ut3M" "workspace_r2v/$s/val_gt" "O3-$s-UT3M"
  fi

  say "$s — O2 L16-XL trên cap tốt hơn (xem 2 dòng O1 ở trên, mặc định lấy 3M nếu chênh <0.002)"
  # L16 train trên chính workspace holdout (chỉ nhìn ảnh TRAIN của nó — val vẫn sạch)
  if ! [ -s "results/r2v_${s}__l16xl/net.pt" ]; then
    $PY tools/enhance_net.py train --workspace "workspace_r2v/$s" \
      --ckpt "results/r2v_${s}__cap3M/ckpts/ckpt_29999_rank0.pt" \
      --out "results/r2v_${s}__l16xl/net.pt" \
      --steps 8000 --ch_mult 2 --patch 320 2>&1 | tee "/tmp/r2v_l16_${s}.log" || die "L16 $s"
  fi
  $PY tools/enhance_net.py apply --net "results/r2v_${s}__l16xl/net.pt" \
    --in_dir "renders_r2v/${s}__cap3M" --out_dir "renders_r2v/${s}__l16" >/dev/null || die "L16 apply $s"
  score "renders_r2v/${s}__l16" "workspace_r2v/$s/val_gt" "O2-$s-3M+L16"
done

echo
echo "########################################################################"
echo "#  VERDICT GATE R2-OBJ (so các dòng ★ ở trên)"
echo "#  [O1] 6M − 3M ≥ +0.002       → CAP_OBJ=6000000 cho run_sub_r2, ngược lại giữ 3M"
echo "#  [O2] L16 − cap3M            → gain L16 trên scene mờ (kỳ vọng ≥ +0.004)"
echo "#  [O3] (nếu chạy) UT − classic → chỉ đổi nhánh nếu ≥ +0.003"
echo "#  Lưu ý: bonsai chỉ 28 test thật nhưng val có ~25 pose — noise ±0.002, đọc số cẩn thận"
echo "########################################################################"
echo "DÁN TOÀN BỘ các dòng [O1][O2][O3] + khối này CHO CLAUDE."
