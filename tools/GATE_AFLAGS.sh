#!/bin/bash
# ============================================================================
# GATE A-FLAGS — 3 nút có SẴN trong gsplat, chưa từng bật (DOC2 §8 Tier A).
# Bàn chấm: HCM0204 public half-res (đúng chế độ round 2, có GT) — cần GATE_R2_HCM
# đã chạy trước để có mốc [H1-6M] (cap6M seed42 @1320×989).
#
#   [F1] --depth-loss --depth-lambda 1e-2  (ghim geometry bằng SfM depth)
#   [F2] 45k steps thay 30k                (knee cap → thêm giờ tinh luyện)
#   [F3] F1+F2 gộp                         (nếu cả 2 dương riêng lẻ)
#
# Chạy SERVER sau GATE_R2_HCM (~2.5-3h): bash tools/GATE_AFLAGS.sh 2>&1 | tee /tmp/gate_aflags.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
CAL=workspace_r2cal
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$CAL/gt_half" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

say "0. tiên quyết"
[ -s "$CAL/test_poses_half.csv" ] || die "thiếu bàn chấm half-res → chạy GATE_R2_HCM trước"
[ -s "results/r2cal_${S}__cap6M_s42/ckpts/ckpt_29999_rank0.pt" ] || die "thiếu mốc cap6M (GATE_R2_HCM chưa xong?)"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 15 ] && die "đĩa ${FREE}GB<15"

tr_flags(){  # $1=tag $2=steps $3=extra
  local tag=$1 steps=$2 extra=${3:-} res="results/r2cal_${S}__${1}"
  local last_ckpt="$res/ckpts/ckpt_$((steps-1))_rank0.pt"
  if ! [ -s "$last_ckpt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 2 \
      --result-dir "$PWD/$res" --max-steps "$steps" --test-every 999999 \
      --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max 6000000 --eval-steps "$steps" --save-steps "$steps" --global-seed 42 \
      $extra 2>&1 | tee "/tmp/aflags_${tag}.log" || die "train $tag"
    find "$res/ckpts" -name "ckpt_*" ! -name "ckpt_$((steps-1))_*" -delete 2>/dev/null
    rm -rf "$res/videos"
  else echo "  ⏩ $tag có"; fi
  if [ "$(ls "renders_r2cal/${tag}" 2>/dev/null | wc -l)" -lt 60 ]; then
    $PY tools/render_test_poses.py --ckpt "$last_ckpt" \
      --csv "$CAL/test_poses_half.csv" --out "renders_r2cal/${tag}" \
      --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 $K1 \
      2>&1 | grep -av "render " || die "render $tag"
  fi
}

# ⚠ 17/07 23h: --depth-loss + --with-ut/--raw-distortion → loss NaN @step600 → opacity sập
# → MCMC multinomial assert. 2 nghi phạm: (a) RGB+ED chưa hỗ trợ đường UT; (b) depth-loss
# dùng keypoint 2D = lớp data lệch thang 4× (DOC3 §2.4). Mặc định TẮT; RUN_DEPTH=1 chỉ khi debug.
if [ "${RUN_DEPTH:-0}" = "1" ]; then
  say "F1. depth-loss @6M 30k (~40ph)"
  tr_flags f1_depth 30000 "--depth-loss --depth-lambda 1e-2"
  score renders_r2cal/f1_depth "F1-depth"
fi

say "F2. 45k steps @6M (~60ph)"
tr_flags f2_45k 45000 ""
score renders_r2cal/f2_45k "F2-45k"

if [ "${RUN_DEPTH:-0}" = "1" ]; then
  say "F3. depth + 45k (~60ph)"
  tr_flags f3_both 45000 "--depth-loss --depth-lambda 1e-2"
  score renders_r2cal/f3_both "F3-both"
fi

echo
echo "########################################################################"
echo "#  VERDICT A-FLAGS — mốc = [H1-6M] trong /tmp/gate_r2hcm.txt (cap6M 30k)"
echo "#  F1 − H1-6M ≥ +0.002 → bật depth-loss cho production R2"
echo "#  F2 − H1-6M ≥ +0.002 → 45k steps (giá +50% giờ train — cân với 3-seed)"
echo "#  F3 ≥ F1,F2 riêng lẻ → gộp cả hai"
echo "########################################################################"
echo "DÁN [F1][F2][F3] + mốc [H1-6M] CHO CLAUDE."

# ===================== F4 (tuỳ chọn, RUN_F4=1) — ERANK REG =====================
if [ "${RUN_F4:-0}" = "1" ]; then
  say "F4. erank-reg 0.02 @6M 30k (trị needle gaussian — cây/dây điện)"
  tr_flags f4_erank 30000 "--erank-reg 0.02"
  score renders_r2cal/f4_erank "F4-erank"
  echo "  [F4] − [H1-6M] ≥ +0.002 → bật --erank-reg 0.02 cho production"
fi
