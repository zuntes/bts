#!/bin/bash
# ============================================================================
# PHASE A — 4 cổng rẻ trên HCM0204, dùng lại ckpt cap12M seed42 (GATE_CAP).
#   A1 stack@12M : seed7@12M → ens(42,7) → L16-XL → v50   (stack +1.24 có chuyển sang 12M?)
#   A2 cap16M    : train 16M → v50                        (knee đã qua chưa?)
#   A3 ft512@12M : finetune LPIPS crop512 từ ckpt 12M     (L40S gỡ OOM của 4060)
#   A4 seed3     : ens(42,7,123) → L16-XL → v50           (seed thứ 3 đáng không?)
# MỐC: plain12M=0.75587 · stack5M=0.75968 · Tổng ~7-9h. RESUME-SAFE.
# Chạy: tmux new -s ga && bash tools/GATE_A.sh 2>&1 | tee /tmp/gate_a.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=HCM0204
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
K1=0.010009930826722385
say(){ echo; echo "[$(date +%H:%M)] ===== $* ====="; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }
train12(){ # $1=seed $2=result_tag
  [ -s "results/${S}__$2/ckpts/ckpt_29999_rank0.pt" ] && { echo "  ⏩ $2 ckpt có"; return; }
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
    --result-dir "$PWD/results/${S}__$2" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max ${3:-12000000} --eval-steps 30000 --save-steps 30000 \
    --global-seed $1 2>&1 | tee /tmp/ga_train.log | tail -2 || die "train $2"
  rm -f "results/${S}__$2/ckpts/ckpt_14999_rank0.pt"; rm -rf "results/${S}__$2/videos"
}
rend(){ # $1=ckpt_tag $2=render_tag
  [ "$(ls renders/${S}__$2 2>/dev/null | wc -l)" -ge 60 ] && { echo "  ⏩ $2 render có"; return; }
  $PY tools/render_test_poses.py --ckpt "results/${S}__$1/ckpts/ckpt_29999_rank0.pt" \
    --csv "$CSV" --out "renders/${S}__$2" --data_dir "workspace_raw/$S" \
    --antialiased --with_ut --radial_k1 $K1 2>&1 | tail -1
}

say "0. tiên quyết"
[ -s "results/${S}__cap12M/ckpts/ckpt_29999_rank0.pt" ] || die "thiếu ckpt cap12M (chạy GATE_CAP trước)"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 20 ] && die "đĩa ${FREE}GB<20"
echo "  ✅ ckpt 12M seed42 có · đĩa ${FREE}GB"

say "A1a. train seed7 @12M (~1.5h)"
train12 7 s7_12M
say "A1b. render seed7 + ensemble(42,7) + L16-XL"
rend s7_12M s7_12M
$PY tools/ensemble.py --dirs renders/${S}__cap12M renders/${S}__s7_12M --out renders/${S}__ens12 --mode mean >/dev/null
score renders/${S}__ens12 "A1-ens2seed-raw"
if ! [ -s results/${S}__l16xl12/net.pt ]; then
  $PY tools/enhance_net.py train --workspace "workspace_raw/$S" \
    --ckpt "results/${S}__cap12M/ckpts/ckpt_29999_rank0.pt" \
    --out results/${S}__l16xl12/net.pt --with_ut --radial_k1 $K1 \
    --steps 8000 --ch_mult 2 --patch 320 2>&1 | grep -aE "VAL-GAIN|val BASE"
fi
$PY tools/enhance_net.py apply --net results/${S}__l16xl12/net.pt \
  --in_dir renders/${S}__ens12 --out_dir renders/${S}__a1 >/dev/null
score renders/${S}__a1 "A1-STACK12M"

say "A2. cap 16M (~1.9h)"
train12 42 cap16M 16000000
rend cap16M cap16M
score renders/${S}__cap16M "A2-cap16M"

say "A3. ft512 @12M (3k steps từ ckpt 12M, ~30-45ph)"
if ! [ -s results/${S}__ft12/ckpt_ft.pt ]; then
  $PY tools/finetune_lpips.py --workspace "workspace_raw/$S" \
    --ckpt "results/${S}__cap12M/ckpts/ckpt_29999_rank0.pt" \
    --out results/${S}__ft12/ckpt_ft.pt --lambda_lpips 0.05 --steps 3000 \
    --antialiased --with_ut --lpips_patch 512 --test_csv "$CSV" 2>&1 | tail -3 \
    || echo "  ❌ ft512 fail (OOM? dán log)"
fi
if [ -s results/${S}__ft12/ckpt_ft.pt ]; then
  [ "$(ls renders/${S}__ft12 2>/dev/null | wc -l)" -ge 60 ] || \
  $PY tools/render_test_poses.py --ckpt results/${S}__ft12/ckpt_ft.pt --csv "$CSV" \
    --out renders/${S}__ft12 --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 $K1 2>&1 | tail -1
  score renders/${S}__ft12 "A3-ft512@12M"
fi

say "A4. seed3 @12M (~1.5h) + ens3 + L16-XL"
train12 123 s123_12M
rend s123_12M s123_12M
$PY tools/ensemble.py --dirs renders/${S}__cap12M renders/${S}__s7_12M renders/${S}__s123_12M --out renders/${S}__ens3x12 --mode mean >/dev/null
$PY tools/enhance_net.py apply --net results/${S}__l16xl12/net.pt \
  --in_dir renders/${S}__ens3x12 --out_dir renders/${S}__a4 >/dev/null
score renders/${S}__a4 "A4-STACK-3SEED"

echo
echo "########################################################################"
echo "#  VERDICT PHASE A — so các dòng ★ ở trên với:"
echo "#    plain12M = 0.75587   ·   stack5M (SUB5-style) = 0.75968"
echo "#  A1 ≥ 0.767 → stack chuyển tốt sang 12M (dự phóng SUB7 ~78)"
echo "#  A2 − plain12M ≥ +0.002 → dùng cap 16M cho production"
echo "#  A3 − plain12M ≥ +0.003 → thêm ft512 vào chuỗi production"
echo "#  A4 − A1 ≥ +0.0015 → dùng 3 seed"
echo "########################################################################"
echo "DÁN TOÀN BỘ CÁC DÒNG [A*] + khối này CHO CLAUDE."
