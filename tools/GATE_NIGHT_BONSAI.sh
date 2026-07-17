#!/bin/bash
# ============================================================================
# GATE ĐÊM 17→18/07 — bonsai train-time levers (chạy SERVER GPU rảnh, ~3h):
#   [N-erank] erank-reg 0.02        (needle gaussian quanh cành cây/mép kính)
#   [N-45k]   45k steps             (thêm giờ tinh luyện ở knee cap)
#   [N-blur]  blur-match σ0.8 TRAIN (N1: geometry sắc, blur áp hậu kỳ) —
#             chấm 4 kiểu: raw + post-blur σ 0.5/0.8/1.1 (σ test tự quét)
# Mốc so: [O1-bonsai-3M] = 0.7087 (SH3) · lưu ý các biến thể này chạy SH3 để so sạch
# Chạy: bash tools/GATE_NIGHT_BONSAI.sh 2>&1 | tee /tmp/gate_night_bonsai.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "workspace_r2v/bonsai/val_gt" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

[ -s workspace_r2v/bonsai/val_poses.csv ] || die "thiếu holdout bonsai (GATE_R2_OBJ đã chạy trên máy này chưa?)"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 10 ] && die "đĩa ${FREE}GB<10"

run(){ # tag steps extra
  local tag=$1 steps=$2 extra=${3:-}
  local res="results/r2v_bonsai__${tag}" last="results/r2v_bonsai__${tag}/ckpts/ckpt_$((steps-1))_rank0.pt"
  say "bonsai $tag (steps=$steps $extra)"
  if ! [ -s "$last" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir workspace_r2v/bonsai --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps "$steps" --test-every 999999 \
      --disable-viewer --antialiased --strategy.cap-max 3000000 \
      --eval-steps "$steps" --save-steps "$steps" --global-seed 42 $extra \
      2>&1 | tee "/tmp/nb_${tag}.log" || die "train $tag"
    find "$res/ckpts" -name "ckpt_*" ! -name "ckpt_$((steps-1))_*" -delete 2>/dev/null; rm -rf "$res/videos"
  else echo "  ⏩ có"; fi
  rm -rf "renders_r2v/bonsai__${tag}"
  $PY tools/render_test_poses.py --ckpt "$last" --csv workspace_r2v/bonsai/val_poses.csv \
    --out "renders_r2v/bonsai__${tag}" --data_dir workspace_r2v/bonsai --antialiased 2>&1 | tail -1
  score "renders_r2v/bonsai__${tag}" "N-$tag"
}

run erank 30000 "--erank-reg 0.02"
run 45k   45000 ""
run blur  30000 "--blur-match 0.8"
say "N-blur: quét σ hậu kỳ (GT mờ — render sắc phải blur lại mới khớp)"
for SG in 0.5 0.8 1.1; do
  $PY tools/apply_blur.py --in_dir renders_r2v/bonsai__blur --out_dir "renders_r2v/bonsai__blur_p${SG/0./}" --sigma $SG >/dev/null
  score "renders_r2v/bonsai__blur_p${SG/0./}" "N-blur+p$SG"
done

echo
echo "########################################################################"
echo "#  VERDICT ĐÊM BONSAI — mốc [O1-bonsai-3M] = 0.70874 (server đo)"
echo "#  [N-erank] ≥ +0.002 → --erank-reg 0.02 vào prod bonsai (kiểm cả chair c1 bên 4060)"
echo "#  [N-45k]   ≥ +0.002 → 45k steps cho scene mờ (cân +50% giờ)"
echo "#  max[N-blur+p*] ≥ +0.003 → BLUR-MATCH vào prod bonsai (σ train 0.8, σ test = argmax)"
echo "#  Các lever độc lập cơ chế → cái nào thắng thì CỘNG vào config prod sáng mai"
echo "########################################################################"
echo "DÁN các dòng [N-*] CHO CLAUDE."
