#!/bin/bash
# ============================================================================
# GATE ĐÊM 18→19/07 (4060, test-only) — 2 kỹ thuật từ ĐỘI THẮNG SoccerNet-NVS'26:
#   [W1-*]  test-proximity loss weighting (w=3): train view gần pose test ăn loss lớn
#   [W2-*]  BRANCHED MEMBERS: từ ckpt 30k CÓ SẴN chạy tiếp +5k với config khác
#           → member giá 1/6 (5k vs 30k steps). Đo: thêm 2 branch vào MEGA có ăn thêm?
# Bàn: holdout chair + bonsai. Mốc: chair sh4 0.66358 · MEGA6+vgg 0.68539
#                                  bonsai sh4 0.71402 · MEGA4+vgg 0.74125
# Chạy: setsid nohup bash tools/GATE_WINNER.sh > results/gate_winner.log 2>&1 &
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "workspace_r2v/$3/val_gt" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }
rend(){ # ckpt out scene
  rm -rf "$2"
  $PY tools/render_test_poses.py --ckpt "$1" --csv "workspace_r2v/$3/val_poses.csv" \
    --out "$2" --data_dir "workspace_r2v/$3" --antialiased 2>&1 | tail -1
}

# ---------- W1: test-proximity weighting (train từ đầu, w=3) ----------
for s in chair bonsai; do
  say "$s — W1 test-weight 3.0 (SH4, 30k)"
  res="results/r2v_${s}__w1"
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2v/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased --sh-degree 4 --strategy.cap-max 3000000 \
      --eval-steps 30000 --save-steps 30000 --global-seed 42 \
      --test-weight 3.0 --test-csv-path "workspace_r2v/$s/val_poses.csv" \
      2>&1 | tee "/tmp/w1_${s}.log" | grep -aE "BTS W1" || die "train W1 $s"
  fi
  rend "$res/ckpts/ckpt_29999_rank0.pt" "renders_r2v/${s}__w1" "$s"
  score "renders_r2v/${s}__w1" "W1-$s" "$s"
done

# ---------- W2: branched members từ ckpt sh4 sẵn có (+5k steps, config lệch) ----------
branch(){ # scene tag extra
  local s=$1 tag=$2 extra=$3
  local src="results/r2v_${s}__sh4/ckpts/ckpt_29999_rank0.pt"
  [ -s "$src" ] || { echo "  ⏭ thiếu ckpt sh4 $s"; return 0; }
  local res="results/r2v_${s}__br_${tag}"
  say "$s — W2 branch:$tag (+5k từ ckpt chung, --init-ckpt)"
  if ! [ -s "$res/ckpts/ckpt_4999_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2v/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 5000 --test-every 999999 \
      --disable-viewer --antialiased --sh-degree 4 --strategy.cap-max 3000000 \
      --eval-steps 5000 --save-steps 5000 --global-seed 42 \
      --init-ckpt "$src" $extra \
      2>&1 | tee "/tmp/br_${s}_${tag}.log" | grep -aE "BTS W2" || { echo "  ⚠ branch $tag fail — bỏ"; return 0; }
  fi
  rend "$res/ckpts/ckpt_4999_rank0.pt" "renders_r2v/${s}__br_${tag}" "$s"
}
for s in chair bonsai; do
  branch "$s" sreg "--scale-reg 0.02"
  branch "$s" oreg "--opacity-reg 0.02"
  # MEGA + 2 branch mới
  BASE=""
  for d in sh4 sh4_s7 sh4_s123 cap3M br_sreg br_oreg; do
    [ -d "renders_r2v/${s}__$d" ] && BASE="$BASE renders_r2v/${s}__$d"
  done
  $PY tools/ensemble.py --dirs $BASE --out "renders_r2v/${s}__megabr" --mode mean >/dev/null
  $PY tools/enhance_net.py apply --net "results/r2v_${s}__enhvgg/net.pt" \
    --in_dir "renders_r2v/${s}__megabr" --out_dir "renders_r2v/${s}__megabrvgg" >/dev/null 2>&1
  score "renders_r2v/${s}__megabrvgg" "W2-MEGA+BR-$s" "$s"
done

echo
echo "########################################################################"
echo "#  VERDICT WINNER-GATE — mốc: chair MEGA6+vgg 0.68539 · bonsai MEGA4+vgg 0.74125"
echo "#  [W1-*] vs sh4 (chair .66358/bonsai .71402): ≥+0.003 → prod thêm --test-weight 3"
echo "#  [W2-MEGA+BR-*] vs mốc MEGA: ≥+0.002 → member-factory phân nhánh vào FINALIZE"
echo "########################################################################"
echo "WINNER-GATE-DONE"
