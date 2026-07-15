#!/bin/bash
# Leo thang nhánh 3DGUT (thắng A/B 10/7) trên HCM0204 — mỗi lần 1 biến:
#   ut1500k_30k = 0.81869 (đã có) → ut3M_30k (tăng CAP) → ut3M_60k (tăng STEPS)
# VRAM đo thật: UT 1.5M = 2.21GB đỉnh → 3M ≈ 3.7-4GB, an toàn trên 8GB.
# Chạy: setsid nohup bash tools/run_ut_scale.sh > results/ut_scale.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K1=0.010009930826722385
CSV=VAI_NVS_DATA/phase1/public_set/HCM0204/test/test_poses.csv
GT=VAI_NVS_DATA/phase1/public_set/HCM0204/test/images

run_ut () {  # $1 = cap, $2 = steps, $3 = tag
  echo "[$(date +%H:%M)] === 3DGUT cap=$1 steps=$2 tag=$3 ==="
  $PY gsplat/examples/simple_trainer.py mcmc \
    --data-dir workspace_raw/HCM0204 --data-factor 1 \
    --result-dir "$PWD/results/HCM0204__$3" \
    --max-steps "$2" --test-every 8 --disable-viewer --antialiased \
    --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max "$1" --eval-steps "$2" --save-steps 15000 "$2" \
  && $PY tools/render_test_poses.py \
    --ckpt "results/HCM0204__$3/ckpts/ckpt_$(($2-1))_rank0.pt" \
    --csv "$CSV" --out "renders/HCM0204__$3" \
    --data_dir workspace_raw/HCM0204 --antialiased --with_ut --radial_k1 $K1 \
  && $PY tools/score_local.py --pred_dir "renders/HCM0204__$3" \
    --gt_dir "$GT" --out "results/HCM0204__$3_score.json" \
  || echo "[$(date +%H:%M)] !!! FAILED: $3"
}

run_ut 3000000 30000 ut3M_30k    # biến 1: cap 1.5M→3M (so với ut1500k_30k)
run_ut 3000000 60000 ut3M_60k    # biến 2: steps 30k→60k (so với ut3M_30k)

echo "[$(date +%H:%M)] UT-SCALE-ALL-DONE"
