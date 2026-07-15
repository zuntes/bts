#!/bin/bash
# Chuỗi thí nghiệm máy 4060 (10/7 tối) — 2 mục tiêu:
#  (A) SCALING: card mạnh + cap cao có ăn điểm không?  classic 2M & 1.5M vs mốc 0.794@700k (máy cũ)
#  (B) A/B quyết định: classic+redist vs 3DGUT raw-distortion, CẶP CÔNG BẰNG cùng cap 1.5M
# Mỗi run độc lập (run sau vẫn chạy nếu run trước lỗi). Save giữa chừng @15k để không mất trắng nếu bị kill.
# Chạy: setsid nohup bash tools/run_scale_ab.sh > results/scale_ab.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K1=0.010009930826722385
CSV=VAI_NVS_DATA/phase1/public_set/HCM0204/test/test_poses.csv
GT=VAI_NVS_DATA/phase1/public_set/HCM0204/test/images

run_classic () {  # $1 = cap, $2 = tag
  echo "[$(date +%H:%M)] === CLASSIC cap=$1 tag=$2 ==="
  $PY gsplat/examples/simple_trainer.py mcmc \
    --data-dir workspace/HCM0204 --data-factor 1 \
    --result-dir "$PWD/results/HCM0204__$2" \
    --max-steps 30000 --test-every 8 --disable-viewer --antialiased --packed \
    --strategy.cap-max "$1" --eval-steps 30000 --save-steps 15000 30000 \
  && $PY tools/render_test_poses.py \
    --ckpt "results/HCM0204__$2/ckpts/ckpt_29999_rank0.pt" \
    --csv "$CSV" --out "renders/HCM0204__$2" \
    --data_dir workspace/HCM0204 --antialiased --redistort_k1 $K1 \
  && $PY tools/score_local.py --pred_dir "renders/HCM0204__$2" \
    --gt_dir "$GT" --out "results/HCM0204__$2_score.json" \
  || echo "[$(date +%H:%M)] !!! FAILED: $2"
}

run_ut () {  # $1 = cap, $2 = tag
  echo "[$(date +%H:%M)] === 3DGUT cap=$1 tag=$2 ==="
  $PY gsplat/examples/simple_trainer.py mcmc \
    --data-dir workspace_raw/HCM0204 --data-factor 1 \
    --result-dir "$PWD/results/HCM0204__$2" \
    --max-steps 30000 --test-every 8 --disable-viewer --antialiased \
    --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max "$1" --eval-steps 30000 --save-steps 15000 30000 \
  && $PY tools/render_test_poses.py \
    --ckpt "results/HCM0204__$2/ckpts/ckpt_29999_rank0.pt" \
    --csv "$CSV" --out "renders/HCM0204__$2" \
    --data_dir workspace_raw/HCM0204 --antialiased --with_ut --radial_k1 $K1 \
  && $PY tools/score_local.py --pred_dir "renders/HCM0204__$2" \
    --gt_dir "$GT" --out "results/HCM0204__$2_score.json" \
  || echo "[$(date +%H:%M)] !!! FAILED: $2"
}

run_classic 2000000 classic2M      # (A) điểm scaling cao nhất an toàn VRAM
run_ut      1500000 ut1500k        # (B) nhánh 3DGUT — UT không packed, 1.5M là mức thận trọng 8GB
run_classic 1500000 classic1500k   # (B) cặp công bằng cho ut1500k + điểm giữa đường scaling

echo "[$(date +%H:%M)] SCALE-AB-ALL-DONE"
