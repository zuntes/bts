#!/bin/bash
# Thí nghiệm quyết định #1 (docs/11 §4.2): classic+redistort vs 3DGUT raw-distortion
# cùng cap 700k × 30k steps trên HCM0204. Chạy: setsid nohup bash tools/run_ab_700k.sh &
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
K1=0.010009930826722385
CSV=VAI_NVS_DATA/phase1/public_set/HCM0204/test/test_poses.csv
GT=VAI_NVS_DATA/phase1/public_set/HCM0204/test/images

echo "[$(date +%H:%M)] === NHÁNH 1: classic + redistort ==="
$PY gsplat/examples/simple_trainer.py mcmc \
  --data-dir workspace/HCM0204 --data-factor 1 \
  --result-dir "$PWD/results/HCM0204__base700k_v153" \
  --max-steps 30000 --test-every 8 --disable-viewer --antialiased --packed \
  --strategy.cap-max 700000 --eval-steps 30000 --save-steps 30000
$PY tools/render_test_poses.py \
  --ckpt results/HCM0204__base700k_v153/ckpts/ckpt_29999_rank0.pt \
  --csv "$CSV" --out renders/HCM0204__base700k_v153 \
  --data_dir workspace/HCM0204 --antialiased --redistort_k1 $K1
$PY tools/score_local.py --pred_dir renders/HCM0204__base700k_v153 \
  --gt_dir "$GT" --out results/HCM0204__base700k_v153_score.json

echo "[$(date +%H:%M)] === NHÁNH 2: 3DGUT raw-distortion ==="
$PY gsplat/examples/simple_trainer.py mcmc \
  --data-dir workspace_raw/HCM0204 --data-factor 1 \
  --result-dir "$PWD/results/HCM0204__ut700k" \
  --max-steps 30000 --test-every 8 --disable-viewer --antialiased \
  --with-ut --with-eval3d --raw-distortion \
  --strategy.cap-max 700000 --eval-steps 30000 --save-steps 30000
$PY tools/render_test_poses.py \
  --ckpt results/HCM0204__ut700k/ckpts/ckpt_29999_rank0.pt \
  --csv "$CSV" --out renders/HCM0204__ut700k \
  --data_dir workspace_raw/HCM0204 --antialiased --with_ut --radial_k1 $K1
$PY tools/score_local.py --pred_dir renders/HCM0204__ut700k \
  --gt_dir "$GT" --out results/HCM0204__ut700k_score.json

echo "[$(date +%H:%M)] AB-700K-ALL-DONE"
