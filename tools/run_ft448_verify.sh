#!/bin/bash
# Xác minh 5M+ft crop448 (gỡ confound p320) + thử chồng L16 lên ckpt ft.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K1=0.010009930826722385
CSV=VAI_NVS_DATA/phase1/public_set/HCM0204/test/test_poses.csv
GT=VAI_NVS_DATA/phase1/public_set/HCM0204/test/images
while ! grep -q "SUB3-ALL-DONE\|PACKAGE-FAILED" results/sub3.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] === ft448 trên ckpt 5M ==="
{ $PY tools/finetune_lpips.py --workspace workspace_raw/HCM0204 \
    --ckpt results/HCM0204__ut5M_30k/ckpts/ckpt_29999_rank0.pt \
    --out results/HCM0204__ut5M_ft448/ckpt_ft.pt \
    --lambda_lpips 0.05 --steps 3000 --antialiased --with_ut --lpips_patch 448 \
    --test_csv "$CSV" \
  && $PY tools/render_test_poses.py --ckpt results/HCM0204__ut5M_ft448/ckpt_ft.pt \
    --csv "$CSV" --out renders/HCM0204__ut5M_ft448 --data_dir workspace_raw/HCM0204 \
    --antialiased --with_ut --radial_k1 $K1 \
  && $PY tools/score_local.py --pred_dir renders/HCM0204__ut5M_ft448 --gt_dir "$GT" \
    --out results/HCM0204__ut5M_ft448_score.json && echo FT448-OK
} || echo "!!! FT448-FAILED"
echo "[$(date +%H:%M)] === L16 trên ckpt ft448 ==="
{ $PY tools/enhance_net.py train --workspace workspace_raw/HCM0204 \
    --ckpt results/HCM0204__ut5M_ft448/ckpt_ft.pt \
    --out results/HCM0204__ft448_l16/net.pt --with_ut --radial_k1 $K1 --steps 3000 \
  && $PY tools/enhance_net.py apply --net results/HCM0204__ft448_l16/net.pt \
    --in_dir renders/HCM0204__ut5M_ft448 --out_dir renders/HCM0204__ft448_l16 \
  && $PY tools/score_local.py --pred_dir renders/HCM0204__ft448_l16 --gt_dir "$GT" \
    --out results/HCM0204__ft448_l16_score.json && echo FT448-L16-OK
} || echo "!!! FT448-L16-FAILED"
echo "[$(date +%H:%M)] FT448-VERIFY-ALL-DONE"
