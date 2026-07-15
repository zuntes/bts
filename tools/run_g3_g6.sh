#!/bin/bash
# G3 (L16-XL ch_mult=2, 8k steps) + G6 (affine transfer) trên 2 calib.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A CK=( [HCM0204]=results/HCM0204__ut5M_30k/ckpts/ckpt_29999_rank0.pt \
                [HCM0181]=results/HCM0181__SUB2/ckpts/ckpt_29999_rank0.pt )
declare -A K1=( [HCM0204]=0.010009930826722385 [HCM0181]=0.00900353942750311 )
for s in HCM0204 HCM0181; do
  CSV=VAI_NVS_DATA/phase1/public_set/$s/test/test_poses.csv
  GT=VAI_NVS_DATA/phase1/public_set/$s/test/images
  echo "[$(date +%H:%M)] === G3 L16-XL $s ==="
  { $PY tools/enhance_net.py train --workspace workspace_raw/$s --ckpt "${CK[$s]}" \
      --out results/${s}__l16xl/net.pt --with_ut --radial_k1 "${K1[$s]}" \
      --steps 8000 --ch_mult 2 --patch 320 \
    && $PY tools/enhance_net.py apply --net results/${s}__l16xl/net.pt \
      --in_dir "$( [ $s = HCM0204 ] && echo renders/HCM0204__ut5M_30k || echo renders/HCM0181__SUB2 )" \
      --out_dir renders/${s}__l16xl \
    && $PY tools/score_local.py --pred_dir renders/${s}__l16xl --gt_dir "$GT" \
      --out results/${s}__l16xl_score.json && echo "G3-OK $s"
  } || echo "!!! G3-FAILED $s"
  echo "[$(date +%H:%M)] === G6 affine $s (chồng lên L16-XL) ==="
  { $PY tools/affine_transfer.py --workspace workspace_raw/$s \
      --renders_train results/${s}__l16xl/renders_train --test_csv "$CSV" \
      --in_dir renders/${s}__l16xl --out_dir renders/${s}__g6 \
      --net results/${s}__l16xl/net.pt \
    && $PY tools/score_local.py --pred_dir renders/${s}__g6 --gt_dir "$GT" \
      --out results/${s}__g6_score.json && echo "G6-OK $s"
  } || echo "!!! G6-FAILED $s"
done
echo "[$(date +%H:%M)] G3G6-ALL-DONE"
