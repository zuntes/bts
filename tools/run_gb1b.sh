#!/bin/bash
# GB1b sạch: 5M seed2 (--global-seed 7) → ensemble 2-seed ± L16. GPU đang rảnh.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K1=0.010009930826722385
CSV=VAI_NVS_DATA/phase1/public_set/HCM0204/test/test_poses.csv
GT=VAI_NVS_DATA/phase1/public_set/HCM0204/test/images
sc () { $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>/dev/null | grep "★"; }
echo "[$(date +%H:%M)] === 5M seed2 (global-seed 7) ==="
$PY gsplat/examples/simple_trainer.py mcmc --data-dir workspace_raw/HCM0204 --data-factor 1 \
  --result-dir "$PWD/results/HCM0204__ut5M_seed2" --max-steps 30000 --test-every 999999 \
  --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
  --strategy.cap-max 5000000 --eval-steps 30000 --save-steps 30000 --global-seed 7 \
&& $PY tools/render_test_poses.py --ckpt results/HCM0204__ut5M_seed2/ckpts/ckpt_29999_rank0.pt \
  --csv "$CSV" --out renders/HCM0204__ut5M_seed2 --data_dir workspace_raw/HCM0204 \
  --antialiased --with_ut --radial_k1 $K1 \
&& { echo -n "  5M-seed2 đơn: "; sc renders/HCM0204__ut5M_seed2; } \
&& $PY tools/ensemble.py --dirs renders/HCM0204__ut5M_30k renders/HCM0204__ut5M_seed2 --out renders/HCM0204__ens_seed --mode mean >/dev/null \
&& { echo -n "  ens(5M-s1,5M-s2): "; sc renders/HCM0204__ens_seed; } \
&& $PY tools/enhance_net.py apply --net results/HCM0204__l16/net.pt --in_dir renders/HCM0204__ens_seed --out_dir renders/HCM0204__ens_seed_l16 >/dev/null \
&& { echo -n "  ens+L16: "; sc renders/HCM0204__ens_seed_l16; } \
&& echo "GB1b-OK" || echo "!!! GB1b-FAILED"
echo "[$(date +%H:%M)] GB1B-ALL-DONE"
