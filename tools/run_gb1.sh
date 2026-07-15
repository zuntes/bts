#!/bin/bash
# GB1 mở rộng: test ensemble các bộ CÓ SẴN (free, áp private ngay nếu thắng)
# TRƯỚC, rồi mới train seed2 (ensemble đúng). Đợi SUB4.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K1=0.010009930826722385
CSV=VAI_NVS_DATA/phase1/public_set/HCM0204/test/test_poses.csv
GT=VAI_NVS_DATA/phase1/public_set/HCM0204/test/images
sc () { $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" --out "$2" 2>/dev/null | grep "★"; }
while ! grep -q "SUB4-ALL-DONE\|PACKAGE-FAILED" results/sub4.log 2>/dev/null; do sleep 120; done
echo "[$(date +%H:%M)] GB1 bắt đầu — baseline l16=0.75107 l16xl=0.75114"

# (A) ensemble các bộ CÓ SẴN trên HCM0204 (giống bộ private đang có)
for combo in "ut5M_30k l16" "l16 l16xl" "ut3M_30k ut5M_30k l16" "ut5M_30k ut3M_60k"; do
  dirs=""; for c in $combo; do dirs="$dirs renders/HCM0204__$c"; done
  tag=$(echo $combo | tr ' ' '+')
  $PY tools/ensemble.py --dirs $dirs --out "renders/HCM0204__ens_$tag" --mode mean >/dev/null
  echo -n "  mean($tag): "; sc "renders/HCM0204__ens_$tag" "results/ens_${tag}.json"
done
echo "GB1existing-OK"

# (B) ensemble ĐÚNG: 5M seed2 (cùng config khác seed → lỗi độc lập)
echo "[$(date +%H:%M)] === train 5M seed2 ==="
$PY gsplat/examples/simple_trainer.py mcmc --data-dir workspace_raw/HCM0204 --data-factor 1 \
  --result-dir "$PWD/results/HCM0204__ut5M_seed2" --max-steps 30000 --test-every 999999 \
  --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
  --strategy.cap-max 5000000 --eval-steps 30000 --save-steps 30000 --seed 7 \
&& $PY tools/render_test_poses.py --ckpt results/HCM0204__ut5M_seed2/ckpts/ckpt_29999_rank0.pt \
  --csv "$CSV" --out renders/HCM0204__ut5M_seed2 --data_dir workspace_raw/HCM0204 \
  --antialiased --with_ut --radial_k1 $K1 \
&& $PY tools/ensemble.py --dirs renders/HCM0204__ut5M_30k renders/HCM0204__ut5M_seed2 --out renders/HCM0204__ens_seed --mode mean >/dev/null \
&& { echo -n "  mean(5M-s1,5M-s2): "; sc renders/HCM0204__ens_seed results/ens_seed.json; } \
&& $PY tools/enhance_net.py apply --net results/HCM0204__l16/net.pt --in_dir renders/HCM0204__ens_seed --out_dir renders/HCM0204__ens_seed_l16 >/dev/null \
&& { echo -n "  +L16: "; sc renders/HCM0204__ens_seed_l16 results/ens_seed_l16.json; } \
&& echo "GB1seed-OK" || echo "!!! GB1seed-FAILED"
echo "[$(date +%H:%M)] GB1-ALL-DONE"
