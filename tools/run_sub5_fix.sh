#!/bin/bash
# Sửa HCM0249 (crash do race tb) sau khi SUB5 chính xong, rồi RE-PACKAGE.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while ! grep -q "SUB5-ALL-DONE\|PACKAGE-FAILED" results/sub5.log 2>/dev/null; do sleep 180; done
s=HCM0249; k1=0.00890987376989828
echo "[$(date +%H:%M)] === FIX SEED2 $s ==="
$PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$s" --data-factor 1 \
  --result-dir "$PWD/results/${s}__seed2" --max-steps 30000 --test-every 999999 \
  --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
  --strategy.cap-max 5000000 --eval-steps 30000 --save-steps 30000 --global-seed 7 \
&& $PY tools/render_test_poses.py --ckpt "results/${s}__seed2/ckpts/ckpt_29999_rank0.pt" \
  --csv "VAI_NVS_DATA/phase1/private_set1/$s/test/test_poses.csv" \
  --out "renders/${s}__seed2" --data_dir "workspace_raw/$s" --antialiased --with_ut --radial_k1 "$k1" \
&& $PY tools/ensemble.py --dirs "renders/${s}__SUB2" "renders/${s}__seed2" --out "renders/${s}__ens" --mode mean >/dev/null \
&& $PY tools/enhance_net.py apply --net "results/${s}__l16/net.pt" --in_dir "renders/${s}__ens" --out_dir "renders/${s}__sub5" >/dev/null \
&& echo "FIX-OK $s" || echo "!!! FIX-FAILED $s"
# RE-PACKAGE toàn bộ SUB5 với HCM0249 đã có
$PY - <<'EOF'
import csv,cv2
from pathlib import Path
flags=[cv2.IMWRITE_JPEG_QUALITY,96];miss=0;fb=0
for sd in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (sd/"test/test_poses.csv").exists():continue
    s=sd.name;out=Path("renders_sub5")/s;out.mkdir(parents=True,exist_ok=True)
    for r in csv.DictReader(open(sd/"test/test_poses.csv")):
        n=r["image_name"];p=Path(f"renders/{s}__sub5")/(Path(n).stem+".png")
        if not p.exists():p=Path(f"renders/{s}__l16xl")/(Path(n).stem+".png");fb+=1
        if not p.exists():print("THIẾU",s,n);miss+=1;continue
        cv2.imwrite(str(out/n),cv2.imread(str(p)),flags)
print(f"repackage: fallback {fb} ảnh, thiếu {miss}")
EOF
$PY tools/make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub5 --ext .same --out submission_SUB5.zip \
  && echo "[$(date +%H:%M)] SUB5-FIXED-DONE" || echo "!!! REPACKAGE-FAILED"
