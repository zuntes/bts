#!/bin/bash
# Fix2: (a) chạy lại HCM0249 seed2 (crash race tb); (b) A/B net l16 vs l16xl trên
# ensemble calib; (c) áp l16xl (bằng chứng SUB4 private) lên __ens; (d) đóng gói SUB5.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GT=VAI_NVS_DATA/phase1/public_set/HCM0204/test/images
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
&& $PY tools/ensemble.py --dirs "renders/${s}__SUB2" "renders/${s}__seed2" --out "renders/${s}__ens" --mode mean \
&& echo "FIX-OK $s" || echo "!!! FIX-FAILED $s"

echo "[$(date +%H:%M)] === A/B net trên ensemble (calib HCM0204, tham khảo) ==="
$PY tools/enhance_net.py apply --net results/HCM0204__l16xl/net.pt \
  --in_dir renders/HCM0204__ens_seed --out_dir renders/HCM0204__ens_seed_l16xl
echo -n "  ens+L16base: "; $PY tools/score_local.py --pred_dir renders/HCM0204__ens_seed_l16 --gt_dir "$GT" 2>/dev/null | grep "★"
echo -n "  ens+L16XL  : "; $PY tools/score_local.py --pred_dir renders/HCM0204__ens_seed_l16xl --gt_dir "$GT" 2>/dev/null | grep "★"

# chọn XL theo bằng chứng SUB4 (private chuyển giao tốt hơn calib)
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  [ -d "renders/${s}__ens" ] || { echo "!! thiếu ens $s"; continue; }
  $PY tools/enhance_net.py apply --net "results/${s}__l16xl/net.pt" \
    --in_dir "renders/${s}__ens" --out_dir "renders/${s}__sub5f" && echo "APPLY-OK $s"
done

$PY - <<'EOF'
import csv, cv2
from pathlib import Path
flags = [cv2.IMWRITE_JPEG_QUALITY, 96]; miss = 0; fb = 0
for sd in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (sd/"test/test_poses.csv").exists(): continue
    s = sd.name; out = Path("renders_sub5")/s; out.mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(sd/"test/test_poses.csv")):
        n = r["image_name"]; p = Path(f"renders/{s}__sub5f")/(Path(n).stem+".png")
        if not p.exists(): p = Path(f"renders/{s}__l16xl")/(Path(n).stem+".png"); fb += 1
        if not p.exists(): print("THIẾU", s, n); miss += 1; continue
        cv2.imwrite(str(out/n), cv2.imread(str(p)), flags)
print(f"repackage: fallback {fb}, thiếu {miss}")
EOF
$PY tools/make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub5 --ext .same --out submission_SUB5.zip \
  && echo "[$(date +%H:%M)] SUB5-FINAL-DONE" || echo "!!! REPACKAGE-FAILED"
