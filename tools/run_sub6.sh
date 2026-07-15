#!/bin/bash
# SUB6 = 3-seed ensemble + L16-XL. Train seed3 (global-seed 123) cho 8 private,
# ensemble seed1(SUB2,42)+seed2(7)+seed3(123) → L16-XL → zip.
# Guard đĩa. Chạy: setsid nohup bash tools/run_sub6.sh > results/sub6.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
k1_of () { $PY - "$1" <<'EOF'
import struct,sys
with open(f"workspace_raw/{sys.argv[1]}/sparse/0/cameras.bin","rb") as f:
    f.read(8);cid,model,w,h=struct.unpack("<iiQQ",f.read(24));p=struct.unpack("<dddd",f.read(32))
print(repr(p[3]))
EOF
}
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  [ -d "renders/${s}__sub6" ] && { echo "⏩ $s"; continue; }
  FREE=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "$FREE" -lt 8 ] && { echo "!!! DISK-LOW ${FREE}GB"; break; }
  cap=5000000; [ "$s" = HCM1439 ] && cap=2500000
  k1=$(k1_of "$s")
  echo "[$(date +%d/%m' '%H:%M)] === SEED3 $s cap=$cap (đĩa ${FREE}GB) ==="
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$s" --data-factor 1 \
    --result-dir "$PWD/results/${s}__seed3" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max "$cap" --eval-steps 30000 --save-steps 30000 --global-seed 123 \
  && $PY tools/render_test_poses.py --ckpt "results/${s}__seed3/ckpts/ckpt_29999_rank0.pt" \
    --csv "VAI_NVS_DATA/phase1/private_set1/$s/test/test_poses.csv" \
    --out "renders/${s}__seed3" --data_dir "workspace_raw/$s" --antialiased --with_ut --radial_k1 "$k1" \
  && $PY tools/ensemble.py --dirs "renders/${s}__SUB2" "renders/${s}__seed2" "renders/${s}__seed3" --out "renders/${s}__ens3" --mode mean \
  && $PY tools/enhance_net.py apply --net "results/${s}__l16xl/net.pt" --in_dir "renders/${s}__ens3" --out_dir "renders/${s}__sub6" \
  && rm -f "results/${s}__seed3/ckpts/ckpt_14999_rank0.pt" \
  && echo "[$(date +%H:%M)] SCENE-OK $s" || echo "[$(date +%H:%M)] !!! SCENE-FAILED $s"
done
$PY - <<'EOF'
import csv, cv2
from pathlib import Path
flags = [cv2.IMWRITE_JPEG_QUALITY, 96]; miss = 0; fb = 0
for sd in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (sd/"test/test_poses.csv").exists(): continue
    s = sd.name; out = Path("renders_sub6")/s; out.mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(sd/"test/test_poses.csv")):
        n = r["image_name"]; p = Path(f"renders/{s}__sub6")/(Path(n).stem+".png")
        if not p.exists(): p = Path("renders_sub5")/s/n; fb += 1  # fallback SUB5
        if not p.exists(): print("THIẾU", s, n); miss += 1; continue
        cv2.imwrite(str(out/n), cv2.imread(str(p)), flags)
print(f"repackage: fallback {fb}, thiếu {miss}")
EOF
$PY tools/make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub6 --ext .same --out submission_SUB6.zip \
  && echo "[$(date +%d/%m' '%H:%M)] SUB6-ALL-DONE" || echo "!!! PACKAGE-FAILED"
