#!/bin/bash
# SUB4 = SUB2 ckpt (5M) + L16-XL (ch_mult=2, 8k steps) per-scene. Ratchet rẻ.
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
  [ -d "renders/${s}__l16xl" ] && { echo "⏩ $s"; continue; }
  echo "[$(date +%H:%M)] === L16-XL $s ==="
  { $PY tools/enhance_net.py train --workspace "workspace_raw/$s" \
      --ckpt "results/${s}__SUB2/ckpts/ckpt_29999_rank0.pt" \
      --out "results/${s}__l16xl/net.pt" --with_ut --radial_k1 "$(k1_of $s)" \
      --steps 8000 --ch_mult 2 --patch 320 \
    && $PY tools/enhance_net.py apply --net "results/${s}__l16xl/net.pt" \
      --in_dir "renders/${s}__SUB2" --out_dir "renders/${s}__l16xl" \
    && rm -rf "results/${s}__l16xl/renders_train" \
    && echo "L16XL-OK $s"
  } || echo "!!! L16XL-FAILED $s"
done
$PY - <<'EOF'
import csv,cv2
from pathlib import Path
flags=[cv2.IMWRITE_JPEG_QUALITY,96];miss=0
for sd in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (sd/"test/test_poses.csv").exists():continue
    s=sd.name;out=Path("renders_sub4")/s;out.mkdir(parents=True,exist_ok=True)
    for r in csv.DictReader(open(sd/"test/test_poses.csv")):
        n=r["image_name"];p=Path(f"renders/{s}__l16xl")/(Path(n).stem+".png")
        if not p.exists():p=Path(f"renders/{s}__l16")/(Path(n).stem+".png")
        if not p.exists():print("THIẾU",s,n);miss+=1;continue
        cv2.imwrite(str(out/n),cv2.imread(str(p)),flags)
print("convert q96 thiếu",miss)
EOF
$PY tools/make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub4 --ext .same --out submission_SUB4.zip \
  && echo "[$(date +%H:%M)] SUB4-ALL-DONE" || echo "!!! PACKAGE-FAILED"
