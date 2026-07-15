#!/bin/bash
# SUB3 = SUB2 (5M) + L16 enhancement net per-scene.
# HCM0204/HCM0181 đã có net+renders; chạy 7 private còn lại rồi đóng gói.
# Chạy: setsid nohup bash tools/run_sub3.sh > results/sub3.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

k1_of () {
  $PY - "$1" <<'EOF'
import struct, sys
with open(f"workspace_raw/{sys.argv[1]}/sparse/0/cameras.bin", "rb") as f:
    f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
    params = struct.unpack("<dddd", f.read(32))
print(repr(params[3]))
EOF
}

for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437; do
  [ -d "renders/${s}__l16" ] && { echo "⏩ $s đã có l16"; continue; }
  echo "[$(date +%d/%m' '%H:%M)] === L16 $s ==="
  { $PY tools/enhance_net.py train --workspace "workspace_raw/$s" \
      --ckpt "results/${s}__SUB2/ckpts/ckpt_29999_rank0.pt" \
      --out "results/${s}__l16/net.pt" --with_ut --radial_k1 "$(k1_of $s)" --steps 3000 \
    && $PY tools/enhance_net.py apply --net "results/${s}__l16/net.pt" \
      --in_dir "renders/${s}__SUB2" --out_dir "renders/${s}__l16" \
    && rm -rf "results/${s}__l16/renders_train" \
    && echo "[$(date +%d/%m' '%H:%M)] L16-OK $s"
  } || echo "[$(date +%d/%m' '%H:%M)] !!! L16-FAILED $s"
done

$PY - <<'EOF'
import csv, cv2
from pathlib import Path
flags = [cv2.IMWRITE_JPEG_QUALITY, 96]
missing = 0
for scene_dir in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (scene_dir / "test/test_poses.csv").exists(): continue
    s = scene_dir.name
    out = Path("renders_sub3") / s; out.mkdir(parents=True, exist_ok=True)
    base = Path(f"renders/{s}__l16")
    fb = Path(f"renders/{s}__SUB2")  # fallback nếu L16 fail scene đó
    for r in csv.DictReader(open(scene_dir / "test/test_poses.csv")):
        name = r["image_name"]
        png = base / (Path(name).stem + ".png")
        if not png.exists(): png = fb / (Path(name).stem + ".png")
        if not png.exists(): print("THIẾU", s, name); missing += 1; continue
        cv2.imwrite(str(out / name), cv2.imread(str(png)), flags)
print("convert q96 xong, thiếu", missing)
EOF
$PY tools/make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub3 --ext .same --out submission_SUB3.zip \
  && echo "[$(date +%d/%m' '%H:%M)] SUB3-ALL-DONE → submission_SUB3.zip" \
  || echo "!!! PACKAGE-FAILED"
