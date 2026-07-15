#!/bin/bash
# Resume SUB2 sau sự cố đĩa đầy 13/7 01:24 (4/9 scene đã xong).
# Thêm: guard đĩa trống trước mỗi scene + xóa ckpt 15k sau khi scene OK.
# Chạy: setsid nohup bash tools/run_sub2_resume.sh > results/sub2_resume.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=SUB2

k1_of () {
  $PY - "$1" <<'EOF'
import struct, sys
with open(f"workspace_raw/{sys.argv[1]}/sparse/0/cameras.bin", "rb") as f:
    f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
    params = struct.unpack("<dddd", f.read(32))
print(repr(params[3]))
EOF
}
csv_of () {
  local src=VAI_NVS_DATA/phase1/private_set1/$1
  [ -d "$src" ] || src=VAI_NVS_DATA/phase1/public_set/$1
  echo "$src/test/test_poses.csv"
}

for s in HNI0131 HNI0265 HNI0366 HNI0437 HCM0181; do
  if [ -f "results/${s}__${TAG}/ckpts/ckpt_29999_rank0.pt" ] && [ -d "renders/${s}__${TAG}" ]; then
    echo "⏩ $s đã xong — bỏ qua"; continue
  fi
  FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  if [ "$FREE_GB" -lt 8 ]; then
    echo "!!! DISK-LOW ${FREE_GB}GB — DỪNG để tránh hỏng run"; break
  fi
  cap=5000000
  echo "[$(date +%d/%m' '%H:%M)] === SCENE $s cap=$cap (đĩa trống ${FREE_GB}GB) ==="
  rm -rf "results/${s}__${TAG}"   # dọn tàn dư run fail
  src=VAI_NVS_DATA/phase1/private_set1/$s
  [ -d "$src" ] || src=VAI_NVS_DATA/phase1/public_set/$s
  { $PY gsplat/examples/simple_trainer.py mcmc \
      --data-dir "workspace_raw/$s" --data-factor 1 \
      --result-dir "$PWD/results/${s}__${TAG}" \
      --max-steps 30000 --test-every 999999 --disable-viewer --antialiased \
      --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$cap" --eval-steps 30000 --save-steps 15000 30000 \
    && $PY tools/render_test_poses.py \
      --ckpt "results/${s}__${TAG}/ckpts/ckpt_29999_rank0.pt" \
      --csv "$(csv_of $s)" --out "renders/${s}__${TAG}" \
      --data_dir "workspace_raw/$s" --antialiased --with_ut --radial_k1 "$(k1_of $s)" \
    && rm -f "results/${s}__${TAG}/ckpts/ckpt_14999_rank0.pt" \
    && echo "[$(date +%d/%m' '%H:%M)] SCENE-OK $s"
  } || echo "[$(date +%d/%m' '%H:%M)] !!! SCENE-FAILED $s"
done

$PY tools/score_local.py --pred_dir renders/HCM0181__SUB2 \
  --gt_dir VAI_NVS_DATA/phase1/public_set/HCM0181/test/images \
  --out results/HCM0181__SUB2_score.json || echo "!!! score HCM0181 lỗi"
echo "[calib] HCM0204 dùng renders/HCM0204__ut5M_30k"

$PY - <<'EOF'
import csv, cv2
from pathlib import Path
flags = [cv2.IMWRITE_JPEG_QUALITY, 96]
srcmap = {"HCM0204": "renders/HCM0204__ut5M_30k"}
missing = 0
for scene_dir in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (scene_dir / "test/test_poses.csv").exists(): continue
    s = scene_dir.name
    out = Path("renders_sub2") / s; out.mkdir(parents=True, exist_ok=True)
    base = Path(srcmap.get(s, f"renders/{s}__SUB2"))
    for r in csv.DictReader(open(scene_dir / "test/test_poses.csv")):
        name = r["image_name"]
        png = base / (Path(name).stem + ".png")
        if not png.exists(): print("THIẾU", s, name); missing += 1; continue
        cv2.imwrite(str(out / name), cv2.imread(str(png)), flags)
print("convert q96 xong, thiếu", missing)
EOF
$PY tools/make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub2 --ext .same --out submission_SUB2.zip \
  && echo "[$(date +%d/%m' '%H:%M)] SUB2-ALL-DONE → submission_SUB2.zip" \
  || echo "!!! PACKAGE-FAILED"
