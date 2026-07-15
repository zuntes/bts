#!/bin/bash
# ============================================================================
# SUB1 — train submission bằng phương pháp tốt nhất đã kiểm chứng: 3DGUT (MCMC
# + --with-ut --with-eval3d --raw-distortion, render méo native, thắng A/B 10/7).
#
#   Scenes : 8 private (nộp) + HCM0181 public (hiệu chỉnh điểm local ↔ BTC)
#            HCM0204 public KHÔNG train lại — tái dùng kết quả ut3M_30k từ thang leo.
#   Config : cap 3M (HCM1439: 1.5M vì chỉ 103 ảnh) × 30k steps; fallback 1.5M nếu 3M fail.
#   Output : submission_SUB1.zip — ảnh .JPG (IN HOA, đúng tên CSV), quality 100.
#
# Tự ĐỢI thang leo ut_scale xong mới chiếm GPU. Mỗi scene độc lập (scene lỗi
# không chặn scene sau). Chạy:  setsid nohup bash tools/run_sub1.sh > results/sub1.log 2>&1 &
# Theo dõi:  tail -f results/sub1.log
# ============================================================================
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date +%d/%m' '%H:%M)] SUB1 khởi động — đợi thang leo ut_scale giải phóng GPU..."
while ! grep -q "UT-SCALE-ALL-DONE" results/ut_scale.log 2>/dev/null; do sleep 120; done
echo "[$(date +%d/%m' '%H:%M)] GPU rảnh. Bắt đầu."

# --- chọn cap: 3M nếu thang leo 3M thành công, ngược lại 1.5M đã kiểm chứng ---
if [ -f results/HCM0204__ut3M_30k_score.json ]; then CAP=3000000; else CAP=1500000; fi
STEPS=30000
TAG=SUB1UT
echo "[config] 3DGUT cap=$CAP steps=$STEPS tag=$TAG"

PRIVATE="HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437"
CALIB="HCM0181"

k1_of () {
  $PY - "$1" <<'EOF'
import struct, sys
with open(f"workspace_raw/{sys.argv[1]}/sparse/0/cameras.bin", "rb") as f:
    f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
    params = struct.unpack("<dddd", f.read(32))  # SIMPLE_RADIAL: f cx cy k1
print(repr(params[3]))
EOF
}

for s in $PRIVATE $CALIB; do
  cap=$CAP; [ "$s" = "HCM1439" ] && cap=$((CAP / 2))
  k1=$(k1_of "$s")
  src=VAI_NVS_DATA/phase1/private_set1/$s
  [ -d "$src" ] || src=VAI_NVS_DATA/phase1/public_set/$s
  echo "[$(date +%d/%m' '%H:%M)] === SCENE $s cap=$cap k1=$k1 ==="
  {
    $PY gsplat/examples/simple_trainer.py mcmc \
      --data-dir "workspace_raw/$s" --data-factor 1 \
      --result-dir "$PWD/results/${s}__${TAG}" \
      --max-steps $STEPS --test-every 999999 --disable-viewer --antialiased \
      --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$cap" --eval-steps $STEPS --save-steps 15000 $STEPS \
    && $PY tools/render_test_poses.py \
      --ckpt "results/${s}__${TAG}/ckpts/ckpt_$((STEPS-1))_rank0.pt" \
      --csv "$src/test/test_poses.csv" --out "renders/${s}__${TAG}" \
      --data_dir "workspace_raw/$s" --antialiased --with_ut --radial_k1 "$k1" \
    && echo "[$(date +%d/%m' '%H:%M)] SCENE-OK $s"
  } || echo "[$(date +%d/%m' '%H:%M)] !!! SCENE-FAILED $s"
done

# --- score 2 public calib (HCM0181 vừa train + HCM0204 tái dùng từ thang leo) ---
$PY tools/score_local.py --pred_dir "renders/${CALIB}__${TAG}" \
  --gt_dir "VAI_NVS_DATA/phase1/public_set/$CALIB/test/images" \
  --out "results/${CALIB}__${TAG}_score.json" || echo "!!! score $CALIB lỗi"
echo "[calib] HCM0204 (tái dùng ut3M_30k): xem results/HCM0204__ut3M_30k_score.json"

# --- convert PNG → .JPG (IN HOA, đúng tên CSV, quality 100, chroma 4:4:4) và đóng gói ---
$PY - <<'EOF'
import csv, cv2
from pathlib import Path
TAG = "SUB1UT"
flags = [cv2.IMWRITE_JPEG_QUALITY, 100]
if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
    flags += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
missing = 0
for scene_dir in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (scene_dir / "test/test_poses.csv").exists():
        continue
    s = scene_dir.name
    out = Path("renders_sub1") / s
    out.mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(scene_dir / "test/test_poses.csv")):
        name = r["image_name"]                      # vd DJI_..._V.JPG (IN HOA sẵn)
        png = Path("renders") / f"{s}__{TAG}" / (Path(name).stem + ".png")
        if not png.exists():
            print(f"THIẾU {s}/{name}"); missing += 1; continue
        cv2.imwrite(str(out / name), cv2.imread(str(png)), flags)
print(f"convert xong, thiếu {missing} ảnh")
EOF

$PY tools/make_submission.py \
  --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub1 --ext .same --out submission_SUB1.zip \
  && echo "[$(date +%d/%m' '%H:%M)] SUB1-ALL-DONE → submission_SUB1.zip" \
  || echo "!!! PACKAGE-FAILED"
