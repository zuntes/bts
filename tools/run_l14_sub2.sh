#!/bin/bash
# ============================================================================
# Giai đoạn cuối 12-13/7:
#   1) A/B L14 (TestFrustumMCMC) trên HCM0204 @ 5M — so với ut5M_30k đã có
#   2) TỰ CHỌN config thắng (L14 phải hơn base ≥0.002 trên CẢ a35 và v35)
#   3) SUB2 production: 9 scenes còn lại @ 5M no-ft (HCM1439: 2.5M) → zip
#      (HCM0204 tái dùng từ nhánh thắng — không train lại)
# Chạy: setsid nohup bash tools/run_l14_sub2.sh > results/l14_sub2.log 2>&1 &
# ============================================================================
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
csv_of () {
  local src=VAI_NVS_DATA/phase1/private_set1/$1
  [ -d "$src" ] || src=VAI_NVS_DATA/phase1/public_set/$1
  echo "$src/test/test_poses.csv"
}

train_render () {  # $1=scene $2=cap $3=tag $4=extra-flags(may be empty)
  local src=VAI_NVS_DATA/phase1/private_set1/$1
  [ -d "$src" ] || src=VAI_NVS_DATA/phase1/public_set/$1
  echo "[$(date +%d/%m' '%H:%M)] === SCENE $1 cap=$2 tag=$3 extra='$4' ==="
  { $PY gsplat/examples/simple_trainer.py mcmc \
      --data-dir "workspace_raw/$1" --data-factor 1 \
      --result-dir "$PWD/results/$1__$3" \
      --max-steps 30000 --test-every 999999 --disable-viewer --antialiased \
      --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$2" --eval-steps 30000 --save-steps 15000 30000 $4 \
    && $PY tools/render_test_poses.py \
      --ckpt "results/$1__$3/ckpts/ckpt_29999_rank0.pt" \
      --csv "$(csv_of $1)" --out "renders/$1__$3" \
      --data_dir "workspace_raw/$1" --antialiased --with_ut --radial_k1 "$(k1_of $1)" \
    && echo "[$(date +%d/%m' '%H:%M)] SCENE-OK $1 ($3)"
  } || echo "[$(date +%d/%m' '%H:%M)] !!! SCENE-FAILED $1 ($3)"
}

# ---- 1) A/B L14 trên HCM0204 (test-every 8 để so công bằng với ut5M_30k) ----
if [ ! -f results/HCM0204__ut5M_l14_score.json ]; then
  echo "[$(date +%d/%m' '%H:%M)] === A/B L14 HCM0204 @5M ==="
  { $PY gsplat/examples/simple_trainer.py mcmc \
      --data-dir workspace_raw/HCM0204 --data-factor 1 \
      --result-dir "$PWD/results/HCM0204__ut5M_l14" \
      --max-steps 30000 --test-every 8 --disable-viewer --antialiased \
      --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max 5000000 --eval-steps 30000 --save-steps 15000 30000 \
      --test-frustum-csv "$(csv_of HCM0204)" \
    && $PY tools/render_test_poses.py \
      --ckpt results/HCM0204__ut5M_l14/ckpts/ckpt_29999_rank0.pt \
      --csv "$(csv_of HCM0204)" --out renders/HCM0204__ut5M_l14 \
      --data_dir workspace_raw/HCM0204 --antialiased --with_ut --radial_k1 "$(k1_of HCM0204)" \
    && $PY tools/score_local.py --pred_dir renders/HCM0204__ut5M_l14 \
      --gt_dir VAI_NVS_DATA/phase1/public_set/HCM0204/test/images \
      --out results/HCM0204__ut5M_l14_score.json \
    && echo "STEP-OK l14_ab"
  } || echo "!!! STEP-FAILED l14_ab"
fi

# ---- 2) chọn config: L14 thắng nếu hơn base ≥0.002 trên CẢ a35 lẫn v35 ----
L14_FLAGS_MARKER=$($PY - <<'EOF'
import csv
def scores(path):
    import numpy as np
    rows = list(csv.DictReader(open(path)))
    ps = np.mean([float(r["psnr"]) for r in rows])
    ss = np.mean([float(r["ssim"]) for r in rows])
    la = np.mean([float(r["lpips_alex"]) for r in rows])
    lv = np.mean([float(r["lpips_vgg"]) for r in rows])
    a35 = 0.4*(1-la) + 0.3*ss + 0.3*min(ps/35, 1)
    v35 = 0.4*(1-lv) + 0.3*ss + 0.3*min(ps/35, 1)
    return a35, v35
try:
    ba, bv = scores("results/HCM0204__ut5M_30k_score.json")
    la_, lv_ = scores("results/HCM0204__ut5M_l14_score.json")
    win = (la_ - ba >= 0.002) and (lv_ - bv >= 0.002)
    print(f"USE_L14={int(win)}  # base a35={ba:.5f} v35={bv:.5f} | l14 a35={la_:.5f} v35={lv_:.5f}")
except Exception as e:
    print(f"USE_L14=0  # lỗi so sánh: {e}")
EOF
)
echo "[quyết định] $L14_FLAGS_MARKER"
USE_L14=$(echo "$L14_FLAGS_MARKER" | grep -o "USE_L14=[01]" | cut -d= -f2)

# ---- 3) SUB2 production: 9 scenes (HCM0204 tái dùng) ----
TAG=SUB2
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437 HCM0181; do
  cap=5000000; [ "$s" = "HCM1439" ] && cap=2500000
  extra=""
  [ "$USE_L14" = "1" ] && extra="--test-frustum-csv $(csv_of $s)"
  train_render "$s" "$cap" "$TAG" "$extra"
done

$PY tools/score_local.py --pred_dir renders/HCM0181__SUB2 \
  --gt_dir VAI_NVS_DATA/phase1/public_set/HCM0181/test/images \
  --out results/HCM0181__SUB2_score.json || echo "!!! score HCM0181 lỗi"

# HCM0204 cho renders_sub2: dùng nhánh thắng
if [ "$USE_L14" = "1" ]; then SRC204=renders/HCM0204__ut5M_l14; else SRC204=renders/HCM0204__ut5M_30k; fi
echo "[calib] HCM0204 dùng $SRC204"

# ---- 4) đóng gói ----
$PY - <<'EOF'
import csv, cv2
from pathlib import Path
flags = [cv2.IMWRITE_JPEG_QUALITY, 96]
missing = 0
for scene_dir in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (scene_dir / "test/test_poses.csv").exists(): continue
    s = scene_dir.name
    out = Path("renders_sub2") / s; out.mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(scene_dir / "test/test_poses.csv")):
        name = r["image_name"]
        png = Path("renders") / f"{s}__SUB2" / (Path(name).stem + ".png")
        if not png.exists(): print("THIẾU", s, name); missing += 1; continue
        cv2.imwrite(str(out / name), cv2.imread(str(png)), flags)
print("convert q96 xong, thiếu", missing)
EOF
$PY tools/make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub2 --ext .same --out submission_SUB2.zip \
  && echo "[$(date +%d/%m' '%H:%M)] SUB2-ALL-DONE → submission_SUB2.zip" \
  || echo "!!! PACKAGE-FAILED"
