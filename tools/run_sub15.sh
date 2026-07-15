#!/bin/bash
# ============================================================================
# Kịch bản 12/7 (phần A):
#   0) ft λ0.05 trên ckpt 5M (retry với --lpips_patch 320 sau OOM crop 512)
#   1) SUB1.5 = ft λ0.05 lên 9 ckpt SUB1UT (8 private + HCM0181 calib) + render
#   2) convert q96 → submission_SUB1_5.zip  (bản nộp thuần cải thiện — best-submission an toàn)
# Chạy: setsid nohup bash tools/run_sub15.sh > results/sub15.log 2>&1 &
# ============================================================================
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# PATCH=512 = đúng config đã đo +0.004 v35 trên ckpt 3M (512 OOM chỉ ở cap 5M);
# HCM1439 ckpt 1.5M càng dư. KHÔNG hạ xuống 320 — ft 320 trên 5M đã đo là kém hơn.
PATCH=512

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

# ---- 0) retry ft 5M (verify cộng dồn ở cap trần) — ĐÃ XONG 12/7: ft làm 5M
# TỆ ĐI (a35 −0.003, v35 −0.001; LPIPS headroom nhỏ không bù nổi SSIM) → skip ----
if [ -f results/HCM0204__ut5M_ft05_score.json ]; then
  echo "ft5M đã có kết quả — bỏ qua"
else
echo "[$(date +%d/%m' '%H:%M)] === FT5M retry patch=$PATCH ==="
{ $PY tools/finetune_lpips.py --workspace workspace_raw/HCM0204 \
    --ckpt results/HCM0204__ut5M_30k/ckpts/ckpt_29999_rank0.pt \
    --out results/HCM0204__ut5M_ft05/ckpt_ft.pt \
    --lambda_lpips 0.05 --steps 3000 --antialiased --with_ut --lpips_patch $PATCH \
    --test_csv "$(csv_of HCM0204)" \
  && $PY tools/render_test_poses.py --ckpt results/HCM0204__ut5M_ft05/ckpt_ft.pt \
    --csv "$(csv_of HCM0204)" --out renders/HCM0204__ut5M_ft05 \
    --data_dir workspace_raw/HCM0204 --antialiased --with_ut --radial_k1 "$(k1_of HCM0204)" \
  && $PY tools/score_local.py --pred_dir renders/HCM0204__ut5M_ft05 \
    --gt_dir VAI_NVS_DATA/phase1/public_set/HCM0204/test/images \
    --out results/HCM0204__ut5M_ft05_score.json \
  && echo "STEP-OK ft5M"
} || echo "!!! STEP-FAILED ft5M"
fi

# ---- 1) SUB1.5: ft + render 9 scenes từ ckpt SUB1UT (3M) ----
for s in HCM0249 HCM0254 HCM0276 HCM1439 HNI0131 HNI0265 HNI0366 HNI0437 HCM0181; do
  echo "[$(date +%d/%m' '%H:%M)] === SUB1.5 $s ==="
  { $PY tools/finetune_lpips.py --workspace "workspace_raw/$s" \
      --ckpt "results/${s}__SUB1UT/ckpts/ckpt_29999_rank0.pt" \
      --out "results/${s}__SUB15/ckpt_ft.pt" \
      --lambda_lpips 0.05 --steps 3000 --antialiased --with_ut --lpips_patch $PATCH \
      --test_csv "$(csv_of $s)" \
    && $PY tools/render_test_poses.py --ckpt "results/${s}__SUB15/ckpt_ft.pt" \
      --csv "$(csv_of $s)" --out "renders/${s}__SUB15" \
      --data_dir "workspace_raw/$s" --antialiased --with_ut --radial_k1 "$(k1_of $s)" \
    && echo "[$(date +%d/%m' '%H:%M)] SCENE-OK $s"
  } || echo "[$(date +%d/%m' '%H:%M)] !!! SCENE-FAILED $s"
done

$PY tools/score_local.py --pred_dir renders/HCM0181__SUB15 \
  --gt_dir VAI_NVS_DATA/phase1/public_set/HCM0181/test/images \
  --out results/HCM0181__SUB15_score.json || echo "!!! score HCM0181 lỗi"

# ---- 2) đóng gói (chỉ 8 private) ----
$PY - <<'EOF'
import csv, cv2
from pathlib import Path
flags = [cv2.IMWRITE_JPEG_QUALITY, 96]
missing = 0
for scene_dir in sorted(Path("VAI_NVS_DATA/phase1/private_set1").iterdir()):
    if not (scene_dir / "test/test_poses.csv").exists(): continue
    s = scene_dir.name
    out = Path("renders_sub15") / s; out.mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(scene_dir / "test/test_poses.csv")):
        name = r["image_name"]
        png = Path("renders") / f"{s}__SUB15" / (Path(name).stem + ".png")
        if not png.exists(): print("THIẾU", s, name); missing += 1; continue
        cv2.imwrite(str(out / name), cv2.imread(str(png)), flags)
print("convert q96 xong, thiếu", missing)
EOF
$PY tools/make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
  --renders_root renders_sub15 --ext .same --out submission_SUB1_5.zip \
  && echo "[$(date +%d/%m' '%H:%M)] SUB15-ALL-DONE → submission_SUB1_5.zip" \
  || echo "!!! PACKAGE-FAILED"
