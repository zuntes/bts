#!/bin/bash
# ============================================================================
# GATE R2-HCM — cap tối ưu cho scene HCM round 2 (1320×989 = round 1 ÷ 2 mỗi chiều).
#
# Round 2 private KHÔNG có GT, nhưng public round 1 CÒN NGUYÊN giá trị:
# hạ HCM0204 public xuống đúng 1320×989 (--data-factor 2, gsplat tự downscale;
# k1 không đổi theo scale) → có bàn chấm GT thật ở ĐÚNG chế độ round 2.
#
# Trả lời: [H1] knee cap ở 1320×989 nằm đâu (3M/6M/9M)?  [H2] ensemble 2-seed còn ăn?
# Chạy SERVER sau GATE_A/B (~4-5h): bash tools/GATE_R2_HCM.sh 2>&1 | tee /tmp/gate_r2hcm.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
CSV_FULL="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT_FULL="VAI_NVS_DATA/phase1/public_set/$S/test/images"
CAL=workspace_r2cal
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$CAL/gt_half" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

say "0. tiên quyết + dựng bàn chấm half-res (CSV ÷2 + GT ÷2)"
[ -x "$PY" ] || die "thiếu .venv"
$PY -c "from pycolmap import SceneManager" 2>/dev/null || die ".venv hỏng pycolmap rmbrualla (DOC3 §3.7)"
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S (public round 1)"
[ -f "$CSV_FULL" ] || die "thiếu $CSV_FULL"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 15 ] && die "đĩa ${FREE}GB<15"
if ! [ -s "$CAL/test_poses_half.csv" ] || [ "$(ls "$CAL/gt_half" 2>/dev/null | wc -l)" -lt 60 ]; then
  $PY - <<EOF || die "dựng bàn chấm half-res"
import csv, cv2
from pathlib import Path
cal = Path("$CAL"); (cal / "gt_half").mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open("$CSV_FULL")))
with open(cal / "test_poses_half.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
    for r in rows:
        for k in ("fx", "fy", "cx", "cy"):
            r[k] = str(float(r[k]) / 2)
        r["width"] = str(int(r["width"]) // 2); r["height"] = str(int(r["height"]) // 2)
        w.writerow(r)
for p in Path("$GT_FULL").iterdir():
    dst = cal / "gt_half" / p.name
    if dst.exists(): continue
    im = cv2.imread(str(p))
    cv2.imwrite(str(dst), cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2),
                interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 100])
print(f"✓ CSV half ({len(rows)} pose, {rows[0]['width']}x{rows[0]['height']}) + gt_half")
EOF
else echo "  ⏩ bàn chấm có"; fi

# gsplat --data-factor 2 ĐÒI thư mục images_2 có sẵn (quy ước MipNeRF — parser không tự tạo,
# raise "Image folder ... does not exist"). Tự sinh PNG ÷2 bằng INTER_AREA (cùng phép với gt_half).
if [ "$(ls "workspace_raw/$S/images_2" 2>/dev/null | wc -l)" -lt 240 ]; then
  echo "  → sinh workspace_raw/$S/images_2 (240 ảnh ÷2, INTER_AREA, ~2-3ph)..."
  $PY - <<EOF || die "sinh images_2"
import cv2
from pathlib import Path
src = Path("workspace_raw/$S/images"); dst = Path("workspace_raw/$S/images_2")
dst.mkdir(exist_ok=True)
for p in sorted(src.iterdir()):
    o = dst / (p.stem + ".png")
    if o.exists(): continue
    im = cv2.imread(str(p))
    cv2.imwrite(str(o), cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2),
                                   interpolation=cv2.INTER_AREA))
print("✓ images_2:", len(list(dst.iterdir())), "ảnh")
EOF
else echo "  ⏩ images_2 có"; fi

train_render(){  # $1=cap_tag $2=cap $3=seed
  local tag=$1 cap=$2 seed=$3 res="results/r2cal_${S}__${1}_s${3}"
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 2 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$cap" --eval-steps 30000 --save-steps 30000 --global-seed "$seed" \
      2>&1 | tee "/tmp/r2cal_${tag}_s${seed}.log" || die "train $tag s$seed"
    rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
  else echo "  ⏩ $tag s$seed có"; fi
  if [ "$(ls "renders_r2cal/${tag}_s${seed}" 2>/dev/null | wc -l)" -lt 60 ]; then
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
      --csv "$CAL/test_poses_half.csv" --out "renders_r2cal/${tag}_s${seed}" \
      --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 $K1 \
      2>&1 | grep -av "render " || die "render $tag s$seed"
  fi
}

say "H1. quét cap 3M / 6M / 9M @1320×989 (seed 42)"
train_render cap3M 3000000 42;  score renders_r2cal/cap3M_s42 "H1-3M"
train_render cap6M 6000000 42;  score renders_r2cal/cap6M_s42 "H1-6M"
train_render cap9M 9000000 42;  score renders_r2cal/cap9M_s42 "H1-9M"

say "H2. seed 7 @6M + ensemble(42,7) — xác nhận ensemble còn ăn ở half-res"
train_render cap6M 6000000 7
$PY tools/ensemble.py --dirs renders_r2cal/cap6M_s42 renders_r2cal/cap6M_s7 \
  --out renders_r2cal/ens6M --mode mean >/dev/null || die "ensemble"
score renders_r2cal/ens6M "H2-ens2x6M"

echo
echo "########################################################################"
echo "#  VERDICT GATE R2-HCM (các dòng ★ ở trên là v50 trên bàn half-res)"
echo "#  [H1] chọn cap nhỏ nhất trong vòng 0.002 của cap tốt nhất → CAP_HCM cho run_sub_r2"
echo "#       (nếu 9M vẫn tăng dốc ≥+0.004 so 6M → chạy thêm cap12M rồi hỏi Claude)"
echo "#  [H2] ens − cap6M ≥ +0.003 → giữ 2 seed; < → cân nhắc 1 seed để tiết kiệm giờ GPU"
echo "#  L16-XL KHÔNG gate ở đây (production train net trên chính scene r2 native-res)"
echo "########################################################################"
echo "DÁN [H1][H2] + khối này CHO CLAUDE."
