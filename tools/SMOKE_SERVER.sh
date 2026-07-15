#!/bin/bash
# ============================================================================
# SMOKE_SERVER.sh — validate TOÀN CHAIN trên máy mới trước khi đốt giờ GPU.
#   prepare → train 500 steps → render test poses → score
#
# MỐC ĐỐI CHIẾU (đo trên 4060, docs/memory gsplat-153-t3-flip):
#   HCM0204, 500 steps → PSNR ≈ 14.9  ✅ chain đúng
#                        PSNR ≈ 9.x   ❌ T3-flip hỏng (normalize_compat sai) → DỪNG
# Cấu hình smoke không khớp bit-exact bản cũ, nên đọc theo NGƯỠNG:
#   >13 = chain OK · 9-11 = pose lệch (T3) · <8 = hỏng nặng
#
#   bash tools/SMOKE_SERVER.sh 2>&1 | tee /tmp/smoke.txt      (~5-10 phút)
# ============================================================================
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

S=HCM0204
SCENE=VAI_NVS_DATA/phase1/public_set/$S
K1=0.010009930826722385          # SIMPLE_RADIAL k1 của HCM0204
CSV=$SCENE/test/test_poses.csv
GT=$SCENE/test/images

echo "########## SMOKE — $(date '+%F %T') — GPU=$CUDA_VISIBLE_DEVICES ##########"
[ -x "$PY" ] || { echo "❌ chưa có .venv → bash tools/SETUP_MAIN_VENV.sh"; exit 1; }

echo "===== 0/4 kiểm SOURCE data (rsync đủ chưa) ====="
# "thư mục tồn tại" KHÔNG đủ: rsync dở dang vẫn tạo thư mục → phải kiểm từng FILE.
FAIL=0
for f in train/sparse/0/cameras.bin train/sparse/0/images.bin train/sparse/0/points3D.bin \
         test/test_poses.csv; do
  if [ -s "$SCENE/$f" ]; then echo "  ✅ $f ($(du -h "$SCENE/$f" | cut -f1))"
  else echo "  ❌ THIẾU/RỖNG: $SCENE/$f"; FAIL=1; fi
done
NTRAIN=$(ls "$SCENE/train/images" 2>/dev/null | wc -l)
NGT=$(ls "$GT" 2>/dev/null | wc -l)
echo "  train/images: $NTRAIN ảnh (kỳ vọng 240) · test/images: $NGT ảnh (kỳ vọng 60)"
[ "$NTRAIN" -lt 240 ] && { echo "  ❌ thiếu ảnh train → rsync CHƯA XONG"; FAIL=1; }
[ "$NGT" -lt 60 ]     && { echo "  ❌ thiếu ảnh test"; FAIL=1; }
[ "$FAIL" -eq 1 ] && { echo; echo "→ DỪNG: data chưa đủ. Đợi rsync xong rồi chạy lại."; exit 1; }

echo; echo "===== 1/4 prepare (lọc ảnh thừa + undistort) ====="
# kiểm OUTPUT THẬT (cameras.bin), không kiểm thư mục: prepare mkdir trước khi ghi .bin,
# nên chết giữa chừng sẽ để lại thư mục rỗng và làm bước sau tưởng đã xong.
if [ -s workspace/$S/sparse/0/cameras.bin ]; then
  echo "workspace/$S đã hoàn chỉnh → bỏ qua"
else
  [ -d workspace/$S ] && { echo "workspace/$S dở dang → xoá, làm lại"; rm -rf workspace/$S; }
  $PY tools/prepare_scene.py --scene_dir "$SCENE" --out_dir workspace/$S
fi
for f in sparse/0/cameras.bin sparse/0/images.bin sparse/0/points3D.bin; do
  [ -s workspace/$S/$f ] || { echo "❌ prepare KHÔNG tạo được workspace/$S/$f"; exit 1; }
done
echo "  ✅ prepare OK: $(ls workspace/$S/images | wc -l) ảnh + sparse/0/*.bin"

echo; echo "===== 2/4 train 500 steps (classic MCMC, cap 200k) ====="
$PY gsplat/examples/simple_trainer.py mcmc \
  --data-dir workspace/$S --data-factor 1 \
  --result-dir "$PWD/results/${S}__smoke" \
  --max-steps 500 --disable-viewer --antialiased --packed \
  --strategy.cap-max 200000 --eval-steps 500 --save-steps 500 --test-every 8

CKPT=results/${S}__smoke/ckpts/ckpt_499_rank0.pt
[ -f "$CKPT" ] || { echo "❌ không thấy $CKPT — train hỏng?"; ls -R results/${S}__smoke/ckpts 2>/dev/null; exit 1; }

echo; echo "===== 3/4 render test poses (COLMAP w2c + T3 + redistort k1) ====="
$PY tools/render_test_poses.py --ckpt "$CKPT" --csv "$CSV" \
  --out renders/${S}__smoke --data_dir workspace/$S --antialiased --redistort_k1 $K1

echo; echo "===== 4/4 score (dòng 'n=... PSNR=...' bên dưới là kết quả) ====="
# LƯU Ý: --out xuất CSV per-image (không phải JSON, dù run cũ đặt đuôi .json).
$PY tools/score_local.py --pred_dir renders/${S}__smoke --gt_dir "$GT" \
  --out results/${S}__smoke_score.csv

echo; echo "########## KẾT LUẬN ##########"
$PY - <<'PY'
import csv, sys
from statistics import mean
try:
    rows = list(csv.DictReader(open("results/HCM0204__smoke_score.csv")))
    p = mean(float(r["psnr"]) for r in rows)
except Exception as e:
    print("❌ không đọc được CSV score:", e); sys.exit(1)
print(f">>> n={len(rows)} ảnh   PSNR trung bình = {p:.2f}   (mốc 4060: ~14.9)")
if   p > 13: print("✅ CHAIN ĐÚNG trên L40S → yên tâm chạy run lớn.")
elif p > 8:  print("❌ NGHI T3-FLIP HỎNG (pose lệch hệ quy chiếu). DỪNG — báo Claude, đừng train tiếp.")
else:        print("❌ HỎNG NẶNG. DỪNG, gửi toàn bộ output cho Claude.")
PY
echo "########## HẾT — gửi output này cho Claude ##########"
