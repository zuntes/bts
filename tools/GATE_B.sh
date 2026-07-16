#!/bin/bash
# ============================================================================
# GATE_B — B1 (restoration prior) + B2 (pose-refine) + B3 (transient-mask), TUẦN TỰ.
# Chạy SAU GATE_A (B1 cần renders/HCM0204__ens12 + ckpt cap12M từ A). ~4h10.
# Chạy: tmux new -s gb && bash tools/GATE_B.sh 2>&1 | tee /tmp/gate_b.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

say "0. tiên quyết"
[ -x "$PY" ] || die "thiếu .venv"
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 30 ] && die "đĩa ${FREE}GB<30"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
echo "  ✅ ok · đĩa ${FREE}GB · GPU $CUDA_VISIBLE_DEVICES"

# ============================== B1 — RESTORATION PRIOR ==============================
say "B1. restoration-prior (encoder VGG16 pretrained) trên ens12 (~40ph)"
if [ -d "renders/${S}__ens12" ]; then
  # tái dùng renders_train của A1 (cùng ckpt cap12M) — đỡ re-render 240 view (~10-15ph)
  if [ -d "results/${S}__l16xl12/renders_train" ] && ! [ -e "results/${S}__b1vgg/renders_train" ]; then
    mkdir -p "results/${S}__b1vgg"
    ln -s "$PWD/results/${S}__l16xl12/renders_train" "results/${S}__b1vgg/renders_train"
  fi
  if ! [ -s results/${S}__b1vgg/net.pt ]; then
    $PY tools/enhance_net.py train --workspace "workspace_raw/$S" \
      --ckpt "results/${S}__cap12M/ckpts/ckpt_29999_rank0.pt" \
      --out results/${S}__b1vgg/net.pt --with_ut --radial_k1 $K1 \
      --arch vgg --steps 8000 --patch 320 2>&1 | tee /tmp/b1_train.log \
      || die "B1 train"
  fi
  $PY tools/enhance_net.py apply --net results/${S}__b1vgg/net.pt \
    --in_dir renders/${S}__ens12 --out_dir renders/${S}__b1 >/dev/null || die "B1 apply"
  score renders/${S}__b1 "B1-vgg-prior"
  echo "  MỐC A1 (ens12+L16-XL, xem /tmp/gate_a.txt) → Δ≥+0.002 thì thay L16-XL bằng B1"
else
  echo "  ⏩ thiếu renders/${S}__ens12 (GATE_A chưa chạy/chưa tới A1) — BỎ QUA B1, làm B2/B3 trước"
fi

# ============================== B2 — POSE REFINE ==============================
say "B2.0 HAI gói 'pycolmap' TRÙNG TÊN → tách 2 venv (bài học 16/07, DOC3 §3.7)"
# 'pycolmap' rmbrualla (version 0.0.1, có SceneManager — gsplat BẮT BUỘC để train) và
# 'pycolmap' chính thức COLMAP (4.1.0, có Reconstruction/BA — pose_refine cần) TRÙNG TÊN.
# Đã trả giá: force-reinstall 4.1.0 vào .venv đè mất SceneManager → MỌI train gsplat gãy.
# → .venv giữ bản rmbrualla; BA chạy bằng venv riêng .venv_ba.

# 0a. .venv phải import được SceneManager (tự chữa 2 bệnh đã gặp 16/07):
#   bệnh 1: pycolmap 4.1.0 đè mất bản rmbrualla → ImportError SceneManager
#   bệnh 2: 4.1.0 kéo numpy 1.26.4→2.2.6; code rmbrualla có np.uint64(-1) →
#           OverflowError NGAY LÚC IMPORT dưới numpy 2.x (NEP 50) — đã tái hiện local
if ! $PY -c "from pycolmap import SceneManager" 2>/dev/null; then
  echo "  ⚠ .venv không import được SceneManager → khôi phục numpy 1.26.4 + pycolmap rmbrualla..."
  $PY -m pip install -q --no-cache-dir "numpy==1.26.4" || die "pin numpy==1.26.4 fail"
  $PY -m pip install -q --force-reinstall --no-deps --no-cache-dir \
    "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e" \
    || die "cài lại pycolmap rmbrualla fail — dán output pip"
fi
# verify KHÔNG nuốt stderr — nếu vẫn fail thì traceback nằm ngay phía trên
$PY -c "from pycolmap import SceneManager; import numpy; print('  ✅ .venv: pycolmap rmbrualla (SceneManager) + numpy', numpy.__version__, '— train gsplat OK')" \
  || die ".venv vẫn không import được SceneManager — DÁN TRACEBACK PHÍA TRÊN cho Claude"

# 0b. venv riêng .venv_ba cho pycolmap chính thức (KHÔNG BAO GIỜ cài 4.1.0 vào .venv)
BA_PY=.venv_ba/bin/python
if ! [ -x "$BA_PY" ]; then
  echo "  → tạo .venv_ba (chỉ chứa pycolmap COLMAP chính thức + numpy)..."
  python3.10 -m venv .venv_ba || die "tạo .venv_ba fail (cần python3.10 trên PATH)"
  $BA_PY -m pip install -q --upgrade pip
fi
if ! $BA_PY -c "import pycolmap; assert hasattr(pycolmap,'Reconstruction')" 2>/dev/null; then
  $BA_PY -m pip install -q --no-cache-dir --index-url https://pypi.org/simple "pycolmap==4.1.0" numpy \
    || die "pip pycolmap==4.1.0 vào .venv_ba fail — mirror nội bộ chặn pypi.org?"
fi
$BA_PY -c "
import pycolmap, importlib.metadata
assert hasattr(pycolmap, 'Reconstruction')
print('  ✅ .venv_ba: pycolmap', importlib.metadata.version('pycolmap'), '(Reconstruction/BA — gói COLMAP thật)')
" || die ".venv_ba pycolmap vẫn không dùng được — dán output pip index versions pycolmap"

say "B2.1 refine STAGE-1 (pose+points, intrinsics cố định) — ĐỌC KỸ dòng in ra"
# Mặc định CHỈ stage 1 (không --refine_intrinsics): cô lập đúng giả thuyết "pose nhiễu",
# tránh rủi ro phân kỳ đã gặp khi refine intrinsics+pose cùng lúc (xem DOC3 §2.3).
if [ -s "workspace_ref/$S/sparse/0/images.bin" ] && [ -f "workspace_ref/$S/REFINE_OK" ]; then
  echo "  ⏩ workspace_ref/$S đã có (marker REFINE_OK — kết quả BA hội tụ)"
else
  if [ -d "workspace_ref/$S" ]; then
    echo "  ⚠ workspace_ref/$S tồn tại nhưng THIẾU marker REFINE_OK → rác từ lần BA fail/phân kỳ, XOÁ làm lại"
    rm -rf "workspace_ref/$S"
  fi
  rm -f /tmp/b2_refine.txt
  $BA_PY tools/pose_refine.py --in_ws "workspace_raw/$S" --out_ws "workspace_ref/$S" 2>&1 \
    | tee /tmp/b2_refine.txt || die "pose_refine fail — dán /tmp/b2_refine.txt"
fi
# đọc k1 THẲNG từ cameras.bin của workspace_ref (không phụ thuộc /tmp/b2_refine.txt —
# file đó không tồn tại nếu đi nhánh skip "đã có")
K1R=$($PY - <<'EOF'
import struct
with open("workspace_ref/HCM0204/sparse/0/cameras.bin","rb") as f:
    f.read(8); cid,model,w,h = struct.unpack("<iiQQ", f.read(24))
    if model == 2:   # SIMPLE_RADIAL: f,cx,cy,k1
        print(repr(struct.unpack("<dddd", f.read(32))[3]))
    else:            # OPENCV (stage 2): fx,fy,cx,cy,k1,k2,p1,p2 → in k1
        print(repr(struct.unpack("<dddddddd", f.read(64))[4]))
EOF
)
# sanity: k1 gốc HCM0204 ~ +0.01; BA phân kỳ từng ghi ra k1=0.314 → chặn giá trị vô lý
$PY -c "import sys; k=float('$K1R'); sys.exit(0 if abs(k) < 0.2 else 1)" \
  || die "k1=$K1R VÔ LÝ (gốc ~+0.01, |k1|≥0.2 = rác BA phân kỳ) → rm -rf workspace_ref/$S rồi chạy lại"
echo "  render B2 sẽ dùng: k1=$K1R (stage 1 → SIMPLE_RADIAL giữ nguyên, k2/tangential=0)"

say "B2.2 train 12M trên workspace REFINED (~90ph)"
if ! [ -s "results/${S}__ref12M/ckpts/ckpt_29999_rank0.pt" ]; then
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_ref/$S" --data-factor 1 \
    --result-dir "$PWD/results/${S}__ref12M" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max 12000000 --eval-steps 30000 --save-steps 30000 \
    2>&1 | tee /tmp/b2_train.log || die "train ref"
  rm -f results/${S}__ref12M/ckpts/ckpt_14999_rank0.pt; rm -rf results/${S}__ref12M/videos
fi

say "B2.3 render (test poses gốc, intrinsics refined) + score"
$PY tools/render_test_poses.py --ckpt "results/${S}__ref12M/ckpts/ckpt_29999_rank0.pt" \
  --csv "$CSV" --out "renders/${S}__ref12M" --data_dir "workspace_ref/$S" \
  --antialiased --with_ut --radial_k1 "$K1R" \
  2>&1 | tee /tmp/b2_render.log
score renders/${S}__ref12M "B2"

# ===================== B2X (tuỳ chọn, RUN_B2X=1) — STAGE-2 INTRINSICS =====================
if [ "${RUN_B2X:-0}" = "1" ]; then
  say "B2X. refine STAGE-2 (OPENCV intrinsics, pose khoá) + train + score (~100ph)"
  if [ -s "workspace_ref2/$S/sparse/0/images.bin" ] && [ -f "workspace_ref2/$S/REFINE_OK" ]; then
    echo "  ⏩ workspace_ref2/$S đã có (marker REFINE_OK)"
  else
    [ -d "workspace_ref2/$S" ] && { echo "  ⚠ workspace_ref2/$S thiếu marker → xoá làm lại"; rm -rf "workspace_ref2/$S"; }
    $BA_PY tools/pose_refine.py --in_ws "workspace_raw/$S" --out_ws "workspace_ref2/$S" \
      --refine_intrinsics 2>&1 | tee /tmp/b2x_refine.txt || die "pose_refine stage2 fail"
  fi
  if ! [ -s "results/${S}__ref2_12M/ckpts/ckpt_29999_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_ref2/$S" --data-factor 1 \
      --result-dir "$PWD/results/${S}__ref2_12M" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max 12000000 --eval-steps 30000 --save-steps 30000 \
      2>&1 | tee /tmp/b2x_train.log || die "train ref2"
    rm -f results/${S}__ref2_12M/ckpts/ckpt_14999_rank0.pt; rm -rf results/${S}__ref2_12M/videos
  fi
  K1X=$(grep -oE "k1=[+-][0-9.]+" /tmp/b2x_refine.txt | tail -1 | cut -d= -f2)
  K2X=$(grep -oE "k2=[+-][0-9.]+" /tmp/b2x_refine.txt | tail -1 | cut -d= -f2)
  P1X=$(grep -oE "p1=[+-][0-9.]+" /tmp/b2x_refine.txt | tail -1 | cut -d= -f2)
  P2X=$(grep -oE "p2=[+-][0-9.]+" /tmp/b2x_refine.txt | tail -1 | cut -d= -f2)
  $PY tools/render_test_poses.py --ckpt "results/${S}__ref2_12M/ckpts/ckpt_29999_rank0.pt" \
    --csv "$CSV" --out "renders/${S}__ref2_12M" --data_dir "workspace_ref2/$S" \
    --antialiased --with_ut --radial_k1 "$K1X" --radial_k2 "$K2X" --tangential "$P1X" "$P2X" \
    2>&1 | tee /tmp/b2x_render.log
  score renders/${S}__ref2_12M "B2X"
fi

# ============================== B3 — TRANSIENT MASK ==============================
say "B3.1 sinh transient masks (deeplabv3, ~10ph — tải weights 160MB lần đầu)"
if [ "$(ls workspace_raw/$S/transient_masks 2>/dev/null | wc -l)" -lt 200 ]; then
  $PY tools/make_transient_masks.py --workspace "workspace_raw/$S" --vis 3 || die "make masks"
else echo "  ⏩ masks đã có"; fi
echo "  → soi mắt: workspace_raw/$S/transient_vis/*_vis.jpg (đỏ = bị mask)"

say "B3.2 train 12M VỚI transient mask (~90ph) — BTS_TMASK=1 tường minh"
if ! [ -s "results/${S}__tmask12M/ckpts/ckpt_29999_rank0.pt" ]; then
  BTS_TMASK=1 $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
    --result-dir "$PWD/results/${S}__tmask12M" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max 12000000 --eval-steps 30000 --save-steps 30000 \
    2>&1 | tee /tmp/b3_train.log || die "train tmask"
  rm -f results/${S}__tmask12M/ckpts/ckpt_14999_rank0.pt; rm -rf results/${S}__tmask12M/videos
fi

say "B3.3 render + score"
$PY tools/render_test_poses.py --ckpt "results/${S}__tmask12M/ckpts/ckpt_29999_rank0.pt" \
  --csv "$CSV" --out "renders/${S}__tmask12M" --data_dir "workspace_raw/$S" \
  --antialiased --with_ut --radial_k1 $K1 2>&1 | tee /tmp/b3_render.log
score renders/${S}__tmask12M "B3"

echo
echo "########################################################################"
echo "#  VERDICT GATE_B — mốc plain12M = 0.75587 · mốc A1 = xem /tmp/gate_a.txt"
echo "#  [B1] ≥ [A1]+0.002 → thay L16-XL bằng B1 cho production"
echo "#  [B2] ≥ 0.75587+0.004 → ĐÒN LỚN: refine toàn scene trước production"
echo "#  [B2] 0.75587+0.001..+0.004 → dùng, cộng dồn miễn phí"
echo "#  [B3] ≥ 0.75587+0.002 → BẬT transient-mask cho production"
echo "#  B2+B3 cộng dồn được (sửa pose khác sửa loss, trực giao)"
echo "########################################################################"
echo "DÁN [B1] [B2] [B3] + khối này CHO CLAUDE."
