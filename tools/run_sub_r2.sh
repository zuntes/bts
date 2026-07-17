#!/bin/bash
# ============================================================================
# ROUND 2 — production: train → selfcheck → render → ensemble → L16-XL → zip.
#
# 2 nhánh chiến thuật (bắt buộc, không phải tuỳ chọn — xem prepare_r2.sh):
#   HCM*        (SIMPLE_RADIAL k1≈+0.009): 3DGUT --with-ut --with-eval3d --raw-distortion
#   bonsai/chair(SIMPLE_PINHOLE, video)  : CLASSIC — parser assert nếu bật raw-distortion
#
# ROUND 2 KHÔNG có GT test → sau seed đầu mỗi scene chạy SELFCHECK (render 3 pose
# train qua đúng đường CSV, PSNR>=20) — chặn bug transform trước khi phí GPU.
#
# Config qua env (mặc định an toàn, chỉnh sau khi có gate round 2):
#   SUBTAG=2 SEEDS="42 7 123"  CAP_HCM=...  CAP_OBJ=...  ENH_ARCH=vgg|unet
#   SUBTAG tách output giữa các lần production (renders __sub$SUBTAG, zip R2_SUB$SUBTAG)
# Chạy: tmux new -s r2 && bash tools/run_sub_r2.sh 2>&1 | tee /tmp/run_r2.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK   # chống lây nhiễm transient-mask (DOC3): production KHÔNG mask trừ khi gate bảo bật
PY=.venv/bin/python
# TỰ DÒ root data R2 (layout local có tầng lồng, server phẳng — xem prepare_r2.sh)
R2=""
for c in VAI_NVS_DATA_ROUND_2/VAI_NVS_DATA_ROUND2 VAI_NVS_DATA_ROUND_2 VAI_NVS_DATA_ROUND2; do
  [ -s "$c/bonsai/test/test_poses.csv" ] && { R2="$c"; break; }
done
[ -n "$R2" ] || { echo "❌ không tìm thấy data round 2"; exit 1; }
# Override được qua env để CHIA VIỆC 2 máy (server: SCENES_OBJ="" · 4060: SCENES_HCM="")
SCENES_HCM=${SCENES_HCM-"HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"}
SCENES_OBJ=${SCENES_OBJ-"bonsai chair"}
OBJ_SH_DEGREE=${OBJ_SH_DEGREE:-3}   # O4 gate quyết: 4 nếu SH4 thắng trên holdout bonsai/chair
SEEDS=${SEEDS:-"42 7"}
CAP_HCM=${CAP_HCM:-6000000}     # 1320×989 = 1/4 pixel của round 1 → knee thấp hơn 12M nhiều
CAP_OBJ=${CAP_OBJ:-3000000}     # scene object-centric nhỏ (54-80k điểm SfM)
SUBTAG=${SUBTAG:-1}             # đổi mỗi lần production để không đè/skip nhầm output cũ
ENH_ARCH=${ENH_ARCH:-vgg}       # GATE_B1 17/07: vgg prior thắng L16-unet +0.0052 → mặc định vgg
ZIP=${ZIP:-submission_R2_SUB${SUBTAG}.zip}
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
disk_guard(){ FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 8 ] && die "đĩa ${FREE}GB<8"; }

k1_of(){ $PY - "$1" <<'EOF'
import struct, sys
with open(f"workspace_r2/{sys.argv[1]}/sparse/0/cameras.bin", "rb") as f:
    f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
    if model == 2:
        print(repr(struct.unpack("<dddd", f.read(32))[3]))
    elif model in (0, 1):
        print("0.0")
    else:
        sys.exit(f"model {model} chưa hỗ trợ")
EOF
}

say "0. tiên quyết"
[ -x "$PY" ] || die "thiếu .venv"
$PY -c "from pycolmap import SceneManager" 2>/dev/null || die ".venv hỏng pycolmap rmbrualla (DOC3 §3.7) — chạy GATE_B.sh B2.0 hoặc pip install lại"
for s in $SCENES_HCM $SCENES_OBJ; do
  [ -s "workspace_r2/$s/sparse/0/images.bin" ] || die "thiếu workspace_r2/$s → bash tools/prepare_r2.sh trước"
done
disk_guard
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'
echo "  ✅ SEEDS=[$SEEDS] CAP_HCM=$CAP_HCM CAP_OBJ=$CAP_OBJ · đĩa ${FREE}GB · GPU $CUDA_VISIBLE_DEVICES"

# ---------- train + render 1 scene × 1 seed ----------
train_render_seed(){   # $1=scene $2=seed $3=cap $4=branch(gut|classic)
  local s=$1 seed=$2 cap=$3 branch=$4
  local res="results/r2_${s}__s${seed}" rend="renders_r2/${s}__s${seed}"
  local UT_TRAIN="" UT_REND="" K1=""
  if [ "$branch" = gut ]; then
    K1=$(k1_of "$s") || die "k1_of $s"
    UT_TRAIN="--with-ut --with-eval3d --raw-distortion"
    UT_REND="--with_ut --radial_k1 $K1"
  else
    UT_TRAIN="--sh-degree $OBJ_SH_DEGREE"   # O4: SH4 cho scene vật liệu khó nếu gate thắng
  fi
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    disk_guard
    echo "[$(date +%H:%M)] train $s seed=$seed cap=$cap branch=$branch"
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased $UT_TRAIN \
      --strategy.cap-max "$cap" --eval-steps 30000 --save-steps 30000 --global-seed "$seed" \
      2>&1 | tee "/tmp/r2_${s}_s${seed}.log" || die "train $s seed$seed — xem /tmp/r2_${s}_s${seed}.log"
    rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
  else echo "  ⏩ $s seed$seed ckpt có"; fi

  # SELFCHECK sau seed ĐẦU TIÊN của scene (round 2 không có GT — đây là lưới an toàn duy nhất)
  if [ "$seed" = "${SEEDS%% *}" ] && ! [ -f "$res/SELFCHECK_OK" ]; then
    $PY tools/r2_selfcheck.py gen --ws "workspace_r2/$s" --n 5 --out "/tmp/r2_sc_${s}.csv" || die "selfcheck gen $s"
    rm -rf "/tmp/r2_sc_${s}_render"
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
      --csv "/tmp/r2_sc_${s}.csv" --out "/tmp/r2_sc_${s}_render" --data_dir "workspace_r2/$s" \
      --antialiased $UT_REND 2>&1 | grep -av "render " || die "selfcheck render $s"
    $PY tools/r2_selfcheck.py score --render_dir "/tmp/r2_sc_${s}_render" --ws "workspace_r2/$s" \
      || die "SELFCHECK FAIL $s — transform/nhánh camera SAI, dừng scene này, DÁN LOG CHO CLAUDE"
    touch "$res/SELFCHECK_OK"
  fi

  if [ "$(ls "$rend" 2>/dev/null | wc -l)" -lt 5 ]; then
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
      --csv "$R2/$s/test/test_poses.csv" --out "$rend" --data_dir "workspace_r2/$s" \
      --antialiased $UT_REND 2>&1 | tee -a /tmp/r2_render.log || die "render $s seed$seed"
  fi
  N_EXPECT=$(($(wc -l < "$R2/$s/test/test_poses.csv") - 1))
  [ "$(ls "$rend" | wc -l)" -eq "$N_EXPECT" ] || die "$s seed$seed render $(ls "$rend" | wc -l)/$N_EXPECT ảnh"
}

# ---------- vòng chính ----------
if [ "${PACK_ONLY:-0}" = "1" ]; then
  echo "PACK_ONLY: bỏ train, đóng gói thẳng từ renders_r2/*__sub${SUBTAG}"
  SCENES_HCM="HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"; SCENES_OBJ="bonsai chair"
  for s in $SCENES_OBJ $SCENES_HCM; do
    [ "$(ls "renders_r2/${s}__sub${SUBTAG}" 2>/dev/null | wc -l)" -ge 5 ] \
      || die "PACK_ONLY nhưng thiếu renders_r2/${s}__sub${SUBTAG} — chưa gộp đủ 7 scene (thiếu = LOẠI BÀI)"
  done
fi
[ "${PACK_ONLY:-0}" = "1" ] || for s in $SCENES_OBJ $SCENES_HCM; do
  case " $SCENES_OBJ " in *" $s "*) branch=classic; cap=$CAP_OBJ;; *) branch=gut; cap=$CAP_HCM;; esac
  # cap riêng per-scene qua env CAP_<scene> (vd CAP_bonsai=6000000 — profiler/O-gate quyết)
  eval "override=\${CAP_${s}:-}"; [ -n "$override" ] && cap=$override
  say "SCENE $s ($branch, cap=$cap)"
  [ -d "renders_r2/${s}__sub${SUBTAG}" ] && { echo "  ⏩ $s hoàn tất"; continue; }

  DIRS=""
  for seed in $SEEDS; do
    train_render_seed "$s" "$seed" "$cap" "$branch"
    DIRS="$DIRS renders_r2/${s}__s${seed}"
  done

  N_SEED=$(echo $SEEDS | wc -w)
  ENS="renders_r2/${s}__ens${SUBTAG}"
  if [ "$N_SEED" -gt 1 ]; then
    $PY tools/ensemble.py --dirs $DIRS --out "$ENS" --mode mean >/dev/null || die "ensemble $s"
  else
    rm -rf "$ENS"; cp -r "renders_r2/${s}__s${SEEDS}" "$ENS"
  fi

  # Enhancer per-scene (ENH_ARCH=vgg: B1 restoration prior thắng L16-unet +0.0052, GATE_B 17/07)
  FIRST_SEED=${SEEDS%% *}
  L16_FLAGS=""
  [ "$branch" = gut ] && L16_FLAGS="--with_ut --radial_k1 $(k1_of "$s")"
  NET="results/r2_${s}__enh_${ENH_ARCH}/net.pt"
  if ! [ -s "$NET" ]; then
    $PY tools/enhance_net.py train --workspace "workspace_r2/$s" \
      --ckpt "results/r2_${s}__s${FIRST_SEED}/ckpts/ckpt_29999_rank0.pt" \
      --out "$NET" $L16_FLAGS --arch "$ENH_ARCH" \
      --steps 8000 --ch_mult 2 --patch 320 2>&1 | tee "/tmp/r2_enh_${s}.log" || die "enh train $s"
  fi
  $PY tools/enhance_net.py apply --net "$NET" \
    --in_dir "$ENS" --out_dir "renders_r2/${s}__sub${SUBTAG}" >/dev/null || die "enh apply $s"
  echo "[$(date +%H:%M)] SCENE-OK $s"
done

# chạy chia máy (SCENES_OBJ="" hoặc SCENES_HCM="") → BỎ đóng gói ở máy này,
# gộp renders về 1 máy rồi PACK_ONLY=1 để đóng gói đủ 7 scene (thiếu scene = LOẠI BÀI)
if [ -z "$SCENES_HCM" ] || [ -z "$SCENES_OBJ" ]; then
  echo; echo "⏸ CHẠY MỘT PHẦN (SCENES_HCM='$SCENES_HCM' · SCENES_OBJ='$SCENES_OBJ')"
  echo "  → KHÔNG đóng gói ở đây. Gộp renders_r2/*__sub${SUBTAG} về 1 máy rồi chạy:"
  echo "     PACK_ONLY=1 SUBTAG=$SUBTAG bash tools/run_sub_r2.sh"
  exit 0
fi

say "ĐÓNG GÓI (PNG → JPEG q96 đúng tên CSV → zip)"
$PY - <<EOF || die "repackage"
import csv, cv2, sys
from pathlib import Path
flags = [cv2.IMWRITE_JPEG_QUALITY, 96]; miss = 0
for sd in sorted(Path("$R2").iterdir()):
    if not (sd / "test/test_poses.csv").exists(): continue
    s = sd.name; out = Path("renders_sub_r2") / s; out.mkdir(parents=True, exist_ok=True)
    for r in csv.DictReader(open(sd / "test/test_poses.csv")):
        n = r["image_name"]; p = Path(f"renders_r2/{s}__sub${SUBTAG}") / (Path(n).stem + ".png")
        if not p.exists(): print("THIẾU", s, n); miss += 1; continue
        cv2.imwrite(str(out / n), cv2.imread(str(p)), flags)
print(f"repackage: thiếu {miss}")
sys.exit(1 if miss else 0)
EOF
$PY tools/make_submission.py --data_root "$R2" --renders_root renders_sub_r2 \
  --ext .same --out "$ZIP" || die "make_submission"
ls -la "$ZIP"
echo "[$(date +%d/%m' '%H:%M)] R2-ALL-DONE → $ZIP (nhớ kiểm <350MB và đủ 7 scene ở output trên)"
