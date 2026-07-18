#!/bin/bash
# ============================================================================
# BAN NGÀY 18/07 — train CONFIG-MEMBERS trên FULL data (workspace_r2).
# Đây vừa là "test ban ngày" vừa là THÀNH VIÊN prod tối nay (CE đã chứng minh
# member thắng: HCM mixed-cap +0.0024, chair 4cfg vượt 3-seed).
#
# Member theo scene (từ verdict các gate):
#   HCM*  : cap3M (~25ph) + cap9M (~60ph)   [mixed-cap như ce4]
#   bonsai: 45k SH4 (~50ph)                 [N-45k +0.0021]
#   chair : 45k SH4 (~42ph) + erank SH4 (~28ph)  [thành viên MEGA6]
#
# CHIA 2 GPU qua env MEMBER_SCENES (mặc định tất cả):
#   GPU0: MEMBER_SCENES="HCM0421 HCM0539 HCM0540" (~4.2h)
#   GPU1: MEMBER_SCENES="HCM0644 HCM0674 bonsai chair" (~4.9h)
# Chạy: MEMBER_SCENES="..." bash tools/ADD_MEMBERS.sh 2>&1 | tee /tmp/members_gpuX.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
R2=""; for c in VAI_NVS_DATA_ROUND_2/VAI_NVS_DATA_ROUND2 VAI_NVS_DATA_ROUND_2; do
  [ -s "$c/bonsai/test/test_poses.csv" ] && { R2="$c"; break; }; done
[ -n "$R2" ] || { echo "❌ không thấy data R2"; exit 1; }
MEMBER_SCENES=${MEMBER_SCENES:-"HCM0421 HCM0539 HCM0540 HCM0644 HCM0674 bonsai chair"}
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
k1_of(){ $PY - "$1" <<'EOF'
import struct, sys
with open(f"workspace_r2/{sys.argv[1]}/sparse/0/cameras.bin","rb") as f:
    f.read(8); cid,model,w,h=struct.unpack("<iiQQ",f.read(24))
    print(repr(struct.unpack("<dddd",f.read(32))[3]) if model==2 else "0.0")
EOF
}

train_member(){ # scene tag steps cap extra_train extra_rend
  local s=$1 tag=$2 steps=$3 cap=$4 extra=$5 rext=${6:-}
  local res="results/r2_${s}__m_${tag}" last="results/r2_${s}__m_${tag}/ckpts/ckpt_$((steps-1))_rank0.pt"
  say "$s member:$tag (steps=$steps cap=$cap)"
  if ! [ -s "$last" ]; then
    FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 8 ] && die "đĩa ${FREE}GB<8"
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps "$steps" --test-every 999999 \
      --disable-viewer --antialiased $extra \
      --strategy.cap-max "$cap" --eval-steps "$steps" --save-steps "$steps" --global-seed 42 \
      2>&1 | tee "/tmp/mem_${s}_${tag}.log" || die "train $s $tag"
    find "$res/ckpts" -name "ckpt_*" ! -name "ckpt_$((steps-1))_*" -delete 2>/dev/null; rm -rf "$res/videos"
  else echo "  ⏩ có"; fi
  if [ "$(ls "renders_r2/${s}__m_${tag}" 2>/dev/null | wc -l)" -lt 5 ]; then
    $PY tools/render_test_poses.py --ckpt "$last" --csv "$R2/$s/test/test_poses.csv" \
      --out "renders_r2/${s}__m_${tag}" --data_dir "workspace_r2/$s" --antialiased $rext \
      2>&1 | tail -1 || die "render $s $tag"
  fi
  echo "[$(date +%H:%M)] MEMBER-OK $s $tag"
}

for s in $MEMBER_SCENES; do
  case $s in
    HCM*)
      K1=$(k1_of "$s")
      UT="--with-ut --with-eval3d --raw-distortion"; UR="--with_ut --radial_k1 $K1"
      train_member "$s" cap3M 30000 3000000 "$UT" "$UR"
      train_member "$s" cap9M 30000 9000000 "$UT" "$UR"
      ;;
    bonsai)
      train_member "$s" 45k 45000 3000000 "--sh-degree 4"
      ;;
    chair)
      train_member "$s" 45k 45000 3000000 "--sh-degree 4"
      train_member "$s" erank 30000 3000000 "--sh-degree 4 --erank-reg 0.02"
      ;;
  esac
done
echo "MEMBERS-DONE ($MEMBER_SCENES)"
