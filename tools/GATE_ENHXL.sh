#!/bin/bash
# ============================================================================
# GATE ENH-XL (server, chạy SAU GATE_AFLAGS trên cùng GPU — cần ckpt r2cal cap6M):
# Prototype "đòn chất tuần 2": enhancer vgg TO HƠN có ăn thêm không?
#   [X0] vgg 8k/patch320  (baseline đúng công thức prod hiện tại)
#   [X1] vgg 20k/patch512 (2.5× steps, patch to — nhìn ngữ cảnh rộng hơn)
# Bàn: half-res HCM0204, áp lên renders_r2cal/ens6M (đã có, mốc raw = 0.81102)
# ~2.5h. Chạy: bash tools/GATE_ENHXL.sh 2>&1 | tee /tmp/gate_enhxl.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "workspace_r2cal/gt_half" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

CKPT="results/r2cal_${S}__cap6M_s42/ckpts/ckpt_29999_rank0.pt"
[ -s "$CKPT" ] || die "thiếu ckpt r2cal cap6M (đừng xoá — AFLAGS/ENHXL cần)"
[ -d renders_r2cal/ens6M ] || die "thiếu renders_r2cal/ens6M (GATE_R2_HCM H2)"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 10 ] && die "đĩa ${FREE}GB<10"

# LƯU Ý: enhancer train trên renders_train của ckpt 6M — render qua Dataset half-res
# (ckpt half-res + workspace_raw full-res: enhance_net render theo cameras.bin =
#  1320x989 — ĐÚNG thang half-res vì cameras.bin vốn lưu 1320, xem DOC3 §2.4)
for V in "x0 8000 320" "x1 20000 512"; do
  set -- $V; tag=$1 steps=$2 patch=$3
  say "ENH-$tag: vgg steps=$steps patch=$patch"
  NET="results/r2cal_enh_${tag}/net.pt"
  if ! [ -s "$NET" ]; then
    $PY tools/enhance_net.py train --workspace "workspace_raw/$S" \
      --ckpt "$CKPT" --out "$NET" --arch vgg --with_ut --radial_k1 $K1 \
      --steps "$steps" --patch "$patch" 2>&1 | tee "/tmp/enhxl_${tag}.log" || die "train $tag"
  fi
  $PY tools/enhance_net.py apply --net "$NET" \
    --in_dir renders_r2cal/ens6M --out_dir "renders_r2cal/ens6M_${tag}" >/dev/null || die "apply $tag"
  score "renders_r2cal/ens6M_${tag}" "X-$tag"
done

echo
echo "########################################################################"
echo "#  VERDICT ENH-XL — mốc ens6M raw = 0.81102"
echo "#  [X-x0] = vgg chuẩn prod trên bench này (điểm tham chiếu)"
echo "#  [X-x1] − [X-x0] ≥ +0.002 → prod nâng enhancer lên 20k/512 (thêm ~40ph/scene)"
echo "#  [X-x1] − [X-x0] ≥ +0.004 → tuần 2 đầu tư tiếp enhancer sâu hơn (đòn chất)"
echo "########################################################################"
echo "DÁN [X-*] CHO CLAUDE."
