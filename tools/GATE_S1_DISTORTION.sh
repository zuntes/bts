#!/bin/bash
# ============================================================================
# GATE S1 — MÉO ỐNG KÍNH ĐÚNG (đòn PSNR lớn nhất chưa đo, nhắm 71% điểm HCM).
#
# PHÁT HIỆN: cameras.bin lưu SIMPLE_RADIAL k1≈+0.010 (fit 1-param của lens thật),
# nhưng audit_distortion đo méo THẬT = k1≈−0.0012 + k2≈+0.0171 (nhất quán 5/5 HCM r2
# + HCM0204 r1). Ta đang train+render với méo SAI → geometry lệch hệ thống → chặn PSNR.
# Sửa: truyền méo ĐO ĐƯỢC vào cả train (--dist-k*-override) lẫn render (--radial_k1/k2).
# KHÁC B2 (đã loại −0.027): B2 dịch POSE phá nhất quán 409-pose; S1 KHÓA pose nguyên gốc,
# chỉ đổi mô hình méo intrinsic — hệ quy chiếu KHÔNG đổi.
#
# Bàn: HCM0204 FULL-RES round-1, GT thật. Mốc base = méo hiện tại [+0.010, 0].
# HCM0204 đo: k1=-0.00117 k2=+0.01710 p1=+0.00016 p2=-0.00034
# Chạy (server GPU1 khuyến nghị, 6M full-res):
#   tmux new -d -s s1 "CUDA_VISIBLE_DEVICES=1 bash tools/GATE_S1_DISTORTION.sh 2>&1 | tee /tmp/gate_s1.txt"
# Env: CAP=6000000 (server) hoặc 3000000 (4060) · SEED=42
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S=HCM0204
K1_STORED=0.010009930826722385
# méo THẬT đo bằng audit_distortion (HCM0204 r1)
K1_TRUE=-0.00117; K2_TRUE=0.01710
CAP=${CAP:-6000000}; SEED=${SEED:-42}
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
# check FREE mem trên ĐÚNG GPU được gán (server share tenant: GPU nào cũng có VLLM/unsloth
# chiếm sẵn vài GB → check "used<2000" SAI; và head -1 luôn đọc GPU0 bất kể CUDA_VISIBLE_DEVICES).
# Cần ~16GB cho 6M full-res → coi là bận nếu FREE < 18GB.
# bận nếu FREE < ngưỡng adaptive = min(18GB, 55% tổng VRAM) trên ĐÚNG GPU được gán
# (18GB cứng làm 4060 8GB treo mãi; 55% total để server share-tenant vẫn qua).
gpu_busy(){ local g free tot thr; g=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  thr=$(( tot*55/100 )); [ "$thr" -gt 18000 ] && thr=18000
  [ "${free:-0}" -lt "$thr" ]; }
for i in $(seq 1 300); do gpu_busy || break; [ "$i" = 1 ] && echo "[$(date +%H:%M)] chờ GPU..."; sleep 60; done
gpu_busy && die "GPU bận >5h"

[ -x "$PY" ] || die "thiếu .venv"
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S"
[ -f "$CSV" ] || die "thiếu $CSV"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 10 ] && die "đĩa ${FREE}GB<10"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
MAXCAP=$(( (VRAM-5000)*1000 )); [ "$CAP" -gt "$MAXCAP" ] && { echo "  ⚠ hạ CAP→$MAXCAP (VRAM ${VRAM})"; CAP=$MAXCAP; }
echo "  CAP=$CAP SEED=$SEED · base méo=[$K1_STORED,0] · S1 méo=[$K1_TRUE,$K2_TRUE]"
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>/dev/null | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

# train (override méo) + render (méo khớp) + score
run(){  # $1=tag $2=train_override_flags $3=render_k1 $4=render_k2
  local tag=$1 ovr=$2 rk1=$3 rk2=$4
  local res="results/s1_${S}__${tag}" rend="renders_s1/${tag}"
  say "$tag — train méo=[$rk1,$rk2]"
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$CAP" --eval-steps 30000 --save-steps 30000 --global-seed "$SEED" \
      $ovr > "/tmp/s1_${tag}.log" 2>&1
    grep -a "raw_distortion" "/tmp/s1_${tag}.log" | head -1 | sed 's/^/    /'
    [ -s "$res/ckpts/ckpt_29999_rank0.pt" ] || die "train $tag — /tmp/s1_${tag}.log"
    rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
  else echo "  ⏩ ckpt có"; fi
  if [ "$(ls "$rend" 2>/dev/null | wc -l)" -lt 60 ]; then
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" --csv "$CSV" \
      --out "$rend" --data_dir "workspace_raw/$S" --antialiased --with_ut \
      --radial_k1 "$rk1" --radial_k2 "$rk2" 2>&1 | grep -av "render " || die "render $tag"
  fi
  score "$rend" "$tag"
}

run base   ""                                              "$K1_STORED" "0"
run k2add  "--dist-k2-override $K2_TRUE"                    "$K1_STORED" "$K2_TRUE"
run true   "--dist-k1-override $K1_TRUE --dist-k2-override $K2_TRUE"  "$K1_TRUE" "$K2_TRUE"

echo
echo "########################################################################"
echo "#  VERDICT S1 (bàn HCM0204 FULL-RES, GT thật, mốc [base]):"
echo "#  [k2add]  = base k1 + k2 đo được   [true] = k1,k2 đo được đầy đủ"
echo "#  ≥ +0.002 → port méo đo được sang 5 scene HCM prod (audit đã có sẵn per-scene)"
echo "#  ⚠ HCM 71% trọng số: +0.002/scene = +0.7 điểm tổng. Nếu [true]>[k2add] → k1 cũng sai."
echo "#  Nếu CẢ HAI < base → convention méo render/train lệch, DÁN LOG cho Claude."
echo "########################################################################"
echo "S1-GATE-DONE"
