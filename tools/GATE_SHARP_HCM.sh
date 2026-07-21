#!/bin/bash
# ============================================================================
# GATE SHARP-HCM — đòn CHỈ dành cho scene NÉT (drone), nơi 71% điểm nằm.
# Chạy trên HCM0204 FULL-RES round-1 (GT thật, KHÔNG lệch như bàn r2cal half-res
# đã đo +2dB lạc quan). Mốc B1 full-res = 0.77158 · plain-stack = xem log.
#
# 3 đòn CHƯA ĐO trên scene nét (một số bị loại TRÊN SCENE MỜ — có thể sống ở nét):
#   [Q1] MCMC reg sweep: opacity_reg/scale_reg đang dùng DEFAULT PAPER (0.01/0.01),
#        CHƯA tune cho data ta (paper: dataset-dependent, DeepBlending dùng 0.001).
#        Reg thấp → giữ nhiều gaussian đóng góp → chi tiết cao tần (SOTA: HiDeGS).
#   [Q2] supersample×2 khi render: LOẠI trên obj mờ (−0.012, "nét hơn = xa GT mờ hơn")
#        NHƯNG GT của HCM NÉT → nét hơn có thể KHỚP hơn. Phase C ghi "đáng retest trên UT".
#   [Q3] noise_lr sweep (MCMC exploration): mặc định 5e5, chưa từng đụng.
#
# Bàn: workspace_raw/HCM0204 full-res + GT test thật. Cap 6M (knee full-res đã đo là 12M
# nhưng ta test tương đối giữa các biến thể cùng cap → 6M rẻ, đủ phân biệt).
# Chạy: setsid nohup bash tools/GATE_SHARP_HCM.sh > results/gate_sharp_hcm.log 2>&1 &
# Env: CAP=6000000 · SEED=42
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
CAP=${CAP:-6000000}; SEED=${SEED:-42}
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }

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
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S (data round-1)"
[ -f "$CSV" ] || die "thiếu $CSV"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 10 ] && die "đĩa ${FREE}GB<10"
# auto-cap theo VRAM (full-res HCM0204 2640×1978 nặng): trần ≈ (VRAM_MiB−5000)×1000.
# 4060 8GB → ~3.2M (6M OOM chắc chắn) · L40S 46GB → thừa cho 6M. Chỉ hạ, không nâng.
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
MAXCAP=$(( (VRAM - 5000) * 1000 ))
if [ "$CAP" -gt "$MAXCAP" ]; then
  echo "  ⚠ VRAM ${VRAM}MiB → hạ CAP $CAP→$MAXCAP (tránh OOM full-res). Muốn 6M: chạy trên L40S."
  CAP=$MAXCAP
fi
echo "  CAP=$CAP · SEED=$SEED · VRAM=${VRAM}MiB"

score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>/dev/null | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

# train 1 biến thể (extra flags cho phần TRAIN)
train(){  # $1=tag $2=extra_train_flags
  local tag=$1 extra=$2
  local res="results/qhcm_${S}__${tag}"
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$CAP" --eval-steps 30000 --save-steps 30000 --global-seed "$SEED" \
      $extra > "/tmp/qhcm_${tag}.log" 2>&1
    [ -s "$res/ckpts/ckpt_29999_rank0.pt" ] || die "train $tag — xem /tmp/qhcm_${tag}.log"
    rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
  else echo "  ⏩ $tag ckpt có"; fi
  echo "$res/ckpts/ckpt_29999_rank0.pt"
}

# render (tuỳ chọn supersample) + score
rend_score(){  # $1=ckpt $2=tag $3=ss(1|2)
  local ck=$1 tag=$2 ss=${3:-1}
  local rend="renders_qhcm/${tag}"
  local ssflag=""; [ "$ss" = 2 ] && ssflag="--supersample 2"
  if [ "$(ls "$rend" 2>/dev/null | wc -l)" -lt 60 ]; then
    $PY tools/render_test_poses.py --ckpt "$ck" --csv "$CSV" --out "$rend" \
      --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 $K1 $ssflag \
      2>&1 | grep -av "render " || die "render $tag"
  fi
  score "$rend" "$tag"
}

say "BASE — cấu hình prod hiện tại (opacity_reg=scale_reg=0.01 default)"
CK=$(train base ""); rend_score "$CK" "base" 1

say "Q1a — opacity_reg thấp 0.001 (giữ nhiều hạt cao tần)"
CK=$(train oreg001 "--opacity-reg 0.001"); rend_score "$CK" "oreg001" 1

say "Q1b — scale_reg thấp 0.001 (cho hạt nhỏ hơn = chi tiết mịn)"
CK=$(train sreg001 "--scale-reg 0.001"); rend_score "$CK" "sreg001" 1

say "Q3 — noise_lr cao 2e6 (MCMC explore mạnh hơn)"
CK=$(train noise2e6 "--strategy.noise-lr 2000000"); rend_score "$CK" "noise2e6" 1

say "Q2 — supersample×2 lúc render BASE (đòn loại trên MỜ, retest trên NÉT)"
CK="results/qhcm_${S}__base/ckpts/ckpt_29999_rank0.pt"
rend_score "$CK" "base_ss2" 2

echo
echo "########################################################################"
echo "#  VERDICT GATE SHARP-HCM (bàn HCM0204 FULL-RES, GT thật):"
echo "#  so mỗi biến thể với [base]. ≥ +0.002 → ĐÁNG port sang 5 scene HCM prod."
echo "#  ⚠ HCM = 71% trọng số: +0.002 v50/scene ≈ +0.7 điểm tổng (đòn bẩy lớn nhất)."
echo "#  Q2 supersample: nếu +→ scene NÉT KHÁC scene mờ đúng như giả thuyết; nếu − → xác nhận loại."
echo "########################################################################"
echo "SHARP-HCM-GATE-DONE"
