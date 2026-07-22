#!/bin/bash
# ============================================================================
# GATE_APP — PHƯƠNG ÁN MỚI: app_opt (appearance embedding) trên HCM0204 GT thật.
# Exposure per-view (240 ảnh drone 1 chuyến bay có biến thiên phơi sáng) hấp thụ
# vào embedding → geometry không "trả giá" → PSNR có thể tăng. Test embed=zero (GLO).
# BƯỚC 0 VALIDATE BRIDGE (train-view PSNR>26) trước khi tin điểm test.
# Mốc base 3M ≈ 0.751 (mega). ≥+0.002 → phương án mới THẮNG.
# Chạy: setsid nohup bash tools/GATE_APP.sh > results/gate_app.log 2>&1 &
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
gpu_busy(){ local g free tot thr; g=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  thr=$(( tot*55/100 )); [ "$thr" -gt 18000 ] && thr=18000; [ "${free:-0}" -lt "$thr" ]; }
for i in $(seq 1 600); do gpu_busy || break; [ "$i" = 1 ] && echo "[$(date +%H:%M)] chờ GPU rảnh..."; sleep 60; done

res="results/app_${S}__ao3M"; ck="$res/ckpts/ckpt_29999_rank0.pt"
say "1. TRAIN app_opt (3M, 30k, HCM0204 UT+k1)"
if ! [ -s "$ck" ]; then
  $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
    --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
    --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
    --strategy.cap-max 3000000 --eval-steps 30000 --save-steps 30000 --global-seed 42 \
    --app-opt > /tmp/app_train.log 2>&1
  [ -s "$ck" ] || { echo "TRAIN FAIL:"; tail -5 /tmp/app_train.log; die "train app_opt"; }
  rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
fi

say "2. VALIDATE BRIDGE (train-view PSNR — bắt render-sai-âm-thầm)"
$PY tools/render_app.py --ckpt "$ck" --ws "workspace_raw/$S" --validate \
  --with_ut --radial_k1 $K1 --antialiased 2>&1 | grep -aE "PSNR|VALIDATE|❌|✅" || die "validate lỗi"
if $PY tools/render_app.py --ckpt "$ck" --ws "workspace_raw/$S" --validate --with_ut --radial_k1 $K1 --antialiased 2>&1 | grep -q "❌"; then
  die "BRIDGE SAI — dừng, không render test (tránh kết luận trên render rác)"
fi

say "3. RENDER test + score"
$PY tools/render_app.py --ckpt "$ck" --ws "workspace_raw/$S" --csv "$CSV" \
  --out "renders_app/ao3M" --with_ut --radial_k1 $K1 --antialiased 2>&1 | tail -2
$PY tools/score_local.py --pred_dir "renders_app/ao3M" --gt_dir "$GT" 2>&1 | grep -aE "n=|★"

echo
echo "########################################################################"
echo "#  VERDICT APP_OPT — so base 3M (~0.751). ≥+0.002 → phương án MỚI thắng → prod."
echo "#  app_opt hấp thụ exposure per-view. Nếu validate <26dB = bridge sai (bỏ điểm test)."
echo "########################################################################"
echo "APP-GATE-DONE"
