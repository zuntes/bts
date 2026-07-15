#!/bin/bash
# ============================================================================
# MAX POWER (12/7, sau điểm BTC 0.7522): vắt kiệt 4060 8GB trên HCM0204.
# Thứ tự theo giá trị/giờ GPU:
#   1) L2 LPIPS-ft (vgg) + L13 trên ckpt ut3M CÓ SẴN — λ 0.1 rồi 0.05 (~40ph/cái)
#   2) L1 cap 4M × 30k  (~2.7h, VRAM dự kiến ~5GB)
#   3) L1 cap 5M × 30k  (~3.5h, VRAM ~6GB — thăm dò trần, chấp nhận rủi ro OOM)
# Mỗi bước độc lập. Chạy: setsid nohup bash tools/run_max_power.sh > results/max_power.log 2>&1 &
# ============================================================================
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
K1=0.010009930826722385
CSV=VAI_NVS_DATA/phase1/public_set/HCM0204/test/test_poses.csv
GT=VAI_NVS_DATA/phase1/public_set/HCM0204/test/images
BASE=results/HCM0204__ut3M_30k/ckpts/ckpt_29999_rank0.pt

render_score () {  # $1 = ckpt, $2 = tag
  $PY tools/render_test_poses.py --ckpt "$1" \
    --csv "$CSV" --out "renders/HCM0204__$2" \
    --data_dir workspace_raw/HCM0204 --antialiased --with_ut --radial_k1 $K1 \
  && $PY tools/score_local.py --pred_dir "renders/HCM0204__$2" \
    --gt_dir "$GT" --out "results/HCM0204__$2_score.json"
}

# ---- 1) LPIPS fine-tune trên ckpt 3M sẵn có ----
for LAM in 0.1 0.05; do
  TAG="ut3M_ft${LAM/0./}"
  echo "[$(date +%d/%m' '%H:%M)] === FT lambda=$LAM tag=$TAG ==="
  { $PY tools/finetune_lpips.py --workspace workspace_raw/HCM0204 \
      --ckpt "$BASE" --out "results/HCM0204__$TAG/ckpt_ft.pt" \
      --lambda_lpips "$LAM" --steps 3000 --antialiased --with_ut \
      --test_csv "$CSV" \
    && render_score "results/HCM0204__$TAG/ckpt_ft.pt" "$TAG" \
    && echo "[$(date +%d/%m' '%H:%M)] STEP-OK $TAG"
  } || echo "[$(date +%d/%m' '%H:%M)] !!! STEP-FAILED $TAG"
done

# ---- 2+3) cap 4M rồi 5M ----
for CAP in 4000000 5000000; do
  TAG="ut$((CAP/1000000))M_30k"
  echo "[$(date +%d/%m' '%H:%M)] === TRAIN cap=$CAP tag=$TAG ==="
  { $PY gsplat/examples/simple_trainer.py mcmc \
      --data-dir workspace_raw/HCM0204 --data-factor 1 \
      --result-dir "$PWD/results/HCM0204__$TAG" \
      --max-steps 30000 --test-every 8 --disable-viewer --antialiased \
      --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$CAP" --eval-steps 30000 --save-steps 15000 30000 \
    && render_score "results/HCM0204__$TAG/ckpts/ckpt_29999_rank0.pt" "$TAG" \
    && echo "[$(date +%d/%m' '%H:%M)] STEP-OK $TAG"
  } || echo "[$(date +%d/%m' '%H:%M)] !!! STEP-FAILED $TAG"
done

echo "[$(date +%d/%m' '%H:%M)] MAX-POWER-ALL-DONE"
