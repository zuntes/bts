#!/bin/bash
# ============================================================================
# GATE DEPTH-CLASSIC — depth-loss trên nhánh CLASSIC (bonsai/chair) — LỖ HỔNG THẬT.
# depth-loss CHỈ từng NaN trên UT/HCM (F1) = incompat kỹ thuật, KHÔNG phải phép đo.
# Classic branch dùng RGB+ED chuẩn → chạy được. Theory: ghim geometry vùng ít texture
# bằng SfM depth — hợp scene video sparse (bonsai 54k điểm). Bàn: holdout val_gt.
# Mốc SH4: chair 0.66358 · bonsai 0.71402
# Chạy: setsid nohup bash tools/GATE_DEPTH_CLASSIC.sh > results/gate_depth.log 2>&1 &
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
gpu_busy(){ local g free tot thr; g=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  thr=$(( tot*55/100 )); [ "$thr" -gt 18000 ] && thr=18000; [ "${free:-0}" -lt "$thr" ]; }
for i in $(seq 1 300); do gpu_busy || break; [ "$i" = 1 ] && echo "chờ GPU..."; sleep 60; done
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "workspace_r2v/$2/val_gt" 2>/dev/null | grep -aE "n=|★" | sed "s/^/  [$3] /"; }

for s in chair bonsai; do
  for lam in 0.01 0.001; do
    tag="depth${lam/./}"
    res="results/r2v_${s}__${tag}"; rend="renders_r2v/${s}__${tag}"
    FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 6 ] && die "đĩa ${FREE}GB<6"
    say "$s — depth-loss lambda=$lam (SH4 cap3M classic)"
    if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
      $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2v/$s" --data-factor 1 \
        --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
        --disable-viewer --antialiased --sh-degree 4 --strategy.cap-max 3000000 \
        --eval-steps 30000 --save-steps 30000 --global-seed 42 \
        --depth-loss --depth-lambda "$lam" > "/tmp/depth_${s}_${tag}.log" 2>&1
      if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
        echo "  ⚠ $s $tag fail:"; tail -3 "/tmp/depth_${s}_${tag}.log" | sed 's/^/    /'; continue
      fi
      rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
    else echo "  ⏩ ckpt có"; fi
    if [ "$(ls "$rend" 2>/dev/null | wc -l)" -lt 5 ]; then
      $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
        --csv "workspace_r2v/$s/val_poses.csv" --out "$rend" \
        --data_dir "workspace_r2v/$s" --antialiased 2>&1 | tail -1
    fi
    score "$rend" "$s" "DEPTH-$s-$lam"
  done
done
echo
echo "########################################################################"
echo "#  VERDICT DEPTH-CLASSIC — mốc SH4 chair 0.66358 bonsai 0.71402"
echo "#  ≥+0.002 → depth-loss vào prod obj (member/config). <0 → đóng depth trên classic luôn."
echo "########################################################################"
echo "DEPTH-CLASSIC-GATE-DONE"
