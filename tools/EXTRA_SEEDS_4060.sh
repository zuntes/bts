#!/bin/bash
# ============================================================================
# EXTRA_SEEDS_4060 — member seed thứ 4-5 cho bonsai/chair (luật 1/√N còn +0.002-0.004/scene).
# Chạy trên 4060 song song với prod server. Output đặt tên __m_s<seed> để
# FINALIZE_SUB2 (glob renders_r2/<s>__m_*) TỰ NHẶT sau khi rsync lên server.
# Chạy: setsid nohup bash tools/EXTRA_SEEDS_4060.sh > results/xseeds_4060.log 2>&1 &
# Theo dõi train: tail -f /tmp/xseed_<scene>_<seed>.log
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
say(){ echo "[$(date +%H:%M)] $*"; }
die(){ echo "❌ $*"; exit 1; }

$PY -c "import torch; assert torch.cuda.is_available(), 'CUDA chưa sống sau driver fix'" || die "torch/CUDA lỗi"
R2=""
for c in VAI_NVS_DATA_ROUND_2/VAI_NVS_DATA_ROUND2 VAI_NVS_DATA_ROUND_2 VAI_NVS_DATA_ROUND2; do
  [ -s "$c/bonsai/test/test_poses.csv" ] && { R2="$c"; break; }
done
[ -n "$R2" ] || die "thiếu data round 2"
for s in chair bonsai; do
  [ -s "workspace_r2/$s/sparse/0/images.bin" ] || die "thiếu workspace_r2/$s"
done

for s in chair bonsai; do
  for seed in 777 2024; do
    res="results/r2_${s}__m_s${seed}"; rend="renders_r2/${s}__m_s${seed}"
    FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 6 ] && die "đĩa ${FREE}GB<6"
    if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
      say "train $s seed=$seed (SH4 cap3M 30k, ~1h40 — log: /tmp/xseed_${s}_${seed}.log)"
      $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2/$s" --data-factor 1 \
        --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
        --disable-viewer --antialiased --sh-degree 4 --strategy.cap-max 3000000 \
        --eval-steps 30000 --save-steps 30000 --global-seed "$seed" \
        > "/tmp/xseed_${s}_${seed}.log" 2>&1
      [ -s "$res/ckpts/ckpt_29999_rank0.pt" ] || die "train $s s$seed không ra ckpt — xem /tmp/xseed_${s}_${seed}.log"
      rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
    else say "⏩ $s s$seed ckpt có"; fi
    if [ "$(ls "$rend" 2>/dev/null | wc -l)" -lt 5 ]; then
      $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
        --csv "$R2/$s/test/test_poses.csv" --out "$rend" --data_dir "workspace_r2/$s" \
        --antialiased > "/tmp/xseed_rend_${s}_${seed}.log" 2>&1
    fi
    N_EXPECT=$(($(wc -l < "$R2/$s/test/test_poses.csv") - 1))
    [ "$(ls "$rend" | wc -l)" -eq "$N_EXPECT" ] || die "$s s$seed render thiếu ($(ls "$rend" | wc -l)/$N_EXPECT)"
    say "XSEED-OK $s s$seed → $rend ($N_EXPECT ảnh)"
  done
done
echo
echo "XSEEDS-DONE — đưa lên server (chạy từ LOCAL):"
echo "  rsync -avP renders_r2/chair__m_s777 renders_r2/chair__m_s2024 renders_r2/bonsai__m_s777 renders_r2/bonsai__m_s2024 pcn-robot@500308175-GPU-02:~/bts/renders_r2/"
echo "  (PHẢI rsync xong TRƯỚC khi FINALIZE chạy trên server — nếu trễ thì thôi, FINALIZE vẫn đủ 5 member)"
