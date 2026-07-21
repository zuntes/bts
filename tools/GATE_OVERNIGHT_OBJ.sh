#!/bin/bash
# ============================================================================
# GATE_OVERNIGHT_OBJ — battery THAM SỐ chưa test trên chair/bonsai (holdout GT).
# chair=69.82 (YẾU NHẤT, neo cứng). Nối tiếp SAU depth-classic (tự chờ GPU).
# Mọi biến thể so SH4-base holdout: chair 0.66358 · bonsai 0.71402.
# Chạy 4060: setsid nohup bash tools/GATE_OVERNIGHT_OBJ.sh > results/night_obj.log 2>&1 &
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
CACHE=results/night_obj_cache; mkdir -p "$CACHE"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
gpu_busy(){ local g free tot thr; g=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  thr=$(( tot*55/100 )); [ "$thr" -gt 18000 ] && thr=18000; [ "${free:-0}" -lt "$thr" ]; }
for i in $(seq 1 600); do gpu_busy || break; [ "$i" = 1 ] && echo "[$(date +%H:%M)] chờ depth-classic xong (GPU)..."; sleep 60; done

declare -A BASE=( [chair]=0.66358 [bonsai]=0.71402 )
run(){  # $1=scene $2=tag $3=steps $4=extra
  local s=$1 tag=$2 steps=$3 extra=$4
  local res="results/nobj_${s}__${tag}" rend="renders_nobj/${s}__${tag}" cf="$CACHE/${s}_${tag}.v50"
  local ck="$res/ckpts/ckpt_$((steps-1))_rank0.pt"
  local stop=$(( steps > 30000 ? steps*5/6 : 25000 ))
  if [ -s "$cf" ]; then local v; v=$(cat "$cf"); printf "  [%-8s %-9s] %s Δ=%s (cache)\n" "$s" "$tag" "$v" "$(.venv/bin/python -c "print(f'{$v-${BASE[$s]}:+.5f}')")"; return 0; fi
  say "$s/$tag — steps=$steps refine_stop=$stop $extra"
  local FREE; FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 6 ] && die "đĩa ${FREE}GB<6"
  if ! [ -s "$ck" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2v/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps "$steps" --test-every 999999 \
      --disable-viewer --antialiased --sh-degree 4 --strategy.cap-max 3000000 \
      --strategy.refine-stop-iter "$stop" --eval-steps "$steps" --save-steps "$steps" --global-seed 42 \
      $extra > "/tmp/nobj_${s}_${tag}.log" 2>&1
    if ! [ -s "$ck" ]; then echo "  ⚠ FAIL:"; tail -3 "/tmp/nobj_${s}_${tag}.log"|sed 's/^/    /'; echo FAIL>"$cf"; return 0; fi
    rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
  fi
  [ "$(ls "$rend" 2>/dev/null|wc -l)" -lt 5 ] && $PY tools/render_test_poses.py --ckpt "$ck" \
    --csv "workspace_r2v/$s/val_poses.csv" --out "$rend" --data_dir "workspace_r2v/$s" --antialiased 2>&1 | tail -1
  local v; v=$($PY tools/score_local.py --pred_dir "$rend" --gt_dir "workspace_r2v/$s/val_gt" 2>/dev/null | grep -a "PSNR_max=50" | grep -oE "[0-9]+\.[0-9]+$")
  [ -n "$v" ] || { echo "  ⚠ score fail"; return 0; }
  echo "$v">"$cf"; printf "  ★ [%-8s %-9s] v50=%s Δ=%s\n" "$s" "$tag" "$v" "$(.venv/bin/python -c "print(f'{$v-${BASE[$s]}:+.5f}')")"
}

for s in chair bonsai; do
  say "SCENE $s (mốc SH4 = ${BASE[$s]})"
  run "$s" base    30000 ""
  run "$s" ssim03  30000 "--ssim-lambda 0.3"
  run "$s" ssim01  30000 "--ssim-lambda 0.1"
  run "$s" sreg001 30000 "--scale-reg 0.001"
  run "$s" s45k    45000 ""
  run "$s" noise2e6 30000 "--strategy.noise-lr 2000000"
done
echo
echo "########################################################################"
echo "#  VERDICT OBJ-NIGHT — chair mốc 0.66358 · bonsai 0.71402. ≥+0.002 → prod obj."
echo "#  chair yếu nhất (69.82 thật) — mọi +0.002 = +0.14 tổng. ssim_lambda/step/reg trên scene mờ."
echo "########################################################################"
echo "OBJ-NIGHT-DONE ($(date +%d/%m' '%H:%M))"
