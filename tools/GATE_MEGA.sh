#!/bin/bash
# ============================================================================
# GATE_MEGA — BATTERY SIÊU LỚN (cả ngày 22/07) trên L40S GPU1. Aim 80.
# Bàn HCM0204 1320×989 (= ĐÚNG res round-2 gốc/4), GT thật. Mọi biến thể so [base].
# Ưu tiên đòn LỚN chạy trước (knee, S1, stack) — kết quả quan trọng có sớm.
# ≥+0.002 → port prod. HCM ×5 = 71% điểm → +0.002/scene = +0.7 tổng.
#
# Chạy: tmux new -d -s mega "CUDA_VISIBLE_DEVICES=1 bash tools/GATE_MEGA.sh 2>&1 | tee /tmp/mega.txt"
# Resumable: cache điểm → chạy lại tiếp. Xem: grep '★' /tmp/mega.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385; K1T=-0.00117; K2T=0.01710
SEED=${SEED:-42}
CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
CACHE=results/mega_cache; mkdir -p "$CACHE"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
gpu_busy(){ local g free tot thr; g=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  thr=$(( tot*55/100 )); [ "$thr" -gt 18000 ] && thr=18000; [ "${free:-0}" -lt "$thr" ]; }
for i in $(seq 1 300); do gpu_busy || break; [ "$i" = 1 ] && echo "chờ GPU..."; sleep 60; done
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S"
[ -f "$CSV" ] || die "thiếu $CSV"
BASEV=""

run(){  # $1=tag $2=cap $3=steps $4=rk1 $5=rk2 $6=ss $7=nseed $8=extra
  local tag=$1 cap=$2 steps=$3 rk1=$4 rk2=$5 ss=$6 nseed=$7 extra=$8
  local cf="$CACHE/${tag}.v50" rend="renders_mega/${tag}"
  local ssflag=""; [ "$ss" = 2 ] && ssflag="--supersample 2"
  if [ -s "$cf" ]; then
    local v; v=$(cat "$cf"); local d=""
    [ -n "$BASEV" ] && [ "$v" != FAIL ] && d=$(.venv/bin/python -c "print(f'{$v-$BASEV:+.5f}')")
    printf "  [%-14s] %s  Δ=%s (cache)\n" "$tag" "$v" "$d"; return 0
  fi
  say "$tag — cap=$cap steps=$steps k=[$rk1,$rk2] ss=$ss seeds=$nseed $extra"
  local FREE; FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 10 ] && die "đĩa ${FREE}GB<10"
  local stop=$(( steps > 30000 ? steps*5/6 : 25000 ))
  local dirs=""
  for sd in $(seq 0 $((nseed-1))); do
    local sv=$((SEED + sd*100))
    local res="results/mega_${S}__${tag}_s${sv}" ck="$res/ckpts/ckpt_$((steps-1))_rank0.pt"
    if ! [ -s "$ck" ]; then
      $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 1 \
        --result-dir "$PWD/$res" --max-steps "$steps" --test-every 999999 \
        --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
        --strategy.cap-max "$cap" --strategy.refine-stop-iter "$stop" \
        --eval-steps "$steps" --save-steps "$steps" --global-seed "$sv" \
        $extra > "/tmp/mega_${tag}_s${sv}.log" 2>&1
      if ! [ -s "$ck" ]; then echo "  ⚠ $tag s$sv FAIL:"; tail -4 "/tmp/mega_${tag}_s${sv}.log"|sed 's/^/    /'; echo FAIL>"$cf"; return 0; fi
      rm -f "$res/ckpts/ckpt_14999_rank0.pt" "$res/ckpts/ckpt_$((steps/2-1))_rank0.pt"; rm -rf "$res/videos"
    fi
    local rd="renders_mega/${tag}_s${sv}"
    if [ "$(ls "$rd" 2>/dev/null|wc -l)" -lt 60 ]; then
      $PY tools/render_test_poses.py --ckpt "$ck" --csv "$CSV" --out "$rd" \
        --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 "$rk1" --radial_k2 "$rk2" $ssflag \
        2>&1 | grep -av "render " || { echo "  ⚠ render fail"; return 0; }
    fi
    dirs="$dirs $rd"
  done
  if [ "$nseed" -gt 1 ]; then $PY tools/ensemble.py --dirs $dirs --out "$rend" --mode mean >/dev/null || { echo "  ⚠ ens fail"; return 0; }
  else rend="renders_mega/${tag}_s${SEED}"; fi   # single-seed: path TRỰC TIẾP (tránh leading-space từ $dirs → score fail)
  local v; v=$($PY tools/score_local.py --pred_dir "$rend" --gt_dir "$GT" 2>/dev/null | grep -a "PSNR_max=50" | grep -oE "[0-9]+\.[0-9]+$")
  [ -n "$v" ] || { echo "  ⚠ score fail $tag"; return 0; }
  echo "$v" > "$cf"
  # PRUNE ckpt sau khi chấm xong (render+điểm đã lưu) — chống đầy đĩa. GIỮ base (supersample reuse).
  [ "$tag" != base ] && rm -rf results/mega_${S}__${tag}_s*/ckpts
  local d=""; [ -n "$BASEV" ] && d=$(.venv/bin/python -c "print(f'{$v-$BASEV:+.5f}')")
  printf "  ★ [%-14s] v50=%s  Δvs_base=%s\n" "$tag" "$v" "$d"
}

# ═══════ A. KNEE trên bàn ĐÚNG (đòn LỚN NHẤT — prod 6M nghi under-cap) ═══════
say "A. KNEE 3→6→9→12→16M (bàn 1320×989 = round-2). GATE_CAP cũ đo knee 12M — xác nhận + có S1?"
run base   3000000 30000 "$K1" 0 1 1 ""
BASEV=$(cat "$CACHE/base.v50"); echo "  → BASE(3M) v50=$BASEV"
run cap6M  6000000 30000 "$K1" 0 1 1 ""
run cap9M  9000000 30000 "$K1" 0 1 1 ""
run cap12M 12000000 30000 "$K1" 0 1 1 ""
run cap16M 16000000 30000 "$K1" 0 1 1 ""

# ═══════ B. S1 méo đúng (re-confirm +0.005) + stack với cap tốt ═══════
say "B. S1 distortion + STACK cap"
run s1_3M   3000000 30000 "$K1T" "$K2T" 1 1 "--dist-k1-override $K1T --dist-k2-override $K2T"
run s1_12M  12000000 30000 "$K1T" "$K2T" 1 1 "--dist-k1-override $K1T --dist-k2-override $K2T"

# ═══════ C. STEP đúng cách (s45k thắng +0.0021 trên chair — confirm HCM) ═══════
say "C. STEP (refine_stop scale — 60k cũ crippled)"
run s45k   3000000 45000 "$K1" 0 1 1 ""
run s60k   3000000 60000 "$K1" 0 1 1 ""

# ═══════ D. THAM SỐ metric-align + MCMC (bỏ oan) — tất cả 3M nhanh ═══════
say "D. THAM SỐ chưa tune"
run ssim01  3000000 30000 "$K1" 0 1 1 "--ssim-lambda 0.1"
run ssim03  3000000 30000 "$K1" 0 1 1 "--ssim-lambda 0.3"
run ssim05  3000000 30000 "$K1" 0 1 1 "--ssim-lambda 0.5"
run oreg001 3000000 30000 "$K1" 0 1 1 "--opacity-reg 0.001"
run oreg02  3000000 30000 "$K1" 0 1 1 "--opacity-reg 0.02"
run sreg001 3000000 30000 "$K1" 0 1 1 "--scale-reg 0.001"
run sreg02  3000000 30000 "$K1" 0 1 1 "--scale-reg 0.02"
run noise1e6 3000000 30000 "$K1" 0 1 1 "--strategy.noise-lr 1000000"
run noise2e6 3000000 30000 "$K1" 0 1 1 "--strategy.noise-lr 2000000"
run minop01 3000000 30000 "$K1" 0 1 1 "--strategy.min-opacity 0.01"
run minop02 3000000 30000 "$K1" 0 1 1 "--strategy.min-opacity 0.02"
run erank   3000000 30000 "$K1" 0 1 1 "--erank-reg 0.1"    # needle-reg: half-res +0.0003, full-res chưa test (anten/cột BTS)
run erank5  3000000 30000 "$K1" 0 1 1 "--erank-reg 0.5"

# ═══════ E. RENDER supersample×2 (không retrain — reuse base ckpt) ═══════
say "E. RENDER supersample×2 trên NÉT"
BCK="results/mega_${S}__base_s${SEED}/ckpts/ckpt_29999_rank0.pt"
if [ -s "$CACHE/base_ss2.v50" ]; then echo "  [base_ss2] $(cat "$CACHE/base_ss2.v50") (cache)"
elif [ -s "$BCK" ]; then
  [ "$(ls renders_mega/base_ss2 2>/dev/null|wc -l)" -lt 60 ] && $PY tools/render_test_poses.py --ckpt "$BCK" \
    --csv "$CSV" --out renders_mega/base_ss2 --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 $K1 --supersample 2 2>&1 | grep -av "render "
  V=$($PY tools/score_local.py --pred_dir renders_mega/base_ss2 --gt_dir "$GT" 2>/dev/null | grep -a "PSNR_max=50" | grep -oE "[0-9]+\.[0-9]+$")
  [ -n "$V" ] && { echo "$V">"$CACHE/base_ss2.v50"; printf "  ★ [base_ss2] v50=%s Δ=%s\n" "$V" "$(.venv/bin/python -c "print(f'{$V-$BASEV:+.5f}')")"; }
fi

# ═══════ F. STACK TỔNG (gộp đòn thắng — cấu hình SUB3 ứng viên) ═══════
say "F. STACK cuối: S1+cap12M+step + ensemble 2-seed (production config thật)"
run s1_12M_45k 12000000 45000 "$K1T" "$K2T" 1 1 "--dist-k1-override $K1T --dist-k2-override $K2T"
run s1_12M_2seed 12000000 30000 "$K1T" "$K2T" 1 2 "--dist-k1-override $K1T --dist-k2-override $K2T"

echo
echo "########################################################################"
echo "#  VERDICT MEGA — BASE(3M)=$BASEV. Mọi ★ Δvs_base ở trên."
echo "#  KNEE: cap6/9/12/16M cho biết prod nên cap bao nhiêu (nghi 6M under-cap → 12M)."
echo "#  S1: s1_3M/s1_12M re-confirm méo đúng. STACK F = cấu hình SUB3 thật (multi-đòn)."
echo "#  ≥+0.002 → prod. DÁN TOÀN BỘ ★ cho Claude → build SUB3."
echo "########################################################################"
echo "MEGA-DONE ($(date +%d/%m' '%H:%M))"
