#!/bin/bash
# ============================================================================
# GATE_OVERNIGHT — BATTERY TOÀN DIỆN đêm 21→22/07 trên L40S GPU1 (~8-10h).
# Mục tiêu: (A) RE-VALIDATE claim chưa chắc  (B) TEST tham số/phương pháp MỚI.
# Bàn: HCM0204 FULL-RES round-1, GT THẬT (native detail, không smooth như half).
# Mọi biến thể single-seed, so [base]. ≥+0.002 → port prod (HCM ×5 = 71% điểm).
# top-4=77.7, ta 76.52 — cách 1.2. Mọi +0.002/scene HCM = +0.7 tổng.
#
# Chạy: tmux new -d -s night "CUDA_VISIBLE_DEVICES=1 bash tools/GATE_OVERNIGHT.sh 2>&1 | tee /tmp/night.txt"
# Resumable: ckpt/score đã có → skip. Env: SEED=42 · CAPB=6000000
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S=HCM0204; K1=0.010009930826722385
K1T=-0.00117; K2T=0.01710
SEED=${SEED:-42}; CAPB=${CAPB:-6000000}
# RES=half (mặc định, NHANH 3× + đúng res round-2 gốc/4) hoặc full (bàn native chậm)
RES=${RES:-half}
if [ "$RES" = half ]; then
  DF=2; CSV="workspace_r2cal/test_poses_half.csv"; GT="workspace_r2cal/gt_half"
  CACHE=results/night_cache_half
else
  DF=1; CSV="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
  GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"; CACHE=results/night_cache
fi
mkdir -p "$CACHE"
echo "  RES=$RES (data-factor $DF) · bàn=$GT"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
gpu_busy(){ local g free tot thr; g=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | cut -d, -f1)
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$g" 2>/dev/null | head -1)
  thr=$(( tot*55/100 )); [ "$thr" -gt 18000 ] && thr=18000; [ "${free:-0}" -lt "$thr" ]; }
for i in $(seq 1 300); do gpu_busy || break; [ "$i" = 1 ] && echo "chờ GPU..."; sleep 60; done
[ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S"
[ -f "$CSV" ] || die "thiếu $CSV"

# train+render+score 1 biến thể; cache điểm để resume. In Δ vs base tự động.
BASEV=""
run(){  # $1=tag $2=cap $3=steps $4=render_k1 $5=render_k2 $6=ss $7=extra_train
  local tag=$1 cap=$2 steps=$3 rk1=$4 rk2=$5 ss=$6 extra=$7
  local res="results/night_${S}__${tag}" rend="renders_night/${tag}" cf="$CACHE/${tag}.v50"
  local ck="$res/ckpts/ckpt_$((steps-1))_rank0.pt"
  local stop=$(( steps > 30000 ? steps*5/6 : 25000 ))
  local ssflag=""; [ "$ss" = 2 ] && ssflag="--supersample 2"
  if [ -s "$cf" ]; then
    local v; v=$(cat "$cf"); local d=""
    [ -n "$BASEV" ] && d=$(.venv/bin/python -c "print(f'{$v-$BASEV:+.5f}')")
    printf "  [%-10s] %s  Δ=%s (cache)\n" "$tag" "$v" "$d"; return 0
  fi
  say "$tag — cap=$cap steps=$steps refine_stop=$stop k=[$rk1,$rk2] ss=$ss $extra"
  local FREE; FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 8 ] && die "đĩa ${FREE}GB<8"
  if ! [ -s "$ck" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor $DF \
      --result-dir "$PWD/$res" --max-steps "$steps" --test-every 999999 \
      --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
      --strategy.cap-max "$cap" --strategy.refine-stop-iter "$stop" \
      --eval-steps "$steps" --save-steps "$steps" --global-seed "$SEED" \
      $extra > "/tmp/night_${tag}.log" 2>&1
    if ! [ -s "$ck" ]; then echo "  ⚠ $tag TRAIN FAIL:"; tail -4 "/tmp/night_${tag}.log"|sed 's/^/    /'; echo "FAIL">"$cf"; return 0; fi
    rm -f "$res/ckpts/ckpt_14999_rank0.pt" "$res/ckpts/ckpt_$((steps/2-1))_rank0.pt"; rm -rf "$res/videos"
  fi
  if [ "$(ls "$rend" 2>/dev/null|wc -l)" -lt 60 ]; then
    $PY tools/render_test_poses.py --ckpt "$ck" --csv "$CSV" --out "$rend" \
      --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 "$rk1" --radial_k2 "$rk2" $ssflag \
      2>&1 | grep -av "render " || { echo "  ⚠ render fail $tag"; return 0; }
  fi
  local v; v=$($PY tools/score_local.py --pred_dir "$rend" --gt_dir "$GT" 2>/dev/null | grep -a "PSNR_max=50" | grep -oE "[0-9]+\.[0-9]+$")
  [ -n "$v" ] || { echo "  ⚠ score fail $tag"; return 0; }
  echo "$v" > "$cf"
  local d=""; [ -n "$BASEV" ] && d=$(.venv/bin/python -c "print(f'{$v-$BASEV:+.5f}')")
  printf "  ★ [%-10s] v50=%s  Δvs_base=%s\n" "$tag" "$v" "$d"
}

# ============ A. NỀN + RE-VALIDATE ============
say "A. BASE + RE-VALIDATE claim cũ"
run base "$CAPB" 30000 "$K1" 0 1 ""
BASEV=$(cat "$CACHE/base.v50")
echo "  → BASE v50 = $BASEV (mọi Δ tính từ đây)"
run s1true   "$CAPB" 30000 "$K1T" "$K2T" 1 "--dist-k1-override $K1T --dist-k2-override $K2T"  # re-confirm S1 +0.005
run cap9M    9000000 30000 "$K1"  0 1 ""   # re-validate knee (native muốn thêm hạt?)
run cap12M   12000000 30000 "$K1" 0 1 ""   # knee native = full-res 12M?
run s45k     "$CAPB" 45000 "$K1"  0 1 ""    # STEP đúng cách (refine_stop scale) — 60k cũ crippled

# ============ B. THAM SỐ CHƯA TUNE (bỏ oan) ============
say "B. THAM SỐ metric-align + MCMC dynamics"
run ssim03   "$CAPB" 30000 "$K1" 0 1 "--ssim-lambda 0.3"   # metric chấm 0.3 SSIM, train chỉ 0.2
run ssim05   "$CAPB" 30000 "$K1" 0 1 "--ssim-lambda 0.5"
run oreg001  "$CAPB" 30000 "$K1" 0 1 "--opacity-reg 0.001"   # default 0.01, paper dataset-dependent
run sreg001  "$CAPB" 30000 "$K1" 0 1 "--scale-reg 0.001"   # hạt nhỏ hơn = chi tiết mịn
run noise2e6 "$CAPB" 30000 "$K1" 0 1 "--strategy.noise-lr 2000000"   # MCMC explore mạnh
run minop01  "$CAPB" 30000 "$K1" 0 1 "--strategy.min-opacity 0.01"   # relocate hạt mờ hơn

# ============ C. RENDER-TIME (KHÔNG retrain — reuse base ckpt) ============
say "C. RENDER supersample×2 trên NÉT (loại trên mờ — retest native)"
BCK="results/night_${S}__base/ckpts/ckpt_29999_rank0.pt"
if [ -s "$CACHE/base_ss2.v50" ]; then
  echo "  [base_ss2 ] $(cat "$CACHE/base_ss2.v50") (cache)"
elif [ -s "$BCK" ]; then
  [ "$(ls renders_night/base_ss2 2>/dev/null|wc -l)" -lt 60 ] && \
    $PY tools/render_test_poses.py --ckpt "$BCK" --csv "$CSV" --out renders_night/base_ss2 \
      --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 $K1 --supersample 2 \
      2>&1 | grep -av "render "
  SSV=$($PY tools/score_local.py --pred_dir renders_night/base_ss2 --gt_dir "$GT" 2>/dev/null | grep -a "PSNR_max=50" | grep -oE "[0-9]+\.[0-9]+$")
  [ -n "$SSV" ] && { echo "$SSV" > "$CACHE/base_ss2.v50"; printf "  ★ [base_ss2 ] v50=%s  Δvs_base=%s\n" "$SSV" "$(.venv/bin/python -c "print(f'{$SSV-$BASEV:+.5f}')")"; }
fi

# ============ D. STACK ứng viên thắng (S1 là nền chắc) ============
say "D. STACK S1 + cap9M (2 đòn nền nếu đều thắng)"
run s1_cap9M 9000000 30000 "$K1T" "$K2T" 1 "--dist-k1-override $K1T --dist-k2-override $K2T"

echo
echo "########################################################################"
echo "#  VERDICT OVERNIGHT — BASE=$BASEV. Mọi dòng ★ Δvs_base ở trên."
echo "#  ≥+0.002 → port prod SUB3 (HCM ×5). Đòn thắng CỘNG DỒN được nếu trực giao."
echo "#  Re-validate: s1true nên ≈+0.005 · s45k cho biết step-đúng có ăn · cap9/12M = knee native."
echo "#  DÁN TOÀN BỘ khối này + các dòng ★ cho Claude."
echo "########################################################################"
echo "OVERNIGHT-DONE ($(date +%d/%m' '%H:%M))"
