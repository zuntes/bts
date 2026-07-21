#!/bin/bash
# ============================================================================
# GATE S3 — SHARPNESS-WEIGHTED LOSS cho scene video mờ-KHÔNG-ĐỀU (đòn nguyên bản).
#
# BỆNH ĐO ĐƯỢC (không phải giả thuyết):
#   chair lapvar p10=383 vs mean=1369 · bonsai p10=50 vs mean=234 → độ mờ LỆCH 3-5×
#   giữa các frame. MCMC coi MỌI train view NGANG NHAU → frame mờ nặng dạy model
#   vẽ mờ. Bằng chứng: render chair lapvar 703 < GT mean 1369 = model MỜ HƠN dữ liệu.
#   S3: w_i = (lapvar_i/median)^gamma, clamp [1/3,3], mean=1 → frame NÉT dạy nhiều hơn.
#
# KHÁC các đòn đã LOẠI: blur-match làm mờ RENDER (sai hướng) · W1 weight theo POSE
# (data không lệch phân bố) · đây weight theo CHẤT LƯỢNG ẢNH (bệnh có thật, đo được).
#
# Bàn: holdout workspace_r2v (GT thật). Mốc 1-seed SH4: chair .66358 · bonsai .71402
# Chạy: setsid nohup bash tools/GATE_S3_SHARP.sh > results/gate_s3.log 2>&1 &
# Env: S3_SCENES="chair bonsai" · S3_GAMMAS="1.0 0.5"
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
S3_SCENES=${S3_SCENES:-"chair bonsai"}
S3_GAMMAS=${S3_GAMMAS:-"1.0 0.5"}
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }

# đợi GPU rảnh (gate khác có thể đang chạy) — tối đa 4h
for i in $(seq 1 240); do
  pgrep -f "ref_enhance|simple_trainer" >/dev/null || break
  [ "$i" = 1 ] && echo "[$(date +%H:%M)] GPU đang bận — xếp hàng chờ..."
  sleep 60
done
pgrep -f "ref_enhance|simple_trainer" >/dev/null && die "GPU vẫn bận sau 4h — dừng"

score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "workspace_r2v/$2/val_gt" 2>/dev/null \
         | grep -aE "n=|★" | sed "s/^/  [$3] /"; }

for s in $S3_SCENES; do
  for g in $S3_GAMMAS; do
    tag="sharp${g/./}"
    res="results/r2v_${s}__${tag}"; rend="renders_r2v/${s}__${tag}"
    FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
    [ "$FREE" -lt 6 ] && die "đĩa ${FREE}GB<6"
    say "$s — S3 sharp-weight gamma=$g (SH4 cap3M 30k, ~1h30)"
    if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
      $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2v/$s" --data-factor 1 \
        --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
        --disable-viewer --antialiased --sh-degree 4 --strategy.cap-max 3000000 \
        --eval-steps 30000 --save-steps 30000 --global-seed 42 \
        --sharp-weight "$g" > "/tmp/s3_${s}_${tag}.log" 2>&1
      grep -a "BTS S3" "/tmp/s3_${s}_${tag}.log" | head -1
      [ -s "$res/ckpts/ckpt_29999_rank0.pt" ] || die "train $s $tag không ra ckpt — /tmp/s3_${s}_${tag}.log"
      rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
    else echo "  ⏩ ckpt có"; fi
    if [ "$(ls "$rend" 2>/dev/null | wc -l)" -lt 5 ]; then
      $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
        --csv "workspace_r2v/$s/val_poses.csv" --out "$rend" \
        --data_dir "workspace_r2v/$s" --antialiased 2>&1 | tail -1
    fi
    score "$rend" "$s" "S3-$s-g$g"
  done
done

echo
echo "########################################################################"
echo "#  VERDICT S3 — mốc 1-seed SH4 KHÔNG weight: chair 0.66358 · bonsai 0.71402"
echo "#  ≥ +0.003 → S3 THẮNG: thêm --sharp-weight vào prod obj (SUB3) + làm member ens"
echo "#  0.000..+0.003 → giữ làm MEMBER config-ensemble (bài học CE: thua solo vẫn góp)"
echo "#  < 0 → LOẠI, đóng hồ sơ hướng 'weight theo chất lượng ảnh'"
echo "########################################################################"
echo "S3-GATE-DONE"
