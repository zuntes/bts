#!/bin/bash
# ============================================================================
# GATE R2-OBJ2 — biến thể VẬT LIỆU cho bonsai/chair (chạy SAU GATE_R2_OBJ).
# Nhận định 17/07: bonsai đặt trên BÀN KÍNH phản chiếu (màu phụ thuộc góc nhìn
# mạnh — SH3 kém), chair lưng LƯỚI bán trong suốt. Khác hẳn scene trạm BTS.
#   [O4] SH bậc 4 @cap3M — tăng sức biểu đạt view-dependent cho phản chiếu
#        (SH4 = 75 float màu/gaussian vs 48 ở SH3 — VRAM tăng ~40% phần màu)
# So với [O1-*-3M] cùng cap trong /tmp/gate_r2obj log (4060: results/gate_r2obj.log).
# Chạy 4060 (~2h): bash tools/GATE_R2_OBJ2.sh 2>&1 | tee -a results/gate_r2obj2.log
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$2" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$3] /"; }

say "0. tiên quyết"
for s in bonsai chair; do
  [ -s "workspace_r2v/$s/val_poses.csv" ] || die "thiếu holdout $s → GATE_R2_OBJ chạy trước"
done
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 8 ] && die "đĩa ${FREE}GB<8"

# ===== O5: enhancer VGG vs UNET trên ĐÚNG 2 scene ưu tiên (rẻ: tái dùng ckpt cap3M) =====
# Vì sao cần: B1 đo vgg thắng +0.0052 nhưng trên HCM (scene NÉT). O2 vừa đo unet trên
# bonsai +0.0101 / chair chỉ +0.0036 → enhancer hành xử KHÁC nhau theo độ mờ scene.
# Production sẽ dùng vgg cho 2 scene này → phải xác thực trước khi đốt 10h train.
for s in bonsai chair; do
  say "$s — O5 enhancer VGG (so unet: bonsai 0.71879 · chair 0.66535)"
  NET="results/r2v_${s}__enhvgg/net.pt"
  if ! [ -s "$NET" ]; then
    $PY tools/enhance_net.py train --workspace "workspace_r2v/$s" \
      --ckpt "results/r2v_${s}__cap3M/ckpts/ckpt_29999_rank0.pt" \
      --out "$NET" --arch vgg --steps 8000 --patch 320 \
      2>&1 | tee "/tmp/r2v_vgg_${s}.log" || die "O5 train $s"
  fi
  $PY tools/enhance_net.py apply --net "$NET" \
    --in_dir "renders_r2v/${s}__cap3M" --out_dir "renders_r2v/${s}__vgg" >/dev/null || die "O5 apply $s"
  score "renders_r2v/${s}__vgg" "workspace_r2v/$s/val_gt" "O5-$s-VGG"
done

for s in bonsai chair; do
  say "$s — O4 SH degree 4 @3M"
  res="results/r2v_${s}__sh4"
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2v/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased --sh-degree 4 \
      --strategy.cap-max 3000000 --eval-steps 30000 --save-steps 30000 --global-seed 42 \
      2>&1 | tee "/tmp/r2v_${s}_sh4.log" || die "train $s sh4"
    rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
  else echo "  ⏩ có"; fi
  if [ "$(ls "renders_r2v/${s}__sh4" 2>/dev/null | wc -l)" -lt 5 ]; then
    $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
      --csv "workspace_r2v/$s/val_poses.csv" --out "renders_r2v/${s}__sh4" \
      --data_dir "workspace_r2v/$s" --antialiased 2>&1 | grep -av "render " || die "render $s sh4"
  fi
  score "renders_r2v/${s}__sh4" "workspace_r2v/$s/val_gt" "O4-$s-SH4"
done

echo
echo "########################################################################"
echo "#  VERDICT O5 (arch enhancer) — mốc unet: bonsai 0.71879 · chair 0.66535 (base .70867/.66176)"
echo "#    O5 > unet → production 2 scene dùng ENH_ARCH=vgg (mặc định) ✓"
echo "#    O5 < unet → ĐỔI: 2 scene lạ dùng unet, HCM dùng vgg (arch per-scene)"
echo "#  VERDICT O4 — so [O4-*-SH4] với [O1-*-3M] (cùng cap, chỉ khác SH độ 4 vs 3)"
echo "#  Δ ≥ +0.002 trên scene có phản chiếu → production bonsai/chair dùng --sh-degree 4"
echo "#  (SH4 chỉ đáng cho 2 scene vật liệu khó; HCM giữ SH3 — drone ít specular)"
echo "########################################################################"
echo "DÁN [O4-*] CHO CLAUDE."
