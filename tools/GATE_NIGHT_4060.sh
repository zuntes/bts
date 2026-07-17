#!/bin/bash
# ============================================================================
# GATE ĐÊM 4060 (tự chờ chair_tests xong rồi chạy, ~00:45 → ~06:30):
#  [CE-*]  CONFIG-ENSEMBLE (SE-GS ICCV'25 bản rẻ): trộn render các BIẾN THỂ đã có
#          (cap3M/SH4/erank/45k) — 0 phút GPU, chỉ mean+score
#  [E2/E3-*] SEED-LAW trên scene lạ: train seed 7+123 (SH4 = config prod) cho
#          bonsai+chair → ens 2/3 seed + vgg → KIỂM chứng +0.0095 có chuyển sang
#          scene mờ không (hiện đang NGOẠI SUY từ HCM — rủi ro dự phóng SUB2)
# Chạy: setsid nohup bash tools/GATE_NIGHT_4060.sh > results/gate_night_4060.log 2>&1 &
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ [ "$(ls "$1" 2>/dev/null | wc -l)" -lt 5 ] && { echo "  [$2] ⏭ thiếu render"; return 0; }
         $PY tools/score_local.py --pred_dir "$1" --gt_dir "workspace_r2v/$3/val_gt" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

say "0. chờ chair_tests (c1/c2) xong để không tranh GPU"
until grep -aq "CHAIR-TESTS-DONE\|❌" results/chair_tests.log 2>/dev/null; do sleep 60; done
grep -aq "❌" results/chair_tests.log && echo "  ⚠ chair_tests có lỗi — vẫn chạy tiếp phần mình"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 8 ] && die "đĩa ${FREE}GB<8"

say "1. CONFIG-ENSEMBLE (0 GPU) — trộn biến thể sẵn có"
# chair: cap3M + sh4 + c1_erank + c2_45k (4 config 1 seed)
D=""
for t in cap3M sh4 c1_erank c2_45k; do [ -d "renders_r2v/chair__$t" ] && D="$D renders_r2v/chair__$t"; done
N=$(echo $D | wc -w)
if [ "$N" -ge 2 ]; then
  $PY tools/ensemble.py --dirs $D --out renders_r2v/chair__cfgens --mode mean >/dev/null
  score renders_r2v/chair__cfgens "CE-chair-${N}cfg" chair
  $PY tools/enhance_net.py apply --net results/r2v_chair__enhvgg/net.pt \
    --in_dir renders_r2v/chair__cfgens --out_dir renders_r2v/chair__cfgens_vgg >/dev/null 2>&1 \
    && score renders_r2v/chair__cfgens_vgg "CE-chair-${N}cfg+vgg" chair
fi
# bonsai: cap3M + sh4
if [ -d renders_r2v/bonsai__sh4 ]; then
  $PY tools/ensemble.py --dirs renders_r2v/bonsai__cap3M renders_r2v/bonsai__sh4 \
    --out renders_r2v/bonsai__cfgens --mode mean >/dev/null
  score renders_r2v/bonsai__cfgens "CE-bonsai-2cfg" bonsai
  $PY tools/enhance_net.py apply --net results/r2v_bonsai__enhvgg/net.pt \
    --in_dir renders_r2v/bonsai__cfgens --out_dir renders_r2v/bonsai__cfgens_vgg >/dev/null 2>&1 \
    && score renders_r2v/bonsai__cfgens_vgg "CE-bonsai-2cfg+vgg" bonsai
fi

say "2. SEED-LAW trên scene lạ — train s7/s123 (SH4 như prod), ens với sh4(=s42)"
train_seed(){ # scene seed minutes-est
  local s=$1 seed=$2 res="results/r2v_${1}__sh4_s${2}"
  if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
    $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_r2v/$s" --data-factor 1 \
      --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
      --disable-viewer --antialiased --sh-degree 4 --strategy.cap-max 3000000 \
      --eval-steps 30000 --save-steps 30000 --global-seed "$seed" \
      2>&1 | tee "/tmp/n4_${s}_s${seed}.log" || die "train $s s$seed"
    find "$res/ckpts" -name "ckpt_14999*" -delete 2>/dev/null; rm -rf "$res/videos"
  fi
  rm -rf "renders_r2v/${s}__sh4_s${seed}"
  $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
    --csv "workspace_r2v/$s/val_poses.csv" --out "renders_r2v/${s}__sh4_s${seed}" \
    --data_dir "workspace_r2v/$s" --antialiased 2>&1 | tail -1
}
for s in chair bonsai; do        # chair trước — scene nghẽn ưu tiên
  train_seed "$s" 7
  $PY tools/ensemble.py --dirs "renders_r2v/${s}__sh4" "renders_r2v/${s}__sh4_s7" \
    --out "renders_r2v/${s}__sh4_e2" --mode mean >/dev/null
  score "renders_r2v/${s}__sh4_e2" "E2-$s" "$s"
  train_seed "$s" 123
  $PY tools/ensemble.py --dirs "renders_r2v/${s}__sh4" "renders_r2v/${s}__sh4_s7" "renders_r2v/${s}__sh4_s123" \
    --out "renders_r2v/${s}__sh4_e3" --mode mean >/dev/null
  score "renders_r2v/${s}__sh4_e3" "E3-$s" "$s"
  $PY tools/enhance_net.py apply --net "results/r2v_${s}__enhvgg/net.pt" \
    --in_dir "renders_r2v/${s}__sh4_e3" --out_dir "renders_r2v/${s}__sh4_e3vgg" >/dev/null 2>&1 \
    && score "renders_r2v/${s}__sh4_e3vgg" "E3-$s+vgg" "$s"
done

echo
echo "########################################################################"
echo "#  VERDICT ĐÊM 4060 — mốc 1seed: bonsai-sh4 0.71402 · chair-sh4 0.66358"
echo "#  [E2/E3-*]: seed-law scene lạ — kỳ vọng +0.007/+0.010; nếu <một nửa → HẠ dự phóng SUB2"
echo "#  [E3-*+vgg]: chính là con số per-scene SUB2 dự kiến THẬT (đo, không ngoại suy)"
echo "#  [CE-*]: config-ens ≥ seed-ens cùng số thành viên → prod trộn config thay thêm seed"
echo "########################################################################"
echo "NIGHT-4060-DONE"
