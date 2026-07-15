#!/bin/bash
# ============================================================================
# CỔNG QUYẾT ĐỊNH: NHT@5M có thắng standard@5M (=0.74727) trên competition poses?
#
#   Đã biết: NHT@1M thắng standard@1M +0.016 (cùng cap) NHƯNG NHT@1M=0.728 < std@5M=0.747
#   Chưa biết: ở cap 5M lợi thế NHT còn giữ không → script này trả lời.
#
# Chạy trên L40S:  bash tools/GATE_NHT.sh 2>&1 | tee /tmp/gate_nht.txt
# Rồi DÁN 20 dòng cuối cho Claude. ~3-5h (train NHT 5M) — chạy trong tmux/nohup.
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"
export CUDA_VISIBLE_DEVICES=0            # GPU 1 bị VLLM chiếm
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # BẮT BUỘC: 4060 OOM lẻ tẻ khi render nếu thiếu
SCENE=HCM0204
CAP=5000000
STD_BASELINE=0.74727                     # standard 5M single-seed, competition v50 (đo trên 4060)

say() { echo; echo "===== $* ====="; }
die() { echo; echo "❌ $*"; echo "   → Dán dòng này cho Claude."; exit 1; }

########## 0. TIÊN QUYẾT — fail sớm, TRƯỚC khi đụng GPU ##########
say "0. Kiểm tiên quyết"
[ -x .venv/bin/python ] || die "thiếu .venv gsplat → bash tools/SETUP_MAIN_VENV.sh"
NHT_PY=~/3dgrut/.venv/bin/python
[ -x "$NHT_PY" ] || die "thiếu ~/3dgrut/.venv → bash tools/BUILD_NHT.sh"
[ -f ~/3dgrut/render_comp.py ] || die "thiếu ~/3dgrut/render_comp.py → cp tools/render_comp.py ~/3dgrut/"
[ -n "${CONDA_PREFIX:-}" ] && die "đang trong conda env → chạy: conda deactivate  rồi lại"
CSV="VAI_NVS_DATA/phase1/public_set/$SCENE/test/test_poses.csv"
GT="VAI_NVS_DATA/phase1/public_set/$SCENE/test/images"
[ -f "$CSV" ] || die "thiếu $CSV (rsync data chưa đủ?)"
N_GT=$(ls "$GT" 2>/dev/null | wc -l); [ "$N_GT" -ge 60 ] || die "GT chỉ $N_GT ảnh (cần 60) — rsync dở dang?"
echo "  ✅ venv gsplat + venv NHT + render_comp.py + data ($N_GT ảnh GT)"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
[ "$FREE" -lt 20 ] && die "đĩa chỉ ${FREE}GB — NHT 5M ckpt lớn, cần ≥20GB"
echo "  ✅ đĩa ${FREE}GB"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  GPU /'

########## 1. WORKSPACE (kiểm FILE THẬT, không kiểm thư mục) ##########
say "1. workspace_raw/$SCENE (giữ méo — cho 3DGUT/NHT)"
NEED=0
for f in sparse/0/cameras.bin sparse/0/images.bin sparse/0/points3D.bin; do
  [ -s "workspace_raw/$SCENE/$f" ] || NEED=1
done
N_IMG=$(ls "workspace_raw/$SCENE/images" 2>/dev/null | wc -l); [ "$N_IMG" -lt 100 ] && NEED=1
if [ "$NEED" = 1 ]; then
  echo "  → chưa có/dở dang (ảnh=$N_IMG) → prepare..."
  SRC=VAI_NVS_DATA/phase1/public_set/$SCENE
  .venv/bin/python tools/prepare_scene.py --scene_dir "$SRC" --out_dir "workspace_raw/$SCENE" --keep_distortion \
    || die "prepare_scene thất bại"
  N_IMG=$(ls "workspace_raw/$SCENE/images" 2>/dev/null | wc -l)
fi
[ "$N_IMG" -ge 100 ] || die "workspace_raw/$SCENE chỉ $N_IMG ảnh — prepare hỏng"
echo "  ✅ $N_IMG ảnh train + sparse/0/*.bin"

########## 2. SYMLINK DATA CHO 3DGRUT ##########
say "2. symlink data → ~/3dgrut/data/$SCENE"
mkdir -p ~/3dgrut/data
ln -sfn "$PROJ/workspace_raw/$SCENE" ~/3dgrut/data/$SCENE
[ -s ~/3dgrut/data/$SCENE/sparse/0/images.bin ] || die "symlink hỏng (path có dấu cách?)"
echo "  ✅ $(readlink ~/3dgrut/data/$SCENE)"

########## 3. TRAIN NHT @ CAP 5M (bước lâu nhất) ##########
say "3. Train NHT cap=$CAP, 30k steps — ~3-5h. VRAM ước tính 12-20GB (L40S 44GB → dư)"
RUN=${SCENE}_nht_5M
if [ -s ~/3dgrut/runs/$RUN/*/ckpt_last.pt ] 2>/dev/null; then
  echo "  ⏩ đã có ckpt → bỏ qua train"
else
  # ⚠ ĐÚNG PATH: max_n_gaussians nằm dưới section `add:` trong configs/strategy/mcmc.yaml
  #   → strategy.add.max_n_gaussians (KHÔNG phải strategy.max_n_gaussians — sai sẽ train
  #     cap 1M mặc định và chỉ lộ ra sau 5h → cổng vô nghĩa).
  ( cd ~/3dgrut && PATH=$HOME/3dgrut/.venv/bin:$PATH "$NHT_PY" train.py \
      --config-name apps/colmap_3dgut_mcmc_nht.yaml \
      path=data/$SCENE out_dir=runs experiment_name=$RUN \
      n_iterations=30000 strategy.add.max_n_gaussians=$CAP 2>&1 | tail -25 ) \
    || die "train NHT thất bại — dán 25 dòng trên"
fi
CKPT=$(ls -t ~/3dgrut/runs/$RUN/*/ckpt_last.pt 2>/dev/null | head -1)
[ -s "$CKPT" ] || die "train xong nhưng không thấy ckpt_last.pt"
echo "  ✅ ckpt: $CKPT ($(du -h "$CKPT" | cut -f1))"
# VERIFY cap thật sự ăn — nếu vẫn 1M thì cổng vô nghĩa, phải biết NGAY
N_GS=$("$NHT_PY" -c "
import torch,sys
try:
    c=torch.load('$CKPT',map_location='cpu',weights_only=False)
    for k in ('positions','means','particles','xyz'):
        for src in (c, c.get('model',{}) if isinstance(c,dict) else {}):
            if isinstance(src,dict) and k in src: print(len(src[k])); sys.exit()
    print('?')
except Exception as e: print('?')
" 2>/dev/null)
echo "  → số gaussians trong ckpt: $N_GS  (kỳ vọng ~$CAP; nếu ~1000000 thì override CAP KHÔNG ăn → báo Claude)"

########## 4. EVAL WORKSPACE (competition poses, pre-transform T240) ##########
say "4. Dựng eval workspace (60 competition poses, transform bằng T train)"
.venv/bin/python tools/build_nht_eval_ws.py --train_ws "workspace_raw/$SCENE" \
  --test_csv "$CSV" --gt_dir "$GT" --out_ws ~/3dgrut/data/${SCENE}_comp \
  || die "build_nht_eval_ws thất bại"
[ -s ~/3dgrut/data/${SCENE}_comp/sparse/0/images.bin ] || die "eval ws thiếu images.bin"

########## 5. RENDER COMPETITION POSES ##########
say "5. Render 60 competition poses bằng model NHT"
rm -rf ~/3dgrut/runs/gate_comp
( cd ~/3dgrut && PATH=$HOME/3dgrut/.venv/bin:$PATH "$NHT_PY" render_comp.py \
    "$CKPT" data/${SCENE}_comp runs/gate_comp 2>&1 | tail -12 ) \
  || die "render_comp thất bại"
RD=$(ls -d ~/3dgrut/runs/gate_comp/*/*/ 2>/dev/null | head -1)
N_R=$(ls "$RD/ours_30000/renders"/*.png 2>/dev/null | wc -l)
[ "$N_R" -eq 60 ] || die "chỉ render $N_R/60 ảnh (OOM? xem log trên)"
echo "  ✅ render đủ 60 ảnh"

########## 6. CHẤM v50 + VERDICT ##########
say "6. Chấm v50 (LPIPS-vgg, PSNR_max=50 — thang BTC)"
.venv/bin/python tools/score_local.py --pred_dir "$RD/ours_30000/renders" \
  --gt_dir "$RD/ours_30000/gt" --out results/${SCENE}__nht5M_score.json 2>&1 \
  | grep -aE "n=|★|Score\[" || die "score_local thất bại"

V50=$(.venv/bin/python -c "
import json,sys
try:
    d=json.load(open('results/${SCENE}__nht5M_score.json'))
    r=d if isinstance(d,list) else d.get('per_image',[])
    import statistics as st
    p=st.mean(float(x['psnr']) for x in r); s=st.mean(float(x['ssim']) for x in r); l=st.mean(float(x['lpips_vgg']) for x in r)
    print(f'{0.4*(1-l)+0.3*s+0.3*min(p/50,1):.5f}')
except Exception as e: print('ERR')
" 2>/dev/null)

echo
echo "########################################################################"
echo "#  VERDICT — CỔNG NHT@5M"
echo "########################################################################"
echo "  NHT cap 5M   competition v50 = ${V50}"
echo "  standard 5M  competition v50 = ${STD_BASELINE}   (mốc đã đo trên 4060)"
if [ "$V50" != "ERR" ] && [ -n "$V50" ]; then
  .venv/bin/python -c "
v=float('$V50'); b=float('$STD_BASELINE'); d=v-b
print(f'  Δ = {d:+.5f}')
print()
if d >= 0.005:
    print('  ✅ NHT THẮNG RÕ → PIVOT: SUB7 = NHT cap cao × multi-seed ensemble + L16-XL')
    print(f'     Dự phóng sau stack: ~{v+0.013:.3f} local → BTC ~{(v+0.013-0.7597)*100+76.97:.1f}')
elif d > 0:
    print('  🟡 NHT nhỉnh hơn nhưng mỏng → cân nhắc: NHT cap 8M? hoặc dùng standard cap cao (rẻ hơn)')
else:
    print('  ❌ NHT KHÔNG thắng ở cap 5M → lợi thế biểu đạt co lại khi cap cao.')
    print('     → Chuyển nhánh AN TOÀN: standard cap 8-12M + ft512@5M + ensemble + L16')
"
fi
echo "########################################################################"
echo "DÁN TOÀN BỘ KHỐI VERDICT NÀY CHO CLAUDE."
