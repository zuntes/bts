#!/bin/bash
# ============================================================================
# GATE S1 — Difix3D+ (NVIDIA CVPR'25 oral): diffusion 1-bước xoá artifact render.
# Lever trần cao nhất còn lại (đánh vào LPIPS 0.4 trọng số; −0.01 LPIPS = +0.4 điểm).
#
# Bàn chấm: renders/HCM0204__ens12 (GT thật) — mốc L16-XL trên cùng input = 0.76641.
# Biến thể đo:
#   [D-raw]   Difix(ens12) nguyên chất (α=1)
#   [D-α]     blend α·Difix + (1−α)·ens12, α ∈ 0.3/0.5/0.7 — nút an toàn chống bịa chi tiết
#   [D-ref]   (RUN_REF=1) biến thể difix_ref có ảnh train tham chiếu
#   [D-onL16] Difix chồng lên renders/HCM0204__a1 (đã qua L16) — 2 enhancer cộng dồn?
#
# ⚠ LICENSE: model NVIDIA (nvidia/difix trên HF) — KIỂM điều lệ BTC trước khi dùng
#   cho bài NỘP; gate đo trước không vi phạm gì.
# Chạy SERVER (~30ph sau khi cài): bash tools/GATE_DIFIX.sh 2>&1 | tee /tmp/gate_difix.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
PY=.venv/bin/python
S=HCM0204
GT="VAI_NVS_DATA/phase1/public_set/$S/test/images"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }
score(){ $PY tools/score_local.py --pred_dir "$1" --gt_dir "$GT" 2>&1 | grep -aE "n=|★" | sed "s/^/  [$2] /"; }

say "0. tiên quyết"
[ -d "renders/${S}__ens12" ] || die "thiếu renders/${S}__ens12 (từ GATE_A)"
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9); [ "$FREE" -lt 15 ] && die "đĩa ${FREE}GB<15"

say "1. venv riêng .venv_difix + repo (KHÔNG đụng .venv chính — DOC3 §3.7)"
if ! [ -x .venv_difix/bin/python ]; then
  python3.10 -m venv .venv_difix || die "tạo .venv_difix (cần python3.10)"
  .venv_difix/bin/pip install -q --upgrade pip
fi
if ! [ -d "$HOME/Difix3D" ]; then
  git clone --depth 1 https://github.com/nv-tlabs/Difix3D "$HOME/Difix3D" || die "clone Difix3D"
fi
if ! .venv_difix/bin/python -c "import diffusers, torch" 2>/dev/null; then
  .venv_difix/bin/pip install -q -r "$HOME/Difix3D/requirements.txt" \
    || die "pip requirements Difix — dán 30 dòng cuối"
fi
.venv_difix/bin/python -c "import torch; assert torch.cuda.is_available(), 'torch trong .venv_difix không thấy CUDA'" \
  || die "torch .venv_difix không có CUDA — có thể requirements cài bản CPU; báo Claude"
DPY=".venv_difix/bin/python"
echo "  ✅ .venv_difix + ~/Difix3D"

say "2. Difix(ens12) — tải weights HF lần đầu (~vài GB, cần mạng)"
if [ "$(ls renders/${S}__difix 2>/dev/null | wc -l)" -lt 60 ]; then
  PYTHONPATH="$HOME/Difix3D/src:${PYTHONPATH:-}" $DPY tools/difix_apply.py \
    --in_dir "renders/${S}__ens12" --out_dir "renders/${S}__difix" 2>&1 | tee /tmp/difix_run.log \
    || die "difix_apply fail — dán 30 dòng cuối /tmp/difix_run.log (nghi: API đổi/OOM/HF chặn)"
fi
score "renders/${S}__difix" "D-raw"

say "3. Quét α (nút an toàn) — so với mốc L16 0.76641 và ens12-raw 0.76277"
for A in 0.3 0.5 0.7; do
  $PY tools/blend.py --dir_a "renders/${S}__difix" --dir_b "renders/${S}__ens12" \
    --alpha $A --out "renders/${S}__difix_a${A/0./}" >/dev/null || die "blend $A"
  score "renders/${S}__difix_a${A/0./}" "D-a$A"
done

say "4. Difix chồng L16 (input = renders/${S}__a1)"
if [ -d "renders/${S}__a1" ]; then
  if [ "$(ls renders/${S}__difix_onl16 2>/dev/null | wc -l)" -lt 60 ]; then
    PYTHONPATH="$HOME/Difix3D/src:${PYTHONPATH:-}" $DPY tools/difix_apply.py \
      --in_dir "renders/${S}__a1" --out_dir "renders/${S}__difix_onl16" || die "difix on L16"
  fi
  score "renders/${S}__difix_onl16" "D-onL16"
  $PY tools/blend.py --dir_a "renders/${S}__difix_onl16" --dir_b "renders/${S}__a1" \
    --alpha 0.5 --out "renders/${S}__difix_onl16_a5" >/dev/null
  score "renders/${S}__difix_onl16_a5" "D-onL16-a0.5"
else echo "  ⏩ thiếu renders/${S}__a1 — bỏ biến thể onL16"; fi

if [ "${RUN_REF:-0}" = "1" ]; then
  say "5. (tuỳ chọn) difix_ref với ảnh train tham chiếu"
  if [ "$(ls renders/${S}__difixref 2>/dev/null | wc -l)" -lt 60 ]; then
    PYTHONPATH="$HOME/Difix3D/src:${PYTHONPATH:-}" $DPY tools/difix_apply.py \
      --in_dir "renders/${S}__ens12" --out_dir "renders/${S}__difixref" \
      --ref_dir "workspace_raw/$S/images" || die "difix_ref"
  fi
  score "renders/${S}__difixref" "D-ref"
fi

echo
echo "########################################################################"
echo "#  VERDICT GATE DIFIX — mốc: L16-XL(ens12)=0.76641 · ens12-raw=0.76277"
echo "#  max(D-*) ≥ 0.76641+0.003 → Difix VÀO pipeline (thay/chồng L16 theo biến thể thắng)"
echo "#  max(D-*) trong ±0.003    → giữ L16, note Difix cho vòng sau"
echo "#  D-raw thắng LPIPS nhưng thua v50 → tăng dần α từ biến thể α tốt nhất"
echo "#  NẾU DÙNG CHO BÀI NỘP: kiểm license NVIDIA + điều lệ BTC trước"
echo "########################################################################"
echo "DÁN các dòng [D-*] + khối này CHO CLAUDE."
