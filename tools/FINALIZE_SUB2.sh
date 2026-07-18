#!/bin/bash
# ============================================================================
# SÁNG 19/07 (~08:15) — RÁP CUỐI SUB2: mean(3 seed + members) → vgg → zip.
# Chạy SAU khi prod đêm (2 tmux) xong. ~30-40ph. Rồi NỘP 9h.
# Chạy: SUBTAG=2 bash tools/FINALIZE_SUB2.sh 2>&1 | tee /tmp/finalize.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
SUBTAG=${SUBTAG:-2}
MIN_SEEDS=${MIN_SEEDS:-2}   # deadline gấp 19/07: chấp nhận 2 seed nếu seed thứ 3 chưa kịp xong
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }

for s in bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674; do
  say "$s — gộp seed + members"
  D=""
  for seed in 42 7 123; do
    [ "$(ls "renders_r2/${s}__s${seed}" 2>/dev/null | wc -l)" -ge 5 ] && D="$D renders_r2/${s}__s${seed}"
  done
  N_SEED_OK=$(echo $D | wc -w)
  [ "$N_SEED_OK" -ge "$MIN_SEEDS" ] || die "$s chỉ có $N_SEED_OK/$MIN_SEEDS seed renders (prod đêm chưa xong đủ?)"
  [ "$N_SEED_OK" -lt 3 ] && echo "  ⚠ chỉ $N_SEED_OK seed (thiếu 1) — vẫn chạy tiếp, điểm sẽ thấp hơn full 3-seed 1 chút"
  for m in renders_r2/${s}__m_*; do
    [ -d "$m" ] && [ "$(ls "$m" | wc -l)" -ge 5 ] && D="$D $m"
  done
  echo "  members: $(echo $D | wc -w) dirs"
  $PY tools/ensemble.py --dirs $D --out "renders_r2/${s}__ensF" --mode mean >/dev/null || die "ens $s"
  NET="results/r2_${s}__enh_vgg/net.pt"
  [ -s "$NET" ] || die "$s thiếu enhancer net (prod tạo ở bước enh)"
  $PY tools/enhance_net.py apply --net "$NET" \
    --in_dir "renders_r2/${s}__ensF" --out_dir "renders_r2/${s}__sub${SUBTAG}" >/dev/null || die "vgg $s"
  echo "  ✅ $s → renders_r2/${s}__sub${SUBTAG}"
done

say "đóng gói + contact sheet"
PACK_ONLY=1 SUBTAG=$SUBTAG bash tools/run_sub_r2.sh || die "pack"
$PY tools/contact_sheet.py --renders_root renders_sub_r2 --out_dir /tmp/sheets_sub${SUBTAG} --cols 8 --thumb 190
echo
echo "✅ submission_R2_SUB${SUBTAG}.zip sẵn — SOI /tmp/sheets_sub${SUBTAG}/*.jpg rồi mới nộp"
