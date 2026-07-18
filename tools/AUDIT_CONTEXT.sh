#!/bin/bash
# ============================================================================
# AUDIT_CONTEXT — rà soát lấy lại context sau sự cố: phân loại scene MỜ/NÉT
# TỰ ĐỘNG (đo lapvar, không hardcode tên scene), re-score mọi render dir có bàn
# chấm GT (cache — cái đã có điểm thì bỏ qua), in bảng trạng thái method theo
# TỪNG NHÓM scene (mờ ≠ nét — không đại diện cho nhau), và (tuỳ chọn) đo gate
# W1 test-weight trên bàn SHARP (HCM0204 half-res) — lỗ hổng GATE_WINNER chỉ đo trên mờ.
#
# Chạy local 4060 hoặc server (đều được):
#   bash tools/AUDIT_CONTEXT.sh 2>&1 | tee /tmp/audit_ctx.txt            # chỉ audit (CPU+GPU score)
#   RUN_W_SHARP=1 bash tools/AUDIT_CONTEXT.sh 2>&1 | tee /tmp/audit_ctx.txt  # + gate W1-sharp (GPU, ~3h)
# Env: BLUR_THR=500 (p10 lapvar dưới ngưỡng = MỜ) · CAP_SHARP=3000000 (fit 4060)
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset BTS_TMASK
PY=.venv/bin/python
BLUR_THR=${BLUR_THR:-500}
CAP_SHARP=${CAP_SHARP:-3000000}
CACHE=results/scores_cache; mkdir -p "$CACHE"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }

say "0. môi trường"
[ -x "$PY" ] || die "thiếu .venv ($PWD)"
echo "  máy: $(hostname) · project: $PWD"
df -h . | tail -1 | awk '{print "  đĩa: "$4" trống / "$2" ("$5" dùng)"}'
FREE=$(df --output=avail -BG . | tail -1 | tr -dc 0-9)
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  GPU /' || echo "  (không thấy GPU)"

# ---------------------------------------------------------------------------
say "1. PHÂN LOẠI SCENE MỜ/NÉT — đo lapvar thật (mean + p10), KHÔNG hardcode"
# p10 quan trọng hơn mean: video có frame mờ KHÔNG ĐỀU (chair mean 1506 nhưng min 63)
$PY - <<EOF || die "phân loại scene"
import cv2, csv, numpy as np
from pathlib import Path
rows = []
for ws_root in ("workspace_r2", "workspace_r2v"):
    root = Path(ws_root)
    if not root.is_dir(): continue
    for sd in sorted(root.iterdir()):
        img_dir = sd / "images"
        if not img_dir.is_dir(): continue
        imgs = sorted(img_dir.iterdir())
        if len(imgs) < 5: continue
        sample = imgs[:: max(1, len(imgs) // 12)][:12]
        lv = []
        for p in sample:
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is None: continue
            lv.append(cv2.Laplacian(im, cv2.CV_64F).var())
        if not lv: continue
        lv = np.array(lv)
        p10 = float(np.percentile(lv, 10)); mean = float(lv.mean())
        cls = "MO" if p10 < $BLUR_THR else "NET"
        rows.append((ws_root, sd.name, mean, p10, cls))
seen = set(); out = []
for r in rows:  # ưu tiên workspace_r2 (bản đầy đủ), r2v (holdout) chỉ khi scene chưa gặp
    if r[1] in seen: continue
    seen.add(r[1]); out.append(r)
print(f"  {'scene':<10} {'lapvar_mean':>12} {'lapvar_p10':>11}  phân loại (ngưỡng p10<{$BLUR_THR})")
with open("/tmp/scene_class.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scene", "lapvar_mean", "lapvar_p10", "class"])
    for ws, s, m, p, c in out:
        label = "MỜ (video)" if c == "MO" else "NÉT (drone)"
        print(f"  {s:<10} {m:>12.0f} {p:>11.0f}  {label}")
        w.writerow([s, f"{m:.0f}", f"{p:.0f}", c])
print("  → /tmp/scene_class.csv (script khác đọc file này thay vì hardcode tên)")
EOF

# ---------------------------------------------------------------------------
say "2. CONFIRM ĐIỂM — re-score render dir có bàn GT; đã có cache thì BỎ QUA"
score_one(){ # $1=render_dir $2=gt_dir $3=tag
  local n; n=$(ls "$1" 2>/dev/null | wc -l)
  [ "$n" -lt 5 ] && return 0
  [ -d "$2" ] || return 0
  local cf="$CACHE/$3.v50"
  if [ -s "$cf" ]; then echo "  ⏩ $3 = $(cat "$cf") (cache)"; return 0; fi
  local v50
  v50=$($PY tools/score_local.py --pred_dir "$1" --gt_dir "$2" 2>/dev/null \
        | grep -a "Score_BTC\[vgg, PSNR_max=50\]" | grep -oE "[0-9]+\.[0-9]+$")
  if [ -n "$v50" ]; then echo "$v50" > "$cf"; echo "  ✅ $3 = $v50 (MỚI ĐO — đối chiếu experiments.csv!)"
  else echo "  ⚠ $3: score lỗi (thiếu torch/lpips? xem tay)"; fi
}
echo "  --- nhóm MỜ (bàn = holdout val_gt per-scene) ---"
for d in renders_r2v/*/; do
  [ -d "$d" ] || continue
  tag=$(basename "$d"); s=${tag%%__*}
  score_one "$d" "workspace_r2v/$s/val_gt" "r2v_$tag"
done
echo "  --- nhóm NÉT (bàn = workspace_r2cal/gt_half, HCM0204 half-res) ---"
for d in renders_r2cal/*/; do
  [ -d "$d" ] || continue
  score_one "$d" "workspace_r2cal/gt_half" "r2cal_$(basename "$d")"
done
echo "  (renders_r2/* KHÔNG có GT — chỉ selfcheck, không score được. Đó là bài nộp.)"

# ---------------------------------------------------------------------------
say "3. BẢN ĐỒ METHOD THEO NHÓM SCENE (nguồn: results/experiments.csv — số đo thật)"
cat <<'TABLE'
  ┌────────────────────────┬───────────────────────┬──────────────────────┐
  │ Method                 │ Scene NÉT (HCM)       │ Scene MỜ (bonsai/chair)│
  ├────────────────────────┼───────────────────────┼──────────────────────┤
  │ 3DGUT raw-distortion   │ ✅ nền (+1.0đ r1)      │ — (pinhole, classic)  │
  │ Cap knee               │ ✅ 6M (H1)             │ ✅ 3M (6M hại bonsai)  │
  │ SH degree 4            │ chưa đo riêng          │ ✅ +0.005/+0.002 (O4)  │
  │ 3-seed ensemble        │ ✅ +0.009 (A4/H2)      │ ✅ +0.011 (E3)         │
  │ Config-ensemble member │ ✅ +0.0024 (CE4)       │ ✅ chair vượt 3-seed   │
  │ Enhancer VGG per-scene │ ✅ +0.005 (B1)         │ ✅ +0.019/+0.010 (O5)  │
  │ ENH2MEAN (2 enhancer)  │ chưa đo                │ ✅ chair +0.0016       │
  │ 45k steps member       │ ~0 solo (F2)           │ 🟡 member (+0.002)     │
  │ erank member           │ ~0 solo (F4)           │ 🟡 member              │
  │ W1 test-weight ×3      │ ❓ CHƯA ĐO ← RUN_W_SHARP│ ❌ +0.0003/+0.0026<ngưỡng│
  │ W2 branched members    │ ❓ chưa đo (ROI thấp)   │ ❌ ÂM cả 2 scene       │
  │ app_opt exposure       │ ❓ chưa đo              │ ❓ chưa đo (cần cầu render)│
  │ blur-match/IBR/Difix-0shot/bilagrid/pose-BA/supersample/q98 │ ❌ loại (đo số) │ ❌ loại │
  └────────────────────────┴───────────────────────┴──────────────────────┘
  NGUYÊN TẮC: kết luận nhóm MỜ không suy ra nhóm NÉT và ngược lại (O2/SS2 đã chứng minh).
TABLE

# ---------------------------------------------------------------------------
if [ "${RUN_W_SHARP:-0}" = "1" ]; then
  say "4. GATE W1-SHARP — test-weight trên bàn NÉT (HCM0204 half, cap ${CAP_SHARP})"
  [ "$FREE" -lt 10 ] && die "đĩa ${FREE}GB<10 — dọn trước khi train"
  S=HCM0204; K1=0.010009930826722385
  CSV_FULL="VAI_NVS_DATA/phase1/public_set/$S/test/test_poses.csv"
  GT_FULL="VAI_NVS_DATA/phase1/public_set/$S/test/images"
  CAL=workspace_r2cal
  [ -s "workspace_raw/$S/sparse/0/images.bin" ] || die "thiếu workspace_raw/$S (máy này không có data round-1 public?)"
  [ -f "$CSV_FULL" ] || die "thiếu $CSV_FULL"

  # dựng bàn half-res nếu thiếu (giống hệt GATE_R2_HCM §0)
  if ! [ -s "$CAL/test_poses_half.csv" ] || [ "$(ls "$CAL/gt_half" 2>/dev/null | wc -l)" -lt 60 ]; then
    $PY - <<EOF || die "dựng bàn half-res"
import csv, cv2
from pathlib import Path
cal = Path("$CAL"); (cal / "gt_half").mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(open("$CSV_FULL")))
with open(cal / "test_poses_half.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
    for r in rows:
        for k in ("fx", "fy", "cx", "cy"): r[k] = str(float(r[k]) / 2)
        r["width"] = str(int(r["width"]) // 2); r["height"] = str(int(r["height"]) // 2)
        w.writerow(r)
for p in Path("$GT_FULL").iterdir():
    dst = cal / "gt_half" / p.name
    if dst.exists(): continue
    im = cv2.imread(str(p))
    cv2.imwrite(str(dst), cv2.resize(im, (im.shape[1]//2, im.shape[0]//2), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 100])
print("✓ bàn half-res sẵn")
EOF
  else echo "  ⏩ bàn chấm có"; fi
  if [ "$(ls "workspace_raw/$S/images_2" 2>/dev/null | wc -l)" -lt 240 ]; then
    echo "  → sinh images_2 (~2-3ph)..."
    $PY - <<EOF || die "sinh images_2"
import cv2
from pathlib import Path
src = Path("workspace_raw/$S/images"); dst = Path("workspace_raw/$S/images_2"); dst.mkdir(exist_ok=True)
for p in sorted(src.iterdir()):
    o = dst / (p.stem + ".png")
    if o.exists(): continue
    im = cv2.imread(str(p))
    cv2.imwrite(str(o), cv2.resize(im, (im.shape[1]//2, im.shape[0]//2), interpolation=cv2.INTER_AREA))
print("✓ images_2 ok")
EOF
  else echo "  ⏩ images_2 có"; fi

  tr_rend(){ # $1=tag $2=extra_flags
    local tag=$1 extra=$2 res="results/r2cal_${S}__${tag}"
    if ! [ -s "$res/ckpts/ckpt_29999_rank0.pt" ]; then
      $PY gsplat/examples/simple_trainer.py mcmc --data-dir "workspace_raw/$S" --data-factor 2 \
        --result-dir "$PWD/$res" --max-steps 30000 --test-every 999999 \
        --disable-viewer --antialiased --with-ut --with-eval3d --raw-distortion \
        --strategy.cap-max "$CAP_SHARP" --eval-steps 30000 --save-steps 30000 --global-seed 42 \
        $extra 2>&1 | tee "/tmp/wsharp_${tag}.log" | grep -aE "BTS W1|Error|error" || die "train $tag"
      rm -f "$res/ckpts/ckpt_14999_rank0.pt"; rm -rf "$res/videos"
    else echo "  ⏩ $tag ckpt có"; fi
    if [ "$(ls "renders_r2cal/wsharp_${tag}" 2>/dev/null | wc -l)" -lt 60 ]; then
      $PY tools/render_test_poses.py --ckpt "$res/ckpts/ckpt_29999_rank0.pt" \
        --csv "$CAL/test_poses_half.csv" --out "renders_r2cal/wsharp_${tag}" \
        --data_dir "workspace_raw/$S" --antialiased --with_ut --radial_k1 $K1 \
        2>&1 | grep -av "render " || die "render $tag"
    fi
    score_one "renders_r2cal/wsharp_${tag}" "$CAL/gt_half" "wsharp_${tag}"
  }
  tr_rend base ""
  tr_rend w1 "--test-weight 3.0 --test-csv-path $CAL/test_poses_half.csv"
  echo
  echo "########################################################################"
  echo "#  VERDICT W1-SHARP: wsharp_w1 − wsharp_base ≥ +0.003 → W1 SỐNG trên scene"
  echo "#  NÉT → thêm member --test-weight 3 cho 5 scene HCM (train thêm 1 lượt/scene,"
  echo "#  KHÔNG phải retrain prod). Mốc lịch sử cap3M_s42 server = 0.80257."
  echo "#  < +0.003 → W1 đóng hồ sơ VĨNH VIỄN (cả mờ lẫn nét đều không ăn)."
  echo "########################################################################"
fi

echo
echo "AUDIT-CONTEXT-DONE ($(date +%d/%m' '%H:%M))"
