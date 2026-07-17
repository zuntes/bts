#!/bin/bash
# ============================================================================
# ROUND 2 — chuẩn bị workspace cho 7 scene mới (VAI_NVS_DATA_ROUND_2).
#   5 HCM drone : SIMPLE_RADIAL k1≈+0.009, 1320×989 (gốc/4) → 3DGUT raw-distortion
#   bonsai/chair: SIMPLE_PINHOLE KHÔNG méo, video frames    → classic (KHÔNG --raw-distortion,
#                 parser sẽ assert "không có camera méo" nếu cố dùng)
# Chạy (local hoặc server, CPU-only, ~5ph): bash tools/prepare_r2.sh 2>&1 | tee /tmp/prep_r2.txt
# ============================================================================
set -uo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; cd "$PROJ"
PY=.venv/bin/python
# TỰ DÒ root data R2 — layout khác nhau giữa 2 máy (local có tầng lồng do giải nén zip,
# server phẳng). Nhận diện bằng SỰ TỒN TẠI của scene thật, không giả định đường dẫn.
R2=""
for c in VAI_NVS_DATA_ROUND_2/VAI_NVS_DATA_ROUND2 VAI_NVS_DATA_ROUND_2 VAI_NVS_DATA_ROUND2; do
  [ -s "$c/bonsai/test/test_poses.csv" ] && { R2="$c"; break; }
done
[ -n "$R2" ] || { echo "❌ không tìm thấy data round 2 (thử: VAI_NVS_DATA_ROUND_2[/VAI_NVS_DATA_ROUND2])"; \
  echo "   thấy gì ở đây: $(ls -d VAI_NVS_DATA_ROUND_2/* 2>/dev/null | head -3)"; exit 1; }
echo "  data R2: $R2"
SCENES="bonsai chair HCM0421 HCM0539 HCM0540 HCM0644 HCM0674"
say(){ echo; echo "[$(date +%H:%M)] ═════ $* ═════"; }
die(){ echo "❌ $*"; exit 1; }

say "0. tiên quyết"
[ -x "$PY" ] || die "thiếu .venv"
[ -d "$R2" ] || die "thiếu $R2 — rsync data round 2 chưa?"
for s in $SCENES; do
  [ -s "$R2/$s/train/sparse/0/images.bin" ] || die "$s thiếu train/sparse/0/images.bin"
  [ -s "$R2/$s/test/test_poses.csv" ] || die "$s thiếu test/test_poses.csv"
done
echo "  ✅ đủ 7 scene"

say "1. prepare workspace_r2/<scene> (--keep_distortion: HCM giữ k1 cho 3DGUT; pinhole = copy nguyên)"
for s in $SCENES; do
  if [ -s "workspace_r2/$s/sparse/0/images.bin" ] && \
     [ "$(ls "workspace_r2/$s/images" 2>/dev/null | wc -l)" -ge 100 ]; then
    echo "  ⏩ $s đã có"; continue
  fi
  $PY tools/prepare_scene.py --scene_dir "$R2/$s" --out_dir "workspace_r2/$s" --keep_distortion \
    || die "prepare $s"
done

say "2. verify chéo: CSV test ≡ cameras.bin (fx/cx/cy/w/h) + đếm ảnh + phân nhánh chiến thuật"
# R2_ROOT qua env: heredoc <<'EOF' KHÔNG nội suy biến shell (cố ý — python có nhiều $),
# nên truyền đường dẫn bằng biến môi trường thay vì hardcode (đã dính bug này 17/07).
R2_ROOT="$R2" $PY - <<'EOF' || die "verify fail — ĐỌC LỖI Ở TRÊN, đừng train tiếp"
import csv, os, struct, sys
from pathlib import Path
R2 = Path(os.environ["R2_ROOT"])
NP = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}
MO = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}
bad = 0
for s in ["bonsai", "chair", "HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674"]:
    with open(f"workspace_r2/{s}/sparse/0/cameras.bin", "rb") as f:
        f.read(8)
        cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
        p = struct.unpack(f"<{NP[model]}d", f.read(8 * NP[model]))
    if model == 2:
        f_, cx, cy, k1 = p
    elif model in (0,):
        f_, cx, cy = p; k1 = 0.0
    else:
        print(f"  ❌ {s}: model {MO.get(model, model)} chưa xử lý trong luồng r2"); bad += 1; continue
    rows = list(csv.DictReader(open(R2 / s / "test/test_poses.csv")))
    r = rows[0]
    for name, a, b in [("fx", float(r["fx"]), f_), ("cx", float(r["cx"]), cx),
                       ("cy", float(r["cy"]), cy), ("w", int(r["width"]), w), ("h", int(r["height"]), h)]:
        if abs(a - b) > 1e-3:
            print(f"  ❌ {s}: CSV {name}={a} ≠ cameras.bin {b} — test KHÔNG cùng camera, render sẽ sai!")
            bad += 1
    n_img = len(list(Path(f"workspace_r2/{s}/images").iterdir()))
    branch = "3DGUT raw-distortion" if model == 2 else "CLASSIC (pinhole, không méo)"
    print(f"  ✅ {s}: {MO[model]} f={f_:.1f} k1={k1:+.4f} {w}x{h} · train={n_img} · test={len(rows)} → {branch}")
sys.exit(1 if bad else 0)
EOF

echo
echo "########################################################################"
echo "✅ workspace_r2/ sẵn sàng. Bước tiếp: bash tools/run_sub_r2.sh"
echo "########################################################################"
