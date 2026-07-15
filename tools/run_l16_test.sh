#!/bin/bash
# L16 test trên 2 scene calib — TỰ ĐỢI SUB2 đóng gói xong mới chiếm GPU.
# HCM0204: ckpt ut5M_30k, render test = renders/HCM0204__ut5M_30k
# HCM0181: ckpt HCM0181__SUB2,  render test = renders/HCM0181__SUB2
# Chạy: setsid nohup bash tools/run_l16_test.sh > results/l16_test.log 2>&1 &
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

while ! grep -q "SUB2-ALL-DONE\|PACKAGE-FAILED" results/sub2_resume.log 2>/dev/null; do sleep 120; done
echo "[$(date +%d/%m' '%H:%M)] GPU rảnh — bắt đầu L16"

l16_one () {  # $1=scene $2=ckpt $3=test_render_dir
  k1=$($PY - "$1" <<'EOF'
import struct, sys
with open(f"workspace_raw/{sys.argv[1]}/sparse/0/cameras.bin", "rb") as f:
    f.read(8); cid, model, w, h = struct.unpack("<iiQQ", f.read(24))
    params = struct.unpack("<dddd", f.read(32))
print(repr(params[3]))
EOF
)
  echo "[$(date +%d/%m' '%H:%M)] === L16 $1 (k1=$k1) ==="
  { $PY tools/enhance_net.py train --workspace "workspace_raw/$1" --ckpt "$2" \
      --out "results/$1__l16/net.pt" --with_ut --radial_k1 "$k1" --steps 3000 \
    && $PY tools/enhance_net.py apply --net "results/$1__l16/net.pt" \
      --in_dir "$3" --out_dir "renders/$1__l16" \
    && $PY tools/score_local.py --pred_dir "renders/$1__l16" \
      --gt_dir "VAI_NVS_DATA/phase1/public_set/$1/test/images" \
      --out "results/$1__l16_score.json" \
    && echo "[$(date +%d/%m' '%H:%M)] L16-OK $1"
  } || echo "[$(date +%d/%m' '%H:%M)] !!! L16-FAILED $1"
}

l16_one HCM0204 results/HCM0204__ut5M_30k/ckpts/ckpt_29999_rank0.pt renders/HCM0204__ut5M_30k
l16_one HCM0181 results/HCM0181__SUB2/ckpts/ckpt_29999_rank0.pt renders/HCM0181__SUB2
echo "[$(date +%d/%m' '%H:%M)] L16-TEST-ALL-DONE"
