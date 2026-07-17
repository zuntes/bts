#!/usr/bin/env python3
"""N1 blur-match nửa sau: áp Gaussian blur σ lên renders (khớp GT mờ lúc chấm/nộp).
Dùng: python apply_blur.py --in_dir renders/X --out_dir renders/X_b08 --sigma 0.8"""
import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sigma", type=float, required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(a.in_dir).iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        im = cv2.imread(str(p))
        cv2.imwrite(str(out / p.name), cv2.GaussianBlur(im, (0, 0), a.sigma))
        n += 1
    assert n, "không có ảnh"
    print(f"✓ blur σ={a.sigma}: {n} ảnh → {out}")


if __name__ == "__main__":
    main()
