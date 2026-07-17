#!/usr/bin/env python3
"""Blend 2 thư mục render: out = α·A + (1−α)·B (pixel-wise, khớp theo stem).

Nút vặn an toàn cho mọi enhancer (Difix/L16): α=0 hoàn nguyên B; quét α trên bàn
chấm có GT rồi mới nhận. Dùng:
  python blend.py --dir_a renders/X_difix --dir_b renders/X --alpha 0.5 --out renders/X_b50
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir_a", required=True)
    ap.add_argument("--dir_b", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="trọng số của dir_a")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    da, db, out = Path(a.dir_a), Path(a.dir_b), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bmap = {p.stem: p for p in db.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    n = 0
    for pa in sorted(da.iterdir()):
        if pa.suffix.lower() not in (".png", ".jpg", ".jpeg") or pa.stem not in bmap:
            continue
        ia = cv2.imread(str(pa)).astype(np.float64)
        ib = cv2.imread(str(bmap[pa.stem])).astype(np.float64)
        assert ia.shape == ib.shape, f"{pa.stem}: {ia.shape} ≠ {ib.shape}"
        cv2.imwrite(str(out / (pa.stem + ".png")),
                    np.clip(a.alpha * ia + (1 - a.alpha) * ib, 0, 255).astype(np.uint8))
        n += 1
    assert n > 0, "không khớp được cặp ảnh nào"
    print(f"✓ blend α={a.alpha}: {n} ảnh → {out}")


if __name__ == "__main__":
    main()
