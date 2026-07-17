#!/usr/bin/env python3
"""Lưới thumbnail (contact sheet) để soi mắt render trước khi nộp — round 2 mù GT
nên đây là lớp kiểm cuối. 1 ảnh JPEG/scene, tên ảnh in nhỏ dưới mỗi ô.

Dùng: python contact_sheet.py --renders_root renders_sub_r2 --out_dir /tmp/sheets [--cols 8 --thumb 200]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--thumb", type=int, default=200, help="chiều rộng thumbnail (px)")
    a = ap.parse_args()
    root, out = Path(a.renders_root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in scene_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if not files:
            continue
        thumbs = []
        for f in files:
            im = cv2.imread(str(f))
            if im is None:
                continue
            h, w = im.shape[:2]
            tw = a.thumb
            th = int(round(h * tw / w))
            t = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            # nhãn tên (12 ký tự cuối, tránh tràn)
            lbl = f.stem[-14:]
            cv2.putText(t, lbl, (3, th - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(t, lbl, (3, th - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (0, 255, 0), 1, cv2.LINE_AA)
            thumbs.append(t)
        if not thumbs:
            continue
        th_h = max(t.shape[0] for t in thumbs)
        th_w = a.thumb
        cols = a.cols
        rows = (len(thumbs) + cols - 1) // cols
        pad = 4
        canvas = np.full(((th_h + pad) * rows + pad, (th_w + pad) * cols + pad, 3),
                         40, np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            y = pad + r * (th_h + pad)
            x = pad + c * (th_w + pad)
            canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
        outp = out / f"{scene_dir.name}.jpg"
        cv2.imwrite(str(outp), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"{scene_dir.name}: {len(thumbs)} ảnh · {canvas.shape[1]}x{canvas.shape[0]} → {outp}")


if __name__ == "__main__":
    main()
