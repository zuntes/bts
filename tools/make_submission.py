#!/usr/bin/env python3
"""Đóng gói submission.zip từ các thư mục render, kèm kiểm tra tự động:
đúng tên scene (giữ nguyên hoa/thường), đúng tên/số lượng/kích thước ảnh theo CSV.

Dùng:
  python make_submission.py --data_root VAI_NVS_DATA/phase1/private_set1 \
      --renders_root renders --out submission.zip [--ext .png]
"""
import argparse
import csv
import zipfile
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="thư mục chứa các scene cần nộp")
    ap.add_argument("--renders_root", required=True, help="renders/<scene>/<image>")
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--ext", default=".png", help="đuôi file render ('.same' = giữ như CSV)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    renders = Path(args.renders_root)
    scenes = sorted(p for p in data_root.iterdir()
                    if p.is_dir() and (p / "test/test_poses.csv").exists())
    assert scenes, "Không tìm thấy scene nào"

    errors = []
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            rows = list(csv.DictReader(open(scene / "test/test_poses.csv")))
            rdir = renders / scene.name
            n_ok = 0
            for r in rows:
                name = r["image_name"]
                if args.ext != ".same":
                    name = str(Path(name).with_suffix(args.ext))
                f = rdir / name
                if not f.exists():
                    errors.append(f"{scene.name}: thiếu {name}")
                    continue
                img = cv2.imread(str(f))
                w, h = int(r["width"]), int(r["height"])
                if img is None or img.shape[1] != w or img.shape[0] != h:
                    errors.append(f"{scene.name}/{name}: kích thước "
                                  f"{None if img is None else img.shape[:2]} ≠ ({h},{w})")
                    continue
                zf.write(f, f"{scene.name}/{name}")
                n_ok += 1
            print(f"{scene.name}: {n_ok}/{len(rows)} ảnh")

    if errors:
        print(f"\n*** {len(errors)} LỖI — submission KHÔNG hợp lệ ***")
        for e in errors[:20]:
            print(" -", e)
    else:
        print(f"\n✓ Hợp lệ. → {args.out}")


if __name__ == "__main__":
    main()
