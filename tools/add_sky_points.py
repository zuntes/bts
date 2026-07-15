#!/usr/bin/env python3
"""Thêm điểm init "far-sphere" cho bầu trời vào points3D.bin (docs/03 §2, giải pháp 2).

Cơ chế: trời không có SfM point → 3DGS mọc gaussians trời ở depth tuỳ ý → floaters.
Ta rải điểm trên bán cầu bán kính R = --radius_mult × extent quanh tâm scene, phía "trên"
(hướng up ước lượng từ trung bình trục -Y của các camera), màu lấy từ dải pixel trên
cùng của vài ảnh (xấp xỉ màu trời).

Chạy SAU prepare_scene.py, TRƯỚC khi train:
  python add_sky_points.py --workspace workspace/HCM0204 --n 60000
Tạo bản sao lưu points3D.bin.bak (chạy lại sẽ khôi phục từ .bak trước — idempotent).
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import read_images_bin, read_points3D_bin, write_points3D_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="thư mục scene đã prepare")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--radius_mult", type=float, default=15.0)
    args = ap.parse_args()

    ws = Path(args.workspace)
    p3d_path = ws / "sparse/0/points3D.bin"
    bak = p3d_path.with_suffix(".bin.bak")
    if bak.exists():
        shutil.copy(bak, p3d_path)  # idempotent
    else:
        shutil.copy(p3d_path, bak)

    ids, xyzs, rgbs, errs, tracks = read_points3D_bin(p3d_path)
    imgs = read_images_bin(ws / "sparse/0/images.bin")

    centers = np.array([im.center for im in imgs.values()])
    centroid = np.median(xyzs, axis=0)
    extent = np.percentile(np.linalg.norm(xyzs - centroid, axis=1), 95)
    R = args.radius_mult * extent
    # up thế giới ≈ trung bình của (-hàng Y của R_w2c) — trục y camera hướng xuống
    up = -np.mean([im.R[1] for im in imgs.values()], axis=0)
    up /= np.linalg.norm(up)
    print(f"extent(p95)={extent:.2f}  R_sky={R:.1f}  up={up.round(3)}")

    # Màu trời: trung bình dải 8% pixel trên cùng của 8 ảnh
    img_files = sorted((ws / "images").iterdir())[:: max(1, len(imgs) // 8)][:8]
    tops = []
    for p in img_files:
        im = cv2.imread(str(p))
        tops.append(im[: int(im.shape[0] * 0.08)].reshape(-1, 3).mean(axis=0))
    sky_bgr = np.mean(tops, axis=0)
    sky_rgb = sky_bgr[::-1]
    print(f"màu trời ước lượng (RGB): {sky_rgb.round(1)}")

    # Rải đều trên chỏm cầu quanh hướng up (từ ~10° trên chân trời tới thiên đỉnh)
    rng = np.random.default_rng(0)
    v = rng.normal(size=(args.n * 3, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cosang = v @ up
    v = v[cosang > np.sin(np.deg2rad(10))][: args.n]
    pts = centroid + R * v
    cols = np.clip(sky_rgb + rng.normal(0, 4, size=(len(v), 3)), 0, 255).astype(np.uint8)

    new_ids = np.arange(len(v), dtype=np.uint64) + ids.max() + 1
    write_points3D_bin(
        p3d_path,
        np.concatenate([ids, new_ids]),
        np.concatenate([xyzs, pts]),
        np.concatenate([rgbs, cols]),
        np.concatenate([errs, np.full(len(v), 1.0)]),
        tracks + [b""] * len(v),
    )
    print(f"→ thêm {len(v):,} điểm trời; tổng {len(ids) + len(v):,} điểm")


if __name__ == "__main__":
    main()
