#!/usr/bin/env python3
"""Vòng 5 — audit dữ liệu toàn bộ 13 scenes, kiểm chứng các giả định:
1. Test poses có thật sự nội suy giữa train poses? (khoảng cách tới train gần nhất
   so với khoảng cách train-train liền kề)
2. Exposure có lệch trong scene không? (cv của mean luminance)
3. Có bao nhiêu ảnh mờ? (Laplacian variance thấp bất thường)
4. Camera model & k1 từng scene (redistort lever cần k1 đúng per-scene)
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import read_cameras_bin, read_images_bin

# Data root: argv[1] (Colab/server), mặc định = đường dẫn máy local
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path("/home/es/BTS Image Reconstruction/VAI_NVS_DATA/phase1")


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


def audit(scene_dir):
    name = scene_dir.name
    sp = scene_dir / "train/sparse/0"
    cams = read_cameras_bin(sp / "cameras.bin")
    imgs = read_images_bin(sp / "images.bin")
    on_disk = sorted(p.name for p in (scene_dir / "train/images").iterdir())
    train = [imgs[n] for n in on_disk if n in imgs]

    # --- pose: test vs train ---
    rows = list(csv.DictReader(open(scene_dir / "test/test_poses.csv")))
    tC, tdir = [], []
    for r in rows:
        q = np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")])
        t = np.array([float(r[k]) for k in ("tx", "ty", "tz")])
        R = qvec2rotmat(q)
        tC.append(-R.T @ t)
        tdir.append(R[2])  # trục nhìn (hàng z của w2c)
    tC, tdir = np.array(tC), np.array(tdir)
    trC = np.array([im.center for im in train])
    trdir = np.array([im.R[2] for im in train])

    # test → train gần nhất
    d = np.linalg.norm(tC[:, None] - trC[None], axis=2)
    nn = d.min(1)
    nn_idx = d.argmin(1)
    ang = np.degrees(np.arccos(np.clip((tdir * trdir[nn_idx]).sum(1), -1, 1)))
    # train-train spacing (láng giềng gần nhất, bỏ chính nó)
    dtt = np.linalg.norm(trC[:, None] - trC[None], axis=2)
    np.fill_diagonal(dtt, np.inf)
    tt = dtt.min(1)

    # --- exposure & blur (đọc nhanh ở 1/4 res) ---
    lum, blur = [], []
    for fn in on_disk:
        g = cv2.imread(str(scene_dir / "train/images" / fn), cv2.IMREAD_GRAYSCALE)
        g = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4))
        lum.append(g.mean())
        blur.append(cv2.Laplacian(g, cv2.CV_64F).var())
    lum, blur = np.array(lum), np.array(blur)
    blur_thresh = np.median(blur) * 0.4
    cam = next(iter(cams.values()))
    k1 = cam.params[3] if cam.model in ("SIMPLE_RADIAL",) else 0.0

    print(f"{name:9s} n_tr={len(train):3d} n_te={len(rows):2d} "
          f"| testNN/trainNN: p50={np.median(nn)/np.median(tt):4.2f} p95={np.percentile(nn,95)/np.median(tt):5.2f} "
          f"| angNN p95={np.percentile(ang,95):5.1f}° "
          f"| expo_cv={lum.std()/lum.mean():.3f} "
          f"| blurry={int((blur<blur_thresh).sum()):2d} "
          f"| {cam.model} k1={k1:+.4f}")
    return np.percentile(nn, 95) / np.median(tt)


print("scene      | test-cách-train (tỉ lệ so spacing train) | góc lệch | exposure | ảnh mờ | camera")
worst = {}
for sset in ("public_set", "private_set1"):
    print(f"--- {sset} ---")
    for sd in sorted((DATA / sset).iterdir()):
        if sd.is_dir():
            worst[sd.name] = audit(sd)
print("\nGiả định 'test = interpolation' ĐÚNG nếu tỉ lệ p95 ~ 1-2 (test nằm giữa các train view).")
print("Scene đáng lo nhất:", max(worst, key=worst.get), f"(p95 ratio = {max(worst.values()):.2f})")
