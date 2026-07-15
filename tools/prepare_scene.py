#!/usr/bin/env python3
"""Chuẩn hoá 1 scene VAI → workspace sạch cho gsplat.

Xử lý 3 pitfall đã xác minh (docs/00 §3):
1. images.bin chứa cả ảnh không có trên đĩa → lọc theo file tồn tại.
2. Camera SIMPLE_RADIAL (k1≈0.01) → undistort ảnh (Lanczos4) và ghi cameras.bin PINHOLE.
3. Đổi đuôi tên ảnh sang .png (lossless) đồng bộ giữa images.bin và file trên đĩa.

Kết quả: <out_dir>/{images/, sparse/0/{cameras,images,points3D}.bin}
Dùng:  python prepare_scene.py --scene_dir .../public_set/HCM0204 --out_dir workspace/HCM0204
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import (Camera, read_cameras_bin, read_images_bin,
                       read_points3D_bin, write_cameras_bin, write_images_bin,
                       write_points3D_bin)


def filter_tracks(src_points3D, dst_points3D, kept_image_ids):
    """Lọc track của points3D chỉ giữ image_id còn trong images.bin đã lọc.
    Không lọc = gsplat Parser KeyError (track trỏ tới ảnh đã bị loại)."""
    ids, xyzs, rgbs, errs, tracks = read_points3D_bin(src_points3D)
    kept = np.array(sorted(kept_image_ids), dtype=np.uint32)
    new_tracks = []
    n_elem_before = n_elem_after = 0
    for tr in tracks:
        if not tr:
            new_tracks.append(tr)
            continue
        arr = np.frombuffer(tr, dtype=np.uint32).reshape(-1, 2)
        mask = np.isin(arr[:, 0], kept)
        n_elem_before += len(arr)
        n_elem_after += int(mask.sum())
        new_tracks.append(arr[mask].tobytes())
    write_points3D_bin(dst_points3D, ids, xyzs, rgbs, errs, new_tracks)
    print(f"  points3D: {len(ids):,} điểm giữ nguyên; track elements "
          f"{n_elem_before:,} → {n_elem_after:,} (lọc theo {len(kept)} ảnh)")


def dist_coeffs(cam):
    """Trả về (K, dist) OpenCV cho các model COLMAP hay gặp."""
    p = cam.params
    if cam.model == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = np.zeros(5)
    elif cam.model == "PINHOLE":
        fx, fy, cx, cy = p[:4]; d = np.zeros(5)
    elif cam.model == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = np.array([p[3], 0, 0, 0, 0.0])
    elif cam.model == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = np.array([p[3], p[4], 0, 0, 0.0])
    elif cam.model == "OPENCV":
        fx, fy, cx, cy = p[:4]; d = np.array([p[4], p[5], p[6], p[7], 0.0])
    else:
        raise ValueError(f"Camera model chưa hỗ trợ: {cam.model}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    return K, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="thư mục scene (chứa train/)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--jpg", action="store_true",
                    help="lưu JPEG q98 thay vì PNG (tiết kiệm đĩa, mất chút chất lượng)")
    ap.add_argument("--keep_distortion", action="store_true",
                    help="3DGUT mode: GIỮ ảnh méo gốc + camera model gốc (không undistort). "
                         "Dùng cho train với gsplat --with_ut --with_eval3d. Vẫn lọc "
                         "images.bin + tracks như thường.")
    args = ap.parse_args()

    scene = Path(args.scene_dir)
    out = Path(args.out_dir)
    sparse_in = scene / "train/sparse/0"
    img_in = scene / "train/images"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse/0").mkdir(parents=True, exist_ok=True)

    cams = read_cameras_bin(sparse_in / "cameras.bin")
    imgs = read_images_bin(sparse_in / "images.bin")
    on_disk = {p.name for p in img_in.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")}

    kept = {k: v for k, v in imgs.items() if k in on_disk}
    print(f"[{scene.name}] images.bin: {len(imgs)} | trên đĩa: {len(on_disk)} | giữ lại: {len(kept)}")
    assert kept, "Không khớp tên ảnh nào — kiểm tra scene_dir"

    if args.keep_distortion:
        # 3DGUT mode: copy nguyên ảnh gốc + cameras.bin gốc, chỉ lọc images/tracks
        for name, im in sorted(kept.items()):
            shutil.copy(img_in / name, out / "images" / name)
        write_cameras_bin(out / "sparse/0/cameras.bin", cams)
        write_images_bin(out / "sparse/0/images.bin", kept)
        filter_tracks(sparse_in / "points3D.bin", out / "sparse/0/points3D.bin",
                      {im.id for im in kept.values()})
        cam = next(iter(cams.values()))
        print(f"  [keep_distortion] giữ {cam.model} params={cam.params.round(4)} — "
              f"train bằng: simple_trainer mcmc --with_ut --with_eval3d")
        return

    # Undistort maps per camera (tất cả scene hiện tại chỉ có 1 camera)
    maps = {}
    new_cams = {}
    for cid, cam in cams.items():
        K, d = dist_coeffs(cam)
        if np.any(d != 0):
            m1, m2 = cv2.initUndistortRectifyMap(
                K, d, None, K, (cam.width, cam.height), cv2.CV_32FC1)
            maps[cid] = (m1, m2)
            print(f"  camera {cid}: {cam.model} k={d[d != 0]} → undistort về PINHOLE")
        else:
            maps[cid] = None
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        new_cams[cid] = Camera(cid, "PINHOLE", cam.width, cam.height,
                               np.array([fx, fy, cx, cy]))

    ext = ".jpg" if args.jpg else ".png"
    new_imgs = {}
    for name, im in sorted(kept.items()):
        img = cv2.imread(str(img_in / name), cv2.IMREAD_COLOR)
        assert img is not None, f"Không đọc được {name}"
        m = maps[im.camera_id]
        if m is not None:
            # BORDER_REPLICATE: với k1>0, góc ảnh undistort lấy mẫu ngoài ảnh gốc —
            # replicate đỡ hại hơn viền đen (viền đen dạy 3DGS haze đen không nhất quán đa view)
            img = cv2.remap(img, m[0], m[1], interpolation=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE)
        new_name = Path(name).stem + ext
        if args.jpg:
            cv2.imwrite(str(out / "images" / new_name), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 98])
        else:
            cv2.imwrite(str(out / "images" / new_name), img)
        im.name = new_name
        new_imgs[new_name] = im

    write_cameras_bin(out / "sparse/0/cameras.bin", new_cams)
    write_images_bin(out / "sparse/0/images.bin", new_imgs)
    filter_tracks(sparse_in / "points3D.bin", out / "sparse/0/points3D.bin",
                  {im.id for im in new_imgs.values()})
    print(f"  → {out} sẵn sàng cho gsplat (--data_dir {out} --data_factor 1)")


if __name__ == "__main__":
    main()
