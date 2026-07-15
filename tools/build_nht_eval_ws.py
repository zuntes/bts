#!/usr/bin/env python3
"""Cầu NHT: dựng eval-workspace COLMAP cho 3DGRUT với 60 competition pose ĐÃ
pre-transform bằng T240 (normalization lúc train). Dùng normalize=false khi render
→ pose khớp đúng frame model NHT. Tái dùng render.py của 3DGRUT (không cần ray code).

Out: <out_ws>/{images/, sparse/0/{cameras.bin,images.bin,points3D.bin}}
  images/ = GT competition (nếu public, để score) HOẶC ảnh đen (private)
"""
import argparse, csv, sys, shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from normalize_compat import compute_parser_transform, transform_points
from colmap_io import (read_images_bin, read_points3D_bin, read_cameras_bin,
                       write_images_bin, write_cameras_bin, write_points3D_bin,
                       Image, Camera)


def rotmat2qvec(R):
    # chuẩn COLMAP
    Rxx,Ryx,Rzx,Rxy,Ryy,Rzy,Rxz,Ryz,Rzz = R.flat
    K = np.array([
        [Rxx-Ryy-Rzz,0,0,0],[Ryx+Rxy,Ryy-Rxx-Rzz,0,0],
        [Rzx+Rxz,Rzy+Ryz,Rzz-Rxx-Ryy,0],[Ryz-Rzy,Rzx-Rxz,Rxy-Ryx,Rxx+Ryy+Rzz]])/3
    vals,vecs = np.linalg.eigh(K)
    q = vecs[[3,0,1,2], np.argmax(vals)]
    if q[0] < 0: q = -q
    return q


def q2R(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*z*w,2*x*z+2*y*w],
                     [2*x*y+2*z*w,1-2*x*x-2*z*z,2*y*z-2*x*w],
                     [2*x*z-2*y*w,2*y*z+2*x*w,1-2*x*x-2*y*y]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_ws", required=True, help="workspace_raw/<scene> (để tính T240)")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--gt_dir", default=None, help="ảnh GT competition (public); None → ảnh đen")
    ap.add_argument("--out_ws", required=True)
    a = ap.parse_args()

    tw = Path(a.train_ws)
    imgs = read_images_bin(tw / "sparse/0/images.bin")
    ids, xyz, rgb, err, tracks = read_points3D_bin(tw / "sparse/0/points3D.bin")
    cams = read_cameras_bin(tw / "sparse/0/cameras.bin")
    T = compute_parser_transform(imgs, xyz)   # = T240, KHỚP 3DGRUT (verified)
    print(f"T240 scale={np.linalg.norm(T[:3,:3][0]):.5f}")

    out = Path(a.out_ws); (out/"sparse/0").mkdir(parents=True, exist_ok=True)
    (out/"images").mkdir(exist_ok=True)

    rows = list(csv.DictReader(open(a.test_csv)))
    # cameras.bin: mỗi pose có thể intrinsics riêng → tạo 1 camera/pose (SIMPLE_RADIAL giữ k1 train)
    k1 = float(cams[list(cams)[0]].params[3]) if cams[list(cams)[0]].model=="SIMPLE_RADIAL" else 0.0
    new_cams, new_imgs = {}, {}
    import cv2
    for i, r in enumerate(rows):
        cid = i+1
        fx,fy,cx,cy = (float(r[k]) for k in ("fx","fy","cx","cy"))
        w,h = int(r["width"]), int(r["height"])
        # SIMPLE_RADIAL params: f, cx, cy, k1 (giữ méo gốc như train)
        new_cams[cid] = Camera(cid, "SIMPLE_RADIAL", w, h, np.array([fx, cx, cy, k1]))
        # pose competition (w2c) → c2w → áp T240 → w2c mới
        q = np.array([float(r[k]) for k in ("qw","qx","qy","qz")])
        t = np.array([float(r[k]) for k in ("tx","ty","tz")])
        w2c = np.eye(4); w2c[:3,:3]=q2R(q); w2c[:3,3]=t
        c2w = np.linalg.inv(w2c)
        c2w_n = T @ c2w
        s = np.linalg.norm(c2w_n[:3,0])           # tách scale khỏi rotation
        c2w_n[:3,:3] /= s
        w2c_n = np.linalg.inv(c2w_n)
        qn = rotmat2qvec(w2c_n[:3,:3]); tn = w2c_n[:3,3]
        name = r["image_name"]
        new_imgs[cid] = Image(cid, qn, tn, cid, name)
        # ảnh: GT nếu có (score), else đen
        dst = out/"images"/name
        if a.gt_dir and (Path(a.gt_dir)/name).exists():
            shutil.copy(Path(a.gt_dir)/name, dst)
        else:
            cv2.imwrite(str(dst), np.zeros((h,w,3), np.uint8))

    write_cameras_bin(out/"sparse/0/cameras.bin", new_cams)
    write_images_bin(out/"sparse/0/images.bin", new_imgs)
    # points3D: áp T240 (để scene extent hợp lý), giữ nguyên tracks
    xyz_n = transform_points(T, xyz)
    write_points3D_bin(out/"sparse/0/points3D.bin", ids, xyz_n, rgb, err, tracks)
    print(f"✓ eval workspace: {out} ({len(rows)} poses, k1={k1:+.5f})")


if __name__ == "__main__":
    main()
