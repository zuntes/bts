#!/usr/bin/env python3
"""ROUND 2 không có GT test → tự tạo bàn chấm: giữ lại 1/N ảnh train làm val.

Từ workspace_r2/<scene> tạo:
  workspace_r2v/<scene>/          : workspace TRAIN thiếu các ảnh val (images.bin +
                                    points3D tracks lọc lại — giống hệt cách BTC gỡ ảnh)
  workspace_r2v/<scene>/val_poses.csv : pose val ĐÚNG format test_poses.csv của BTC
  workspace_r2v/<scene>/val_gt/       : ảnh thật của các pose val (để score_local)

Model train trên workspace này CHƯA TỪNG thấy ảnh val → điểm v50 trên val ≈ điểm test
thật (cùng phân bố pose, cùng camera). Đây là bàn A/B duy nhất cho bonsai/chair.

Dùng: python make_holdout.py --ws workspace_r2/bonsai --out_ws workspace_r2v/bonsai --every 10
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import read_cameras_bin, read_images_bin, write_cameras_bin, write_images_bin
from prepare_scene import filter_tracks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", required=True, help="workspace nguồn (đủ ảnh train)")
    ap.add_argument("--out_ws", required=True)
    ap.add_argument("--every", type=int, default=10, help="giữ 1/N ảnh làm val (mặc định 10%%)")
    a = ap.parse_args()
    ws, out = Path(a.ws), Path(a.out_ws)

    cams = read_cameras_bin(ws / "sparse/0/cameras.bin")
    imgs = read_images_bin(ws / "sparse/0/images.bin")
    cam = next(iter(cams.values()))
    if cam.model == "SIMPLE_RADIAL":
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    elif cam.model in ("SIMPLE_PINHOLE",):
        f, cx, cy = cam.params[0], cam.params[1], cam.params[2]
    elif cam.model == "PINHOLE":
        f, cx, cy = cam.params[0], cam.params[2], cam.params[3]
        print("⚠ PINHOLE fx≠fy: CSV dùng fx cho cả 2 (khớp cách BTC xuất CSV)")
    else:
        sys.exit(f"❌ model {cam.model} chưa hỗ trợ")

    names = sorted(imgs.keys())  # thứ tự thời gian quỹ đạo (tên file tăng dần)
    val_names = names[a.every // 2::a.every]  # offset giữa bin — tránh mép quỹ đạo
    train_names = [n for n in names if n not in set(val_names)]
    print(f"[{ws.name}] {len(names)} ảnh → train {len(train_names)} / val {len(val_names)} (1/{a.every})")

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse/0").mkdir(parents=True, exist_ok=True)
    (out / "val_gt").mkdir(exist_ok=True)
    for n in train_names:
        dst = out / "images" / n
        if not dst.exists():
            shutil.copy(ws / "images" / n, dst)
    for n in val_names:
        dst = out / "val_gt" / n
        if not dst.exists():
            shutil.copy(ws / "images" / n, dst)

    kept = {n: imgs[n] for n in train_names}
    write_cameras_bin(out / "sparse/0/cameras.bin", cams)
    write_images_bin(out / "sparse/0/images.bin", kept)
    filter_tracks(ws / "sparse/0/points3D.bin", out / "sparse/0/points3D.bin",
                  {im.id for im in kept.values()})

    with open(out / "val_poses.csv", "w") as fh:
        fh.write("image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height\n")
        for n in val_names:
            im = imgs[n]
            q, t = im.qvec, im.tvec
            fh.write(f"{n},{q[0]},{q[1]},{q[2]},{q[3]},{t[0]},{t[1]},{t[2]},"
                     f"{f},{f},{cx},{cy},{cam.width},{cam.height}\n")
    print(f"✓ {out} · val_poses.csv ({len(val_names)} pose) · val_gt/ ({len(val_names)} ảnh)")


if __name__ == "__main__":
    main()
