#!/usr/bin/env python3
"""HYBRID IBR — lá bài round-1 (user đề xuất, đánh nhãn trong docs cũ, CHƯA TỪNG TEST).

Ý tưởng: splats cho DEPTH tốt nhưng màu tần số cao bị mất. Ảnh train THẬT có đủ
chi tiết — warp chúng sang pose test bằng depth render được, blend vào chỗ nào
nhất quán (photometric), giữ render GS ở chỗ che khuất/lệch. Hợp lệ: chỉ dùng
ảnh train + pose train (không đụng test).

Điều kiện thuận: test spacing p50 ≈ 1× train spacing → luôn có ảnh train sát pose test.

Dùng:
  python ibr_blend.py --ckpt CKPT --ws WORKSPACE --csv POSES.csv \
      --render_dir DIR_GS_RENDER --out_dir OUT [--k 2] [--tau 20] [--alpha 0.7]
"""
import argparse
import csv as csvmod
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import read_cameras_bin, read_images_bin, read_points3D_bin
from normalize_compat import (colmap_w2c_to_normalized_viewmat,
                              compute_parser_transform)


def qvec2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ws", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--render_dir", required=True, help="render GS tốt nhất hiện có (nền blend)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=2, help="số ảnh train gần nhất để warp")
    ap.add_argument("--tau", type=float, default=20.0, help="ngưỡng nhất quán (0-255)")
    ap.add_argument("--alpha", type=float, default=0.7, help="trọng số ảnh warp ở vùng nhất quán")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ws = Path(a.ws)

    from gsplat import rasterization
    ck = torch.load(a.ckpt, map_location=dev, weights_only=True)["splats"]
    means = ck["means"].to(dev)
    quats = torch.nn.functional.normalize(ck["quats"].to(dev), dim=-1)
    scales = torch.exp(ck["scales"].to(dev))
    opac = torch.sigmoid(ck["opacities"].to(dev))
    colors = torch.cat([ck["sh0"], ck["shN"]], 1).to(dev)
    shd = int(np.sqrt(colors.shape[1]) - 1)

    cams = read_cameras_bin(ws / "sparse/0/cameras.bin")
    imgs = read_images_bin(ws / "sparse/0/images.bin")
    _, xyz, _, _, _ = read_points3D_bin(ws / "sparse/0/points3D.bin")
    T = compute_parser_transform(imgs, xyz)
    cam = next(iter(cams.values()))

    # pose + viewmat (chuẩn hoá) của mọi ảnh TRAIN
    train = []
    for name, im in sorted(imgs.items()):
        p = ws / "images" / name
        if not p.exists():
            continue
        vm = colmap_w2c_to_normalized_viewmat(im.R, im.tvec, T)
        C = -im.R.T @ im.tvec
        train.append(dict(name=name, path=p, vm=vm, C=C, z=im.R[2]))
    assert train, "không có ảnh train"

    rows = list(csvmod.DictReader(open(a.csv)))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rd = Path(a.render_dir)
    n_used_total = 0

    for r in rows:
        stem = Path(r["image_name"]).stem
        base_p = next(rd.glob(stem + ".*"), None)
        if base_p is None:
            print(f"  ⚠ thiếu render nền {stem}"); continue
        base = cv2.imread(str(base_p)).astype(np.float32)
        H, W = base.shape[:2]
        fx, fy, cx, cy = (float(r[k]) for k in ("fx", "fy", "cx", "cy"))
        q = np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")])
        t = np.array([float(r[k]) for k in ("tx", "ty", "tz")])
        Rt = qvec2R(q)
        vm_test = colmap_w2c_to_normalized_viewmat(Rt, t, T)
        C_test = -Rt.T @ t

        # 1. render DEPTH tại pose test (classic — đủ cho obj scenes)
        Kt = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]).float()[None].to(dev)
        vmt = torch.from_numpy(vm_test).float()[None].to(dev)
        with torch.no_grad():
            rend, _, _ = rasterization(means, quats, scales, opac, colors, vmt, Kt,
                                       W, H, sh_degree=shd, near_plane=0.01, far_plane=1e10,
                                       rasterize_mode="antialiased", render_mode="RGB+ED")
        depth = rend[0, ..., 3].cpu().numpy()  # z-depth hệ camera (đã chuẩn hoá)

        # 2. K ảnh train gần nhất (khoảng cách tâm + cùng hướng nhìn)
        cand = sorted(train, key=lambda d: np.linalg.norm(d["C"] - C_test))
        cand = [c for c in cand if float(np.dot(c["z"], Rt[2])) > 0.6][: a.k]

        # 3. unproject test → world (chuẩn hoá) → project vào từng train view
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        Xc = np.stack([(us - cx) / fx * depth, (vs - cy) / fy * depth, depth,
                       np.ones_like(depth)], -1)          # [H,W,4] hệ cam test
        inv_vm = np.linalg.inv(vm_test)
        Xw = Xc @ inv_vm.T                                 # [H,W,4] world chuẩn hoá

        best = base.copy()
        best_diff = np.full((H, W), 1e9, np.float32)
        for c in cand:
            Xc2 = Xw @ c["vm"].T                           # hệ cam train
            z2 = Xc2[..., 2]
            u2 = Xc2[..., 0] / np.maximum(z2, 1e-6) * fx + cx
            v2 = Xc2[..., 1] / np.maximum(z2, 1e-6) * fy + cy
            tr_im = cv2.imread(str(c["path"])).astype(np.float32)
            if tr_im.shape[0] != H:                        # phòng camera bin khác thang ảnh
                sc = H / tr_im.shape[0]
                tr_im = cv2.resize(tr_im, (W, H))
            warped = cv2.remap(tr_im, u2.astype(np.float32), v2.astype(np.float32),
                               cv2.INTER_LINEAR, borderValue=-1)
            valid = (z2 > 0) & (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H) & (warped[..., 0] >= 0)
            diff = np.abs(warped - base).mean(-1)
            diff[~valid] = 1e9
            upd = diff < best_diff
            best_diff = np.where(upd, diff, best_diff)
            best = np.where(upd[..., None], warped, best)

        mask = (best_diff < a.tau)[..., None]
        n_used = float(mask.mean())
        n_used_total += n_used
        outim = np.where(mask, a.alpha * best + (1 - a.alpha) * base, base)
        cv2.imwrite(str(out / (stem + ".png")), np.clip(outim, 0, 255).astype(np.uint8))

    print(f"✓ IBR: {len(rows)} ảnh → {out} · tỉ lệ pixel dùng warp trung bình "
          f"{n_used_total / max(len(rows), 1) * 100:.1f}% (tau={a.tau}, alpha={a.alpha}, k={a.k})")


if __name__ == "__main__":
    main()
