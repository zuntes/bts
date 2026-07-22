#!/usr/bin/env python3
"""Render bridge cho ckpt app_opt (appearance embedding) — PHƯƠNG ÁN MỚI.

app_opt: mỗi ảnh train có embedding riêng hấp thụ exposure/màu → geometry sạch hơn
(không "trả giá" cho biến thiên phơi sáng). Test: embed_ids=None → embedding ZERO
= "phơi sáng trung bình" (GLO chuẩn, arXiv 2512.23998).

Ckpt app_opt LƯU khác: splats['features'] (32-dim) + splats['colors'] (base) +
ckpt['app_module'] (state_dict). colors cuối = sigmoid(app_module(features,dirs)+colors).
KHÁC render thường (đọc sh0/shN) → cần bridge riêng.

⚠ AN TOÀN: --validate render 3 TRAIN view, PSNR phải >26dB. Nếu <20 = bridge SAI
(giống bẫy T3-flip render 14dB im lặng) → DỪNG, không tin kết quả test.

Dùng:
  python render_app.py --ckpt ck.pt --ws workspace_raw/HCM0204 --validate   # kiểm bridge
  python render_app.py --ckpt ck.pt --ws ... --csv test_poses.csv --out renders/x \
      [--with_ut --radial_k1 K1 --radial_k2 K2]
"""
import argparse
import csv as csvmod
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import read_cameras_bin, read_images_bin, read_points3D_bin
from normalize_compat import (colmap_w2c_to_normalized_viewmat,
                              compute_parser_transform)
from gsplat import rasterization


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_app_ckpt(ckpt_path, dev):
    from utils import AppearanceOptModule
    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    s = ck["splats"]
    assert "features" in s and "app_module" in ck, "ckpt KHÔNG phải app_opt (thiếu features/app_module)"
    means = s["means"].to(dev)
    quats = F.normalize(s["quats"].to(dev), dim=-1)
    scales = torch.exp(s["scales"].to(dev))
    opac = torch.sigmoid(s["opacities"].to(dev))
    features = s["features"].to(dev)
    base_colors = s["colors"].to(dev)          # [N, 3] hoặc [N, K, 3]?
    # app_module: đọc dims từ state_dict
    st = ck["app_module"]
    n_embed, embed_dim = st["embeds.weight"].shape
    # feature_dim + embed_dim + (sh+1)^2 = first linear in_features
    first_w = st["color_head.0.weight"]
    in_feat = first_w.shape[1]
    feat_dim = features.shape[1]
    K = in_feat - embed_dim - feat_dim
    sh_degree = int(round(K ** 0.5)) - 1
    app = AppearanceOptModule(n_embed, feat_dim, embed_dim, sh_degree).to(dev)
    app.load_state_dict(st); app.eval()
    print(f"app_opt: {means.shape[0]:,} gaussians · feat={feat_dim} embed={embed_dim} sh={sh_degree}")
    return dict(means=means, quats=quats, scales=scales, opac=opac,
                features=features, base=base_colors, app=app, sh=sh_degree)


def render_one(M, viewmat, K, w, h, ut_kw, antialiased):
    """viewmat = world→cam (normalized). Tính colors qua app_module (embed=None=zero)."""
    dev = M["means"].device
    camtoworld = torch.linalg.inv(torch.from_numpy(viewmat).float().to(dev))[None]  # [1,4,4]
    dirs = M["means"][None] - camtoworld[:, None, :3, 3]                            # [1,N,3]
    with torch.no_grad():
        colors = M["app"](features=M["features"], embed_ids=None, dirs=dirs, sh_degree=M["sh"])  # [1,N,3]
        colors = torch.sigmoid(colors + M["base"])                                  # [1,N,3]
        out, _, _ = rasterization(
            M["means"], M["quats"], M["scales"], M["opac"], colors[0],
            torch.from_numpy(viewmat).float()[None].to(dev),
            torch.from_numpy(K).float()[None].to(dev), w, h,
            near_plane=0.01, far_plane=1e10,
            rasterize_mode="antialiased" if antialiased else "classic",
            render_mode="RGB", **ut_kw)
    return out[0].clamp(0, 1).cpu().numpy()


def ut_kwargs(a, dev):
    kw = {}
    if a.with_ut:
        kw = dict(with_ut=True, with_eval3d=True, packed=False)
        if a.radial_k1 is not None:
            kw["radial_coeffs"] = torch.tensor([[a.radial_k1, a.radial_k2, 0, 0, 0, 0]], device=dev).float()
    return kw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ws", required=True)
    ap.add_argument("--csv"); ap.add_argument("--out")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--with_ut", action="store_true")
    ap.add_argument("--radial_k1", type=float, default=None)
    ap.add_argument("--radial_k2", type=float, default=0.0)
    ap.add_argument("--antialiased", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    M = load_app_ckpt(a.ckpt, dev)
    ws = Path(a.ws)
    cams = read_cameras_bin(ws / "sparse/0/cameras.bin")
    imgs = read_images_bin(ws / "sparse/0/images.bin")
    _, xyz, _, _, _ = read_points3D_bin(ws / "sparse/0/points3D.bin")
    transform = compute_parser_transform(imgs, xyz)
    kw = ut_kwargs(a, dev)

    if a.validate:
        gt = {Path(im.name).stem: p for im in imgs.values() for p in [ws / "images" / im.name]}
        items = sorted(imgs.values(), key=lambda i: i.name)
        sel = items[:: max(1, len(items) // 3)][:3]
        ps = []
        for im in sel:
            cam = cams[im.camera_id]
            if cam.model == "SIMPLE_RADIAL": f, cx, cy, _ = cam.params; fx = fy = f
            elif cam.model == "SIMPLE_PINHOLE": f, cx, cy = cam.params; fx = fy = f
            else: fx, fy, cx, cy = cam.params[:4]
            vm = colmap_w2c_to_normalized_viewmat(im.R, im.tvec, transform)
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
            rgb = render_one(M, vm, K, cam.width, cam.height, kw, a.antialiased)
            g = cv2.cvtColor(cv2.imread(str(ws / "images" / im.name)), cv2.COLOR_BGR2RGB) / 255.0
            mse = float(((rgb - g) ** 2).mean()); psnr = 10 * np.log10(1 / max(mse, 1e-10))
            ps.append(psnr); print(f"  {im.name}: PSNR={psnr:.2f}dB")
        m = np.mean(ps)
        print(f"★ VALIDATE train-view PSNR TB = {m:.2f}dB " +
              ("✅ bridge ĐÚNG (>26)" if m > 26 else "❌ bridge SAI (<26) — KHÔNG tin kết quả test!"))
        return

    rows = list(csvmod.DictReader(open(a.csv)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows):
        q = np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")])
        t = np.array([float(r[k]) for k in ("tx", "ty", "tz")])
        fx, fy, cx, cy = (float(r[k]) for k in ("fx", "fy", "cx", "cy"))
        w, h = int(r["width"]), int(r["height"])
        vm = colmap_w2c_to_normalized_viewmat(qvec2rotmat(q), t, transform)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        rgb = render_one(M, vm, K, w, h, kw, a.antialiased)
        img = (rgb * 255).round().astype(np.uint8)[..., ::-1]
        cv2.imwrite(str(out / (Path(r["image_name"]).stem + ".png")), img)
        if (i + 1) % 20 == 0: print(f"  render {i+1}/{len(rows)}")
    print(f"→ {out} ({len(rows)} ảnh)")


if __name__ == "__main__":
    main()
