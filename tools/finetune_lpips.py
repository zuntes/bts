#!/usr/bin/env python3
"""Giai đoạn 2 — fine-tune checkpoint bằng loss hỗn hợp có LPIPS (docs/06 L2).

Ý tưởng: score cuộc thi nặng LPIPS (0.4). Sau khi train chuẩn (L1+SSIM) xong,
fine-tune ngắn KHÔNG densification với:
    L = (1-λs)·L1 + λs·(1-SSIM) + λp·LPIPS_vgg
λp chọn bằng composite score trên public set (KHÔNG chọn bằng mắt).
Cảnh báo từ paper Drop-In Perceptual Optimization: perceptual loss kéo tụt PSNR/SSIM
→ giữ λp nhỏ (0.05–0.2) và LR thấp.

Dùng:
  python finetune_lpips.py --workspace workspace/HCM0204 \
      --ckpt results/HCM0204/ckpts/ckpt_29999_rank0.pt \
      --out results/HCM0204/ckpts/ckpt_ft.pt --lambda_lpips 0.1 --steps 3000
Sau đó render bằng render_test_poses.py --ckpt .../ckpt_ft.pt như thường.
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import read_cameras_bin, read_images_bin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lambda_ssim", type=float, default=0.2)
    ap.add_argument("--lambda_lpips", type=float, default=0.1)
    ap.add_argument("--lr_scale", type=float, default=0.1,
                    help="nhân LR chuẩn của 3DGS với hệ số này (fine-tune nhẹ tay)")
    ap.add_argument("--antialiased", action="store_true")
    ap.add_argument("--with_ut", action="store_true",
                    help="3DGUT: fine-tune model train bằng --with-ut --with-eval3d "
                         "--raw-distortion trên workspace_raw (SIMPLE_RADIAL giữ méo)")
    ap.add_argument("--lpips_patch", type=int, default=512,
                    help="LPIPS tính trên crop ngẫu nhiên cỡ này (tiết kiệm VRAM)")
    # L13 — test-pose-aware sampling (docs/09 §3): sample ảnh train theo độ gần test poses
    ap.add_argument("--test_csv", default=None,
                    help="đường dẫn test_poses.csv → bật L13 (sampling có trọng số)")
    ap.add_argument("--pose_temp", type=float, default=0.5,
                    help="nhiệt độ trọng số L13: nhỏ = dồn mạnh vào train view gần test")
    args = ap.parse_args()

    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
    from gsplat import rasterization

    dev = "cuda"
    ws = Path(args.workspace)
    cams = read_cameras_bin(ws / "sparse/0/cameras.bin")
    imgs = read_images_bin(ws / "sparse/0/images.bin")
    items = sorted(imgs.values(), key=lambda im: im.name)
    print(f"fine-tune trên {len(items)} ảnh train")

    # Transform chuẩn hoá world space — model từ simple_trainer sống trong hệ này
    # (normalize_world_space=True mặc định); pose thô sẽ phá model khi fine-tune!
    from colmap_io import read_points3D_bin
    from normalize_compat import (colmap_w2c_to_normalized_viewmat,
                                  compute_parser_transform)
    _, xyz, _, _, _ = read_points3D_bin(ws / "sparse/0/points3D.bin")
    transform = compute_parser_transform(imgs, xyz)
    print(f"normalize transform: scale={np.linalg.norm(transform[0, :3]):.4f}")

    # Cache ảnh dạng uint8 trên RAM
    frames, viewmats, Ks, whs = [], [], [], []
    for im in items:
        arr = cv2.cvtColor(cv2.imread(str(ws / "images" / im.name)), cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(arr))
        vm = colmap_w2c_to_normalized_viewmat(im.R, im.tvec, transform)
        viewmats.append(torch.from_numpy(vm).float())
        cam = cams[im.camera_id]
        if cam.model == "SIMPLE_RADIAL":
            assert args.with_ut, (
                "workspace SIMPLE_RADIAL (ảnh méo gốc) cần --with_ut; "
                "hoặc dùng workspace đã undistort")
            f, cx, cy, _k1 = cam.params
            fx = fy = f
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params
            fx = fy = f
        else:
            fx, fy, cx, cy = cam.params[:4]
        Ks.append(torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]).float())
        whs.append((cam.width, cam.height))

    ut_kwargs = {}
    if args.with_ut:
        rad = {float(cams[im.camera_id].params[3]) for im in items
               if cams[im.camera_id].model == "SIMPLE_RADIAL"}
        assert len(rad) == 1, f"--with_ut cần đúng 1 camera SIMPLE_RADIAL, thấy: {rad}"
        k1 = rad.pop()
        ut_kwargs = dict(with_ut=True, with_eval3d=True, packed=False,
                         radial_coeffs=torch.tensor(
                             [[k1, 0, 0, 0, 0, 0]], device=dev).float())
        print(f"[with_ut] fine-tune với distortion native k1={k1:+.6f}")

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=True)
    splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(dev)) for k, v in ckpt["splats"].items()})
    # scene_scale từ TÂM CAMERA trong HỆ ĐÃ CHUẨN HOÁ (khớp LR mà simple_trainer dùng);
    # không lấy từ means của gaussians (sky far-sphere làm nổ LR ×15)
    centers = np.array([np.linalg.inv(vm.numpy())[:3, 3] for vm in viewmats])
    scene_scale = float(np.linalg.norm(
        centers - centers.mean(0), axis=1).max()) * 1.1
    # LR chuẩn simple_trainer × lr_scale
    lrs = dict(means=1.6e-4 * scene_scale, quats=1e-3, scales=5e-3,
               opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    opt = torch.optim.Adam(
        [{"params": [splats[k]], "lr": v * args.lr_scale} for k, v in lrs.items()
         if k in splats], eps=1e-15)

    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    # L13: trọng số sampling ∝ exp(−d/temp), d = khoảng cách chuẩn hoá tới test pose gần nhất
    probs = None
    if args.test_csv:
        import csv as csvmod

        def q2R(q):
            w, x, y, z = q
            return np.array([
                [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

        tC, tD = [], []
        for r in csvmod.DictReader(open(args.test_csv)):
            q = np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")])
            t = np.array([float(r[k]) for k in ("tx", "ty", "tz")])
            R = q2R(q)
            tC.append(-R.T @ t)
            tD.append(R[2])
        tC, tD = np.array(tC), np.array(tD)
        trC = np.array([im.center for im in items])
        trD = np.array([im.R[2] for im in items])
        dpos = np.linalg.norm(trC[:, None] - tC[None], axis=2)
        dpos /= np.median(np.sort(dpos, axis=0)[1])  # chuẩn hoá theo spacing
        dang = np.arccos(np.clip(trD @ tD.T, -1, 1))  # radian
        d = (dpos + 0.5 * dang).min(axis=1)           # tới test pose gần nhất
        w = np.exp(-(d - d.min()) / max(args.pose_temp, 1e-3))
        probs = w / w.sum()
        eff = 1.0 / (probs ** 2).sum() / len(items)
        print(f"L13 bật: effective sample size = {eff:.0%} tập train "
              f"(temp={args.pose_temp}; giảm temp để dồn mạnh hơn)")

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(args.steps):
        i = int(rng.choice(len(items), p=probs)) if probs is not None \
            else int(rng.integers(len(items)))
        gt = frames[i].to(dev).float() / 255  # [H,W,3]
        w, h = whs[i]
        colors = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        render, _, _ = rasterization(
            splats["means"],
            torch.nn.functional.normalize(splats["quats"], dim=-1),
            torch.exp(splats["scales"]),
            torch.sigmoid(splats["opacities"]),
            colors,
            viewmats[i][None].to(dev), Ks[i][None].to(dev), w, h,
            sh_degree=int(np.sqrt(colors.shape[1]) - 1),
            near_plane=0.01, far_plane=1e10,
            rasterize_mode="antialiased" if args.antialiased else "classic",
            **ut_kwargs,
        )
        pred = render[0].clamp(0, 1)
        l1 = (pred - gt).abs().mean()
        a = pred.permute(2, 0, 1)[None]
        b = gt.permute(2, 0, 1)[None]
        ssim = tm_ssim(a, b, data_range=1.0)
        # LPIPS trên crop ngẫu nhiên để vừa VRAM
        ps = min(args.lpips_patch, h, w)
        y0 = int(rng.integers(0, h - ps + 1)); x0 = int(rng.integers(0, w - ps + 1))
        lpv = lp(a[..., y0:y0 + ps, x0:x0 + ps] * 2 - 1,
                 b[..., y0:y0 + ps, x0:x0 + ps] * 2 - 1).mean()
        loss = ((1 - args.lambda_ssim) * l1 + args.lambda_ssim * (1 - ssim)
                + args.lambda_lpips * lpv)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 200 == 0:
            print(f"step {step + 1}/{args.steps}  L1={l1.item():.4f} "
                  f"SSIM={ssim.item():.4f} LPIPS={lpv.item():.4f} "
                  f"({(time.time() - t0) / (step + 1):.2f}s/it)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"splats": {k: v.detach().cpu() for k, v in splats.items()}}, args.out)
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
