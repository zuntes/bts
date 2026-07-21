#!/usr/bin/env python3
"""REF-ENHANCER — enhancer 6 kênh: [render, ẢNH TRAIN THẬT warp về pose đích].

BỐI CẢNH (vì sao đòn này khác IBR đã LOẠI):
  IBR (G7/G8) warp ảnh train rồi BLEND THẲNG pixel → lệch sub-pixel → SSIM/LPIPS
  phạt lệch-vị nặng hơn phạt mờ → −0.02..−0.03. Nhưng phép đo đó cũng chứng minh
  THÔNG TIN CÓ THẬT: độ phủ warp 79-91%. Ở đây ta KHÔNG blend — đưa ảnh warp vào
  làm KÊNH ĐIỀU KIỆN để mạng TỰ HỌC chỗ nào lấy chi tiết, chỗ nào bỏ (reference-based
  restoration: RASR'25, ERVSR). Enhancer hiện tại bão hoà vì chỉ nhìn 1 ảnh render —
  nút thắt là THÔNG TIN (DOC1 §B6), và đây là nguồn thông tin mới duy nhất chưa khai thác.

AN TOÀN (2 lớp, để không bao giờ tệ hơn enhancer 3 kênh đang dùng):
  1. Trọng số 3 kênh reference của conv đầu khởi tạo ZERO → lúc step 0 mạng
     TOÁN HỌC TƯƠNG ĐƯƠNG bản 3 kênh; chỉ đi lên nếu reference thực sự có ích.
  2. Vùng warp hỏng (ngoài khung/sau lưng camera) điền bằng CHÍNH ẢNH RENDER
     → reference thoái hoá mượt về "không có thông tin mới", không tạo cạnh giả.

CHỐNG RÒ RỈ: reference của train view i là train view GẦN NHẤT j≠i (không bao giờ
là chính nó — nếu không mạng chỉ học copy = GT, vô dụng lúc test).

Dùng:
  python ref_enhance.py train --workspace workspace_r2v/chair --ckpt ck.pt \
      --out results/chair__refenh/net.pt --steps 8000
  python ref_enhance.py apply --net results/chair__refenh/net.pt \
      --workspace workspace_r2v/chair --ckpt ck.pt --csv val_poses.csv \
      --in_dir renders_r2v/chair__mega6 --out_dir renders_r2v/chair__refenh
"""
import argparse
import csv as csvmod
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))


# --------------------------------------------------------------------------- net
class UNetVGGRef(nn.Module):
    """UNetVGG (B1, đã thắng unet 3/3 scene) + 3 kênh reference.

    conv1 mở rộng 3→6 kênh; 3 kênh mới zero-init ⇒ tương đương bản 3 kênh ở step 0.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        old = feats[0]                                    # Conv2d(3, 64, 3, padding=1)
        new = nn.Conv2d(6, 64, 3, padding=1)
        with torch.no_grad():
            new.weight[:, :3] = old.weight                # nhánh render: y hệt VGG gốc
            new.weight[:, 3:] = 0.0                       # nhánh reference: TẮT lúc đầu
            new.bias.copy_(old.bias)
        self.e1 = nn.Sequential(new, *list(feats[1:4]))   # 64ch  full-res
        self.e2 = feats[4:9]                              # 128ch 1/2
        self.e3 = feats[9:16]                             # 256ch 1/4

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GELU())
        self.d2 = block(256 + 128, 128)
        self.d1 = block(128 + 64, 64)
        self.head = nn.Conv2d(64, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, x, ref):
        z = torch.cat([(x - self.mean) / self.std, (ref - self.mean) / self.std], 1)
        e1 = self.e1(z); e2 = self.e2(e1); e3 = self.e3(e2)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([up(e3), e2], 1))
        d1 = self.d1(torch.cat([up(d2), e1], 1))
        return (x + self.head(d1)).clamp(0, 1)            # vẫn là residual trên RENDER


# ------------------------------------------------------------------- hình học/warp
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_scene(ws, ckpt_path, dev="cuda"):
    """Trả về splats + danh sách train view (viewmat đã chuẩn hoá, K, ảnh GT)."""
    from colmap_io import read_cameras_bin, read_images_bin, read_points3D_bin
    from normalize_compat import (colmap_w2c_to_normalized_viewmat,
                                  compute_parser_transform)
    ws = Path(ws)
    cams = read_cameras_bin(ws / "sparse/0/cameras.bin")
    imgs = read_images_bin(ws / "sparse/0/images.bin")
    _, xyz, _, _, _ = read_points3D_bin(ws / "sparse/0/points3D.bin")
    transform = compute_parser_transform(imgs, xyz)

    ck = torch.load(ckpt_path, map_location=dev, weights_only=True)["splats"]
    splats = dict(
        means=ck["means"].to(dev),
        quats=F.normalize(ck["quats"].to(dev), dim=-1),
        scales=torch.exp(ck["scales"].to(dev)),
        opac=torch.sigmoid(ck["opacities"].to(dev)),
        colors=torch.cat([ck["sh0"], ck["shN"]], 1).to(dev))
    splats["shd"] = int(np.sqrt(splats["colors"].shape[1]) - 1)

    views = []
    for im in sorted(imgs.values(), key=lambda i: i.name):
        cam = cams[im.camera_id]
        if cam.model == "SIMPLE_RADIAL":
            f, cx, cy, _ = cam.params; fx = fy = f
        elif cam.model == "SIMPLE_PINHOLE":
            f, cx, cy = cam.params; fx = fy = f
        else:
            fx, fy, cx, cy = cam.params[:4]
        vm = colmap_w2c_to_normalized_viewmat(im.R, im.tvec, transform)
        views.append(dict(
            stem=Path(im.name).stem, name=im.name,
            vm=torch.from_numpy(vm).float().to(dev),
            K=torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]).float().to(dev),
            w=cam.width, h=cam.height,
            C=np.linalg.inv(vm)[:3, 3]))            # tâm camera (hệ đã chuẩn hoá)
    return splats, views, transform, cams


def render_rgbd(splats, vm, K, w, h, with_ut=False, radial_k1=None, antialiased=True):
    """Render RGB + expected-depth tại 1 pose bất kỳ."""
    from gsplat import rasterization
    kw = {}
    if with_ut:
        kw = dict(with_ut=True, with_eval3d=True, packed=False)
        if radial_k1 is not None:
            kw["radial_coeffs"] = torch.tensor(
                [[radial_k1, 0, 0, 0, 0, 0]], device=vm.device).float()
    with torch.no_grad():
        out, _, _ = rasterization(
            splats["means"], splats["quats"], splats["scales"], splats["opac"],
            splats["colors"], vm[None], K[None], w, h, sh_degree=splats["shd"],
            near_plane=0.01, far_plane=1e10,
            rasterize_mode="antialiased" if antialiased else "classic",
            render_mode="RGB+ED", **kw)
    return out[0, ..., :3].clamp(0, 1), out[0, ..., 3]      # (H,W,3), (H,W)


def warp_source_to_target(depth_t, K_t, vm_t, src_img, K_s, vm_s, fallback):
    """Warp ảnh nguồn (train GT) về khung nhìn đích bằng depth của đích.

    depth_t (H,W) · src_img (Hs,Ws,3) float[0,1] cuda · fallback (H,W,3) = ảnh render.
    Vùng không hợp lệ (depth<=0, sau lưng camera nguồn, ra ngoài khung) → fallback.
    """
    dev = depth_t.device
    H, W = depth_t.shape
    ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                            torch.arange(W, device=dev, dtype=torch.float32),
                            indexing="ij")
    ones = torch.ones_like(xs)
    pix = torch.stack([xs, ys, ones], -1).reshape(-1, 3)               # (N,3)
    # unproject: điểm trong hệ camera đích
    Xc = (torch.linalg.inv(K_t) @ pix.T).T * depth_t.reshape(-1, 1)     # (N,3)
    # sang thế giới rồi vào camera nguồn:  X_s = R_s (R_t^T (X_t − t_t)) + t_s
    R_t, t_t = vm_t[:3, :3], vm_t[:3, 3]
    R_s, t_s = vm_s[:3, :3], vm_s[:3, 3]
    Xw = (R_t.T @ (Xc - t_t).T).T
    Xs = (R_s @ Xw.T).T + t_s
    z = Xs[:, 2]
    proj = (K_s @ Xs.T).T
    u = proj[:, 0] / z.clamp(min=1e-6)
    v = proj[:, 1] / z.clamp(min=1e-6)
    Hs, Ws = src_img.shape[:2]
    valid = (z > 1e-4) & (u >= 0) & (u <= Ws - 1) & (v >= 0) & (v <= Hs - 1) \
        & (depth_t.reshape(-1) > 1e-6)
    gx = (u / max(Ws - 1, 1)) * 2 - 1
    gy = (v / max(Hs - 1, 1)) * 2 - 1
    grid = torch.stack([gx, gy], -1).reshape(1, H, W, 2).clamp(-2, 2)
    samp = F.grid_sample(src_img.permute(2, 0, 1)[None], grid,
                         mode="bilinear", padding_mode="zeros", align_corners=True)
    samp = samp[0].permute(1, 2, 0)                                     # (H,W,3)
    m = valid.reshape(H, W, 1).float()
    return samp * m + fallback * (1 - m), float(m.mean())


def nearest_views(views, target_C, exclude_stem=None, k=1):
    d = [(np.linalg.norm(v["C"] - target_C), i) for i, v in enumerate(views)
         if v["stem"] != exclude_stem]
    d.sort()
    return [views[i] for _, i in d[:k]]


def build_ref_for_view(splats, views, tgt_vm, tgt_K, w, h, render_rgb,
                       exclude_stem, with_ut, radial_k1):
    """Sinh kênh reference cho 1 pose: warp train-view gần nhất (≠ chính nó)."""
    _, depth = render_rgbd(splats, tgt_vm, tgt_K, w, h, with_ut, radial_k1)
    tgt_C = np.linalg.inv(tgt_vm.cpu().numpy())[:3, 3]
    src = nearest_views(views, tgt_C, exclude_stem, k=1)[0]
    ref, cov = warp_source_to_target(depth, tgt_K, tgt_vm, src["img"],
                                     src["K"], src["vm"], render_rgb)
    return ref, cov


# ------------------------------------------------------------------------- train
def cmd_train(a):
    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
    dev = "cuda"
    ws = Path(a.workspace)
    splats, views, _, _ = load_scene(ws, a.ckpt, dev)
    gt_map = {p.stem: p for p in (ws / "images").iterdir()}

    # nạp ảnh GT train lên GPU (cần cho warp) — dùng chính nó làm nguồn reference
    print(f"[ref-enh] nạp {len(views)} train view + render + warp reference ...")
    data = []
    for i, v in enumerate(views):
        img = cv2.imread(str(gt_map[v["stem"]]))
        v["img"] = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    ).float().to(dev) / 255
    covs = []
    for i, v in enumerate(views):
        rgb, _ = render_rgbd(splats, v["vm"], v["K"], v["w"], v["h"],
                             a.with_ut, a.radial_k1)
        ref, cov = build_ref_for_view(splats, views, v["vm"], v["K"], v["w"], v["h"],
                                      rgb, v["stem"], a.with_ut, a.radial_k1)
        covs.append(cov)
        # lưu uint8 (giảm 4× RAM: 185 ảnh × 3 luồng float32 ≈ 6GB → 1.5GB)
        q = lambda t: (t.clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        data.append((q(rgb), q(ref), q(v["img"])))
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(views)} · phủ warp TB {np.mean(covs):.1%}")
    print(f"[ref-enh] phủ warp trung bình = {np.mean(covs):.1%} "
          f"(IBR đo 79-91% — cùng cỡ là đúng)")

    n_val = max(4, len(data) // 10)
    tr, va = data[:-n_val], data[-n_val:]
    print(f"[ref-enh] {len(tr)} train / {len(va)} val pairs")

    net = UNetVGGRef().to(dev)
    enc = [p for n, p in net.named_parameters() if n.startswith("e")]
    dec = [p for n, p in net.named_parameters() if not n.startswith("e")]
    opt = torch.optim.AdamW([{"params": enc, "lr": 2e-5}, {"params": dec, "lr": 2e-4}],
                            weight_decay=1e-5)
    lp = lpips.LPIPS(net="vgg").to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(0)

    def batch(pool, ps, bs):
        xs, rs, ys = [], [], []
        for _ in range(bs):
            r, ref, g = pool[rng.integers(len(pool))]
            h, w = r.shape[:2]
            y0 = int(rng.integers(0, h - ps + 1)); x0 = int(rng.integers(0, w - ps + 1))
            f = lambda t: t[y0:y0 + ps, x0:x0 + ps].permute(2, 0, 1).float() / 255
            xs.append(f(r)); rs.append(f(ref)); ys.append(f(g))
        return (torch.stack(xs).to(dev), torch.stack(rs).to(dev), torch.stack(ys).to(dev))

    def score_val(apply_net):
        P, S, L = [], [], []
        with torch.no_grad():
            for r, ref, g in va:
                f = lambda t: t.permute(2, 0, 1)[None].to(dev).float() / 255
                x = f(r); rf = f(ref); y = f(g)
                if apply_net:
                    h, w = x.shape[-2:]
                    ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
                    x = net(F.pad(x, (0, pw, 0, ph), mode="replicate"),
                            F.pad(rf, (0, pw, 0, ph), mode="replicate"))[..., :h, :w]
                mse = ((x - y) ** 2).mean().item()
                P.append(10 * np.log10(1 / max(mse, 1e-10)))
                S.append(tm_ssim(x, y, data_range=1.0).item())
                L.append(lp(x * 2 - 1, y * 2 - 1).mean().item())
                del x, y, rf; torch.cuda.empty_cache()
        p, s, l = float(np.mean(P)), float(np.mean(S)), float(np.mean(L))
        return 0.4 * (1 - l) + 0.3 * s + 0.3 * min(p / 50, 1), p, s, l

    base = score_val(False)
    print(f"val BASE:    v50={base[0]:.5f} PSNR={base[1]:.3f} SSIM={base[2]:.4f} LPIPS={base[3]:.4f}")
    best = (-1, None)
    for step in range(1, a.steps + 1):
        x, rf, y = batch(tr, a.patch, a.batch)
        out = net(x, rf)
        loss = (0.4 * lp(out * 2 - 1, y * 2 - 1).mean()
                + 0.3 * (1 - tm_ssim(out, y, data_range=1.0))
                + 0.3 * (out - y).abs().mean())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps:
            sc = score_val(True)
            mk = ""
            if sc[0] > best[0]:
                best = (sc[0], {k: v.detach().cpu() for k, v in net.state_dict().items()})
                mk = " ← best"
            print(f"step {step}: val v50={sc[0]:.5f} PSNR={sc[1]:.3f} "
                  f"SSIM={sc[2]:.4f} LPIPS={sc[3]:.4f}{mk}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best[1], "base_v50": base[0], "best_v50": best[0],
                "warp_cov": float(np.mean(covs))}, a.out)
    print(f"VAL-GAIN {best[0] - base[0]:+.5f} → {a.out}")


# ------------------------------------------------------------------------- apply
def cmd_apply(a):
    dev = "cuda"
    ws = Path(a.workspace)
    splats, views, transform, _ = load_scene(ws, a.ckpt, dev)
    gt_map = {p.stem: p for p in (ws / "images").iterdir()}
    for v in views:
        img = cv2.imread(str(gt_map[v["stem"]]))
        v["img"] = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                    ).float().to(dev) / 255
    from normalize_compat import colmap_w2c_to_normalized_viewmat
    ck = torch.load(a.net, map_location=dev, weights_only=True)
    net = UNetVGGRef().to(dev).eval(); net.load_state_dict(ck["state"])

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csvmod.DictReader(open(a.csv)))
    in_dir = Path(a.in_dir)
    covs = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            stem = Path(r["image_name"]).stem
            src_png = in_dir / f"{stem}.png"
            if not src_png.exists():
                cand = list(in_dir.glob(f"{stem}.*"))
                if not cand:
                    print(f"  ⚠ thiếu render {stem} — bỏ"); continue
                src_png = cand[0]
            bgr = cv2.imread(str(src_png))
            rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                                   ).float().to(dev) / 255
            w, h = int(r["width"]), int(r["height"])
            q = np.array([float(r[k]) for k in ("qw", "qx", "qy", "qz")])
            t = np.array([float(r[k]) for k in ("tx", "ty", "tz")])
            vm = torch.from_numpy(
                colmap_w2c_to_normalized_viewmat(qvec2rotmat(q), t, transform)
            ).float().to(dev)
            fx, fy = float(r["fx"]), float(r["fy"])
            cx, cy = float(r["cx"]), float(r["cy"])
            K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]).float().to(dev)
            ref, cov = build_ref_for_view(splats, views, vm, K, w, h, rgb,
                                          None, a.with_ut, a.radial_k1)
            covs.append(cov)
            x = rgb.permute(2, 0, 1)[None]
            rf = ref.permute(2, 0, 1)[None]
            ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
            y = net(F.pad(x, (0, pw, 0, ph), mode="replicate"),
                    F.pad(rf, (0, pw, 0, ph), mode="replicate"))[..., :h, :w]
            img = (y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                   ).round().astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{stem}.png"), img[..., ::-1])
    print(f"applied {len(covs)} ảnh → {out_dir} (phủ warp TB {np.mean(covs):.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--workspace", required=True)
    t.add_argument("--ckpt", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=8000)
    t.add_argument("--patch", type=int, default=320)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--with_ut", action="store_true")
    t.add_argument("--radial_k1", type=float, default=None)
    p = sub.add_parser("apply")
    p.add_argument("--net", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--in_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--with_ut", action="store_true")
    p.add_argument("--radial_k1", type=float, default=None)
    a = ap.parse_args()
    (cmd_train if a.cmd == "train" else cmd_apply)(a)
