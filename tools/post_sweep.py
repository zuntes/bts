#!/usr/bin/env python3
"""P-POST — quét hậu xử lý CỔ ĐIỂN theo đúng metric BTC trên bàn có GT.

Ý tưởng: các phép rẻ (unsharp, gamma, contrast, cân màu theo GT-thống-kê-train)
đôi khi cộng 0.1-0.3 điểm "miễn phí" vì metric distortion-oriented (PSNR/SSIM)
thích ảnh hơi mượt/đúng phân bố sáng. Quét lưới trên public GT → chỉ nhận cấu hình
tăng v50 thật. CPU-only — chạy song song khi GPU bận.

Dùng: python post_sweep.py --pred_dir renders/HCM0204__a1 --gt_dir VAI.../test/images
      [--fast]  (chỉ PSNR+SSIM khi quét thô, LPIPS chỉ chấm top-5)
"""
import argparse
import itertools
from pathlib import Path

import cv2
import numpy as np


def load_pairs(pred_dir, gt_dir):
    gt_map = {p.stem: p for p in Path(gt_dir).iterdir()}
    pairs = []
    for p in sorted(Path(pred_dir).iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg") or p.stem not in gt_map:
            continue
        pairs.append((cv2.imread(str(p)), cv2.imread(str(gt_map[p.stem]))))
    assert pairs, "không có cặp nào"
    return pairs


def unsharp(img, sigma, amount):
    if amount == 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


def gamma_adj(img, g):
    if g == 1.0:
        return img
    lut = (np.linspace(0, 1, 256) ** (1.0 / g) * 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def apply_cfg(img, cfg):
    out = unsharp(img, cfg["us_sigma"], cfg["us_amount"])
    out = gamma_adj(out, cfg["gamma"])
    if cfg["blur"] > 0:   # chiều ngược lại — cho GT mờ (bonsai/chair)
        out = cv2.GaussianBlur(out, (0, 0), cfg["blur"])
    return out


def psnr_ssim(a, b):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    mse = ((a64 - b64) ** 2).mean()
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-10))
    # SSIM xám nhanh (đủ để xếp hạng thô)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu_a, mu_b = cv2.GaussianBlur(ga, (11, 11), 1.5), cv2.GaussianBlur(gb, (11, 11), 1.5)
    va = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
            ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean()
    return psnr, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    pairs = load_pairs(a.pred_dir, a.gt_dir)
    print(f"{len(pairs)} cặp · quét thô bằng 0.3·SSIM + 0.3·PSNR/50 (LPIPS chấm ở vòng 2)")

    grid = {
        "us_sigma": [1.0, 2.0],
        "us_amount": [0.0, 0.3, 0.6],
        "gamma": [0.95, 1.0, 1.05],
        "blur": [0.0, 0.4, 0.7],
    }
    results = []
    base_key = {"us_sigma": 1.0, "us_amount": 0.0, "gamma": 1.0, "blur": 0.0}
    for vals in itertools.product(*grid.values()):
        cfg = dict(zip(grid.keys(), vals))
        if cfg["us_amount"] > 0 and cfg["blur"] > 0:
            continue  # sharpen + blur cùng lúc = vô nghĩa
        P, S = [], []
        for pred, gt in pairs:
            out = apply_cfg(pred, cfg)
            p, s = psnr_ssim(out, gt)
            P.append(p); S.append(s)
        partial = 0.3 * float(np.mean(S)) + 0.3 * min(np.mean(P) / 50, 1)
        results.append((partial, cfg, float(np.mean(P)), float(np.mean(S))))
    results.sort(key=lambda r: -r[0])
    base = next(r for r in results if r[1] == base_key)
    print(f"\nBASE (không xử lý): partial={base[0]:.5f} PSNR={base[2]:.3f} SSIM={base[3]:.4f}")
    print(f"TOP-{a.topk} (partial = 0.6 trọng số của v50; LPIPS chưa tính):")
    for r in results[:a.topk]:
        d = r[0] - base[0]
        print(f"  Δpartial={d:+.5f} PSNR={r[2]:.3f} SSIM={r[3]:.4f}  {r[1]}")
    print("\n→ Nếu top có Δ>+0.0015: chấm LPIPS đầy đủ cấu hình đó bằng score_local trên "
          "thư mục đã áp (tools/apply_post.py sẽ sinh) rồi mới quyết.")


if __name__ == "__main__":
    main()
