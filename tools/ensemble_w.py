#!/usr/bin/env python3
"""WEIGHTED ENSEMBLE — fit trọng số member trên bàn có GT (0 GPU, CPU-only).

Mean đều chia 1/N cho mọi member; nhưng member yếu (vd erank chair −0.001 solo)
có thể xứng đáng trọng số thấp hơn. Hill-climb trọng số trên simplex, mục tiêu
proxy = 0.3·SSIM + 0.3·PSNR/50 (nhanh, CPU; LPIPS chấm lại sau bằng score_local
trên thư mục xuất). Trọng số fit trên HOLDOUT → dùng lại cho prod cùng cấu trúc member.

Dùng:
  python ensemble_w.py --dirs A B C D --gt_dir GT --out_dir OUT [--iters 40]
In: trọng số tốt nhất + Δproxy vs mean đều; xuất OUT với trọng số đó.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def load_stack(dirs, gt_dir):
    gt_map = {p.stem: p for p in Path(gt_dir).iterdir()}
    stems = None
    for d in dirs:
        s = {p.stem for p in Path(d).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
        stems = s if stems is None else stems & s
    stems = sorted(stems & set(gt_map))
    assert stems, "không có stem chung"
    stacks, gts = [], []
    for st in stems:
        ims = [cv2.imread(str(next(Path(d).glob(st + ".*")))).astype(np.float32) for d in dirs]
        stacks.append(np.stack(ims))            # [M,H,W,3]
        gts.append(cv2.imread(str(gt_map[st])).astype(np.float32))
    return stems, stacks, gts


def proxy(img, gt):
    mse = ((img - gt) ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
    ga = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    mu_a, mu_b = cv2.GaussianBlur(ga, (11, 11), 1.5), cv2.GaussianBlur(gb, (11, 11), 1.5)
    va = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a ** 2
    vb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b ** 2
    cov = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim = (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
            ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean()
    return 0.3 * ssim + 0.3 * min(psnr / 50, 1)


def eval_w(w, stacks, gts):
    return float(np.mean([proxy(np.tensordot(w, s, axes=1), g) for s, g in zip(stacks, gts)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--iters", type=int, default=40)
    a = ap.parse_args()
    stems, stacks, gts = load_stack(a.dirs, a.gt_dir)
    M = len(a.dirs)
    w = np.ones(M) / M
    base = eval_w(w, stacks, gts)
    print(f"{len(stems)} ảnh · {M} member · proxy(mean đều) = {base:.5f}")
    rng = np.random.default_rng(0)
    best_w, best = w.copy(), base
    step = 0.15
    for it in range(a.iters):
        i = rng.integers(M)
        for sgn in (+1, -1):
            cand = best_w.copy()
            cand[i] = max(0.0, cand[i] + sgn * step)
            if cand.sum() == 0:
                continue
            cand /= cand.sum()
            v = eval_w(cand, stacks, gts)
            if v > best:
                best, best_w = v, cand
        if it % 10 == 9:
            step *= 0.6
    print("trọng số:", {Path(d).name: round(float(x), 3) for d, x in zip(a.dirs, best_w)})
    print(f"proxy: {base:.5f} → {best:.5f} (Δ={best - base:+.5f})")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for st, s in zip(stems, stacks):
        cv2.imwrite(str(out / (st + ".png")),
                    np.clip(np.tensordot(best_w, s, axes=1), 0, 255).astype(np.uint8))
    print(f"✓ xuất {len(stems)} ảnh → {out} (chấm LPIPS đầy đủ bằng score_local)")


if __name__ == "__main__":
    main()
