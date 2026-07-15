#!/usr/bin/env python3
"""Chấm điểm cục bộ trên public set theo công thức BTC:
    Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*clamp(PSNR/PSNR_max, 0, 1)

Vì chưa biết BTC dùng backbone LPIPS nào và PSNR_max bao nhiêu (docs/04 §5),
đo cả LPIPS-alex lẫn LPIPS-vgg và báo score với PSNR_max ∈ {35, 40}.
Metric tính trên uint8→float[0,1], khớp ảnh theo stem (đuôi file có thể khác).

Dùng: python score_local.py --pred_dir renders/HCM0204 \
        --gt_dir .../HCM0204/test/images [--out results/HCM0204_score.csv]
"""
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import csv as csvmod
from pathlib import Path

import cv2
import numpy as np
import torch


def load_pairs(pred_dir, gt_dir):
    preds = {p.stem: p for p in Path(pred_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    gts = {p.stem: p for p in Path(gt_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    common = sorted(set(preds) & set(gts))
    missing = sorted(set(gts) - set(preds))
    if missing:
        print(f"CẢNH BÁO: thiếu {len(missing)} ảnh render: {missing[:5]}...")
    assert common, "Không có cặp ảnh nào khớp tên"
    return [(preds[s], gts[s]) for s in common]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", default=None, help="CSV per-image")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import lpips  # pip install lpips
    from torchmetrics.functional import peak_signal_noise_ratio as tm_psnr
    from torchmetrics.functional import structural_similarity_index_measure as tm_ssim

    dev = args.device
    lp_alex = lpips.LPIPS(net="alex").to(dev).eval()
    lp_vgg = lpips.LPIPS(net="vgg").to(dev).eval()

    rows = []
    for pred_p, gt_p in load_pairs(args.pred_dir, args.gt_dir):
        pred = cv2.cvtColor(cv2.imread(str(pred_p)), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(str(gt_p)), cv2.COLOR_BGR2RGB)
        assert pred.shape == gt.shape, f"Kích thước lệch {pred_p.name}: {pred.shape} vs {gt.shape}"
        a = torch.from_numpy(pred).float().permute(2, 0, 1)[None].to(dev) / 255
        b = torch.from_numpy(gt).float().permute(2, 0, 1)[None].to(dev) / 255
        with torch.no_grad():
            psnr = tm_psnr(a, b, data_range=1.0).item()
            ssim = tm_ssim(a, b, data_range=1.0).item()
            l_alex = lp_alex(a * 2 - 1, b * 2 - 1).item()
            l_vgg = lp_vgg(a * 2 - 1, b * 2 - 1).item()
        rows.append(dict(name=pred_p.stem, psnr=psnr, ssim=ssim,
                         lpips_alex=l_alex, lpips_vgg=l_vgg))

    def score(lp_key, pmax):
        return float(np.mean([0.4 * (1 - r[lp_key]) + 0.3 * r["ssim"]
                              + 0.3 * min(r["psnr"] / pmax, 1.0) for r in rows]))

    m = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "lpips_alex", "lpips_vgg")}
    print(f"n={len(rows)}  PSNR={m['psnr']:.3f}  SSIM={m['ssim']:.4f}  "
          f"LPIPS(alex)={m['lpips_alex']:.4f}  LPIPS(vgg)={m['lpips_vgg']:.4f}")
    # ★ THANG THẬT CỦA BTC — giải mã 12/7 từ metrics SUB1.5 (vgg + PSNR_max=50, sai số 0)
    print(f"  ★ Score_BTC[vgg, PSNR_max=50] = {score('lpips_vgg', 50):.5f}")
    for lp in ("lpips_alex", "lpips_vgg"):
        for pmax in (35, 40):
            print(f"  Score[{lp}, PSNR_max={pmax}] = {score(lp, pmax):.5f}")

    worst = sorted(rows, key=lambda r: 0.4 * (1 - r["lpips_vgg"]) + 0.3 * r["ssim"]
                   + 0.3 * min(r["psnr"] / 50, 1.0))[:5]
    print("5 ảnh tệ nhất (mở ra xem bằng mắt!):", [r["name"] for r in worst])

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wr = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader(); wr.writerows(rows)
        print(f"→ {args.out}")


if __name__ == "__main__":
    main()
