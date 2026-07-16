#!/usr/bin/env python3
"""B3 — Sinh transient mask cho ảnh train: xe/người di chuyển giữa các frame (cách ~1s)
gây supervision mâu thuẫn → ghost/blur mặt đường. Mask chúng khỏi loss.

Model: torchvision deeplabv3_resnet50 (pretrained VOC/COCO — tự tải lần đầu ~160MB).
Class transient (VOC): person=15, car=7, bus=6, motorbike=14, bicycle=2, boat=4.
Ra: <workspace>/transient_masks/<stem>.png (255 = transient). Dilate 15px (bóng đổ/viền).

Dùng: python make_transient_masks.py --workspace workspace_raw/HCM0204 [--vis 3]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--half_res", action="store_true", default=True,
                    help="segment ở 1/2 độ phân giải (đủ cho xe/người, nhanh 4×)")
    ap.add_argument("--dilate", type=int, default=15)
    ap.add_argument("--vis", type=int, default=0, help="lưu N ảnh overlay để soi mắt")
    a = ap.parse_args()

    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    w = DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    model = deeplabv3_resnet50(weights=w).cuda().eval()
    TRANSIENT = {2, 4, 6, 7, 14, 15}   # bicycle, boat, bus, car, motorbike, person
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]

    ws = Path(a.workspace)
    out = ws / "transient_masks"; out.mkdir(exist_ok=True)
    vis_dir = ws / "transient_vis"; a.vis and vis_dir.mkdir(exist_ok=True)
    files = sorted((ws / "images").iterdir())
    kern = np.ones((a.dilate, a.dilate), np.uint8)
    stats = []
    with torch.no_grad():
        for i, f in enumerate(files):
            bgr = cv2.imread(str(f))
            h, w0 = bgr.shape[:2]
            img = cv2.resize(bgr, (w0 // 2, h // 2)) if a.half_res else bgr
            x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).cuda()
            x = (x.float().permute(2, 0, 1) / 255 - mean) / std
            pred = model(x[None])["out"][0].argmax(0).cpu().numpy()
            m = np.isin(pred, list(TRANSIENT)).astype(np.uint8) * 255
            m = cv2.dilate(m, kern)
            m = cv2.resize(m, (w0, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / (f.stem + ".png")), m)
            frac = (m > 0).mean(); stats.append(frac)
            if a.vis and i < a.vis:
                ov = bgr.copy(); ov[m > 0] = (0, 0, 255)
                cv2.imwrite(str(vis_dir / f"{f.stem}_vis.jpg"),
                            cv2.addWeighted(bgr, 0.6, ov, 0.4, 0), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if (i + 1) % 60 == 0:
                print(f"  {i+1}/{len(files)} (transient TB {np.mean(stats)*100:.1f}% diện tích)")
    print(f"✓ {len(files)} masks → {out}")
    print(f"  diện tích transient: mean {np.mean(stats)*100:.1f}%  p90 {np.percentile(stats,90)*100:.1f}%  max {np.max(stats)*100:.1f}%")
    print(f"  (nếu mean <0.5%: scene này ít xe → B3 sẽ không ăn ở đây, thử scene khác)")


if __name__ == "__main__":
    main()
