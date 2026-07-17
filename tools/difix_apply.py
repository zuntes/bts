#!/usr/bin/env python3
"""S1 — áp Difix3D+ (NVIDIA CVPR'25) lên 1 thư mục render: xoá artifact 3DGS bằng
diffusion 1 bước. Chạy trong VENV RIÊNG .venv_difix (xem GATE_DIFIX.sh — không được
cài diffusers vào .venv chính, bài học pycolmap DOC3 §3.7).

Dùng:
  python difix_apply.py --in_dir renders/X --out_dir renders/X_difix [--ref_dir <train imgs>]
--ref_dir: dùng model 'nvidia/difix_ref' với ảnh tham chiếu (ghép theo tên gần nhất
           alphabet — đủ tốt vì tên file tăng theo quỹ đạo).
"""
import argparse
from pathlib import Path

import torch
from diffusers.utils import load_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ref_dir", default=None, help="thư mục ảnh train làm reference (bật model difix_ref)")
    ap.add_argument("--prompt", default="remove degradation")
    a = ap.parse_args()
    ind, outd = Path(a.in_dir), Path(a.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in ind.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    assert files, f"không có ảnh trong {ind}"

    from pipeline_difix import DifixPipeline  # từ repo Difix3D (PYTHONPATH=src)
    model = "nvidia/difix_ref" if a.ref_dir else "nvidia/difix"
    pipe = DifixPipeline.from_pretrained(model, trust_remote_code=True)
    pipe.to("cuda")
    print(f"Difix: {model} · {len(files)} ảnh · {ind} → {outd}", flush=True)

    refs = sorted(Path(a.ref_dir).iterdir()) if a.ref_dir else None

    def nearest_ref(stem):
        # tên file tăng theo quỹ đạo → ref gần nhất theo thứ tự alphabet
        lo, hi = 0, len(refs) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if refs[mid].stem < stem:
                lo = mid + 1
            else:
                hi = mid
        return refs[lo]

    for i, p in enumerate(files):
        out_p = outd / (p.stem + ".png")
        if out_p.exists():
            continue
        img = load_image(str(p))
        kw = {}
        if refs:
            kw["ref_image"] = load_image(str(nearest_ref(p.stem)))
        with torch.no_grad():
            out = pipe(a.prompt, image=img, num_inference_steps=1,
                       timesteps=[199], guidance_scale=0.0, **kw).images[0]
        if out.size != img.size:   # pipeline có thể resize nội bộ — trả về đúng cỡ để chấm
            out = out.resize(img.size)
        out.save(out_p)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    print(f"✓ Difix xong → {outd}")


if __name__ == "__main__":
    main()
