#!/usr/bin/env python3
"""Phân loại scene NÉT (SHARP) vs MỜ (BLUR) — nguồn sự thật DUY NHẤT cho mọi script.

Vì sao cần: đòn bẩy trên scene mờ KHÔNG ánh xạ sang scene nét và ngược lại (đã chứng
minh: supersample −0.012 obj nhưng chưa đo nét; enhancer ăn 3× trên mờ; blur-match chỉ
hợp mờ). Mỗi nhánh cấu hình train/hậu-xử-lý riêng.

2 chế độ (bài toán "private set đưa trước để train, nhưng lần thi thật sẽ ẩn"):
  - classify(scene)          : HARDCODE theo tên (round-2 hiện tại — chắc chắn, không đoán)
  - classify_auto(ws_dir)    : ĐO lapvar p10 (cho scene private LẠ chưa biết tên)
Cả hai trả 'SHARP' hoặc 'BLUR'. classify() ưu tiên; rơi về auto nếu tên lạ.

Ngưỡng: lapvar p10 < 500 = BLUR (chair p10=383, bonsai 50 · HCM đều >1900).
p10 (không phải mean) vì video mờ KHÔNG ĐỀU: chair mean 1369 "có vẻ nét" nhưng 10%
frame tệ nhất chỉ 383 → chính mấy frame đó kéo tụt model.
"""
import sys
from pathlib import Path

BLUR_THR = 500

# HARDCODE round-2 (7 scene đã biết) — camera model + lapvar đã đo, chắc chắn.
# SHARP = drone SIMPLE_RADIAL 1320×989 nét · BLUR = video SIMPLE_PINHOLE mờ.
_HARD = {
    "HCM0421": "SHARP", "HCM0539": "SHARP", "HCM0540": "SHARP",
    "HCM0644": "SHARP", "HCM0674": "SHARP",
    "bonsai": "BLUR", "chair": "BLUR",
    # round-1 public (bàn calib)
    "HCM0204": "SHARP", "HCM0181": "SHARP", "HCM0193": "SHARP",
    "hcm0031": "SHARP", "hcm0034": "SHARP",
    # round-1 private (đều drone nét)
    "HCM0249": "SHARP", "HCM0254": "SHARP", "HCM0276": "SHARP", "HCM1439": "SHARP",
    "HNI0131": "SHARP", "HNI0265": "SHARP", "HNI0366": "SHARP", "HNI0437": "SHARP",
}


def _lapvar_p10(ws_dir):
    import cv2
    import numpy as np
    img_dir = Path(ws_dir) / "images"
    imgs = sorted(img_dir.iterdir()) if img_dir.is_dir() else []
    if len(imgs) < 5:
        return None
    sample = imgs[:: max(1, len(imgs) // 12)][:12]
    lv = []
    for p in sample:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is not None:
            lv.append(cv2.Laplacian(im, cv2.CV_64F).var())
    return float(np.percentile(lv, 10)) if lv else None


def classify_auto(ws_dir):
    p10 = _lapvar_p10(ws_dir)
    if p10 is None:
        return "SHARP"   # mặc định an toàn (đa số scene BTS là drone nét)
    return "BLUR" if p10 < BLUR_THR else "SHARP"


def classify(scene, ws_dir=None):
    """Ưu tiên hardcode; tên lạ → đo auto (nếu có ws_dir) → mặc định SHARP."""
    if scene in _HARD:
        return _HARD[scene]
    if ws_dir and Path(ws_dir).is_dir():
        return classify_auto(ws_dir)
    return "SHARP"


# cấu hình prod theo lớp — script train đọc ĐÂY thay vì hardcode rải rác
PROFILE = {
    "SHARP": dict(branch="gut", cap=6000000, sh_degree=3, raw_distortion=True),
    "BLUR":  dict(branch="classic", cap=3000000, sh_degree=4, raw_distortion=False),
}


def profile(scene, ws_dir=None):
    return PROFILE[classify(scene, ws_dir)]


if __name__ == "__main__":
    # CLI: in phân loại cho 1 scene hoặc quét cả workspace
    if len(sys.argv) >= 2 and sys.argv[1] not in ("--scan",):
        s = sys.argv[1]
        ws = sys.argv[2] if len(sys.argv) > 2 else None
        print(f"{s}: {classify(s, ws)}  profile={profile(s, ws)}")
    else:
        for root in ("workspace_r2", "workspace_r2v"):
            r = Path(root)
            if not r.is_dir():
                continue
            for sd in sorted(r.iterdir()):
                if (sd / "images").is_dir():
                    hard = _HARD.get(sd.name, "?")
                    auto = classify_auto(sd)
                    flag = "" if hard == "?" or hard == auto else "  ⚠ HARDCODE≠AUTO"
                    print(f"  {root}/{sd.name:<10} hardcode={hard:<5} auto={auto:<5}{flag}")
