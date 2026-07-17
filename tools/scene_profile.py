#!/usr/bin/env python3
"""HỒ SƠ SCENE — bước "khám bệnh" TỰ ĐỘNG trước khi train bất kỳ scene nào.

Trả lời câu hỏi 17/07: "trên tập private không nhìn được test, pipeline có tự phát
hiện scene ngoại lai (bonsai kính phản chiếu, chair video dọc mờ) mà xử lý riêng không?"
→ CÓ, vì mọi tính chất đều đo được từ ảnh TRAIN (luôn có) + sparse. Tool này đo và
suy config từ SỐ ĐO thay vì hardcode theo tên scene:

  - camera model → nhánh train (gut/classic)   [thay danh sách tên]
  - ngân sách pixel (Σ pixel train / ~85px mỗi gaussian) → CAP khuyến nghị per-scene
  - sharpness (Laplacian var) → hồ sơ MỜ (video) → cần O2-enhancer đậm / blur-match / SH4
  - độ lệch sáng giữa các ảnh → cờ exposure (app_opt)
  - test poses: đếm + khoảng cách tới train (mức "novel")
  - (méo THẬT: chạy riêng audit_distortion.py — cần .venv_ba, chậm hơn)

Dùng: .venv/bin/python tools/scene_profile.py --data_root VAI_NVS_DATA_ROUND_2/VAI_NVS_DATA_ROUND2
      (hoặc --ws workspace_r2/bonsai ... cho từng workspace đã prepare)
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import read_cameras_bin, read_images_bin


def qvec2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def profile_scene(img_dir, sparse_dir, test_csv):
    cams = read_cameras_bin(sparse_dir / "cameras.bin")
    imgs = read_images_bin(sparse_dir / "images.bin")
    cam = next(iter(cams.values()))
    on_disk = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    n_train = len(on_disk)

    # ảnh mẫu: sharpness + brightness
    sharp, bright = [], []
    for p in on_disk[:: max(1, n_train // 8)][:8]:
        im = cv2.imread(str(p))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        sharp.append(cv2.Laplacian(g, cv2.CV_64F).var())
        bright.append(g.mean())
    H, W = im.shape[:2]
    sharp_m, bright_std = float(np.mean(sharp)), float(np.std(bright))

    # ngân sách pixel → cap (luật ~85px/gaussian, xác nhận bởi GATE_CAP r1 + A2)
    budget = n_train * W * H
    cap_rec = int(np.clip(budget / 85, 1e6, 12e6))
    cap_rec = int(round(cap_rec / 5e5) * 5e5)   # làm tròn 0.5M

    # test poses: novel tới đâu
    disk_names = {p.name for p in on_disk}
    train_c = np.array([-qvec2R(im2.qvec).T @ im2.tvec
                        for n, im2 in imgs.items() if n in disk_names])
    test_c = []
    rows = list(csv.DictReader(open(test_csv)))
    for r in rows:
        q = [float(r[k]) for k in ("qw", "qx", "qy", "qz")]
        t = np.array([float(r[k]) for k in ("tx", "ty", "tz")])
        test_c.append(-qvec2R(q).T @ t)
    test_c = np.array(test_c)
    from scipy.spatial import cKDTree
    tree = cKDTree(train_c)
    spacing = np.median(tree.query(train_c, k=2)[0][:, 1])
    ratio = tree.query(test_c, k=1)[0] / max(spacing, 1e-9)

    prof = dict(model=cam.model, W=W, H=H, n_train=n_train, n_test=len(rows),
                sharp=sharp_m, bright_std=bright_std, budget_mpx=budget / 1e6,
                cap_rec=cap_rec, nov_p50=float(np.median(ratio)),
                nov_p95=float(np.percentile(ratio, 95)))
    # --- suy CONFIG từ số đo (không dùng tên scene) ---
    flags = []
    prof["branch"] = "gut" if "RADIAL" in cam.model or "OPENCV" in cam.model else "classic"
    if sharp_m < 800:
        flags.append("MỜ(video)→enhancer-vgg đậm + xét blur-match/N1 + SH4-test")
    if bright_std > 10:
        flags.append("EXPO lệch→xét app_opt")
    if len(rows) < 40:
        flags.append(f"ÍT test({len(rows)})→soi tay từng ảnh trước nộp")
    if prof["branch"] == "classic":
        flags.append("pinhole→CHẠY audit_distortion (méo video có thể chưa mô hình hoá)")
    if prof["nov_p95"] > 3.5:
        flags.append("test XA train (p95>3.5×)→rủi ro vùng thiếu ràng buộc, xét Difix-distill")
    prof["flags"] = flags
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", help="root chứa <scene>/train + <scene>/test")
    ap.add_argument("--ws", nargs="*", help="hoặc list workspace đã prepare (cần test_csv cạnh data gốc)")
    a = ap.parse_args()
    jobs = []
    if a.data_root:
        root = Path(a.data_root)
        for sd in sorted(p for p in root.iterdir() if (p / "test/test_poses.csv").exists()):
            jobs.append((sd.name, sd / "train/images", sd / "train/sparse/0", sd / "test/test_poses.csv"))
    assert jobs, "không có scene nào (--data_root?)"

    print(f"{'scene':10s} {'camera':14s} {'res':10s} {'train/test':10s} {'sharp':>6s} "
          f"{'±expo':>5s} {'Mpx':>6s} {'CAP rec':>8s} {'nov p95':>7s}  nhánh")
    for name, img_dir, sp, tc in jobs:
        p = profile_scene(img_dir, sp, tc)
        print(f"{name:10s} {p['model']:14s} {p['W']}x{p['H']:<5d} "
              f"{p['n_train']}/{p['n_test']:<6d} {p['sharp']:6.0f} {p['bright_std']:5.1f} "
              f"{p['budget_mpx']:6.0f} {p['cap_rec']/1e6:6.1f}M {p['nov_p95']:6.2f}×  {p['branch']}")
        for f in p["flags"]:
            print(f"{'':10s} ⚑ {f}")
    print("\nCONFIG per-scene suy từ SỐ ĐO — dùng bảng này thay hardcode tên scene."
          "\nMéo thật: .venv_ba/bin/python tools/audit_distortion.py --ws workspace_r2/<scene>")


if __name__ == "__main__":
    main()
