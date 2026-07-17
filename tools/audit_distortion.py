#!/usr/bin/env python3
"""AUDIT MÉO THỰC — chạy TRƯỚC khi chốt tham số train cho bất kỳ scene nào.

Bài học 17/07 (DOC3 §2.4): cameras.bin của BTC không đáng tin mù quáng —
(1) export lệch thang keypoint 4× ở scene HCM; (2) SIMPLE_RADIAL k1 chỉ là méo
"trung bình hoá" — HCM0204 đo được méo thật là k1≈-0.001, k2≈+0.017 (khác hẳn
k1=+0.010 lưu trong file). Đưa k1 lưu sẵn vào train = train trên model méo sai.

Tool này, cho MỖI scene (CPU-only, ~2-4 phút/scene, không ghi gì ra workspace):
  1. dò thang obs/proj (bug export)
  2. BA stage-1 (pose+points, SOFT_L1, gauge khoá) → pose sạch
  3. nâng OPENCV, BA stage-2 (intrinsics-only, pose khoá) → méo THẬT k1 k2 p1 p2
  4. bảng verdict: méo lưu vs méo đo + khuyến nghị nhánh train

Dùng: .venv_ba/bin/python tools/audit_distortion.py --ws workspace_r2/HCM0421 [workspace_r2/...]
Kết quả người đọc: dòng AUDIT cuối mỗi scene + bảng tổng.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pose_refine import detect_obs_scale, rescale_cameras  # dùng chung, tránh lệch logic


def audit_one(ws, max_iters=150):
    import pycolmap
    rec = pycolmap.Reconstruction(str(Path(ws) / "sparse/0"))
    cam = next(iter(rec.cameras.values()))
    model0 = cam.model.name
    stored = list(cam.params)

    s = detect_obs_scale(rec, pycolmap)
    s_int = round(s)
    if abs(s - 1.0) > 0.02:
        assert abs(s - s_int) < 0.02, f"thang {s:.3f} không nguyên"
        rescale_cameras(rec, s_int)

    om = pycolmap.ObservationManager(rec)
    om.filter_observations_with_negative_depth()
    om.filter_all_points3D(6.0, 0.5)
    for pid in [pid for pid, p in rec.points3D.items() if p.track.length() < 2]:
        rec.delete_point3D(pid)
    if rec.num_points3D() < 10000:
        return dict(scene=Path(ws).name, err=f"còn {rec.num_points3D()} điểm sau lọc — bỏ audit")

    img_ids = sorted(rec.images.keys())

    def ba(refine_pose, refine_intr):
        cfg = pycolmap.BundleAdjustmentConfig()
        for i in img_ids:
            cfg.add_image(i)
        cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        o = pycolmap.BundleAdjustmentOptions()
        o.refine_rig_from_world = refine_pose
        o.refine_points3D = True
        o.refine_focal_length = refine_intr
        o.refine_extra_params = refine_intr
        o.refine_principal_point = False
        o.ceres.loss_function_type = pycolmap.LossFunctionType.SOFT_L1
        o.ceres.loss_function_scale = 1.0
        try:
            o.ceres.solver_options.max_num_iterations = max_iters
        except AttributeError:
            pass
        pycolmap.create_default_bundle_adjuster(o, cfg, rec).solve()
        rec.update_point_3d_errors()
        return rec.compute_mean_reprojection_error()

    rec.update_point_3d_errors()
    e0 = rec.compute_mean_reprojection_error()
    e1 = ba(True, False)                     # stage 1: pose
    # stage 2: nâng OPENCV
    for cid, c in rec.cameras.items():
        if c.model.name == "SIMPLE_RADIAL":
            f, cx, cy, k1 = c.params
            c.model = pycolmap.CameraModelId.OPENCV
            c.params = [f, f, cx, cy, k1, 0.0, 0.0, 0.0]
        elif c.model.name == "SIMPLE_PINHOLE":
            f, cx, cy = c.params
            c.model = pycolmap.CameraModelId.OPENCV
            c.params = [f, f, cx, cy, 0.0, 0.0, 0.0, 0.0]
    e2 = ba(False, True)                     # stage 2: intrinsics

    p = np.array(next(iter(rec.cameras.values())).params, dtype=float)
    p[:4] /= s_int if abs(s - 1) > 0.02 else 1   # fx fy cx cy về thang workspace
    fx, fy, cx, cy, k1, k2, p1, p2 = p
    complex_dist = abs(k2) > 0.004 or abs(p1) > 3e-4 or abs(p2) > 3e-4
    return dict(scene=Path(ws).name, model0=model0, s=s, stored=stored,
                e0=e0, e1=e1, e2=e2, fx=fx, fy=fy,
                k1=k1, k2=k2, p1=p1, p2=p2, complex=complex_dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", nargs="+", required=True)
    a = ap.parse_args()
    rows = []
    for ws in a.ws:
        print(f"\n===== {ws} =====", flush=True)
        try:
            r = audit_one(ws)
        except Exception as e:
            r = dict(scene=Path(ws).name, err=f"{type(e).__name__}: {e}")
        rows.append(r)
        if "err" in r:
            print(f"  ⚠ {r['err']}")
            continue
        st = ", ".join(f"{v:.4g}" for v in r["stored"])
        print(f"  model gốc: {r['model0']} params=[{st}] · thang obs/proj s={r['s']:.3f}")
        print(f"  reproj err: {r['e0']:.3f} → pose {r['e1']:.3f} → +intrinsics {r['e2']:.3f} px")
        print(f"  MÉO THỰC: fx={r['fx']:.2f} fy={r['fy']:.2f} k1={r['k1']:+.5f} k2={r['k2']:+.5f} "
              f"p1={r['p1']:+.5f} p2={r['p2']:+.5f}")

    print("\n" + "#" * 78)
    print("#  AUDIT MÉO — VERDICT (đưa vào quyết định tham số train)")
    print("#" * 78)
    for r in rows:
        if "err" in r:
            print(f"  {r['scene']}: ⚠ {r['err']}"); continue
        tag = ("MÉO PHỨC TẠP → train nên dùng workspace refined stage-2 (B2X) hoặc "
               "truyền k1,k2,p1,p2 đo được" if r["complex"]
               else "méo đơn giản — SIMPLE_RADIAL k1 lưu sẵn tạm đủ")
        print(f"  {r['scene']}: s={r['s']:.1f} · err {r['e0']:.2f}→{r['e2']:.2f}px · "
              f"k1={r['k1']:+.4f} k2={r['k2']:+.4f} → {tag}")
    print("DÁN TOÀN BỘ OUTPUT CHO CLAUDE nếu chạy trên server.")


if __name__ == "__main__":
    main()
