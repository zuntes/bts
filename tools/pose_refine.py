#!/usr/bin/env python3
"""B2 — Refine pose GAUGE-LOCKED trên data BTC (DOC2 §3, DOC3 §2.3-2.4).

LỊCH SỬ 3 LẦN PHÂN KỲ (đắt — đọc trước khi sửa):
1. 16/07: refine pose+intrinsics(OPENCV) cùng lúc → tách 2 stage. VẪN nổ.
2. 17/07: stage-1 pose-only vẫn nổ (err → 10^148). Đào tận gốc thì ra:
3. **BTC export cameras.bin đã CHIA 4 (1320×989, f≈925) nhưng keypoints 2D trong
   images.bin vẫn ở thang ẢNH GỐC 5280×3956** (fit obs = 4.000·proj, rms<1px).
   → residual "thật" ~2700px, BA tối ưu trên rác → điểm bay vô cực.
   (bonsai/chair export sạch — scale 1/1; chỉ scene HCM dính.)

Fix v3 (file này):
  a. TỰ PHÁT HIỆN thang s bằng lstsq obs~proj trên vài trăm điểm (s≈1 → bỏ qua);
     nhân camera lên s để BA nhất quán, GHI RA thì chia lại s (workspace giữ nguyên
     quy ước cũ — gsplat/render không biết gì về vụ này).
  b. Lọc điểm degenerate TRƯỚC BA: track<2, depth âm, reproj lớn, góc tam giác <0.5°.
  c. BAConfig tường minh + fix_gauge(TWO_CAMS_FROM_WORLD) — bịt 7-DOF gauge.
  d. Robust loss SOFT_L1 — điểm outlier không kéo nổ toàn hệ.
  e. check_converged TRƯỚC khi ghi + neo Umeyama + marker REFINE_OK (GATE_B tin marker).

(KHÁC pose-opt bị cấm: không đụng test poses, chỉ làm sạch SfM train — hợp lệ.)

Dùng: .venv_ba/bin/python pose_refine.py --in_ws workspace_raw/HCM0204 --out_ws workspace_ref/HCM0204
      [--refine_intrinsics]  (stage 2 tuỳ chọn, pose khoá)
"""
import argparse, math, sys
from pathlib import Path
import numpy as np


def umeyama(src, dst):
    """Similarity s,R,t: dst ≈ s·R·src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3); S[2, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (sc ** 2).sum() * len(src)
    t = mu_d - s * R @ mu_s
    return s, R, t


def check_converged(tag, err0, err1):
    if not math.isfinite(err1):
        sys.exit(f"❌ [{tag}] BA phân kỳ: error = {err1}. DỪNG.")
    if err1 > err0 * 5:
        sys.exit(f"❌ [{tag}] BA phân kỳ: {err0:.3f}→{err1:.3f}px (TĂNG). DỪNG.")
    print(f"  [{tag}] reprojection error: {err0:.4f} → {err1:.4f} px "
          f"({'giảm' if err1 < err0 else 'TĂNG'} {abs(1 - err1 / max(err0, 1e-9)) * 100:.1f}%)", flush=True)


def detect_obs_scale(rec, pycolmap, n_img=3, n_pts=300):
    """BTC export bug: keypoints ở thang ảnh GỐC, camera đã chia. Fit obs≈a·proj."""
    cam = next(iter(rec.cameras.values()))
    ratios = []
    for iid in sorted(rec.images.keys())[:: max(1, rec.num_images() // n_img)][:n_img]:
        im = rec.images[iid]
        obs, proj = [], []
        for p2 in im.points2D:
            if not rec.exists_point3D(p2.point3D_id):
                continue
            pc = (im.cam_from_world() * rec.point3D(p2.point3D_id).xyz).reshape(1, 3)
            xy = cam.img_from_cam(pc)
            if xy is None:
                continue
            obs.append(np.asarray(p2.xy)); proj.append(xy[0])
            if len(obs) >= n_pts:
                break
        if len(obs) < 50:
            continue
        obs, proj = np.array(obs), np.array(proj)
        for ax in (0, 1):
            A = np.stack([proj[:, ax], np.ones(len(proj))], 1)
            (a, b), *_ = np.linalg.lstsq(A, obs[:, ax], rcond=None)
            ratios.append(a)
    if not ratios:
        sys.exit("❌ không đủ điểm để dò thang obs/proj")
    return float(np.median(ratios))


def rescale_cameras(rec, s):
    for cid, cam in rec.cameras.items():
        p = np.array(cam.params, dtype=float)
        if cam.model.name == "SIMPLE_RADIAL":       # f cx cy k1 — k1 bất biến theo scale
            p[:3] *= s
        elif cam.model.name == "SIMPLE_PINHOLE":
            p[:3] *= s
        elif cam.model.name == "PINHOLE":
            p[:4] *= s
        elif cam.model.name == "OPENCV":            # fx fy cx cy k1 k2 p1 p2
            p[:4] *= s
        else:
            sys.exit(f"❌ rescale: model {cam.model.name} chưa hỗ trợ")
        cam.params = list(p)
        cam.width = int(round(cam.width * s))
        cam.height = int(round(cam.height * s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ws", required=True)
    ap.add_argument("--out_ws", required=True)
    ap.add_argument("--max_iters", type=int, default=200)
    ap.add_argument("--refine_intrinsics", action="store_true")
    ap.add_argument("--max_gauge_drift_frac", type=float, default=0.05)
    a = ap.parse_args()
    import pycolmap
    if not hasattr(pycolmap, "Reconstruction"):
        sys.exit("❌ pycolmap này là bản rmbrualla (gsplat) — chạy bằng .venv_ba/bin/python (DOC3 §3.7)")

    in_ws, out_ws = Path(a.in_ws), Path(a.out_ws)
    rec = pycolmap.Reconstruction(str(in_ws / "sparse/0"))
    print(f"vào: {rec.num_images()} ảnh, {rec.num_points3D()} điểm", flush=True)

    # --- a. THANG OBS/PROJ (bug export BTC — xem docstring) ---
    s_obs = detect_obs_scale(rec, pycolmap)
    print(f"  thang keypoint/camera: s = {s_obs:.4f}", flush=True)
    scaled = abs(s_obs - 1.0) > 0.02
    if scaled:
        print(f"  ⚠ BTC export lệch thang (keypoints ở ảnh gốc, camera đã chia {s_obs:.1f}) "
              f"→ nhân camera ×{s_obs:.1f} cho BA, ghi ra sẽ chia lại", flush=True)
        s_int = round(s_obs)
        assert abs(s_obs - s_int) < 0.02, f"thang {s_obs} không nguyên — dữ liệu lạ, dừng"
        rescale_cameras(rec, s_int)
    err_check = rec.compute_mean_reprojection_error()  # giá trị stored, chỉ để log

    # --- b. LỌC degenerate TRƯỚC BA ---
    om = pycolmap.ObservationManager(rec)
    n_neg = om.filter_observations_with_negative_depth()
    n_bad = om.filter_all_points3D(6.0, 0.5)   # reproj>6px (thang đã chỉnh), góc<0.5°
    deg = [pid for pid, p in rec.points3D.items() if p.track.length() < 2]
    for pid in deg:
        rec.delete_point3D(pid)
    print(f"  lọc: depth_âm={n_neg} · reproj/góc={n_bad} obs · track<2={len(deg)} điểm "
          f"→ còn {rec.num_points3D():,} điểm", flush=True)
    if rec.num_points3D() < 20000:
        sys.exit(f"❌ sau lọc chỉ còn {rec.num_points3D()} điểm — ngưỡng lọc sai hoặc data hỏng, DỪNG")

    img_ids = sorted(rec.images.keys())
    C0 = np.array([rec.images[i].projection_center() for i in img_ids])

    def run_ba(opt, tag):
        cfg = pycolmap.BundleAdjustmentConfig()
        for i in img_ids:
            cfg.add_image(i)
        cfg.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
        opt.ceres.loss_function_type = pycolmap.LossFunctionType.SOFT_L1
        opt.ceres.loss_function_scale = 1.0
        try:
            opt.ceres.solver_options.max_num_iterations = a.max_iters
        except AttributeError:
            pass
        rec.update_point_3d_errors()   # BẮT BUỘC: compute_mean_... đọc giá trị STORED
        err0 = rec.compute_mean_reprojection_error()
        adj = pycolmap.create_default_bundle_adjuster(opt, cfg, rec)
        summ = adj.solve()
        try:
            print("  " + summ.brief_report(), flush=True)
        except Exception:
            pass
        rec.update_point_3d_errors()
        err1 = rec.compute_mean_reprojection_error()
        check_converged(tag, err0, err1)

    # ===================== STAGE 1 — POSE + ĐIỂM =====================
    print("\n=== STAGE 1: pose + points3D (intrinsics CỐ ĐỊNH, gauge khoá 2 cam, SOFT_L1) ===", flush=True)
    o1 = pycolmap.BundleAdjustmentOptions()
    o1.refine_focal_length = False
    o1.refine_principal_point = False
    o1.refine_extra_params = False
    o1.refine_rig_from_world = True
    o1.refine_points3D = True
    run_ba(o1, "stage1-pose")

    C1 = np.array([rec.images[i].projection_center() for i in img_ids])
    su, R, t = umeyama(C1, C0)
    rec.transform(pycolmap.Sim3d(su, pycolmap.Rotation3d(R), t))
    C2 = np.array([rec.images[i].projection_center() for i in img_ids])
    drift = np.linalg.norm(C2 - C0, axis=1)
    scene = np.linalg.norm(C0 - C0.mean(0), axis=1).mean()
    print(f"  gauge sau neo: p50={np.median(drift):.4f} p95={np.percentile(drift, 95):.4f} (scene ~{scene:.2f})", flush=True)
    if np.median(drift) > a.max_gauge_drift_frac * scene:
        sys.exit(f"❌ gauge drift {np.median(drift):.4f} > {a.max_gauge_drift_frac}×scene — DỪNG.")
    print(f"  ✅ Stage 1 ổn — drift {np.median(drift) / scene * 100:.2f}% scene", flush=True)

    # ===================== STAGE 2 (tuỳ chọn) — INTRINSICS, POSE KHOÁ =====================
    if a.refine_intrinsics:
        print("\n=== STAGE 2: nâng OPENCV, refine focal+distortion (pose KHOÁ, cp KHOÁ) ===", flush=True)
        for cid, cam in rec.cameras.items():
            if "SIMPLE_RADIAL" not in cam.model.name:
                print(f"  cam {cid}: {cam.model.name} — bỏ nâng cấp"); continue
            f, cx, cy, k1 = cam.params
            cam.model = pycolmap.CameraModelId.OPENCV
            cam.params = [f, f, cx, cy, k1, 0.0, 0.0, 0.0]
        o2 = pycolmap.BundleAdjustmentOptions()
        o2.refine_focal_length = True
        o2.refine_principal_point = False
        o2.refine_extra_params = True
        o2.refine_rig_from_world = False
        o2.refine_points3D = True
        run_ba(o2, "stage2-intrinsics")

    # --- ghi ra: TRẢ camera về thang cũ để workspace nhất quán với ảnh/pipeline ---
    if scaled:
        rescale_cameras(rec, 1.0 / s_int)
    (out_ws / "sparse/0").mkdir(parents=True, exist_ok=True)
    if not (out_ws / "images").exists():
        (out_ws / "images").symlink_to((in_ws / "images").resolve())
    rec.write(str(out_ws / "sparse/0"))
    for cid, cam in rec.cameras.items():
        p = list(cam.params)
        if len(p) == 4:
            print(f"REFINED_INTRINSICS cam{cid}: f={p[0]:.2f} cx={p[1]:.2f} cy={p[2]:.2f} k1={p[3]:+.6f}")
        else:
            print(f"REFINED_INTRINSICS cam{cid}: fx={p[0]:.2f} fy={p[1]:.2f} cx={p[2]:.2f} cy={p[3]:.2f} "
                  f"k1={p[4]:+.6f} k2={p[5]:+.6f} p1={p[6]:+.6f} p2={p[7]:+.6f}")
    (out_ws / "REFINE_OK").write_text(f"stage2={a.refine_intrinsics} s_obs={s_obs:.4f}\n")
    print(f"✓ {out_ws} (+ marker REFINE_OK)")


if __name__ == "__main__":
    main()
