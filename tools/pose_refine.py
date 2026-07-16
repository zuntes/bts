#!/usr/bin/env python3
"""B2 — Refine pose GAUGE-LOCKED, 2 GIAI ĐOẠN TÁCH BIỆT (docs DOC2 §3, DOC3 §2.3).

Bài học 16/07: refine pose+intrinsics(OPENCV 8-tham-số) CÙNG LÚC trong 1 BA từ cold-start
làm bài toán thiếu ràng buộc → phân kỳ (reprojection error → 10^156, NO_CONVERGENCE).
Sửa bằng tách 2 giai đoạn kiểu COLMAP chuẩn:

  STAGE 1 (mặc định, luôn chạy): giữ NGUYÊN camera model SIMPLE_RADIAL, chỉ refine
    pose (rig_from_world) + points3D — bài toán ràng buộc tốt, ổn định. Đây CHÍNH LÀ
    bài test giả thuyết P1 "pose nhiễu" một cách sạch, tách biệt khỏi rủi ro camera model.
  STAGE 2 (--refine_intrinsics, tuỳ chọn): SAU KHI stage 1 ổn, khoá pose lại
    (refine_rig_from_world=False), nâng model lên OPENCV, chỉ refine focal+distortion
    (KHÔNG refine principal point — dễ trôi, giữ cố định), poses không đổi nên không
    cần neo gauge lại lần 2.

Mỗi giai đoạn có sanity-check TRƯỚC khi dùng kết quả (isfinite + không phân kỳ) —
dừng cứng thay vì âm thầm ghi ra workspace hỏng.

(KHÁC pose-opt bị cấm: không đụng test poses, chỉ làm sạch SfM train — hợp lệ.)

Dùng: python pose_refine.py --in_ws workspace_raw/HCM0204 --out_ws workspace_ref/HCM0204
      [--refine_intrinsics]  (mặc định KHÔNG — chỉ stage 1)
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
    """Chặn phân kỳ NGAY, trước khi dùng kết quả BA hỏng."""
    if not math.isfinite(err1):
        sys.exit(f"❌ [{tag}] BA phân kỳ: reprojection error = {err1} (không hữu hạn). DỪNG.")
    if err1 > err0 * 5:
        sys.exit(f"❌ [{tag}] BA phân kỳ: error {err0:.3f}→{err1:.3f}px (TĂNG {err1/err0:.1f}×, "
                  f"đáng lẽ phải giảm). DỪNG — đừng dùng kết quả này.")
    print(f"  [{tag}] reprojection error: {err0:.4f} → {err1:.4f} px "
          f"({'giảm' if err1 < err0 else 'TĂNG'} {abs(1-err1/max(err0,1e-9))*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ws", required=True)
    ap.add_argument("--out_ws", required=True)
    ap.add_argument("--max_iters", type=int, default=100)
    ap.add_argument("--refine_intrinsics", action="store_true",
                    help="Stage 2: sau khi pose ổn, refine thêm focal+distortion (pose khoá cứng)")
    ap.add_argument("--max_gauge_drift_frac", type=float, default=0.05,
                    help="drift/scene tối đa chấp nhận được trước khi báo lỗi")
    a = ap.parse_args()
    import pycolmap
    if not hasattr(pycolmap, "Reconstruction"):
        sys.exit(
            "❌ Module 'pycolmap' import được nhưng KHÔNG có Reconstruction — bạn đang import "
            "pycolmap của rmbrualla (bản gsplat cần cho SceneManager, TRÙNG TÊN với pycolmap "
            "COLMAP chính thức). ĐỪNG cài 4.1.0 đè vào .venv (sẽ phá train gsplat)! "
            "Chạy script này bằng venv riêng: .venv_ba/bin/python tools/pose_refine.py ... "
            "(GATE_B.sh tự tạo .venv_ba)."
        )
    try:
        ver = pycolmap.__version__
    except AttributeError:
        import importlib.metadata
        try:
            ver = importlib.metadata.version("pycolmap")
        except Exception:
            ver = "?"
    print(f"pycolmap {ver}", flush=True)

    in_ws, out_ws = Path(a.in_ws), Path(a.out_ws)
    rec = pycolmap.Reconstruction(str(in_ws / "sparse/0"))
    print(f"vào: {rec.num_images()} ảnh, {rec.num_points3D()} điểm", flush=True)

    # --- 0. Loại điểm 3D track suy biến (track.length() < 2 sau khi BTC gỡ 169/409 ảnh
    #     làm nhiều track tụt còn 1 quan sát — vô nghĩa hình học, Ceres BA từ chối cứng).
    degenerate = [pid for pid, p in rec.points3D.items() if p.track.length() < 2]
    for pid in degenerate:
        rec.delete_point3D(pid)
    if degenerate:
        print(f"  loại {len(degenerate)} điểm track<2 (còn {rec.num_points3D()} điểm) — "
              f"hệ quả của việc BTC gỡ 169/409 ảnh khỏi tập train", flush=True)

    img_ids = sorted(rec.images.keys())
    C0 = np.array([rec.images[i].projection_center() for i in img_ids])

    def solver_opts(opt):
        try:
            opt.ceres.solver_options.max_num_iterations = a.max_iters
            opt.ceres.solver_options.minimizer_progress_to_stdout = False
        except AttributeError:
            pass  # field name khác version — không chặn chạy, chỉ mất control iter count
        return opt

    # ===================== STAGE 1 — CHỈ POSE + ĐIỂM, INTRINSICS CỐ ĐỊNH =====================
    print("\n=== STAGE 1: refine pose + points3D (intrinsics CỐ ĐỊNH, model SIMPLE_RADIAL nguyên) ===",
          flush=True)
    err0 = rec.compute_mean_reprojection_error()
    opt1 = pycolmap.BundleAdjustmentOptions()
    opt1.refine_focal_length = False
    opt1.refine_principal_point = False
    opt1.refine_extra_params = False
    opt1.refine_rig_from_world = True   # = pose, đây là thứ ta muốn sửa
    opt1.refine_points3D = True
    opt1 = solver_opts(opt1)
    pycolmap.bundle_adjustment(rec, opt1)
    err1 = rec.compute_mean_reprojection_error()
    check_converged("stage1-pose", err0, err1)

    # --- NEO GAUGE: similarity refined→gốc trên camera centers ---
    C1 = np.array([rec.images[i].projection_center() for i in img_ids])
    s, R, t = umeyama(C1, C0)
    sim = pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t)
    rec.transform(sim)
    C2 = np.array([rec.images[i].projection_center() for i in img_ids])
    drift = np.linalg.norm(C2 - C0, axis=1)
    scene = np.linalg.norm(C0 - C0.mean(0), axis=1).mean()
    print(f"  gauge sau neo: residual p50={np.median(drift):.4f} p95={np.percentile(drift,95):.4f}"
          f"  (scene scale ~{scene:.2f})", flush=True)
    if np.median(drift) > a.max_gauge_drift_frac * scene:
        sys.exit(f"❌ gauge drift lớn ({np.median(drift):.4f} > {a.max_gauge_drift_frac}×scene) — "
                  f"neo KHÔNG đáng tin. DỪNG, đừng train trên workspace này.")
    print(f"  ✅ Stage 1 ổn — drift {np.median(drift)/scene*100:.2f}% scene scale", flush=True)

    # ===================== STAGE 2 (tuỳ chọn) — INTRINSICS, POSE KHOÁ =====================
    if a.refine_intrinsics:
        print("\n=== STAGE 2: nâng OPENCV, refine focal+distortion (pose ĐÃ KHOÁ) ===", flush=True)
        for cam_id, cam in rec.cameras.items():
            name = getattr(cam, "model_name", None) or getattr(cam.model, "name", None) or str(cam.model)
            if "SIMPLE_RADIAL" not in name:
                print(f"  cam {cam_id}: model={name} (bỏ qua nâng cấp)"); continue
            f, cx, cy, k1 = cam.params
            cam.model = pycolmap.CameraModelId.OPENCV
            cam.params = [f, f, cx, cy, k1, 0.0, 0.0, 0.0]
            print(f"  cam {cam_id}: SIMPLE_RADIAL → OPENCV, giữ f={f:.1f} k1={k1:+.5f}")

        err0b = rec.compute_mean_reprojection_error()
        opt2 = pycolmap.BundleAdjustmentOptions()
        opt2.refine_focal_length = True
        opt2.refine_principal_point = False   # CỐ Ý tắt — nguồn phân kỳ đã gặp, giữ cố định
        opt2.refine_extra_params = True       # k1,k2,p1,p2
        opt2.refine_rig_from_world = False    # pose ĐÃ KHOÁ — chỉ intrinsics
        opt2.refine_points3D = True
        opt2 = solver_opts(opt2)
        pycolmap.bundle_adjustment(rec, opt2)
        err1b = rec.compute_mean_reprojection_error()
        check_converged("stage2-intrinsics", err0b, err1b)
        print("  ✅ Stage 2 ổn (pose không đổi trong stage này → không cần neo gauge lại)", flush=True)

    # --- ghi workspace mới ---
    (out_ws / "sparse/0").mkdir(parents=True, exist_ok=True)
    if not (out_ws / "images").exists():
        (out_ws / "images").symlink_to((in_ws / "images").resolve())
    rec.write(str(out_ws / "sparse/0"))
    for cam_id, cam in rec.cameras.items():
        p = list(cam.params)
        if len(p) == 4:  # vẫn SIMPLE_RADIAL (chỉ chạy stage 1)
            print(f"REFINED_INTRINSICS cam{cam_id}: f={p[0]:.2f} cx={p[1]:.2f} cy={p[2]:.2f} k1={p[3]:+.6f}")
        else:
            print(f"REFINED_INTRINSICS cam{cam_id}: fx={p[0]:.2f} fy={p[1]:.2f} cx={p[2]:.2f} cy={p[3]:.2f} "
                  f"k1={p[4]:+.6f} k2={p[5]:+.6f} p1={p[6]:+.6f} p2={p[7]:+.6f}")
    # Marker "kết quả này đã qua MỌI check hội tụ" — GATE_B chỉ tin workspace có file này;
    # thiếu marker = run fail giữa chừng / phân kỳ → GATE_B tự xoá làm lại.
    (out_ws / "REFINE_OK").write_text(f"stage2={a.refine_intrinsics}\n")
    print(f"✓ {out_ws} (+ marker REFINE_OK)")


if __name__ == "__main__":
    main()
