#!/usr/bin/env python3
"""B2 — Refine pose/intrinsics GAUGE-LOCKED (docs DOC2 §3).

Vấn đề: COLMAP của BTC dùng SIMPLE_RADIAL (1 tham số k1) — quá thô cho ống kính thật.
Pose/intrinsics sai 0.5px = blur toàn cục = trần PSNR bị chặn bất kể train tốt cỡ nào.

Cách: nâng camera model → OPENCV (fx,fy,cx,cy,k1,k2,p1,p2), bundle-adjust lại
(pose + intrinsics + points) bằng pycolmap, rồi NEO GAUGE: similarity (Umeyama) đưa
camera centers refined về khớp frame GỐC → test poses (ở frame gốc) vẫn dùng được.
Phần "sửa" còn lại là hiệu chỉnh PER-CAMERA phi-rigid — chính là thứ ta muốn giữ.
(KHÁC pose-opt bị cấm: không đụng test poses, chỉ làm sạch SfM train — hợp lệ.)

Dùng: python pose_refine.py --in_ws workspace_raw/HCM0204 --out_ws workspace_ref/HCM0204
"""
import argparse, shutil, sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_ws", required=True)
    ap.add_argument("--out_ws", required=True)
    ap.add_argument("--max_iters", type=int, default=100)
    a = ap.parse_args()
    import pycolmap
    if not hasattr(pycolmap, "Reconstruction"):
        sys.exit(
            "❌ Module 'pycolmap' đã import nhưng KHÔNG có Reconstruction — đây là gói "
            "GIẢ/TRÙNG TÊN (đã gặp: 'pycolmap 0.0.1' rỗng), không phải thư viện SfM thật. "
            "Sửa: pip uninstall -y pycolmap && "
            "pip install --no-cache-dir --index-url https://pypi.org/simple pycolmap==4.1.0"
        )
    try:
        ver = pycolmap.__version__
    except AttributeError:
        import importlib.metadata
        try:
            ver = importlib.metadata.version("pycolmap")
        except Exception:
            ver = "?"
    print(f"pycolmap {ver}")

    in_ws, out_ws = Path(a.in_ws), Path(a.out_ws)
    rec = pycolmap.Reconstruction(str(in_ws / "sparse/0"))
    print(f"vào: {rec.num_images()} ảnh, {rec.num_points3D()} điểm")

    # --- 0. Loại điểm 3D track suy biến (track.length() < 2 sau khi BTC gỡ 169/409 ảnh
    #     làm nhiều track tụt còn 1 quan sát — vô nghĩa hình học, Ceres BA từ chối cứng).
    degenerate = [pid for pid, p in rec.points3D.items() if p.track.length() < 2]
    for pid in degenerate:
        rec.delete_point3D(pid)
    if degenerate:
        print(f"  loại {len(degenerate)} điểm track<2 (còn {rec.num_points3D()} điểm) — "
              f"hệ quả của việc BTC gỡ 169/409 ảnh khỏi tập train")

    # --- 1. nâng camera model SIMPLE_RADIAL(f,cx,cy,k1) → OPENCV(fx,fy,cx,cy,k1,k2,p1,p2)
    #     (API pycolmap đổi giữa version — dò nhiều cách lấy tên/set model, in rõ cách nào ăn)
    for cam_id, cam in rec.cameras.items():
        name = getattr(cam, "model_name", None)
        if name is None:
            m = cam.model
            name = getattr(m, "name", None) or str(m)
        if "SIMPLE_RADIAL" not in name:
            print(f"  cam {cam_id}: model={name} (không phải SIMPLE_RADIAL, bỏ qua nâng cấp)")
            continue
        f, cx, cy, k1 = cam.params
        new_params = [f, f, cx, cy, k1, 0.0, 0.0, 0.0]
        ok = False
        for setter in (
            lambda: setattr(cam, "model", pycolmap.CameraModelId.OPENCV),
            lambda: setattr(cam, "model_name", "OPENCV"),
            lambda: setattr(cam, "model_id", pycolmap.CameraModelId.OPENCV.value),
        ):
            try:
                setter(); ok = True; break
            except Exception:
                continue
        if not ok:
            sys.exit(f"❌ không set được camera.model=OPENCV trên cam {cam_id} — "
                      f"pycolmap API lạ, dán 'python3 -c \"import pycolmap; help(pycolmap.Camera)\"' cho Claude")
        cam.params = new_params
        print(f"  cam {cam_id}: SIMPLE_RADIAL → OPENCV, giữ f={f:.1f} k1={k1:+.5f}")

    # --- 2. lưu centers gốc (để neo gauge)
    img_ids = sorted(rec.images.keys())
    C0 = np.array([rec.images[i].projection_center() for i in img_ids])

    # --- 3. bundle adjustment (pose + intrinsics + points)
    err0 = rec.compute_mean_reprojection_error()
    opt = pycolmap.BundleAdjustmentOptions()
    for k in ("refine_focal_length", "refine_principal_point", "refine_extra_params",
              "refine_extrinsics"):
        if hasattr(opt, k): setattr(opt, k, True)
    if hasattr(opt, "solver_options") and hasattr(opt.solver_options, "max_num_iterations"):
        opt.solver_options.max_num_iterations = a.max_iters
    pycolmap.bundle_adjustment(rec, opt)
    err1 = rec.compute_mean_reprojection_error()
    print(f"reprojection error: {err0:.4f} → {err1:.4f} px  (giảm {100*(1-err1/max(err0,1e-9)):.1f}%)")

    # --- 4. NEO GAUGE: similarity refined→gốc trên camera centers
    C1 = np.array([rec.images[i].projection_center() for i in img_ids])
    s, R, t = umeyama(C1, C0)
    T = np.eye(4); T[:3, :3] = s * R; T[:3, 3] = t
    sim = pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t) if hasattr(pycolmap, "Sim3d") else None
    if sim is not None and hasattr(rec, "transform"):
        rec.transform(sim)
    else:
        sys.exit("❌ pycolmap thiếu Sim3d/transform — dán version cho Claude")
    C2 = np.array([rec.images[i].projection_center() for i in img_ids])
    drift = np.linalg.norm(C2 - C0, axis=1)
    scene = np.linalg.norm(C0 - C0.mean(0), axis=1).mean()
    print(f"gauge sau neo: residual per-camera p50={np.median(drift):.4f} p95={np.percentile(drift,95):.4f}"
          f"  (scene scale ~{scene:.2f}; residual = hiệu chỉnh thật, PHẢI nhỏ hơn scene nhiều)")
    if np.median(drift) > 0.05 * scene:
        print("⚠ drift lớn bất thường — khả năng gauge không neo được, XEM KỸ trước khi train")

    # --- 5. ghi workspace mới (ảnh symlink, sparse mới)
    (out_ws / "sparse/0").mkdir(parents=True, exist_ok=True)
    if not (out_ws / "images").exists():
        (out_ws / "images").symlink_to((in_ws / "images").resolve())
    rec.write(str(out_ws / "sparse/0"))
    # in intrinsics mới để render dùng
    for cam_id, cam in rec.cameras.items():
        p = list(cam.params)
        print(f"REFINED_INTRINSICS cam{cam_id}: fx={p[0]:.2f} fy={p[1]:.2f} cx={p[2]:.2f} cy={p[3]:.2f} "
              f"k1={p[4]:+.6f} k2={p[5]:+.6f} p1={p[6]:+.6f} p2={p[7]:+.6f}")
    print(f"✓ {out_ws}")


if __name__ == "__main__":
    main()
