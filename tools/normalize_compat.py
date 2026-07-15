"""Tái tạo CHÍNH XÁC phép chuẩn hoá world space của gsplat Parser
(sao chép từ gsplat/examples/datasets/, pin theo bản clone v1.5.3 — gồm cả T3 flip).

Vì sao cần: simple_trainer mặc định `normalize_world_space=True` → model được train
trong hệ toạ độ đã xoay+dịch+scale (transform = T2 @ T1). Mọi pose đưa vào render/
fine-tune từ ngoài (test_poses.csv, images.bin) là COLMAP THÔ — phải biến đổi sang
hệ chuẩn hoá trước, nếu không camera chỉ vào khoảng không (ảnh đen/lệch).

compute_parser_transform(images, points) phải cho kết quả GIỐNG HỆT Parser:
- T1 = similarity_from_cameras(c2w của TOÀN BỘ ảnh trong images.bin workspace)
- points sau T1 → T2 = align_principle_axes(points)
(các phép này dùng mean/median — không phụ thuộc thứ tự ảnh/điểm)
"""
import numpy as np


def similarity_from_cameras(c2w, strict_scaling=False, center_method="focus"):
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)
    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]
    if center_method == "focus":
        nearest = t + (fwds * -t).sum(-1)[:, None] * fwds
        translate = -np.median(nearest, axis=0)
    elif center_method == "poses":
        translate = -np.median(t, axis=0)
    else:
        raise ValueError(center_method)
    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))
    transform[:3, :] *= scale
    return transform


def align_principle_axes(point_cloud):
    centroid = np.median(point_cloud, axis=0)
    translated = point_cloud - centroid
    covariance_matrix = np.cov(translated, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sort_indices = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, sort_indices]
    if np.linalg.det(eigenvectors) < 0:
        eigenvectors[:, 0] *= -1
    rotation_matrix = eigenvectors.T
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = -rotation_matrix @ centroid
    return transform


def transform_points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_camera(matrix, c2w):
    """Áp transform lên MỘT c2w 4×4 rồi tái chuẩn hoá rotation (đúng logic
    transform_cameras của gsplat: chia [:3,:3] cho norm của HÀNG đầu)."""
    out = matrix @ c2w
    scaling = np.linalg.norm(out[0, :3])
    out[:3, :3] = out[:3, :3] / scaling
    return out


def compute_parser_transform(images, points_xyz):
    """images: dict từ colmap_io.read_images_bin của WORKSPACE (đã lọc);
    points_xyz: [N,3] từ points3D.bin của workspace. Trả về transform 4×4.

    ⚠ gsplat v1.5.x thêm T3 (lật 180° quanh x khi median(z) > mean(z) sau T2@T1
    — "fix up side down"). Với scene BTS chụp từ dưới lên, T3 KÍCH HOẠT thường
    xuyên (đã đo HCM0204: median 0.0104 > mean −0.0401). Thiếu T3 → mọi render
    lệch pose hoàn toàn (PSNR ~9). Logic dưới đây pin theo clone v1.5.3."""
    c2ws = []
    for im in images.values():
        w2c = np.eye(4)
        w2c[:3, :3] = im.R
        w2c[:3, 3] = im.tvec
        c2ws.append(np.linalg.inv(w2c))
    c2ws = np.stack(c2ws)
    T1 = similarity_from_cameras(c2ws)
    # áp T1 lên points như Parser rồi tính T2
    pts = transform_points(T1, points_xyz)
    T2 = align_principle_axes(pts)
    transform = T2 @ T1
    # T3 của v1.5.x: kiểm tra trên points ĐÃ qua T2@T1
    pts = transform_points(T2, pts)
    if np.median(pts[:, 2]) > np.mean(pts[:, 2]):
        T3 = np.diag([1.0, -1.0, -1.0, 1.0])
        transform = T3 @ transform
    return transform


def colmap_w2c_to_normalized_viewmat(qR, tvec, transform):
    """Từ (R w2c, tvec) COLMAP thô + transform của Parser → viewmat trong hệ model."""
    w2c = np.eye(4)
    w2c[:3, :3] = qR
    w2c[:3, 3] = tvec
    c2w_norm = transform_camera(transform, np.linalg.inv(w2c))
    return np.linalg.inv(c2w_norm)
