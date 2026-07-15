"""Đọc/ghi COLMAP binary format (cameras.bin, images.bin, points3D.bin).

Tự viết để kiểm soát hoàn toàn 2 pitfall của dữ liệu VAI:
- images.bin chứa nhiều ảnh hơn số ảnh có trên đĩa (phải lọc);
- cần ghi lại cameras.bin dạng PINHOLE sau khi undistort.
"""
import struct
from dataclasses import dataclass, field

import numpy as np

CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5), 4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12), 7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12),
}
MODEL_IDS = {name: mid for mid, (name, _) in CAMERA_MODELS.items()}


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray  # theo thứ tự COLMAP của model


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (w,x,y,z), world-to-camera
    tvec: np.ndarray  # world-to-camera (KHÔNG phải camera center)
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def R(self):
        return qvec2rotmat(self.qvec)

    @property
    def center(self):
        return -self.R.T @ self.tvec


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras_bin(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            name, np_ = CAMERA_MODELS[model_id]
            params = np.array(struct.unpack("<" + "d" * np_, f.read(8 * np_)))
            cams[cid] = Camera(cid, name, w, h, params)
    return cams


def write_cameras_bin(path, cams):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cams)))
        for cam in cams.values():
            mid = MODEL_IDS[cam.model]
            f.write(struct.pack("<iiQQ", cam.id, mid, cam.width, cam.height))
            f.write(struct.pack("<" + "d" * len(cam.params), *cam.params))


def read_images_bin(path):
    imgs = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (iid,) = struct.unpack("<i", f.read(4))
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            (cam_id,) = struct.unpack("<i", f.read(4))
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            data = np.frombuffer(f.read(24 * npts), dtype=np.float64).reshape(-1, 3)
            xys = data[:, :2].copy()
            p3d = data[:, 2].view(np.int64).copy() if npts else np.zeros(0, np.int64)
            imgs[name.decode()] = Image(iid, qvec, tvec, cam_id, name.decode(), xys, p3d)
    return imgs


def write_images_bin(path, imgs):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for im in imgs.values():
            f.write(struct.pack("<i", im.id))
            f.write(struct.pack("<dddd", *im.qvec))
            f.write(struct.pack("<ddd", *im.tvec))
            f.write(struct.pack("<i", im.camera_id))
            f.write(im.name.encode() + b"\x00")
            n = len(im.xys)
            f.write(struct.pack("<Q", n))
            if n:
                data = np.empty((n, 3), dtype=np.float64)
                data[:, :2] = im.xys
                data[:, 2] = im.point3D_ids.view(np.float64)
                f.write(data.tobytes())


def read_points3D_bin(path):
    """Trả về (ids, xyz[N,3], rgb[N,3], error[N], tracks list-of-bytes để ghi lại verbatim)."""
    ids, xyzs, rgbs, errs, tracks = [], [], [], [], []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            (pid,) = struct.unpack("<Q", f.read(8))
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            (err,) = struct.unpack("<d", f.read(8))
            (tl,) = struct.unpack("<Q", f.read(8))
            track = f.read(8 * tl)
            ids.append(pid); xyzs.append(xyz); rgbs.append(rgb); errs.append(err); tracks.append(track)
    return (np.array(ids, dtype=np.uint64), np.array(xyzs), np.array(rgbs, dtype=np.uint8),
            np.array(errs), tracks)


def write_points3D_bin(path, ids, xyzs, rgbs, errs, tracks):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(ids)))
        for pid, xyz, rgb, err, track in zip(ids, xyzs, rgbs, errs, tracks):
            f.write(struct.pack("<Q", int(pid)))
            f.write(struct.pack("<ddd", *xyz))
            f.write(struct.pack("<BBB", *(int(c) for c in rgb)))
            f.write(struct.pack("<d", float(err)))
            f.write(struct.pack("<Q", len(track) // 8))
            f.write(track)
