import struct, sys, csv, numpy as np
from pathlib import Path

def read_next_bytes(f, n, fmt):
    return struct.unpack("<" + fmt, f.read(n))

def read_cameras_bin(p):
    cams = {}
    MODEL_PARAMS = {0:3,1:4,2:4,3:5,4:8,5:8,6:12,7:5,8:4,9:5,10:12}
    MODEL_NAMES = {0:"SIMPLE_PINHOLE",1:"PINHOLE",2:"SIMPLE_RADIAL",3:"RADIAL",4:"OPENCV",5:"OPENCV_FISHEYE",6:"FULL_OPENCV",7:"FOV",8:"SIMPLE_RADIAL_FISHEYE",9:"RADIAL_FISHEYE",10:"THIN_PRISM_FISHEYE"}
    with open(p, "rb") as f:
        n = read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            cid, model, w, h = read_next_bytes(f, 24, "iiQQ")
            params = read_next_bytes(f, 8*MODEL_PARAMS[model], "d"*MODEL_PARAMS[model])
            cams[cid] = (MODEL_NAMES[model], w, h, params)
    return cams

def read_images_bin(p):
    imgs = {}
    with open(p, "rb") as f:
        n = read_next_bytes(f, 8, "Q")[0]
        for _ in range(n):
            iid = read_next_bytes(f, 4, "i")[0]
            q = read_next_bytes(f, 32, "dddd")
            t = read_next_bytes(f, 24, "ddd")
            cam_id = read_next_bytes(f, 4, "i")[0]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00": break
                name += c
            npts = read_next_bytes(f, 8, "Q")[0]
            f.read(24 * npts)
            imgs[name.decode()] = (np.array(q), np.array(t), cam_id)
    return imgs

def read_points3D_count(p):
    with open(p, "rb") as f:
        n = read_next_bytes(f, 8, "Q")[0]
        # read first point's xyz for sanity, and sample extent
        pts = []
        for i in range(n):
            pid = read_next_bytes(f, 8, "Q")[0]
            xyz = read_next_bytes(f, 24, "ddd")
            rgb = f.read(3)
            err = read_next_bytes(f, 8, "d")[0]
            tl = read_next_bytes(f, 8, "Q")[0]
            f.read(8 * tl)
            if i % max(1, n // 2000) == 0:
                pts.append(xyz)
        return n, np.array(pts)

scene = Path(sys.argv[1])
cams = read_cameras_bin(scene/"train/sparse/0/cameras.bin")
imgs = read_images_bin(scene/"train/sparse/0/images.bin")
npts, sample = read_points3D_count(scene/"train/sparse/0/points3D.bin")
print(f"scene={scene.name}")
print(f"cameras: {cams}")
print(f"registered train images: {len(imgs)}")
print(f"points3D: {npts}, extent p1-p99: {np.percentile(sample,1,axis=0).round(2)} .. {np.percentile(sample,99,axis=0).round(2)}")

# camera centers from train (COLMAP w2c: C = -R^T t)
def qvec2rot(q):
    w,x,y,z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])
train_C = np.array([-qvec2rot(q).T @ t for q,(t) in [(v[0], v[1]) for v in imgs.values()]])
train_T = np.array([v[1] for v in imgs.values()])
print(f"train camera centers (=-R^T t) range: {train_C.min(0).round(2)} .. {train_C.max(0).round(2)}")
print(f"train raw tvec range:               {train_T.min(0).round(2)} .. {train_T.max(0).round(2)}")

# test poses
rows = list(csv.DictReader(open(scene/"test/test_poses.csv")))
tq = np.array([[float(r[k]) for k in ("qw","qx","qy","qz")] for r in rows])
tt = np.array([[float(r[k]) for k in ("tx","ty","tz")] for r in rows])
test_C = np.array([-qvec2rot(q).T @ t for q,t in zip(tq,tt)])
print(f"test raw tvec range:      {tt.min(0).round(2)} .. {tt.max(0).round(2)}")
print(f"test as-w2c centers range:{test_C.min(0).round(2)} .. {test_C.max(0).round(2)}")
print(f"fx,fy,cx,cy,w,h of row0: {rows[0]['fx']}, {rows[0]['fy']}, {rows[0]['cx']}, {rows[0]['cy']}, {rows[0]['width']}, {rows[0]['height']}")
