#!/usr/bin/env python3
"""G6 — Affine color transfer theo pose (L6-lite, không retrain).

Với mỗi test pose: tìm train view gần nhất (vị trí + hướng nhìn), fit affine màu
per-channel (a·x+b) giữa render(train_NN) → photo(train_NN), áp lên render(test).
Nếu --net: áp L16 net lên render train trước khi fit (để G6 chồng lên L16).

Dùng:
  python affine_transfer.py --workspace workspace_raw/SC \
      --renders_train results/SC__l16/renders_train \
      --test_csv .../test_poses.csv --in_dir renders/SC__l16 \
      --out_dir renders/SC__g6 [--net results/SC__l16/net.pt] [--topk 1]
"""
import argparse, csv, sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from colmap_io import read_images_bin


def q2R(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True)
    ap.add_argument("--renders_train", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--net", default=None, help="L16 net áp lên render train trước khi fit")
    ap.add_argument("--topk", type=int, default=1)
    a = ap.parse_args()

    ws = Path(a.workspace)
    imgs = read_images_bin(ws / "sparse/0/images.bin")
    items = sorted(imgs.values(), key=lambda im: im.name)
    trC = np.array([-im.R.T @ im.tvec for im in items])
    trD = np.array([im.R[2] for im in items])

    net = None
    if a.net:
        import torch
        import torch.nn.functional as Fn
        from enhance_net import UNetSmall
        ck = torch.load(a.net, map_location="cuda", weights_only=True)
        net = UNetSmall(ch_mult=ck.get("ch_mult", 1)).cuda().eval()
        net.load_state_dict(ck["state"])

        def enhance(bgr):
            with torch.no_grad():
                x = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).float().permute(2,0,1)[None].cuda()/255
                h, w = x.shape[-2:]
                ph, pw = (4-h%4)%4, (4-w%4)%4
                y = net(Fn.pad(x, (0,pw,0,ph), mode="replicate"))[..., :h, :w]
                return cv2.cvtColor((y[0].clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    spacing = None
    rows = list(csv.DictReader(open(a.test_csv)))
    for r in rows:
        q = [float(r[k]) for k in ("qw","qx","qy","qz")]
        t = np.array([float(r[k]) for k in ("tx","ty","tz")])
        R = q2R(q); C = -R.T @ t; D = R[2]
        dpos = np.linalg.norm(trC - C, axis=1)
        if spacing is None:
            spacing = np.median(np.sort(np.linalg.norm(trC[:,None]-trC[None], axis=2), axis=1)[:,1])
        dang = np.arccos(np.clip(trD @ D, -1, 1))
        d = dpos/spacing + 0.5*dang
        nn = np.argsort(d)[:a.topk]
        # fit affine per-channel trên (render_trainNN → photo_trainNN), downsample 4x
        A, B = np.ones(3), np.zeros(3)
        xs, ys = [], []
        for j in nn:
            im = items[j]
            rt = cv2.imread(str(Path(a.renders_train)/(Path(im.name).stem+".png")))
            if rt is None: continue
            if net is not None: rt = enhance(rt)
            ph = cv2.imread(str(ws/"images"/im.name))
            xs.append(rt[::4,::4].reshape(-1,3).astype(np.float64))
            ys.append(ph[::4,::4].reshape(-1,3).astype(np.float64))
        if xs:
            X = np.concatenate(xs); Y = np.concatenate(ys)
            for c in range(3):
                A[c], B[c] = np.polyfit(X[:,c], Y[:,c], 1)
        te = cv2.imread(str(Path(a.in_dir)/(Path(r["image_name"]).stem+".png"))).astype(np.float64)
        te = np.clip(te*A + B, 0, 255).astype(np.uint8)
        cv2.imwrite(str(out/(Path(r["image_name"]).stem+".png")), te)
    print(f"G6 applied {len(rows)} ảnh → {out}")


if __name__ == "__main__":
    main()
