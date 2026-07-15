#!/usr/bin/env python3
"""Render NHT model tại competition poses (eval-workspace đã pre-transform T240).
Sửa config: normalize=false + all-test, rồi dùng Renderer.render_all.
Usage: python render_comp.py <ckpt> <eval_ws> <out_dir>
"""
import sys, torch
from pathlib import Path
from omegaconf import OmegaConf, open_dict

ckpt_path, eval_ws, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]

# 1) sửa config trong checkpoint
ckpt = torch.load(ckpt_path, weights_only=False)
conf = ckpt["config"]
with open_dict(conf):
    conf.dataset.normalize_world_space = False   # pose đã ở frame model (pre-transform T240)
    conf.dataset.test_split_interval = 1         # mọi frame = val (render hết 60)
    conf.path = eval_ws
tmp = Path(ckpt_path).parent / "ckpt_comp_tmp.pt"
ckpt["config"] = conf
torch.save(ckpt, tmp)
print(f"temp ckpt: {tmp} (normalize=false, split=1)")

# 2) render bằng Renderer.render_all
from threedgrut.render import Renderer
renderer = Renderer.from_checkpoint(
    checkpoint_path=str(tmp), path=eval_ws, out_dir=out_dir,
    save_gt=True, computes_extra_metrics=True)
renderer.render_all()
print("RENDER-COMP-DONE")
