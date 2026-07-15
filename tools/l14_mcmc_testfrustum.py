"""L14 (docs/09 §3) — MCMCStrategy hướng test-frustum. TRẠNG THÁI: Bậc 4, CHƯA tích hợp.

Ý tưởng: cap_max của MCMC là ngân sách hữu hạn. Định kỳ ép giảm opacity các gaussians
nằm ngoài MỌI test frustum → chúng "chết" → cơ chế relocation của MCMC dời ngân sách
vào vùng được chấm điểm. LƯU Ý: chỉ có tác dụng GIỮA training (khi relocation còn chạy);
prune sau khi train xong là vô nghĩa với score (docs/09 §3 L14).

Cách tích hợp (khi tới Bậc 4): trong gsplat examples/simple_trainer.py, thay
`MCMCStrategy(...)` bằng `TestFrustumMCMC(..., test_viewmats=..., test_Ks=..., wh=...)`
(load từ test_poses.csv bằng tools/colmap_io + qvec2rotmat của render_test_poses).
A/B bắt buộc: rủi ro là vùng ngoài test-frustum vẫn xuất hiện trong ẢNH TRAIN,
ép chết chúng làm train loss nhiễu → decay phải nhẹ (0.995) và không đụng gaussian α cao.
"""
import torch

try:
    from gsplat.strategy import MCMCStrategy
except ImportError:  # cho phép đọc/compile file khi chưa cài gsplat
    MCMCStrategy = object


def visible_from_any(means, viewmats, Ks, width, height, margin=0.1):
    """Mask [N] — gaussian center nằm trong ÍT NHẤT một test frustum (nới margin 10%)."""
    N = means.shape[0]
    vis = torch.zeros(N, dtype=torch.bool, device=means.device)
    ones = torch.ones(N, 1, device=means.device)
    homo = torch.cat([means, ones], dim=1)  # [N,4]
    for vm, K in zip(viewmats, Ks):
        cam = (vm.to(means.device) @ homo.T).T[:, :3]
        z = cam[:, 2]
        front = z > 0.01
        uv = (K.to(means.device) @ cam.T).T
        u, v = uv[:, 0] / z.clamp(min=1e-6), uv[:, 1] / z.clamp(min=1e-6)
        inside = front & (u > -margin * width) & (u < (1 + margin) * width) \
                       & (v > -margin * height) & (v < (1 + margin) * height)
        vis |= inside
        if vis.all():
            break
    return vis


class TestFrustumMCMC(MCMCStrategy):
    """MCMC + ép opacity decay cho gaussians vô hình từ mọi test pose."""

    def __init__(self, *args, test_viewmats=None, test_Ks=None, wh=(1320, 989),
                 decay=0.995, every=500, alpha_protect=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_viewmats = test_viewmats
        self.test_Ks = test_Ks
        self.wh = wh
        self.decay = decay
        self.every = every
        self.alpha_protect = alpha_protect  # không đụng gaussian đã chắc chắn

    def step_post_backward(self, params, optimizers, state, step, info, lr):
        super().step_post_backward(params, optimizers, state, step, info, lr)
        if step % self.every != 0 or self.test_viewmats is None:
            return
        with torch.no_grad():
            vis = visible_from_any(params["means"], self.test_viewmats,
                                   self.test_Ks, *self.wh)
            alpha = torch.sigmoid(params["opacities"])
            target = (~vis) & (alpha < self.alpha_protect)
            if target.any():
                new_alpha = (alpha[target] * self.decay).clamp(1e-4, 1 - 1e-4)
                params["opacities"][target] = torch.log(new_alpha / (1 - new_alpha))
