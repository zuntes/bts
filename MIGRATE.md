# SERVER L40S — trạng thái & cách làm việc

> Migration ĐÃ XONG (15/07/2026). File này là nguồn sự thật về server.
> Lịch sử kế hoạch migrate cũ: xem git log của file này.

## 0. MÔ HÌNH LÀM VIỆC (đọc trước tiên — quyết định mọi thứ khác)

**Claude chạy trên máy LOCAL (RTX 4060), KHÔNG có trên server L40S.** Vòng lặp:

```
Claude sửa code + COMMIT LOCAL  →  [bạn] git push  →  [bạn] git pull trên server
                                                             ↓
Claude đọc & sửa tiếp  ←  bạn DÁN OUTPUT  ←  [bạn] chạy script
```

⚠ **Claude KHÔNG push.** Repo `zuntes/bts` là **public** — Claude chỉ `git commit` ở local,
**bạn tự `git push`**. Lý do: repo chứa `experiments.csv` (thang đo BTC đã giải mã, config
thắng, điểm từng sub) + `docs/` (sổ phương án) = toàn bộ playbook thi đấu. Bạn giữ quyền
quyết định cái gì lên internet.

Hệ quả bắt buộc, đừng phá:
- **Mọi script phải TỰ CHẨN ĐOÁN và in kết luận rõ ràng.** Claude không nhìn thấy server —
  chỉ thấy đúng cái bạn dán về. Script kiểu "chạy im lặng rồi lỗi mơ hồ" là vô dụng.
- **Kiểm FILE THẬT, không kiểm "thư mục có tồn tại".** Đã trả giá: rsync dở dang và
  `prepare` chết giữa chừng đều để lại thư mục rỗng → bước sau tưởng xong → train chết.
- **Fail sớm, trước khi đụng GPU.** Lỗi lộ ra sau 30-60 phút build hoặc vài giờ train là đắt.
- Code chỉ sửa ở local rồi push. **Đừng sửa tay trên server** — lần pull sau là mất.

## 1. TOẠ ĐỘ

| | Local (có Claude) | Server L40S (chỉ chạy) |
|---|---|---|
| Host/user | máy 4060 của bạn | `pcn-robot@500308175-GPU-02` |
| Project | `/home/vt02/BTS Image Reconstruction` (có dấu cách) | `~/bts` (**không dấu cách** — 3DGRUT vỡ nếu có) |
| Repo | `github.com/zuntes/bts` (origin) | clone từ cùng repo |
| NHT | `~/3dgrut` | `~/3dgrut` (ngoài repo, không dính git) |

## 2. MÔI TRƯỜNG SERVER (đã dựng & kiểm chứng 15/07)

**Phần cứng**: 2× L40S 46GB · driver 590.48.01 (CUDA 13.1) · Ubuntu 22.04.5 · RAM 94GB · đĩa 107GB trống.
- ⚠ **GPU 1 bị VLLM chiếm 44GB** → **chỉ dùng GPU 0** (trống ~44.4GB): `export CUDA_VISIBLE_DEVICES=0`.
- L40S = Ada **sm_89**, GIỐNG RTX 4060 → wheel/`TORCH_CUDA_ARCH_LIST="8.9"` dùng chung được.

**Toolchain**: nvcc **12.4** (`/usr/local/cuda` → symlink `cuda-12.4`, khớp CUDA_HOME) · gcc 10.5 · uv 0.11.x (`~/.local/bin`).

**HAI .venv RIÊNG — KHÔNG TRỘN** (torch khác version):

| | gsplat pipeline | NHT / 3DGRUT |
|---|---|---|
| Đường dẫn | `~/bts/.venv` | `~/3dgrut/.venv` |
| Python | 3.10.12 | 3.11 |
| torch | 2.4.1+cu124 | 2.6.0+cu124 |
| Chính | `gsplat 1.5.3+pt24cu124` (wheel dựng sẵn), fused-ssim ✅ | kaolin ✅, tinycudann ✅ |

⚠ **`python3` mặc định trên server = Python 3.14.6 (conda base)** — KHÔNG dùng được:
torch/gsplat pin **cp310**. Script luôn gọi `python3.10` hoặc `.venv/bin/python` tuyệt đối.

⚠ **`conda deactivate` BẮT BUỘC trước khi build 3DGRUT**: `install_env_uv.sh:150` là
`if [ -z "$CONDA_PREFIX" ]` → còn conda thì nó BỎ QUA việc tạo .venv py3.11 và đổ vào
conda base py3.14 → hỏng, và chỉ lộ ra sau 30-60 phút. `BUILD_NHT.sh` đã chặn cứng.

## 3. GIT — KHÁC KẾ HOẠCH CŨ, ĐỌC KỸ

- **gsplat đã patch nằm THẲNG trong repo** (`.git`/`.gitignore`/`.gitmodules`/`.github` của
  upstream đã xoá). → **KHÔNG `git apply`, KHÔNG `git clone gsplat`, KHÔNG `cd gsplat && git branch`**
  (không phải repo nữa — báo lỗi là bình thường). Patch có sẵn: `raw_distortion`,
  `global_seed`, fused-ssim fallback, T3-flip, L14.
- ⚠ **`gsplat/gsplat/cuda/csrc/third_party/glm` RỖNG** (submodule chưa từng init).
  → **TUYỆT ĐỐI KHÔNG `pip install -e gsplat/`** (build from source sẽ chết vì thiếu header).
  Ta dùng wheel dựng sẵn; clone chỉ để chạy `examples/simple_trainer.py`. Không cần sửa.
- Git **KHÔNG** mang: `VAI_NVS_DATA/` (3.2GB), `.venv/`, `workspace*/`, `renders/`,
  `results/*` (trừ `experiments.csv`), `*.zip/*.pt/*.ckpt`, `~/3dgrut`.
- Data chuyển bằng rsync:
  ```bash
  rsync -avP "/home/vt02/BTS Image Reconstruction/VAI_NVS_DATA/" \
    pcn-robot@<host>:/home/pcn-robot/bts/VAI_NVS_DATA/
  ```

## 4. SCRIPT BÀN GIAO (chạy trên server, đều tự chẩn đoán + in kết luận)

```bash
bash tools/VERIFY_SERVER.sh   # chỉ đọc, ~10s — soi toàn bộ máy, in "việc tiếp theo"
bash tools/SETUP_NVCC.sh      # CUDA 12.4 (cần sudo) — ĐÃ CHẠY, không cần lại
bash tools/SETUP_MAIN_VENV.sh # .venv gsplat py3.10 — ĐÃ CHẠY, không cần lại
bash tools/BUILD_NHT.sh       # 3DGRUT+NHT (conda deactivate trước) — ĐÃ CHẠY, không cần lại
bash tools/SMOKE_SERVER.sh    # validate chain, ~5-10ph, mốc PSNR ≈14.9
```

## 5. TRẠNG THÁI (15/07/2026)

**✅ XONG — không cần cài gì thêm:**
- .venv gsplat + fused-ssim · nvcc 12.4 · NHT/3DGRUT (torch 2.6, kaolin, tinycudann)
- symlink header tcnn đã vá (nếu thiếu → lỗi `vector_types.h not found` GIỮA lúc train)
- `render_comp.py` (cầu render competition-pose) đã copy vào `~/3dgrut/`
- **3DGUT KHÔNG phải cài gì** — chỉ là flag gsplat: `--with-ut --with-eval3d --raw-distortion`
- **Chain đã kiểm chứng end-to-end**: SMOKE HCM0204 500 steps → **PSNR 14.930** vs mốc 4060
  **14.9** → prepare/undistort/T3-normalize/render-pose/score đều đúng trên L40S.

**⬜ CÒN LẠI (chuẩn bị dữ liệu, KHÔNG phải cài đặt):**
1. Xác nhận rsync đủ **13 scene** (smoke mới chỉ chứng minh HCM0204):
   `ls VAI_NVS_DATA/phase1/public_set VAI_NVS_DATA/phase1/private_set1`
2. Dựng workspace cho các scene cần dùng, **2 biến thể**:
   ```bash
   .venv/bin/python tools/prepare_scene.py --scene_dir VAI_NVS_DATA/phase1/<set>/<S> --out_dir workspace/<S>
   .venv/bin/python tools/prepare_scene.py --scene_dir VAI_NVS_DATA/phase1/<set>/<S> --out_dir workspace_raw/<S> --keep_distortion
   ```
   (`workspace/` = undistort → classic+redistort · `workspace_raw/` = giữ méo → 3DGUT/NHT)

**Việc thực nghiệm tiếp theo** — xem `results/experiments.csv` + §E lịch sử:
NHT@1M competition v50 = 0.72841 < standard 5M = 0.74727 vì **capacity-limited trên 4060**
(NHT thắng standard cùng cap 1M cả 3 metric). L40S 44GB mở được **NHT cap 4-6M** — đó là
canh bạc chính. Điểm BTC tốt nhất hiện tại: **SUB5 = 76.9654** (5M×2seed ensemble + L16-XL).
