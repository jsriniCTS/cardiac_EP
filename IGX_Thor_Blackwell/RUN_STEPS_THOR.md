# Pre-training TRUNet on cardiac CT — NVIDIA IGX Thor / Blackwell edition

Same task as the parent folder (train [TRUNet](https://github.com/ljollans/TRUNet)
on the [STACOM2025 Public Cardiac CT Dataset](https://github.com/Bjonze/Public-Cardiac-CT-Dataset)),
tuned for:

| | |
|---|---|
| **Platform** | NVIDIA IGX Thor (aarch64 / Grace-class CPU) |
| **OS** | Ubuntu 24.04.4 LTS |
| **GPU** | RTX PRO 6000 Blackwell Max-Q — sm_120, 96 GB VRAM |

## What's different from the macOS/CPU version in `../`

| Concern | macOS/CPU version | This (Thor/Blackwell) version |
|---|---|---|
| PyTorch | any (repo's torch==2.0.1 ok on CPU) | **must be CUDA 12.8 / cu128, torch ≥ 2.7** — 2.0.1 has no sm_120 kernels |
| Arch | x86/arm64 CPU | **aarch64 + Blackwell dGPU** |
| Training loop | repo fp32 loop (fork workaround) | **built-in bf16 AMP loop** + TF32 + cuDNN autotune + `channels_last_3d` |
| Defaults | `img_size 96`, batch 2 | **`img_size 128`, batch 2** (uses the 96 GB) |
| Diagnostics | — | prints GPU name / `sm_120` / VRAM and warns if torch can't target Blackwell |

The data-prep logic and the model itself are identical.

---

## Step 0 — One-command environment setup

```bash
cd IGX_Thor_Blackwell
bash setup_thor.sh
source .venv/bin/activate
```

`setup_thor.sh` will:
1. check `nvidia-smi` (tells you to install the **570+ "open"** driver if missing),
2. create `.venv`,
3. install **PyTorch cu128** (stable, falling back to nightly) — the only
   Blackwell-capable option,
4. install `requirements_thor.txt`,
5. verify on-device that `sm_120` kernels exist and a matmul runs on the GPU.

> **Why not the repo's `requirements.txt`?** It pins `torch==2.0.1`, which has no
> Blackwell kernels — you'd get `CUDA error: no kernel image is available for
> execution on the device`. Always install torch from the cu128 index on this box.

Manual equivalent, if you prefer:
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -U pip wheel
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements_thor.txt
```

---

## Step 1 — Get the data (same as parent)

- **Images (ImageCAS CCTA):** <https://www.kaggle.com/datasets/xiaoweixumedicalai/imagecas> → `1.img.nii.gz`, …
- **Labels (STACOM2025, 576 MB):**
  ```bash
  curl -L -o ImageCAS-STACOM2025.zip \
    https://people.compute.dtu.dk/rapa/STACOM2025/ImageCAS-STACOM2025-02-10-2025.zip
  unzip ImageCAS-STACOM2025.zip -d stacom_labels
  ```
Image ↔ label are matched by the leading integer id (`1.img.nii.gz` ↔ `1.nii.gz`).

---

## Step 2 — Preprocess to `.npz` (default 128³)

```bash
python preprocess_to_npz.py \
  --images-dir /data/ImageCAS/images \
  --labels-dir /data/stacom_labels \
  --out-root   /data/trunet_cardiac \
  --size 128 --val-frac 0.15
```

Keep `--size` equal to `--img-size` in Step 3. The 96 GB GPU handles 128³
easily; 160³ is also feasible (must stay divisible by 16).

---

## Step 3 — Train (bf16, GPU)

```bash
python train_trunet_cardiac.py \
  --root-path /data/trunet_cardiac \
  --trunet-root ../TRUNet-main \
  --num-classes 11 \
  --img-size 128 \
  --batch-size 2 \
  --precision bf16 \
  --max-epochs 100 \
  --num-workers 8 \
  --save-path ./runs/thor_run1
```

Startup prints a Blackwell check, e.g.:
```
[thor] GPU        : NVIDIA RTX PRO 6000 Blackwell Max-Q ...
[thor] capability : sm_120   VRAM: 96 GB
[thor] torch      : 2.7.x   CUDA: 12.8
[thor] precision  : bf16
```

Outputs in `./runs/thor_run1/`: `best_metric_model.pth`, periodic `epoch_<n>.pth`,
`log.txt`, and TensorBoard events under `log/`.
```bash
tensorboard --logdir ./runs/thor_run1/log
```

Key flags:
- `--precision {bf16,fp16,fp32}` — **bf16** is native to Blackwell (no loss
  scaling, best stability). fp16 uses a GradScaler; fp32 for debugging.
- `--batch-size` — 2 is conservative for 128³; with 96 GB you can often push
  to 4–8. If you ever OOM, lower this first, then `--img-size`.
- `--num-workers` — 8 is a good start on the Grace CPU; raise if the GPU is
  data-starved (watch `nvidia-smi` utilization).
- `--repo-trainer` — use the repo's exact fp32 loop instead of the AMP loop
  (parity/debugging; slower, no bf16).
- `--checkpoint path.pth` — resume or fine-tune.

---

## Step 4 — Quick synthetic smoke test (no dataset, no GPU needed)

Verifies the full chain in seconds; runs on GPU if present, else CPU/fp32:
```bash
python - <<'PY'
import numpy as np, nibabel as nib, os
for i in (1,2,3,7):
    os.makedirs('smoke/img',exist_ok=True); os.makedirs('smoke/lab',exist_ok=True)
    nib.save(nib.Nifti1Image((np.random.rand(60,64,50)*2000-1000).astype('float32'),np.eye(4)), f'smoke/img/{i}.img.nii.gz')
    nib.save(nib.Nifti1Image(np.random.randint(0,11,(60,64,50)).astype('uint8'),np.eye(4)), f'smoke/lab/{i}.nii.gz')
PY
python preprocess_to_npz.py --images-dir smoke/img --labels-dir smoke/lab --out-root smoke/npz --size 64 --val-frac 0.25
python train_trunet_cardiac.py --root-path smoke/npz --trunet-root ../TRUNet-main \
  --num-classes 11 --img-size 64 --batch-size 1 --max-epochs 1 --num-workers 0 --save-path smoke/run
```
Expect `Training Finished!` and a `best_metric_model.pth`.

---

## Step 5 — Inference

Same as parent: copy your best checkpoint to `../TRUNet-main/models/trunet_model.pth`
and run `segment_file.py` (edit its hard-coded `numClasses=7` and `img_size=224`
to match what you trained — `11` and `128`). On this GPU use `--gpu`:
```bash
cd ../TRUNet-main && python segment_file.py /path/to/scan.nii.gz --gpu
```

---

## Blackwell / IGX Thor troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available for execution on the device` | Wrong torch — reinstall from the **cu128** index (`setup_thor.sh`). torch 2.0.1 / cu11x / cu121 do **not** have sm_120 kernels. |
| `torch.cuda.is_available()` is False | NVIDIA **570+ open** driver not installed / needs reboot; check `nvidia-smi`. |
| `arch_list` has no `sm_120` | You installed a non-cu128 wheel; use the nightly cu128 line in `setup_thor.sh`. |
| pip installs a CPU-only torch | You're on aarch64 and pip fell back — force the cu128 `--index-url` explicitly. |
| bf16 NaNs / unstable | bf16 rarely needs it, but try `--precision fp32` to confirm it's precision-related. |
| GPU under-utilized (`nvidia-smi` low) | Raise `--num-workers`, `--batch-size`; ensure data is on a fast NVMe. |
| OOM at 128³ | Lower `--batch-size` to 1, then `--img-size`/preprocess `--size` to 96. |
