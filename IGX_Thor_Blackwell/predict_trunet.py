#!/usr/bin/env python3
"""
predict_trunet.py
=================
Run inference with a TRUNet checkpoint trained by train_trunet_cardiac.py, with
a config that MATCHES training (11 classes, 128³, HU-window [-1000,1000]→[0,1]).

Use this instead of the repo's segment_file.py, which cannot run an 11-class /
128³ model as-is:
  * it passes num_classes= to VisionTransformer3d, which takes no such arg (TypeError);
  * it hard-codes numClasses=7 and img_size=224 → state_dict shape mismatch;
  * it feeds RAW Hounsfield units (no normalization) → distribution mismatch vs training.

Input : a NIfTI CT volume (.nii/.nii.gz).
Output: an 11-class label map NIfTI on the ORIGINAL grid (same shape + affine).

Example
-------
python predict_trunet.py \
    --input  /data/new_scans/patient001.nii.gz \
    --checkpoint ./runs/thor_run1/best_metric_model.pth \
    --trunet-root ./TRUNet-main \
    --num-classes 11 --img-size 128 \
    --output /data/new_scans/patient001_seg.nii.gz
"""

import argparse
import os
import sys

import ml_collections
import numpy as np
import torch

try:
    import nibabel as nib
except ImportError:
    sys.exit("[ERROR] nibabel required:  pip install nibabel")
try:
    from scipy.ndimage import zoom
except ImportError:
    sys.exit("[ERROR] scipy required:  pip install scipy")


def build_config(img_size, num_classes):
    c = ml_collections.ConfigDict()
    c.resnet = ml_collections.ConfigDict()
    c.resnet.num_layers = (3, 4, 9)
    c.resnet.width_factor = 1
    c.transformer_mlp_dim = 3072
    c.transformer_num_heads = 12
    c.transformer_num_layers = 12
    c.transformer_attention_dropout_rate = 0.0
    c.transformer_dropout_rate = 0.1
    c.classifier = "seg"
    c.decoder_channels = (256, 128, 64, 16)
    c.n_classes = num_classes                       # <- output channels
    c.n_skip = 3
    c.skip_channels = [512, 256, 64, 16]
    c.activation = "softmax"
    c.patches = ml_collections.ConfigDict()
    c.hidden_size = 768
    c.patches.size = 16
    c.patch_size = c.patches.size
    g = int(img_size / c.patches.size)
    c.patches.grid = (g, g, g)
    c.hybrid = True
    return c


def pick_device(preferred=None):
    if not torch.cuda.is_available():
        print("[pred] CUDA not available — using CPU")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if preferred is not None and 0 <= preferred < n:
        idx = preferred
    else:
        disc = [i for i in range(n)
                if ("rtx" in torch.cuda.get_device_name(i).lower()
                    or "6000" in torch.cuda.get_device_name(i).lower())
                and "thor" not in torch.cuda.get_device_name(i).lower()]
        idx = disc[0] if disc else 0
    print(f"[pred] using cuda:{idx}  {torch.cuda.get_device_name(idx)}")
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def main():
    ap = argparse.ArgumentParser(description="TRUNet inference (config matched to training)")
    ap.add_argument("--input", required=True, help="Input CT NIfTI (.nii/.nii.gz)")
    ap.add_argument("--checkpoint", required=True, help="Trained .pth (e.g. best_metric_model.pth)")
    ap.add_argument("--output", default=None, help="Output label NIfTI [<input>_seg.nii.gz]")
    ap.add_argument("--trunet-root", default="./TRUNet-main", help="Folder containing TRUNet_network/")
    ap.add_argument("--num-classes", type=int, default=11, help="Must match training [11]")
    ap.add_argument("--img-size", type=int, default=128, help="Must match training [128]")
    ap.add_argument("--hu-min", type=float, default=-1000.0)
    ap.add_argument("--hu-max", type=float, default=1000.0)
    ap.add_argument("--gpu-index", type=int, default=None)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    a = ap.parse_args()

    if not os.path.isfile(a.input):
        sys.exit(f"[ERROR] input not found: {a.input}")
    root = os.path.abspath(a.trunet_root)
    if not os.path.isdir(os.path.join(root, "TRUNet_network")):
        # allow ../TRUNet-main fallback
        alt = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TRUNet-main"))
        root = alt if os.path.isdir(os.path.join(alt, "TRUNet_network")) else root
    if not os.path.isdir(os.path.join(root, "TRUNet_network")):
        sys.exit(f"[ERROR] TRUNet_network/ not found under {root}; pass --trunet-root")
    sys.path.insert(0, root)
    from TRUNet_network.model.ViT import VisionTransformer3d

    out_path = a.output or (a.input.replace(".nii.gz", "").replace(".nii", "") + "_seg.nii.gz")
    device = pick_device(a.gpu_index)

    # ---- load + preprocess EXACTLY as training did ----
    img = nib.load(a.input)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    orig_shape = vol.shape
    print(f"[pred] input {a.input}  shape {orig_shape}")
    vol = np.clip(vol, a.hu_min, a.hu_max)
    vol = (vol - a.hu_min) / float(a.hu_max - a.hu_min)     # → [0,1]
    S = a.img_size
    x, y, z = orig_shape
    vol_rs = zoom(vol, (S / x, S / y, S / z), order=1) if orig_shape != (S, S, S) else vol

    # ---- model ----
    cfg = build_config(S, a.num_classes)
    model = VisionTransformer3d(cfg, img_size=S, zero_head=False, vis=False)
    state = torch.load(a.checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[pred] WARNING {len(missing)} missing keys (first: {missing[:3]})")
    if unexpected:
        print(f"[pred] WARNING {len(unexpected)} unexpected keys (first: {unexpected[:3]})")
    model = model.to(device).eval()

    on_cuda = device.type == "cuda"
    if a.precision == "bf16" and on_cuda:
        ctx = torch.autocast("cuda", dtype=torch.bfloat16)
    elif a.precision == "fp16" and on_cuda:
        ctx = torch.autocast("cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    # ---- inference ----
    xt = torch.from_numpy(vol_rs.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad(), ctx:
        logits = model(xt)
        pred = torch.argmax(logits, dim=1).squeeze(0).to(torch.uint8).cpu().numpy()

    # ---- resample label back to original grid (nearest) ----
    if pred.shape != orig_shape:
        pred = np.rint(zoom(pred.astype(np.float32),
                            (x / S, y / S, z / S), order=0)).astype(np.uint8)
    counts = {int(c): int((pred == c).sum()) for c in np.unique(pred)}
    print(f"[pred] label voxel counts: {counts}")

    nib.save(nib.Nifti1Image(pred, img.affine, img.header), out_path)
    print(f"[pred] wrote {out_path}  shape {pred.shape}")


if __name__ == "__main__":
    main()
