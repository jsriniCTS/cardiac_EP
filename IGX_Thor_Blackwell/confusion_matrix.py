#!/usr/bin/env python3
"""
confusion_matrix.py
===================
Evaluate a trained TRUNet checkpoint on a preprocessed split and produce:
  * an N x N voxel confusion matrix (rows = ground truth, cols = prediction)
  * per-class Dice / IoU / precision / recall derived from it
  * a heatmap PNG (row-normalized %) and two CSVs (raw counts + metrics)

The confusion matrix aggregates every voxel of every volume in the split, so
element [i, j] = number of voxels whose true class is i and predicted class j.

Example
-------
python confusion_matrix.py \
    --root-path ./data/trunet_cardiac \
    --checkpoint ./runs/thor_run1/best_metric_model.pth \
    --trunet-root ./TRUNet-main \
    --num-classes 11 --img-size 128 --split val \
    --out ./runs/thor_run1/confusion
"""

import argparse
import os
import sys
from glob import glob

import ml_collections
import numpy as np
import torch

# STACOM2025 label names (index = class id)
CLASS_NAMES = ["background", "myocardium", "LA", "LV", "RA", "RV",
               "aorta", "PA", "LAA", "coronary", "PV"]


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
    c.n_classes = num_classes
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
        print("[cm] CUDA not available — using CPU")
        return torch.device("cpu")
    n = torch.cuda.device_count()
    if preferred is not None and 0 <= preferred < n:
        idx = preferred
    else:
        discrete = [i for i in range(n)
                    if ("rtx" in torch.cuda.get_device_name(i).lower()
                        or "6000" in torch.cuda.get_device_name(i).lower())
                    and "thor" not in torch.cuda.get_device_name(i).lower()]
        idx = discrete[0] if discrete else 0
    print(f"[cm] using cuda:{idx}  {torch.cuda.get_device_name(idx)}")
    torch.cuda.set_device(idx)
    return torch.device(f"cuda:{idx}")


def load_model(trunet_root, img_size, num_classes, checkpoint, device):
    sys.path.insert(0, os.path.abspath(trunet_root))
    from TRUNet_network.model.ViT import VisionTransformer3d
    model = VisionTransformer3d(build_config(img_size, num_classes),
                                img_size=img_size, zero_head=False, vis=False)
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    # tolerate checkpoints saved from a compiled/DDP-wrapped model
    state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[cm] WARNING: {len(missing)} missing keys (first: {missing[:3]})")
    if unexpected:
        print(f"[cm] WARNING: {len(unexpected)} unexpected keys (first: {unexpected[:3]})")
    return model.to(device).eval()


def metrics_from_cm(cm):
    """Return per-class dict from confusion matrix (rows=true, cols=pred)."""
    tp = np.diag(cm).astype(np.float64)
    row = cm.sum(1).astype(np.float64)   # true totals
    col = cm.sum(0).astype(np.float64)   # predicted totals
    fp = col - tp
    fn = row - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        recall = tp / row
        precision = tp / col
    return dice, iou, precision, recall


def main():
    ap = argparse.ArgumentParser(description="Confusion matrix for a trained TRUNet checkpoint")
    ap.add_argument("--root-path", required=True, help="Folder with train/ and val/ npz dirs")
    ap.add_argument("--checkpoint", required=True, help="Path to .pth (e.g. best_metric_model.pth)")
    ap.add_argument("--trunet-root", default="./TRUNet-main", help="Folder containing TRUNet_network/")
    ap.add_argument("--num-classes", type=int, default=11)
    ap.add_argument("--img-size", type=int, default=128, help="Must match training img-size")
    ap.add_argument("--split", default="val", choices=["val", "train"], help="Which split to evaluate")
    ap.add_argument("--gpu-index", type=int, default=None)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--out", default=None, help="Output prefix [<checkpoint-dir>/confusion]")
    a = ap.parse_args()

    split_dir = os.path.join(a.root_path, a.split)
    files = sorted(glob(os.path.join(split_dir, "*.npz")))
    if not files:
        sys.exit(f"[ERROR] no .npz files in {split_dir}")
    out_prefix = a.out or os.path.join(os.path.dirname(os.path.abspath(a.checkpoint)), "confusion")
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    device = pick_device(a.gpu_index)
    model = load_model(a.trunet_root, a.img_size, a.num_classes, a.checkpoint, device)
    print(f"[cm] evaluating {len(files)} {a.split} volumes  (img {a.img_size}^3, {a.num_classes} classes)")

    on_cuda = device.type == "cuda"
    if a.precision == "bf16" and on_cuda:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    elif a.precision == "fp16" and on_cuda:
        autocast = torch.autocast("cuda", dtype=torch.float16)
    else:
        from contextlib import nullcontext
        autocast = nullcontext()

    C = a.num_classes
    cm = np.zeros((C, C), dtype=np.int64)
    with torch.no_grad():
        for n, f in enumerate(files, 1):
            d = np.load(f)
            image, label = d["arr_0"].astype(np.float32), d["arr_1"].astype(np.int64)
            x = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
            with autocast:
                logits = model(x)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
            t = label.ravel()
            p = pred.ravel()
            # accumulate: flat index true*C + pred
            binc = np.bincount(t * C + p, minlength=C * C)
            cm += binc.reshape(C, C)
            print(f"[cm] [{n:>4}/{len(files)}] {os.path.basename(f)}")

    # ---- save raw counts ----
    names = CLASS_NAMES[:C] if C <= len(CLASS_NAMES) else [str(i) for i in range(C)]
    np.savetxt(out_prefix + "_counts.csv", cm, fmt="%d", delimiter=",",
               header=",".join(names), comments="")
    print(f"[cm] wrote {out_prefix}_counts.csv")

    # ---- per-class metrics ----
    dice, iou, precision, recall = metrics_from_cm(cm)
    with open(out_prefix + "_metrics.csv", "w") as fh:
        fh.write("class,dice,iou,precision,recall\n")
        for i, nm in enumerate(names):
            fh.write(f"{nm},{dice[i]:.4f},{iou[i]:.4f},{precision[i]:.4f},{recall[i]:.4f}\n")
    print(f"[cm] wrote {out_prefix}_metrics.csv")

    print("\n  class          dice    iou     prec    recall")
    print("  " + "-" * 46)
    for i, nm in enumerate(names):
        print(f"  {nm:<13} {dice[i]:6.3f}  {iou[i]:6.3f}  {precision[i]:6.3f}  {recall[i]:6.3f}")
    fg = slice(1, C)  # exclude background
    print("  " + "-" * 46)
    print(f"  mean (fg)     {np.nanmean(dice[fg]):6.3f}  {np.nanmean(iou[fg]):6.3f}")

    # ---- heatmap PNG (row-normalized %) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        row = cm.sum(1, keepdims=True)
        cmn = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row != 0) * 100.0
        fig, ax = plt.subplots(figsize=(1.1 * C + 2, 1.1 * C + 2))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(C)); ax.set_yticks(range(C))
        ax.set_xticklabels(names, rotation=45, ha="right"); ax.set_yticklabels(names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Ground truth")
        ax.set_title(f"Confusion matrix (row-normalized %) — {a.split}")
        for i in range(C):
            for j in range(C):
                v = cmn[i, j]
                if v >= 0.5:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            color="white" if v > 50 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_prefix + ".png", dpi=150)
        print(f"[cm] wrote {out_prefix}.png")
    except ImportError:
        print("[cm] matplotlib not installed — skipped PNG "
              "(pip install matplotlib). CSVs are still written.")


if __name__ == "__main__":
    main()
