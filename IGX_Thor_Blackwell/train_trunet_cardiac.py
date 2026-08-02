#!/usr/bin/env python3
"""
train_trunet_cardiac.py  —  NVIDIA IGX Thor / Blackwell edition
===============================================================
(Pre-)train the TRUNet 3D segmentation model on the STACOM2025 Public Cardiac
CT dataset, tuned for:

    Platform : NVIDIA IGX Thor  (aarch64 / Grace-class CPU)
    OS       : Ubuntu 24.04.4 LTS
    GPU      : RTX PRO 6000 Blackwell Max-Q  (sm_120, 96 GB VRAM)

Difference vs. the macOS/CPU launcher in ../:
  * Requires a Blackwell-capable PyTorch (CUDA 12.8, torch >= 2.7, sm_120).
    The TRUNet repo's pinned torch==2.0.1 does NOT support Blackwell — see
    requirements_thor.txt / setup_thor.sh.
  * Built-in mixed-precision training loop (bf16 autocast, the native format
    for Blackwell) + TF32 matmul + cuDNN autotuning, instead of the repo's
    fp32-only loop. Use --repo-trainer to fall back to the exact repo loop.
  * Larger defaults (img-size 128, batch 2) to use the 96 GB of VRAM.
  * Startup diagnostics that verify the GPU is actually Blackwell and that this
    PyTorch build can target sm_120.

Reuses the repo's model, augmentations, and metric helpers unchanged.

Example
-------
python train_trunet_cardiac.py \
    --root-path /data/trunet_cardiac \
    --trunet-root ../TRUNet-main \
    --num-classes 11 --img-size 128 \
    --batch-size 2 --max-epochs 100 \
    --precision bf16 \
    --save-path ./runs/thor_run1
"""

import argparse
import logging
import os
import random
import sys
from contextlib import nullcontext
from datetime import datetime
from glob import glob
from statistics import fmean

import ml_collections
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  Dataset (self-contained: the repo's fetch_dataset lives in trunet_main.py,
#  which has broken top-level imports and can't be imported).
# --------------------------------------------------------------------------- #
class NpzDataset:
    """Loads pt<ID>_*.npz files (arr_0=image, arr_1=label) from a folder."""

    def __init__(self, base_dir, transform=None):
        self.transform = transform
        self.sample_list = sorted(glob(os.path.join(base_dir, "*.npz")))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data = np.load(self.sample_list[idx])
        image, label = data["arr_0"], data["arr_1"]
        case = os.path.basename(self.sample_list[idx]).split(".npz")[0]
        sample = {"image": image, "label": label, "case_name": case}
        if self.transform:
            sample = self.transform(sample)
        return sample


def _worker_init_fn(worker_id):
    # module-level (picklable) so it works under any DataLoader start method
    random.seed(42 + worker_id)


def build_config(img_size, num_classes):
    """TRUNet 3D config with the output-class count wired to `num_classes`."""
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


# --------------------------------------------------------------------------- #
#  GPU diagnostics — confirm we're on Blackwell and this torch can target it.
# --------------------------------------------------------------------------- #
def report_device():
    if not torch.cuda.is_available():
        print("[thor] WARNING: CUDA not available — running on CPU. On IGX Thor "
              "this means the Blackwell PyTorch build is not installed correctly "
              "(see setup_thor.sh).")
        return torch.device("cpu")

    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)          # (12, 0) for Blackwell
    vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    arches = torch.cuda.get_arch_list()
    print(f"[thor] GPU        : {name}")
    print(f"[thor] capability : sm_{cap[0]}{cap[1]}   VRAM: {vram:.0f} GB")
    print(f"[thor] torch      : {torch.__version__}   CUDA: {torch.version.cuda}")
    print(f"[thor] arch_list  : {arches}")

    sm = f"sm_{cap[0]}{cap[1]}"
    if cap[0] < 12:
        print(f"[thor] NOTE: detected {sm} — expected sm_120 for RTX PRO 6000 Blackwell.")
    if not any(a.startswith(f"sm_{cap[0]}") for a in arches):
        print(f"[thor] *** WARNING: this PyTorch build lists {arches} and may NOT "
              f"support {sm}. If you see 'no kernel image is available for "
              f"execution on the device', install a CUDA 12.8 / cu128 build "
              f"(see setup_thor.sh). ***")
    return dev


def dice_score(pred, gt):  # data in shape [batch, classes, h, w, d]
    dice = []
    for b in range(gt.shape[0]):
        tmp = []
        for roi in range(gt.shape[1]):
            if roi > 0:  # skip background
                p, g = pred[b, roi], gt[b, roi]
                a, bb, cc = np.sum(p[g == 1]), np.sum(p), np.sum(g)
                tmp.append(0 if a == 0 else float(a * 2.0 / (bb + cc)))
        dice.append(fmean(tmp))
    return fmean(dice)


def one_hot_encoder(input_tensor, n_classes):
    return torch.cat([(input_tensor == i) for i in range(n_classes)], dim=1).float()


# --------------------------------------------------------------------------- #
#  Blackwell-optimized training loop (bf16 autocast + TF32 + cuDNN autotune).
# --------------------------------------------------------------------------- #
def trainer_amp(args, config, model, device, precision):
    from torch.utils.data import DataLoader
    from tensorboardX import SummaryWriter

    model.to(device, memory_format=torch.channels_last_3d)

    loss_fn = config["loss_function"]
    optimizer = config["optimizer"]
    ds_train, ds_val = config["ds_train"], config["ds_val"]
    save_interval = config["save_interval"]

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0,
                              worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False,
                            num_workers=min(2, args.num_workers), pin_memory=True,
                            worker_init_fn=_worker_init_fn)

    max_iter = args.max_epochs * max(1, len(train_loader))
    os.makedirs(args.save_path, exist_ok=True)
    logging.basicConfig(filename=args.save_path + "/log.txt", level=logging.INFO,
                        format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.info(str(args))
    writer = SummaryWriter(args.save_path + "/log")

    # precision setup
    on_cuda = device.type == "cuda"
    if precision == "bf16" and on_cuda:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
        scaler = None  # bf16 needs no loss scaling
    elif precision == "fp16" and on_cuda:
        autocast = torch.autocast("cuda", dtype=torch.float16)
        scaler = torch.cuda.amp.GradScaler()
    else:
        autocast = nullcontext()
        scaler = None
    print(f"[thor] precision  : {precision if on_cuda else 'fp32 (cpu)'}")

    best_metric, best_epoch, iter_num = -1.0, -1, 0
    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0.0
        for sampled in train_loader:
            inputs = sampled["image"].unsqueeze(1).to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last_3d)
            targets = sampled["label"].unsqueeze(1).to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with (autocast if on_cuda else nullcontext()):
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            lr_ = args.base_lr * (1.0 - iter_num / max_iter) ** 0.9
            for pg in optimizer.param_groups:
                pg["lr"] = lr_
            writer.add_scalar("info/lr", lr_, iter_num)
            writer.add_scalar("info/total_loss", loss.item(), iter_num)
            iter_num += 1

        epoch_loss /= max(1, len(train_loader))
        logging.info("epoch %d : mean loss : %f" % (epoch, epoch_loss))

        # ---- validation ----
        model.eval()
        dices = []
        with torch.no_grad():
            for sampled in val_loader:
                inputs = sampled["image"].to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last_3d)
                targets = one_hot_encoder(sampled["label"].unsqueeze(1), args.num_classes).to(device)
                with (autocast if on_cuda else nullcontext()):
                    out = nn.Softmax(dim=1)(model(inputs))
                dices.append(dice_score(out.float().cpu().numpy(), targets.cpu().numpy()))
        metric = fmean(dices) if dices else 0.0
        writer.add_scalar("info/validation_metric", metric, epoch)
        logging.info("epoch %d : dice score : %f" % (epoch, metric))

        if metric > best_metric:
            best_metric, best_epoch = metric, epoch + 1
            torch.save(model.state_dict(), os.path.join(args.save_path, "best_metric_model.pth"))
            print("saved new best metric model")
        print(f"epoch {epoch + 1}/{args.max_epochs}  loss {epoch_loss:.4f}  "
              f"mean dice {metric:.4f}  (best {best_metric:.4f} @ {best_epoch})  lr {lr_:.2e}")

        if (epoch + 1) % save_interval == 0 or epoch == args.max_epochs - 1:
            path = os.path.join(args.save_path, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)
            logging.info("save model to {}".format(path))

    writer.close()
    return "Training Finished!"


def main():
    ap = argparse.ArgumentParser(description="Train TRUNet on STACOM2025 cardiac CT (IGX Thor / Blackwell)")
    ap.add_argument("--root-path", required=True, help="Folder containing train/ and val/ npz dirs")
    ap.add_argument("--trunet-root", default=None,
                    help="Path to TRUNet-main (folder containing TRUNet_network/). "
                         "Defaults to ./TRUNet-main or ../TRUNet-main.")
    ap.add_argument("--num-classes", type=int, default=11, help="Label classes incl. background [11]")
    ap.add_argument("--img-size", type=int, default=128, help="Cube edge; divisible by 16 [128]")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--base-lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8, help="DataLoader workers [8]")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="Mixed precision; bf16 is native to Blackwell [bf16]")
    ap.add_argument("--save-interval", type=int, default=50)
    ap.add_argument("--checkpoint", default="None", help="Path to a .pth to resume/fine-tune from")
    ap.add_argument("--repo-trainer", action="store_true",
                    help="Use the repo's exact fp32 trainer loop instead of the AMP loop")
    ap.add_argument("--save-path", default=None, help="Output dir [./runs/thor_run_<timestamp>]")
    a = ap.parse_args()

    if a.img_size % 16 != 0:
        sys.exit(f"[ERROR] --img-size must be divisible by 16 (got {a.img_size}); try 64, 96, 128, 160.")

    # locate the repo
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [a.trunet_root] if a.trunet_root else [
        os.path.join(here, "TRUNet-main"), os.path.join(here, "..", "TRUNet-main")]
    trunet_root = next((os.path.abspath(c) for c in candidates
                        if c and os.path.isdir(os.path.join(c, "TRUNet_network"))), None)
    if trunet_root is None:
        sys.exit("[ERROR] Could not find TRUNet_network/. Pass --trunet-root /path/to/TRUNet-main")
    sys.path.insert(0, trunet_root)
    print(f"[thor] TRUNet repo : {trunet_root}")

    from torchvision import transforms
    from TRUNet_network.model.ViT import VisionTransformer3d
    from TRUNet_network.augmentations import RandomGenerator3d_zoom, Reshape3d_zoom
    try:
        from monai.losses import DiceCELoss
        from monai.metrics import DiceMetric
    except ImportError:
        sys.exit("[ERROR] MONAI required:  pip install monai")

    # Blackwell perf knobs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device = report_device()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    save_path = a.save_path or os.path.join(here, "runs", "thor_run_" + datetime.now().strftime("%m%d%Y_%H%M%S"))
    os.makedirs(os.path.join(save_path, "log"), exist_ok=True)

    for split in ("train", "val"):
        d = os.path.join(a.root_path, split)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        if n == 0:
            sys.exit(f"[ERROR] {d} missing or empty. Run preprocess_to_npz.py first.")
        print(f"[thor] {split}: {n} npz files")

    args = ml_collections.ConfigDict()
    args.max_epochs, args.save_path, args.root_path = a.max_epochs, save_path, a.root_path
    args.num_classes, args.batch_size, args.base_lr = a.num_classes, a.batch_size, a.base_lr
    args.seed, args.img_size, args.num_workers = a.seed, a.img_size, a.num_workers

    img = a.img_size
    train_tf = transforms.Compose([RandomGenerator3d_zoom(output_size=(img, img, img))])
    val_tf = transforms.Compose([Reshape3d_zoom(output_size=[img, img, img])])

    cfg = build_config(img, a.num_classes)
    model = VisionTransformer3d(cfg, img_size=img, zero_head=False, vis=False)
    print(f"[thor] model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M | "
          f"classes {a.num_classes} | img {img}^3 | batch {a.batch_size}")

    config = {
        "ds_train": NpzDataset(os.path.join(a.root_path, "train"), transform=train_tf),
        "ds_val": NpzDataset(os.path.join(a.root_path, "val"), transform=val_tf),
        "loss_function": DiceCELoss(include_background=False, to_onehot_y=True, softmax=True),
        "metric": DiceMetric(include_background=False, reduction="mean"),
        "optimizer": torch.optim.Adam(model.parameters(), a.base_lr),
        "save_interval": a.save_interval,
    }

    if a.checkpoint and a.checkpoint != "None":
        print("[thor] loading checkpoint:", a.checkpoint)
        model.load_state_dict(torch.load(a.checkpoint, map_location="cpu"))

    print("[thor] starting…  ->", save_path)
    if a.repo_trainer:
        import torch.multiprocessing as mp
        try:
            mp.set_start_method("fork", force=True)
        except (RuntimeError, ValueError):
            pass
        from TRUNet_network.trunet_train import trainer as repo_trainer
        msg = repo_trainer(args, config, model, save_path)
    else:
        msg = trainer_amp(args, config, model, device, a.precision)
    print("[thor]", msg)


if __name__ == "__main__":
    main()
