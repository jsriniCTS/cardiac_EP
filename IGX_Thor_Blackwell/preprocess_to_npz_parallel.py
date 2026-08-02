#!/usr/bin/env python3
"""
preprocess_to_npz_parallel.py
====================
Convert the STACOM2025 "Public Cardiac CT Dataset" (ImageCAS CCTA volumes +
multi-label heart segmentations) into the .npz format expected by TRUNet.

TRUNet's data loader (TRUNet_network/trunet_main.py -> fetch_dataset) expects:
    <root>/train/pt<ID>_<name>.npz   each with  arr_0 = image, arr_1 = label
    <root>/val/pt<ID>_<name>.npz
The filename MUST start with "pt" followed by an integer id (the loader parses
`int(name.split('_')[0][2:])`).

Dataset layout you provide:
    --images-dir  folder of CCTA volumes   e.g.  1.img.nii.gz, 2.img.nii.gz, ...
    --labels-dir  folder of label maps      e.g.  1.nii.gz / 1.label.nii.gz, ...
Images come from ImageCAS (Kaggle); labels from the STACOM2025 zip:
    https://people.compute.dtu.dk/rapa/STACOM2025/ImageCAS-STACOM2025-02-10-2025.zip

Label classes (0..10):
    0 background 1 myocardium 2 LA 3 LV 4 RA 5 RV 6 aorta 7 PA 8 LAA 9 coronary 10 PV

Each image is intensity-windowed (Hounsfield units) and min-max normalized to
[0,1], then both image and label are resized to a fixed cube (default 96^3) so
that TRUNet's on-the-fly zoom becomes a no-op and training is fast + low-memory.

Example
-------
python preprocess_to_npz_parallel.py \
    --images-dir /data/ImageCAS/images \
    --labels-dir /data/STACOM2025/labels \
    --out-root   /data/trunet_cardiac \
    --size 128 --val-frac 0.15 \
    --workers 16 --skip-existing

Runs cases in parallel (--workers), logs progress + ETA to the console AND to
<out-root>/preprocess.log (or --log-file), and can resume a partial run with
--skip-existing.
"""

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("[ERROR] nibabel is required:  pip install nibabel")

try:
    from scipy.ndimage import zoom
except ImportError:
    sys.exit("[ERROR] scipy is required:  pip install scipy")


_LEADING_INT = re.compile(r"^(\d+)")


def leading_id(path):
    """Return the leading integer id in a file's basename, or None."""
    base = os.path.basename(path)
    m = _LEADING_INT.match(base)
    return int(m.group(1)) if m else None


def index_by_id(folder):
    """Map {integer id -> filepath} for every .nii/.nii.gz in `folder`."""
    files = sorted(glob(os.path.join(folder, "*.nii.gz")) +
                   glob(os.path.join(folder, "*.nii")))
    out = {}
    for f in files:
        i = leading_id(f)
        if i is None:
            continue
        out.setdefault(i, f)   # first match wins (sorted -> deterministic)
    return out


def load_volume(path):
    """Load a NIfTI volume as a float32 numpy array (canonical orientation)."""
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj).astype(np.float32)


def window_normalize(vol, hu_min, hu_max):
    """Clip to [hu_min, hu_max] Hounsfield units and scale to [0, 1]."""
    vol = np.clip(vol, hu_min, hu_max)
    vol = (vol - hu_min) / float(hu_max - hu_min)
    return vol.astype(np.float32)


def resize_to(vol, size, order):
    """Resize a 3D volume to (size, size, size) with spline `order`."""
    x, y, z = vol.shape
    if (x, y, z) == (size, size, size):
        return vol
    factors = (size / x, size / y, size / z)
    return zoom(vol, factors, order=order)


def process_one(image_path, label_path, size, hu_min, hu_max):
    image = load_volume(image_path)
    label = load_volume(label_path)
    if image.shape != label.shape:
        # resize label to image grid first (nearest) so voxels stay aligned
        lx, ly, lz = label.shape
        ix, iy, iz = image.shape
        label = zoom(label, (ix / lx, iy / ly, iz / lz), order=0)

    image = window_normalize(image, hu_min, hu_max)
    image = resize_to(image, size, order=1)                 # linear for image
    label = np.rint(resize_to(label, size, order=0))        # nearest for label
    return image.astype(np.float32), label.astype(np.uint8)


def setup_logging(log_file):
    """Log to stdout and (append) to `log_file`."""
    logger = logging.getLogger("prep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _process_task(task):
    """Worker entry point — runs in a separate process, one case per call."""
    i, img_path, lab_path, dst, size, hu_min, hu_max = task
    try:
        image, label = process_one(img_path, lab_path, size, hu_min, hu_max)
        np.savez_compressed(dst, image, label)          # arr_0=image, arr_1=label
        present = sorted(int(v) for v in np.unique(label))
        return {"id": i, "dst": dst, "labels": present, "error": None}
    except Exception as e:
        return {"id": i, "dst": dst, "labels": None, "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(description="STACOM2025 -> TRUNet .npz preprocessor")
    ap.add_argument("--images-dir", required=True, help="Folder of CCTA image volumes")
    ap.add_argument("--labels-dir", required=True, help="Folder of segmentation label maps")
    ap.add_argument("--out-root", required=True, help="Output root (train/ and val/ created inside)")
    ap.add_argument("--size", type=int, default=128,
                    help="Cube edge length; must be divisible by 16 "
                         "[default 128 — the RTX Pro 6000 Blackwell's 96 GB handles it comfortably]")
    ap.add_argument("--val-frac", type=float, default=0.15, help="Fraction of cases used for validation [0.15]")
    ap.add_argument("--hu-min", type=float, default=-1000.0, help="Lower HU clip [-1000]")
    ap.add_argument("--hu-max", type=float, default=1000.0, help="Upper HU clip [1000]")
    ap.add_argument("--id-list", type=str, default=None,
                    help="Optional .txt of image ids (one per line) to restrict processing")
    ap.add_argument("--seed", type=int, default=42, help="Split shuffle seed [42]")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N cases (smoke test)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                    help="Parallel worker processes [min(8, ncpu)]; use 1 to disable")
    ap.add_argument("--log-file", default=None, help="Log file path [<out-root>/preprocess.log]")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip cases whose output .npz already exists (resume a partial run)")
    args = ap.parse_args()

    if args.size % 16 != 0:
        sys.exit(f"[ERROR] --size must be divisible by 16 (got {args.size}); try 64, 96, or 128.")

    images = index_by_id(args.images_dir)
    labels = index_by_id(args.labels_dir)
    ids = sorted(set(images) & set(labels))
    if not ids:
        sys.exit(f"[ERROR] No id-matched image/label pairs found.\n"
                 f"  images matched: {len(images)}   labels matched: {len(labels)}\n"
                 f"  Check that filenames start with the same integer id.")

    if args.id_list:
        with open(args.id_list) as fh:
            keep = {int(_LEADING_INT.match(l.strip()).group(1))
                    for l in fh if l.strip() and _LEADING_INT.match(l.strip())}
        ids = [i for i in ids if i in keep]
        print(f"[prep] id-list restricts to {len(ids)} cases")

    if args.limit:
        ids = ids[:args.limit]

    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * args.val_frac)))
    val_ids = set(ids[:n_val])
    train_dir = os.path.join(args.out_root, "train")
    val_dir = os.path.join(args.out_root, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(args.out_root, "preprocess.log")
    logger = setup_logging(log_file)
    workers = max(1, args.workers)
    logger.info(f"[prep] {len(ids)} cases -> train={len(ids) - n_val} val={n_val} "
                f"(size={args.size}^3, workers={workers}, log={log_file})")

    # build the work list, honouring --skip-existing (resume)
    tasks, skipped = [], 0
    for i in ids:
        dst = os.path.join(val_dir if i in val_ids else train_dir, f"pt{i}_case.npz")
        if args.skip_existing and os.path.exists(dst):
            skipped += 1
            continue
        tasks.append((i, images[i], labels[i], dst, args.size, args.hu_min, args.hu_max))
    if skipped:
        logger.info(f"[prep] --skip-existing: {skipped} already done, {len(tasks)} remaining")

    total = len(tasks)
    ok, failed, t0 = 0, [], time.time()

    def handle(n, r):
        nonlocal ok
        tag = "val " if r["id"] in val_ids else "train"
        if r["error"]:
            logger.info(f"[{n:>4}/{total}] id={r['id']}  FAILED: {r['error']}")
            failed.append(r["id"])
        else:
            rate = n / max(1e-9, time.time() - t0)
            eta = (total - n) / rate if rate > 0 else 0
            logger.info(f"[{n:>4}/{total}] {tag} id={r['id']:<4} -> "
                        f"{os.path.basename(r['dst'])}  labels={r['labels']}  "
                        f"({rate:.2f} case/s, ETA {eta / 60:.1f} min)")
            ok += 1

    if workers <= 1:
        for n, t in enumerate(tasks, 1):
            handle(n, _process_task(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_task, t) for t in tasks]
            for n, fut in enumerate(as_completed(futures), 1):
                handle(n, fut.result())

    dt = time.time() - t0
    logger.info(f"[prep] done. wrote {ok}/{total} files to {args.out_root} in {dt / 60:.1f} min")
    if failed:
        logger.info(f"[prep] {len(failed)} FAILED: {sorted(failed)}")
    logger.info(f"[prep] full log saved to {log_file}")


if __name__ == "__main__":
    main()
