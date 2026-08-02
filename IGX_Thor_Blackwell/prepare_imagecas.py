#!/usr/bin/env python3
"""
prepare_imagecas.py
===================
One-shot preparation of the Kaggle ImageCAS download for the TRUNet pipeline.

The Kaggle ImageCAS dataset is delivered as several *spanned (multi-part) ZIP*
archives, one per batch of scans, e.g. for the 1-200 batch:

    1-200.change2zip     <- the MAIN zip; extension mangled, must become 1-200.zip
    1-200.z01            <- split part 1
    1-200.z02            <- split part 2
    ...

Each such group is ONE archive split across files (not separate datasets), so
it must be reassembled before extraction. This script does the whole thing:

    1. rename every  *.change2zip  ->  *.zip
    2. group the files into batches by their base name (1-200, 201-400, ...)
    3. reassemble + extract each batch
         - preferred: system `zip -s 0 ... --out` then `unzip`
         - fallback : `7z`
         - last resort: pure-Python concatenation (+ spanning-marker fixup)
    4. move every  *.img.nii.gz  into a single clean output folder
    5. verify the count (and, if given, overlap with the STACOM label ids)

Usage (run from inside IGX_Thor_Blackwell/):
    python prepare_imagecas.py
    # or with explicit paths:
    python prepare_imagecas.py \
        --src ./data/ImagesCAS/images \
        --out ./data/ImageCAS/images \
        --labels-dir /data/stacom_labels

Notes
-----
* Extraction can need a LOT of disk (ImageCAS is large); --work defaults to a
  sibling of --out. Use --keep-extracted to keep the raw extracted tree.
* Requires Info-ZIP `zip`/`unzip` or `7z` for split archives:
      sudo apt-get install -y zip unzip p7zip-full
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import defaultdict
from glob import glob


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def sh(cmd, cwd=None):
    """Run a command, streaming output; return the exit code."""
    print("    $", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd).returncode


def has_content(path):
    """True if `path` contains at least one regular file (recursively)."""
    for _root, _dirs, files in os.walk(path):
        if files:
            return True
    return False


def tools():
    return {t: shutil.which(t) for t in ("zip", "unzip", "7z", "7za")}


def rename_change2zip(src):
    renamed = 0
    for f in glob(os.path.join(src, "*.change2zip")):
        dst = f[: -len(".change2zip")] + ".zip"
        print(f"[rename] {os.path.basename(f)} -> {os.path.basename(dst)}")
        os.rename(f, dst)
        renamed += 1
    if renamed == 0:
        print("[rename] no *.change2zip files (already renamed or not present)")
    return renamed


_PART_RE = re.compile(r"^(?P<base>.+)\.z(?P<num>\d{2})$", re.IGNORECASE)


def group_batches(src):
    """Return {base: {'main': path|None, 'parts': [paths sorted]}}."""
    batches = defaultdict(lambda: {"main": None, "parts": []})
    for f in sorted(glob(os.path.join(src, "*"))):
        name = os.path.basename(f)
        if name.lower().endswith(".zip"):
            batches[name[:-4]]["main"] = f
        else:
            m = _PART_RE.match(name)
            if m:
                batches[m.group("base")]["parts"].append((int(m.group("num")), f))
    # sort split parts by their numeric index
    for b in batches.values():
        b["parts"] = [p for _, p in sorted(b["parts"], key=lambda x: x[0])]
    return dict(batches)


def reassemble_pure_python(main, parts, out_zip):
    """Last-resort: concatenate parts + main into one zip, fixing the leading
    4-byte spanning signature (PK\\x07\\x08) that trips up Python's zipfile."""
    print("    [fallback] concatenating parts in Python (no zip/7z found)")
    SPAN_SIG = b"PK\x07\x08"
    with open(out_zip, "wb") as w:
        for i, p in enumerate(parts + [main]):
            with open(p, "rb") as r:
                if i == 0:
                    head = r.read(4)
                    if head != SPAN_SIG:      # keep bytes if it wasn't the marker
                        w.write(head)
                shutil.copyfileobj(r, w, length=8 * 1024 * 1024)
    return zipfile.is_zipfile(out_zip)


def extract_batch(base, info, work, tl):
    """Reassemble (if split) and extract one batch into work/<base>/."""
    main, parts = info["main"], info["parts"]
    dest = os.path.join(work, base)
    os.makedirs(dest, exist_ok=True)

    if main is None:
        print(f"[skip] {base}: no main .zip found (have parts: {len(parts)}). "
              f"Rename the batch's main file to {base}.zip and re-run.")
        return False

    # NOTE: success is judged by whether files were actually extracted, not by
    # exit code — `unzip` returns 1 on harmless warnings even when it fully
    # succeeds, so an exit-code-only check gives false failures.
    sevenz = tl["7z"] or tl["7za"]

    # --- simple, non-split archive -------------------------------------------
    if not parts:
        print(f"[extract] {base}: single-file zip")
        if tl["unzip"]:
            sh([tl["unzip"], "-o", main, "-d", dest])
        elif sevenz:
            sh([sevenz, "x", main, f"-o{dest}", "-y"])
        else:
            with zipfile.ZipFile(main) as z:
                z.extractall(dest)
        return has_content(dest)

    # --- split / spanned archive ---------------------------------------------
    print(f"[extract] {base}: split archive ({len(parts)} parts + main)")
    single = os.path.join(work, f"{base}_single.zip")

    # preferred: Info-ZIP reassembly (handles the spanning marker correctly)
    if tl["zip"] and tl["unzip"]:
        sh([tl["zip"], "-s", "0", os.path.basename(main), "--out", single],
           cwd=os.path.dirname(main))
        if os.path.exists(single):
            sh([tl["unzip"], "-o", single, "-d", dest])
        if has_content(dest):
            os.path.exists(single) and os.remove(single)
            return True
        print("    [warn] zip/unzip route produced nothing, trying 7z…")

    # 7z reads split parts directly when pointed at the main .zip
    if sevenz:
        sh([sevenz, "x", main, f"-o{dest}", "-y"])
        if has_content(dest):
            return True
        print("    [warn] 7z route produced nothing, trying pure-Python…")

    # last resort
    if reassemble_pure_python(main, parts, single):
        try:
            with zipfile.ZipFile(single) as z:
                z.extractall(dest)
            os.remove(single)
        except zipfile.BadZipFile as e:
            print(f"    [error] Python extraction failed: {e}")
    if has_content(dest):
        return True
    print(f"[FAIL] {base}: could not extract. Install tools:  "
          f"sudo apt-get install -y zip unzip p7zip-full")
    return False


def pool_images(work, out, pattern):
    os.makedirs(out, exist_ok=True)
    found = glob(os.path.join(work, "**", pattern), recursive=True)
    if not found:
        print(f"[pool] no files matching '{pattern}'. Falling back to '*.nii.gz'.")
        found = [f for f in glob(os.path.join(work, "**", "*.nii.gz"), recursive=True)
                 if ".label." not in os.path.basename(f).lower()]
    moved = 0
    for f in found:
        dst = os.path.join(out, os.path.basename(f))
        if os.path.abspath(f) != os.path.abspath(dst):
            shutil.move(f, dst)
        moved += 1
    print(f"[pool] moved {moved} image volumes -> {out}")
    return moved


def ids_in(folder, patt="*.nii.gz"):
    ids = set()
    for f in glob(os.path.join(folder, patt)):
        m = re.match(r"^(\d+)", os.path.basename(f))
        if m:
            ids.add(int(m.group(1)))
    return ids


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Reassemble + extract + pool the ImageCAS split zips")
    ap.add_argument("--src", default="./data/ImagesCAS/images",
                    help="Folder containing the downloaded zip parts [./data/ImagesCAS/images]")
    ap.add_argument("--out", default="./data/ImageCAS/images",
                    help="Output folder for pooled *.img.nii.gz volumes [./data/ImageCAS/images]")
    ap.add_argument("--work", default=None,
                    help="Scratch extraction dir [<out>/../_extracted]")
    ap.add_argument("--pattern", default="*.img.nii.gz",
                    help="Glob for image volumes to pool [*.img.nii.gz]")
    ap.add_argument("--labels-dir", default=None,
                    help="Optional STACOM labels folder — reports id overlap for pairing")
    ap.add_argument("--keep-extracted", action="store_true",
                    help="Keep the raw extracted tree (default: delete to save disk)")
    a = ap.parse_args()

    src = os.path.abspath(a.src)
    out = os.path.abspath(a.out)
    work = os.path.abspath(a.work) if a.work else os.path.join(os.path.dirname(out), "_extracted")

    if not os.path.isdir(src):
        sys.exit(f"[ERROR] --src not found: {src}")
    os.makedirs(work, exist_ok=True)

    tl = tools()
    print(f"[env] src={src}")
    print(f"[env] out={out}")
    print(f"[env] work={work}")
    print(f"[env] tools: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in tl.items()))
    if not (tl["zip"] and tl["unzip"]) and not (tl["7z"] or tl["7za"]):
        print("[env] WARNING: no zip/unzip or 7z found — will attempt pure-Python "
              "fallback, which may fail on some spanned zips.\n"
              "      Recommended:  sudo apt-get install -y zip unzip p7zip-full")

    # 1) rename
    print("\n=== 1) rename *.change2zip ===")
    rename_change2zip(src)

    # 2) group
    print("\n=== 2) group into batches ===")
    batches = group_batches(src)
    if not batches:
        sys.exit(f"[ERROR] No .zip / .z01 files found in {src}")
    for base, info in batches.items():
        kind = "split" if info["parts"] else "single"
        print(f"  batch '{base}': {kind}, main={'yes' if info['main'] else 'MISSING'}, "
              f"parts={len(info['parts'])}")

    # 3) extract
    print("\n=== 3) reassemble + extract ===")
    ok, failed = [], []
    for base, info in batches.items():
        (ok if extract_batch(base, info, work, tl) else failed).append(base)

    # 4) pool
    print("\n=== 4) pool image volumes ===")
    n = pool_images(work, out, a.pattern)

    # 5) verify
    print("\n=== 5) verify ===")
    img_ids = ids_in(out)
    print(f"[verify] {len(img_ids)} image volumes in {out}")
    if a.labels_dir and os.path.isdir(a.labels_dir):
        lab_ids = ids_in(a.labels_dir)
        pairs = img_ids & lab_ids
        print(f"[verify] {len(lab_ids)} label maps in {a.labels_dir}")
        print(f"[verify] {len(pairs)} matched image/label pairs (these become training cases)")
        only_img = sorted(img_ids - lab_ids)[:10]
        if only_img:
            print(f"[verify] examples with image but no label (skipped later): {only_img}")

    if not a.keep_extracted and not failed:
        print(f"[cleanup] removing scratch tree {work}")
        shutil.rmtree(work, ignore_errors=True)

    print("\n=== summary ===")
    print(f"  extracted OK : {ok}")
    if failed:
        print(f"  FAILED       : {failed}")
    print(f"  images ready : {n} in {out}")
    print("\nNext:")
    print(f"  python preprocess_to_npz.py \\")
    print(f"    --images-dir {out} \\")
    print(f"    --labels-dir {a.labels_dir or '/data/stacom_labels'} \\")
    print(f"    --out-root ./data/trunet_cardiac --size 128 --val-frac 0.15")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
