#!/usr/bin/env python3
"""
validate_features.py
====================
Validate model-derived morphometry against ground truth by comparing two
feature tables produced by extract_features.py:

    ground truth : features on the STACOM2025 labels        (features_gt/features_full.csv)
    prediction   : features on the TRUNet predicted masks   (features_pred/features_full.csv)

Subjects are joined by `subject_id`. For every shared feature it reports:
  * numeric columns  → n, Pearson r, MAE, bias (pred−gt), 95% limits of
    agreement (Bland-Altman), and relative MAE (%).
  * categorical columns (category / class / shape / side / region / dominant)
    → agreement accuracy.

Outputs:
    validation_report.csv   one row per feature
    (printed summary of the key Left-Atrium features)

Example
-------
python validate_features.py \
    --gt   ./features_gt/features_full.csv \
    --pred ./features_pred/features_full.csv \
    --out  ./validation
"""

import argparse
import csv
import math
import os
import sys

import numpy as np

CAT_HINTS = ("category", "_class", "_shape", "_side", "_region", "dominant", "anatomy", "_type")
SKIP = {"subject_id", "file", "spacing_mm"}
KEY_LA = ["la_volume_ml", "la_ap_mm", "la_ml_mm", "la_si_mm", "la_major_axis_mm",
          "la_minor_axis_mm", "la_elongation", "la_flatness", "la_sphericity"]


def norm_id(s):
    s = (s or "").strip()
    return s[:-2] if s.endswith(".0") else s   # tolerate "1" vs "1.0"


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return {norm_id(r.get("subject_id")): r for r in rows if r.get("subject_id")}


def is_categorical(col):
    return any(h in col for h in CAT_HINTS)


def to_float(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser(description="Validate predicted vs ground-truth features")
    ap.add_argument("--gt", required=True, help="ground-truth features_full.csv")
    ap.add_argument("--pred", required=True, help="prediction features_full.csv")
    ap.add_argument("--out", default="./validation")
    a = ap.parse_args()

    for p in (a.gt, a.pred):
        if not os.path.isfile(p):
            sys.exit(f"[ERROR] not found: {p}")
    gt, pr = load(a.gt), load(a.pred)
    ids = sorted(set(gt) & set(pr), key=lambda s: (len(s), s))
    if not ids:
        sys.exit("[ERROR] no overlapping subject_id between the two tables")
    os.makedirs(a.out, exist_ok=True)
    print(f"[val] {len(ids)} paired subjects "
          f"(gt={len(gt)}, pred={len(pr)}, common={len(ids)})")

    cols = [c for c in gt[ids[0]].keys() if c not in SKIP and c in pr[ids[0]]]
    report = []
    for c in cols:
        gvals = [gt[i].get(c, "") for i in ids]
        pvals = [pr[i].get(c, "") for i in ids]
        if is_categorical(c):
            pairs = [(g, p) for g, p in zip(gvals, pvals) if g not in ("", "NA") and p not in ("", "NA")]
            if not pairs:
                continue
            acc = float(np.mean([g == p for g, p in pairs]))
            report.append({"feature": c, "type": "categorical", "n": len(pairs),
                           "accuracy": round(acc, 4)})
        else:
            g = np.array([to_float(v) for v in gvals])
            p = np.array([to_float(v) for v in pvals])
            mask = np.isfinite(g) & np.isfinite(p)
            n = int(mask.sum())
            if n < 3:
                continue
            g, p = g[mask], p[mask]
            diff = p - g
            bias, sd = float(diff.mean()), float(diff.std(ddof=1)) if n > 1 else 0.0
            mae = float(np.abs(diff).mean())
            denom = float(np.abs(g).mean())
            report.append({
                "feature": c, "type": "numeric", "n": n,
                "pearson_r": round(pearson(g, p), 4),
                "mae": round(mae, 4),
                "bias_pred_minus_gt": round(bias, 4),
                "loa_low": round(bias - 1.96 * sd, 4),
                "loa_high": round(bias + 1.96 * sd, 4),
                "rel_mae_pct": round(mae / denom * 100, 2) if denom > 0 else "",
            })

    # write report
    fields = ["feature", "type", "n", "pearson_r", "mae", "bias_pred_minus_gt",
              "loa_low", "loa_high", "rel_mae_pct", "accuracy"]
    out_csv = os.path.join(a.out, "validation_report.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in report:
            w.writerow(r)
    print(f"[val] wrote {out_csv}  ({len(report)} features compared)")

    # printed summary of key LA features
    idx = {r["feature"]: r for r in report}
    print("\n  Key Left-Atrium agreement (predicted vs ground truth)")
    print("  feature            n     r      MAE      rel-MAE%   bias")
    print("  " + "-" * 60)
    for f in KEY_LA:
        r = idx.get(f)
        if r and r["type"] == "numeric":
            print(f"  {f:<17} {r['n']:>3}  {r['pearson_r']:>5}  {r['mae']:>8}  "
                  f"{str(r['rel_mae_pct']):>7}   {r['bias_pred_minus_gt']:>7}")
    # categorical highlights
    cats = [r for r in report if r["type"] == "categorical"]
    if cats:
        print("\n  Categorical agreement (accuracy):")
        for r in sorted(cats, key=lambda x: -x["accuracy"])[:8]:
            print(f"    {r['feature']:<28} {r['accuracy']*100:5.1f}%  (n={r['n']})")


if __name__ == "__main__":
    main()
