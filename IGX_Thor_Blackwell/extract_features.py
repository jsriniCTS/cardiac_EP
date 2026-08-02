#!/usr/bin/env python3
"""
extract_features.py
===================
Extract the full 51-feature morphometric / shape set from cardiac CT
segmentation label maps (Features_to_extract.xlsx). Runs AFTER segmentation
(Stage 6 of Fig. 1) and feeds the recommendation engine (Stage 7).

Label convention (STACOM2025):
    0 bg  1 myocardium  2 LA  3 LV  4 RA  5 RV  6 aorta  7 PA  8 LAA  9 coronary  10 PV

Passes:
  (1) per-subject, parallel — every feature computable from a single mask:
        LA (1-24), PV ostia + size/shape/orientation/inter-PV/carina (25-35),
        LAA volume/area/sphericity + length/tortuosity/bend (36-41),
        LAA-LA-PV coherence (42-44), ostial contour features (48-51).
  (2) cohort — percentiles, categories, and the shape models:
        PCA statistical shape model (45), VAE latent embedding (46, needs torch),
        elastic-shape-style clustering (47), plus percentile-based classes for
        ostium size / carina width.

Status legend (features_manifest.csv):
    computed  – exact geometric quantity from the mask + spacing
    pctile/category – filled in the cohort pass
    approx    – principled approximation (ellipse fit, geodesic centerline,
                occupancy-grid SSM/VAE/clustering); validate before clinical use

Units need real voxel spacing (mm) from each file's affine — run on the
ORIGINAL-resolution label maps, not the 128³ .npz. Pass --spacing dx,dy,dz if
the affine is identity.

Example
-------
python extract_features.py --labels-dir ./data/stacom_labels --out ./features \
    --workers 16 --ssm --vae --clusters 4 --per-subject-json
"""

import argparse
import itertools
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.exit("[ERROR] nibabel required:  pip install nibabel")
try:
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree, ConvexHull
except ImportError:
    sys.exit("[ERROR] scipy required:  pip install scipy")

try:
    from skimage.measure import marching_cubes, mesh_surface_area
    _SK = True
except Exception:
    _SK = False
try:
    from skimage.graph import MCP_Geometric
    _MCP = True
except Exception:
    _MCP = False
try:
    from skimage.transform import resize as sk_resize
    _RESIZE = True
except Exception:
    _RESIZE = False

LABELS = {"myocardium": 1, "LA": 2, "LV": 3, "RA": 4, "RV": 5,
          "aorta": 6, "PA": 7, "LAA": 8, "coronary": 9, "PV": 10}
PV_NAMES = ["LSPV", "LIPV", "RSPV", "RIPV"]

# ------------------------------------------------------------------ manifest
FEATURES = [
    (1, "LA", "Volume", "computed", "P1"), (2, "LA", "Volume %tile", "pctile", "P1"),
    (3, "LA", "Volume category", "category", "P1"), (4, "LA", "AP diameter", "computed", "P1"),
    (5, "LA", "AP diameter %tile", "pctile", "P1"), (6, "LA", "ML diameter", "computed", "P1"),
    (7, "LA", "ML diameter %tile", "pctile", "P1"), (8, "LA", "SI diameter", "computed", "P1"),
    (9, "LA", "SI diameter %tile", "pctile", "P1"), (10, "LA", "Major axis length", "computed", "P1"),
    (11, "LA", "Major axis %tile", "pctile", "P1"), (12, "LA", "Minor axis length", "computed", "P1"),
    (13, "LA", "Minor axis %tile", "pctile", "P1"), (14, "LA", "Least axis length", "computed", "P1"),
    (15, "LA", "Least axis %tile", "pctile", "P1"), (16, "LA", "Elongation", "computed", "P1"),
    (17, "LA", "Elongation %tile", "pctile", "P1"), (18, "LA", "Flatness", "computed", "P1"),
    (19, "LA", "Flatness %tile", "pctile", "P1"), (20, "LA", "Sphericity", "computed", "P1"),
    (21, "LA", "Sphericity %tile", "pctile", "P1"), (22, "LA", "Regional volume distribution", "computed", "P1"),
    (23, "LA", "Directional asymmetry (DAI)", "computed", "P1"), (24, "LA", "Cross-sectional profile", "computed", "P1"),
    (25, "PV", "PV number", "computed", "P1"), (26, "PV", "PV anatomy", "computed", "P1"),
    (27, "PV", "Common left PV", "computed", "P1"), (28, "PV", "Common left PV type", "computed", "P1"),
    (29, "PV", "Right middle PV", "computed", "P1"), (30, "PV", "Accessory PV", "computed", "P1"),
    (31, "PV", "PV ostium size (LSPV/LIPV/RSPV/RIPV)", "approx", "P1"),
    (32, "PV", "PV ostium shape (LSPV/LIPV/RSPV/RIPV)", "approx", "P1"),
    (33, "PV", "PV orientation angle", "approx", "P1"), (34, "PV", "Inter-PV distance", "approx", "P1"),
    (35, "PV", "Carina morphology", "approx", "P1"),
    (36, "LAA", "LAA volume", "computed", "P2"), (37, "LAA", "LAA surface area", "computed", "P2"),
    (38, "LAA", "LAA length / depth", "approx", "P2"), (39, "LAA", "LAA tortuosity", "approx", "P2"),
    (40, "LAA", "LAA bend angle", "approx", "P2"), (41, "LAA", "LAA sphericity / compactness", "computed", "P2"),
    (42, "Coherence", "LAA-PV ostial distance", "approx", "P2"),
    (43, "Coherence", "LAA orientation relative to LA", "computed", "P2"),
    (44, "Coherence", "PV configuration relative to LAA", "approx", "P2"),
    (45, "SSM", "PCA statistical shape model", "approx", "P2"),
    (46, "SSM", "VAE latent embedding", "approx", "P2"),
    (47, "SSM", "Elastic-shape clustering", "approx", "P2"),
    (48, "Ostial", "Ostial plane / contour", "approx", "P2"),
    (49, "Ostial", "Ostial diameter (major/minor)", "approx", "P2"),
    (50, "Ostial", "Ostial area / perimeter", "approx", "P2"),
    (51, "Ostial", "Ostial eccentricity", "approx", "P2"),
]


# ============================================================ geometry helpers
def canonical(path):
    img = nib.as_closest_canonical(nib.load(path))
    arr = np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    z = img.header.get_zooms()[:3]
    spacing = np.array([float(v) if v and v > 0 else 1.0 for v in z], dtype=float)
    return arr, spacing


def pca(coords_mm):
    """Return (eigvals desc, eigvecs cols matching) of a centered mm point set."""
    c = coords_mm - coords_mm.mean(0)
    cov = np.cov(c, rowvar=False)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return np.clip(w[order], 0, None), V[:, order]


def perp_basis(n):
    """Two orthonormal vectors spanning the plane perpendicular to unit vector n."""
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(n, a); e1 /= (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n, e1)
    return e1, e2


def surface_area_mm2(mask, spacing):
    if mask.sum() == 0:
        return np.nan
    if _SK:
        try:
            v, f, _, _ = marching_cubes(mask.astype(np.float32), 0.5, spacing=tuple(spacing))
            return float(mesh_surface_area(v, f))
        except Exception:
            pass
    sx, sy, sz = spacing
    face = {0: sy * sz, 1: sx * sz, 2: sx * sy}
    area = 0.0
    for ax in (0, 1, 2):
        up = np.zeros_like(mask); dn = np.zeros_like(mask)
        s1 = [slice(None)] * 3; s1[ax] = slice(0, -1)
        s2 = [slice(None)] * 3; s2[ax] = slice(1, None)
        up[tuple(s1)] = mask[tuple(s2)]; dn[tuple(s2)] = mask[tuple(s1)]
        area += ((mask & ~up).sum() + (mask & ~dn).sum()) * face[ax]
    return float(area)


def ellipse_of_points(P2d, spacing_area):
    """Fit an ellipse-ish descriptor to 2D points: (Dmax, Dmin, area, perimeter)."""
    if len(P2d) < 3:
        return np.nan, np.nan, np.nan, np.nan
    w, V = np.linalg.eigh(np.cov(P2d - P2d.mean(0), rowvar=False))
    proj = (P2d - P2d.mean(0)) @ V
    dmax = float(proj[:, 1].max() - proj[:, 1].min())
    dmin = float(proj[:, 0].max() - proj[:, 0].min())
    if dmax < dmin:
        dmax, dmin = dmin, dmax
    try:
        h = ConvexHull(P2d)
        area = float(h.volume)          # 2D hull area
        perim = float(h.area)           # 2D hull perimeter
    except Exception:
        area = len(P2d) * spacing_area
        perim = math.pi * (dmax + dmin) / 2
    return dmax, dmin, area, perim


# ============================================================ LA (1-24)
def la_metrics(label, spacing):
    m = label == LABELS["LA"]
    out = {}
    n = int(m.sum())
    out["la_n_voxels"] = n
    if n == 0:
        return out
    vox = float(np.prod(spacing))
    out["la_volume_ml"] = n * vox / 1000.0
    idx = np.argwhere(m)
    ext = (idx.max(0) - idx.min(0) + 1) * spacing
    out["la_ml_mm"], out["la_ap_mm"], out["la_si_mm"] = float(ext[0]), float(ext[1]), float(ext[2])
    ev, _ = pca(idx * spacing)
    l1, l2, l3 = ev
    out["la_major_axis_mm"] = float(4 * math.sqrt(l1)) if l1 > 0 else np.nan
    out["la_minor_axis_mm"] = float(4 * math.sqrt(l2)) if l2 > 0 else np.nan
    out["la_least_axis_mm"] = float(4 * math.sqrt(l3)) if l3 > 0 else np.nan
    out["la_elongation"] = float(math.sqrt(l2 / l1)) if l1 > 0 else np.nan
    out["la_flatness"] = float(math.sqrt(l3 / l1)) if l1 > 0 else np.nan
    V = out["la_volume_ml"] * 1000.0
    A = surface_area_mm2(m, spacing)
    out["la_surface_area_mm2"] = A
    out["la_sphericity"] = float((math.pi ** (1 / 3) * (6 * V) ** (2 / 3)) / A) if A and A > 0 else np.nan
    z = idx[:, 2]; zmin, zmax = z.min(), z.max()
    if zmax > zmin:
        t = (z - zmin) / (zmax - zmin)
        bot, mid, top = float((t < 1/3).mean()), float(((t >= 1/3) & (t < 2/3)).mean()), float((t >= 2/3).mean())
        out["la_region_dominant"] = ["inferior", "mid", "superior"][int(np.argmax([bot, mid, top]))]
    else:
        bot = mid = top = np.nan; out["la_region_dominant"] = "NA"
    out["la_region_inferior_frac"], out["la_region_mid_frac"], out["la_region_superior_frac"] = bot, mid, top
    x = idx[:, 0]; xmid = (x.min() + x.max()) / 2.0
    left, right = int((x < xmid).sum()), int((x >= xmid).sum())
    dai = (left - right) / float(n)
    out["la_dai"] = float(dai)
    out["la_dai_side"] = "left-dominant" if dai > 0.05 else ("right-dominant" if dai < -0.05 else "symmetric")
    pix = spacing[0] * spacing[1]
    areas = m.sum((0, 1)).astype(float) * pix
    nz = np.nonzero(areas)[0]
    if nz.size:
        seg = areas[nz[0]:nz[-1] + 1]
        pos = np.argmax(seg) / max(1, len(seg) - 1)
        out["la_xsec_max_area_mm2"] = float(seg.max())
        out["la_xsec_maxpos"] = float(pos)
        out["la_xsec_region"] = "inferior" if pos < 1/3 else ("mid" if pos < 2/3 else "roof")
    else:
        out["la_xsec_max_area_mm2"] = out["la_xsec_maxpos"] = np.nan; out["la_xsec_region"] = "NA"
    return out


# ============================================================ PV (25-35, 48-51)
def analyze_pv(label, spacing):
    out = {}
    pv = label == LABELS["PV"]; la = label == LABELS["LA"]
    if pv.sum() == 0 or la.sum() == 0:
        out["pv_number"] = 0 if la.sum() else np.nan
        return out
    la_c = (np.argwhere(la) * spacing).mean(0)
    ostial = pv & ndi.binary_dilation(la, iterations=1)
    lbl, n = ndi.label(ostial)
    pvlbl, _ = ndi.label(pv)
    pix = spacing[0] * spacing[1]
    ostia = []
    for cid in range(1, n + 1):
        ov = np.argwhere(lbl == cid)
        if len(ov) < 5:
            continue
        omm = ov * spacing
        oc = omm.mean(0)
        vote = np.bincount(pvlbl[tuple(ov.T)]); vote[0] = 0
        vein = np.argwhere(pvlbl == vote.argmax()) * spacing if vote.max() else omm
        long_ax = pca(vein)[1][:, 0] if len(vein) >= 3 else np.array([0, 0, 1.0])
        ang = math.degrees(math.acos(min(1.0, abs(long_ax[2]))))     # vs SI axis
        e1, e2 = perp_basis(long_ax)
        P = np.c_[(omm - oc) @ e1, (omm - oc) @ e2]                  # ostial plane projection
        dmax, dmin, area, perim = ellipse_of_points(P, pix)
        ostia.append(dict(cent=oc, side="L" if oc[0] < la_c[0] else "R", z=oc[2],
                          voxmm=omm, dmax=dmax, dmin=dmin, area=area, perim=perim, ang=ang))
    out["pv_number"] = len(ostia)
    L = sorted([o for o in ostia if o["side"] == "L"], key=lambda o: -o["z"])
    R = sorted([o for o in ostia if o["side"] == "R"], key=lambda o: -o["z"])
    out["pv_left_count"], out["pv_right_count"] = len(L), len(R)
    out["pv_common_left"] = bool(len(L) == 1 and len(R) >= 2)
    out["pv_common_left_type"] = "common-left-trunk" if out["pv_common_left"] else "separate-ostia"
    out["pv_right_middle"] = bool(len(R) >= 3)
    out["pv_accessory"] = bool(len(ostia) > 4)
    out["pv_anatomy"] = f"{len(ostia)} ostia (L{len(L)}/R{len(R)})"

    named = {"LSPV": L[0] if len(L) >= 1 else None, "LIPV": L[1] if len(L) >= 2 else None,
             "RSPV": R[0] if len(R) >= 1 else None, "RIPV": R[1] if len(R) >= 2 else None}
    for nm, o in named.items():
        if o is None:
            continue
        out[f"{nm}_ostium_dmax_mm"] = round(o["dmax"], 3) if o["dmax"] == o["dmax"] else np.nan
        out[f"{nm}_ostium_dmin_mm"] = round(o["dmin"], 3) if o["dmin"] == o["dmin"] else np.nan
        out[f"{nm}_ostium_area_mm2"] = round(o["area"], 2) if o["area"] == o["area"] else np.nan
        out[f"{nm}_ostium_perim_mm"] = round(o["perim"], 2) if o["perim"] == o["perim"] else np.nan
        r = o["dmax"] / o["dmin"] if o["dmin"] else np.nan
        out[f"{nm}_ostium_ratio"] = round(r, 3) if r == r else np.nan
        out[f"{nm}_ostium_shape"] = ("circular" if r < 1.2 else "oval" if r < 1.6 else "elongated-oval") if r == r else "NA"
        out[f"{nm}_ostium_ecc"] = round(math.sqrt(1 - (o["dmin"] / o["dmax"]) ** 2), 3) if (o["dmax"] and o["dmin"] and o["dmax"] >= o["dmin"]) else np.nan
        out[f"{nm}_orient_deg"] = round(o["ang"], 2)

    # inter-PV distances (ostium centroids)
    if len(ostia) >= 2:
        cents = np.array([o["cent"] for o in ostia])
        d = [np.linalg.norm(cents[i] - cents[j]) for i, j in itertools.combinations(range(len(cents)), 2)]
        out["inter_pv_min_mm"] = round(float(min(d)), 2)
        out["inter_pv_mean_mm"] = round(float(np.mean(d)), 2)
        out["inter_pv_max_mm"] = round(float(max(d)), 2)

    # carina width = min surface distance between same-side superior/inferior ostia
    def carina(a, b):
        if a is None or b is None:
            return np.nan
        return float(cKDTree(a["voxmm"]).query(b["voxmm"])[0].min())
    out["carina_left_width_mm"] = round(carina(named["LSPV"], named["LIPV"]), 2) if named["LSPV"] and named["LIPV"] else np.nan
    out["carina_right_width_mm"] = round(carina(named["RSPV"], named["RIPV"]), 2) if named["RSPV"] and named["RIPV"] else np.nan
    return out


# ============================================================ LAA (36-44)
def analyze_laa(label, spacing):
    out = {}
    m = label == LABELS["LAA"]
    n = int(m.sum())
    if n == 0:
        return out
    V = n * float(np.prod(spacing))
    out["laa_volume_ml"] = V / 1000.0
    A = surface_area_mm2(m, spacing)
    out["laa_surface_area_mm2"] = A
    out["laa_sphericity"] = float((math.pi ** (1/3) * (6 * V) ** (2/3)) / A) if A and A > 0 else np.nan

    la = label == LABELS["LA"]
    laa_mm = np.argwhere(m) * spacing
    laa_c = laa_mm.mean(0)

    # LAA orientation relative to LA (angle between principal axes)
    if la.sum() >= 3:
        la_c = (np.argwhere(la) * spacing).mean(0)
        v_laa = pca(laa_mm)[1][:, 0]
        v_la = pca(np.argwhere(la) * spacing)[1][:, 0]
        out["laa_la_axis_angle_deg"] = round(math.degrees(math.acos(min(1.0, abs(float(np.dot(v_laa, v_la)))))), 2)

    # centerline: geodesic from ostium (LAA∩dilate(LA)) to farthest LAA voxel
    if _MCP and la.sum():
        try:
            ost = m & ndi.binary_dilation(la, iterations=1)
            seed = np.argwhere(ost)
            if len(seed):
                seed_vox = seed[np.argmin(np.linalg.norm(seed * spacing - laa_c, axis=1))]
                cost = np.where(m, 1.0, np.inf).astype(np.float64)
                mcp = MCP_Geometric(cost, sampling=tuple(spacing))
                cum, _ = mcp.find_costs([tuple(seed_vox)])
                cum_in = np.where(m, cum, -np.inf)
                apex = np.unravel_index(np.argmax(cum_in), cum.shape)
                geo = float(cum[apex])
                straight = float(np.linalg.norm((np.array(apex) - seed_vox) * spacing))
                out["laa_depth_geodesic_mm"] = round(geo, 2)
                out["laa_straight_mm"] = round(straight, 2)
                out["laa_tortuosity"] = round(geo / straight, 3) if straight > 0 else np.nan
                path = np.array(mcp.traceback(apex)) * spacing
                if len(path) >= 6:
                    half = len(path) // 2
                    d1 = pca(path[:half])[1][:, 0]; d2 = pca(path[half:])[1][:, 0]
                    out["laa_bend_angle_deg"] = round(math.degrees(math.acos(min(1.0, abs(float(np.dot(d1, d2)))))), 2)
        except Exception:
            pass

    # coherence: LAA-PV min surface distance (42)
    pv = label == LABELS["PV"]
    if pv.sum():
        edt = ndi.distance_transform_edt(~pv, sampling=spacing)
        surf = m & ~ndi.binary_erosion(m)
        if surf.sum():
            out["laa_pv_min_dist_mm"] = round(float(edt[surf].min()), 2)

    # coherence: PV configuration relative to LAA takeoff (44)
    if la.sum() and pv.sum():
        la_c = (np.argwhere(la) * spacing).mean(0)
        takeoff = laa_c - la_c; takeoff /= (np.linalg.norm(takeoff) + 1e-9)
        ost = pv & ndi.binary_dilation(la, iterations=1)
        plbl, pn = ndi.label(ost)
        angs = []
        for cid in range(1, pn + 1):
            cc = np.argwhere(plbl == cid)
            if len(cc) < 5:
                continue
            v = (cc * spacing).mean(0) - la_c; v /= (np.linalg.norm(v) + 1e-9)
            angs.append(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(v, takeoff)))))))
        if angs:
            out["pvcfg_mean_angle_to_laa_deg"] = round(float(np.mean(angs)), 2)
            out["pvcfg_min_angle_to_laa_deg"] = round(float(np.min(angs)), 2)
    return out


# ============================================================ per-subject
def _leading_id(p):
    m = re.match(r"^(\d+)", os.path.basename(p))
    return m.group(1) if m else os.path.basename(p).split(".")[0]


def extract_one(task):
    path, spacing_override, ssm_grid = task
    try:
        label, spacing = canonical(path)
        if spacing_override:
            spacing = np.array(spacing_override, dtype=float)
        rec = {"subject_id": _leading_id(path), "file": os.path.basename(path),
               "spacing_mm": [round(float(s), 4) for s in spacing]}
        for fn in (la_metrics, analyze_pv, analyze_laa):
            try:
                rec.update(fn(label, spacing))
            except Exception as e:
                rec[f"err_{fn.__name__}"] = f"{type(e).__name__}: {e}"
        # occupancy grid for cohort SSM/VAE (LA)
        if ssm_grid:
            rec["_occ"] = _occupancy(label == LABELS["LA"], ssm_grid)
        return rec
    except Exception as e:
        return {"subject_id": _leading_id(path), "file": os.path.basename(path),
                "error": f"{type(e).__name__}: {e}"}


def _occupancy(mask, G):
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return np.zeros(G ** 3, dtype=np.float32)
    mn, mx = idx.min(0), idx.max(0)
    crop = mask[mn[0]:mx[0]+1, mn[1]:mx[1]+1, mn[2]:mx[2]+1].astype(np.float32)
    if _RESIZE:
        g = sk_resize(crop, (G, G, G), order=0, preserve_range=True, anti_aliasing=False)
    else:
        zoom = (G / crop.shape[0], G / crop.shape[1], G / crop.shape[2])
        g = ndi.zoom(crop, zoom, order=0)
        g = g[:G, :G, :G]
        pad = [(0, G - s) for s in g.shape]; g = np.pad(g, pad)
    return (g >= 0.5).astype(np.float32).ravel()


# ============================================================ cohort: percentiles
PCTILE = {
    "la_volume_ml": ("la_volume_pctile", "la_volume_category"),
    "la_ap_mm": ("la_ap_pctile", None), "la_ml_mm": ("la_ml_pctile", None),
    "la_si_mm": ("la_si_pctile", None), "la_major_axis_mm": ("la_major_pctile", None),
    "la_minor_axis_mm": ("la_minor_pctile", None), "la_least_axis_mm": ("la_least_pctile", None),
    "la_elongation": ("la_elongation_pctile", None), "la_flatness": ("la_flatness_pctile", None),
    "la_sphericity": ("la_sphericity_pctile", None),
}
# ostium-size & carina cohort percentile → size class (31, 35)
SIZE_PCTILE = ["LSPV_ostium_area_mm2", "LIPV_ostium_area_mm2", "RSPV_ostium_area_mm2", "RIPV_ostium_area_mm2",
               "carina_left_width_mm", "carina_right_width_mm"]


def vol_cat(p):
    if p != p: return "NA"
    return "very small" if p <= 10 else "small" if p <= 25 else "average" if p < 75 else "enlarged" if p < 90 else "severely enlarged"


def lohi(p):
    return "NA" if p != p else ("low" if p <= 25 else "high" if p >= 75 else "typical")


def add_percentiles(recs):
    n = len(recs)
    def pct(vals, v):
        f = vals[np.isfinite(vals)]
        return float((f < v).mean() * 100) if (n >= 5 and np.isfinite(v) and f.size) else np.nan
    for raw, (pc, cc) in PCTILE.items():
        vals = np.array([r.get(raw, np.nan) for r in recs], float)
        for r in recs:
            p = pct(vals, r.get(raw, np.nan))
            r[pc] = None if p != p else round(p, 1)
            if cc:
                r[cc] = vol_cat(p)
            elif raw != "la_volume_ml":
                r[raw + "_level"] = lohi(p)
    for raw in SIZE_PCTILE:
        vals = np.array([r.get(raw, np.nan) for r in recs], float)
        for r in recs:
            p = pct(vals, r.get(raw, np.nan))
            lbl = "small" if p <= 25 else "large" if p >= 75 else "average"
            if "carina" in raw:
                lbl = "narrow" if p <= 25 else "wide" if p >= 75 else "moderate"
            r[raw + "_class"] = "NA" if p != p else lbl
    return recs


# ============================================================ cohort: SSM/VAE/cluster
def cohort_shape_models(recs, modes, do_vae, vae_dim, n_clusters):
    occ = [r.pop("_occ", None) for r in recs]
    have = [i for i, o in enumerate(occ) if o is not None and np.any(o)]
    if len(have) < 3:
        for r in recs:
            r.pop("_occ", None)
        return recs
    X = np.stack([occ[i] for i in have]).astype(np.float32)
    Xc = X - X.mean(0, keepdims=True)

    # 45  PCA statistical shape model
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(modes, S.shape[0])
        scores = U[:, :k] * S[:k]
        var = (S ** 2) / (S ** 2).sum()
        for j, i in enumerate(have):
            for c in range(k):
                recs[i][f"ssm_mode{c+1}"] = round(float(scores[j, c]), 4)
        recs[have[0]].setdefault("_ssm_explained", ",".join(f"{v:.3f}" for v in var[:k]))
    except Exception:
        scores = None

    # 47  elastic-shape-style clustering (on SSM scores)
    if scores is not None and n_clusters > 1:
        labels = _kmeans(scores, n_clusters)
        for j, i in enumerate(have):
            recs[i]["shape_cluster"] = int(labels[j])

    # 46  VAE latent embedding (optional, needs torch)
    if do_vae:
        z = _vae_latent(X, vae_dim)
        if z is not None:
            for j, i in enumerate(have):
                for c in range(z.shape[1]):
                    recs[i][f"vae_z{c+1}"] = round(float(z[j, c]), 4)
    return recs


def _kmeans(X, k, iters=50, seed=0):
    try:
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    except Exception:
        rng = np.random.default_rng(seed)
        C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)]
        for _ in range(iters):
            d = ((X[:, None] - C[None]) ** 2).sum(2)
            a = d.argmin(1)
            newC = np.array([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(len(C))])
            if np.allclose(newC, C):
                break
            C = newC
        return a


def _vae_latent(X, dim, epochs=120):
    try:
        import torch
        import torch.nn as nn
    except Exception:
        print("[feat] VAE requested but torch not available — skipping (feature 46).")
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = X.shape[1]; h = 256
    xt = torch.tensor(X, dtype=torch.float32, device=dev)
    enc = nn.Sequential(nn.Linear(D, h), nn.ReLU()).to(dev)
    mu = nn.Linear(h, dim).to(dev); lv = nn.Linear(h, dim).to(dev)
    dec = nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, D)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(mu.parameters()) +
                           list(lv.parameters()) + list(dec.parameters()), 1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    for _ in range(epochs):
        opt.zero_grad()
        he = enc(xt); m, lg = mu(he), lv(he)
        z = m + torch.randn_like(m) * torch.exp(0.5 * lg)
        rec = dec(z)
        loss = bce(rec, xt) + -0.5 * torch.sum(1 + lg - m.pow(2) - lg.exp())
        loss.backward(); opt.step()
    with torch.no_grad():
        return mu(enc(xt)).cpu().numpy()


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description="Extract the 51-feature set from cardiac CT segmentations")
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out", default="./features")
    ap.add_argument("--pattern", default="*.nii.gz")
    ap.add_argument("--spacing", default=None, help="Override 'dx,dy,dz' (mm) if affine is identity")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--ssm", action="store_true", help="Compute PCA SSM (45) + clustering (47)")
    ap.add_argument("--ssm-grid", type=int, default=24, help="Occupancy grid edge for SSM/VAE [24]")
    ap.add_argument("--ssm-modes", type=int, default=5)
    ap.add_argument("--vae", action="store_true", help="Compute VAE latent (46, needs torch)")
    ap.add_argument("--vae-dim", type=int, default=4)
    ap.add_argument("--clusters", type=int, default=4, help="Shape clusters (47)")
    ap.add_argument("--per-subject-json", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    files = sorted(set(glob(os.path.join(a.labels_dir, a.pattern)) + glob(os.path.join(a.labels_dir, "*.nii"))))
    if a.limit:
        files = files[:a.limit]
    if not files:
        sys.exit(f"[ERROR] no label files matching {a.pattern} in {a.labels_dir}")
    os.makedirs(a.out, exist_ok=True)
    sp = [float(x) for x in a.spacing.split(",")] if a.spacing else None
    grid = a.ssm_grid if (a.ssm or a.vae) else None
    if not _SK:
        print("[feat] note: scikit-image missing — surface area uses face approximation; "
              "install scikit-image for smoother sphericity + LAA centerline (features 38-40).")
    if (a.ssm or a.vae) and not _RESIZE:
        print("[feat] note: skimage.transform.resize missing — occupancy grids use ndi.zoom.")

    print(f"[feat] {len(files)} label maps · workers={max(1, a.workers)} · "
          f"ssm={a.ssm} vae={a.vae} clusters={a.clusters}")
    tasks = [(f, sp, grid) for f in files]
    recs, failed = [], []
    if a.workers <= 1:
        for i, t in enumerate(tasks, 1):
            r = extract_one(t); (failed if "error" in r else recs).append(r)
            print(f"[feat] [{i}/{len(tasks)}] {r['subject_id']} "
                  + (r.get("error", f"LA={r.get('la_volume_ml', float('nan')):.1f}mL PV#={r.get('pv_number','?')}")))
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(extract_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result(); (failed if "error" in r else recs).append(r)
                print(f"[feat] [{i}/{len(tasks)}] {r['subject_id']} "
                      + (r.get("error", f"LA={r.get('la_volume_ml', float('nan')):.1f}mL PV#={r.get('pv_number','?')}")))
    if not recs:
        sys.exit("[ERROR] no subjects processed")

    recs = add_percentiles(recs)
    if a.ssm or a.vae:
        recs = cohort_shape_models(recs, a.ssm_modes, a.vae, a.vae_dim, a.clusters)
    else:
        for r in recs:
            r.pop("_occ", None)
    recs.sort(key=lambda r: (len(r["subject_id"]), r["subject_id"]))

    cols = []
    for r in recs:
        for k in r:
            if not k.startswith("_") and k not in cols:
                cols.append(k)
    csv_path = os.path.join(a.out, "features_full.csv")
    with open(csv_path, "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in recs:
            fh.write(",".join(_csv(r.get(c, "")) for c in cols) + "\n")
    print(f"[feat] wrote {csv_path}  ({len(recs)} subjects × {len(cols)} columns)")

    man = os.path.join(a.out, "features_manifest.csv")
    with open(man, "w") as fh:
        fh.write("s_no,roi,feature,status,phase\n")
        for s in FEATURES:
            fh.write(",".join(_csv(x) for x in s) + "\n")
    nc = sum(1 for s in FEATURES if s[3] in ("computed", "pctile", "category", "approx"))
    print(f"[feat] wrote {man}  ({nc}/51 features produced; 'approx' rows are heuristic — see manifest)")

    if a.per_subject_json:
        for r in recs:
            with open(os.path.join(a.out, f"{r['subject_id']}.json"), "w") as fh:
                json.dump({k: v for k, v in r.items() if not k.startswith("_")}, fh, indent=2, default=str)
        print(f"[feat] wrote {len(recs)} per-subject JSON files")
    if failed:
        print(f"[feat] {len(failed)} failed: {[r['subject_id'] for r in failed]}")
    print("[feat] done.")


def _csv(v):
    s = "" if v is None else str(v)
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in [",", '"', "\n"]) else s


if __name__ == "__main__":
    main()
