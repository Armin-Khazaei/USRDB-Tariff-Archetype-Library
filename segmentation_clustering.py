"""
segmentation_clustering.py
==========================
Temporal segmentation analysis for TOU tariff period maps.

Two approaches:
1. Rule-based classification (k=2, k=3 plans) into seasonal/diurnal/
   summer-afternoon/irregular archetypes.
2. DBSCAN clustering (k>=4 plans) with centroid-ordered label
   canonicalization and medoid selection.

Public API
----------
  classify_tou_patterns(df, col)  -> df_annotated, summary
  dbscan_segmentation(df, ...)    -> df_annotated, meta
"""

import json
import ast
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_12x24(x):
    """Parse a JSON/Python-literal string into a 12x24 numpy array."""
    if x is None:
        raise ValueError("None input")
    s = (x.decode("utf-8", errors="ignore").strip()
         if isinstance(x, (bytes, bytearray)) else str(x).strip())
    if not s:
        raise ValueError("Empty string")
    try:
        v = json.loads(s)
    except Exception:
        v = ast.literal_eval(s)
    arr = np.array([[float(z) for z in row] for row in v], dtype=float)
    if arr.shape != (12, 24):
        raise ValueError(f"Expected (12,24), got {arr.shape}")
    return arr


def count_periods(M):
    """Count distinct non-NaN labels in a 12x24 matrix."""
    vals = M[~np.isnan(M)]
    return int(len(np.unique(vals))) if vals.size else 0


# ---------------------------------------------------------------------------
# Rule-based classification helpers (k=2, k=3)
# ---------------------------------------------------------------------------

def _row_const_ratio(M):
    """Fraction of rows where all valid cells share one label."""
    cnt = 0
    for r in range(M.shape[0]):
        row = M[r, :][~np.isnan(M[r, :])]
        if row.size > 0 and len(np.unique(row)) <= 1:
            cnt += 1
    return cnt / M.shape[0]


def _col_const_ratio(M):
    """Fraction of columns where all valid cells share one label."""
    cnt = 0
    for c in range(M.shape[1]):
        col = M[:, c][~np.isnan(M[:, c])]
        if col.size > 0 and len(np.unique(col)) <= 1:
            cnt += 1
    return cnt / M.shape[1]


def _dominant_label_and_share(M):
    vals = M[~np.isnan(M)].astype(int)
    if vals.size == 0:
        return None, 0.0
    lab, n = Counter(vals).most_common(1)[0]
    return lab, n / vals.size


def _bounding_rect_density(mask):
    """Bounding-rectangle density for a boolean mask."""
    idx = np.argwhere(mask)
    if idx.size == 0:
        return False, 0, 0, 0.0, 0.0
    rmin, cmin = idx.min(axis=0)
    rmax, cmax = idx.max(axis=0)
    rect_area = (rmax - rmin + 1) * (cmax - cmin + 1)
    nonbg_count = mask.sum()
    density = nonbg_count / rect_area if rect_area > 0 else 0.0
    row_span = rmax - rmin + 1
    col_span = cmax - cmin + 1
    cover = nonbg_count / mask.size
    is_rect = (density >= 0.85 and cover >= 0.05
               and row_span >= 2 and col_span >= 2)
    return is_rect, row_span, col_span, density, cover


def _classify_k2(M, row_thr=0.95, col_thr=0.95):
    rr = _row_const_ratio(M)
    cc = _col_const_ratio(M)
    if rr >= row_thr and cc < col_thr:
        return "k2_month_only"
    if cc >= col_thr and rr < row_thr:
        return "k2_hour_only"
    bg, _ = _dominant_label_and_share(M)
    if bg is not None:
        nonbg = (M != bg) & (~np.isnan(M))
        is_rect, *_ = _bounding_rect_density(nonbg)
        if is_rect:
            return "k2_block_pm"
    return "k2_other"


def _classify_k3(M, row_thr=0.95, col_thr=0.95):
    rr = _row_const_ratio(M)
    cc = _col_const_ratio(M)
    if rr >= row_thr and cc < col_thr:
        return "k3_month_only"
    if cc >= col_thr and rr < row_thr:
        return "k3_hour_only"
    bg, _ = _dominant_label_and_share(M)
    if bg is not None:
        nonbg = (M != bg) & (~np.isnan(M))
        is_rect, *_ = _bounding_rect_density(nonbg)
        if is_rect:
            labs = np.unique(M[nonbg].astype(int))
            if len(labs) >= 2:
                return "k3_block_pm_transition"
            return "k3_block_pm"
    return "k3_other"


# ---------------------------------------------------------------------------
# Public: rule-based classification
# ---------------------------------------------------------------------------

def classify_tou_patterns(df, col, out_prefix=None, row_thr=0.95, col_thr=0.95):
    """
    Classify period maps into segmentation archetypes (k=2/3/4+).

    Parameters
    ----------
    df  : DataFrame with a column containing 12x24 period-map strings
    col : column name of the period map

    Returns
    -------
    df_out  : annotated DataFrame with pattern labels
    summary : value_counts of pattern types
    """
    if out_prefix is None:
        out_prefix = col

    ks, cats = [], []
    for x in df[col]:
        try:
            M = parse_12x24(x)
        except Exception:
            ks.append(pd.NA)
            cats.append(pd.NA)
            continue
        k = count_periods(M)
        if k <= 1:
            ks.append(int(k))
            cats.append(pd.NA)
            continue
        ks.append(int(k))
        if k == 2:
            cats.append(_classify_k2(M, row_thr, col_thr))
        elif k == 3:
            cats.append(_classify_k3(M, row_thr, col_thr))
        else:
            cats.append("k4plus")

    df_out = df.copy()
    df_out[f"{out_prefix}_k"] = pd.array(ks, dtype="Int64")
    df_out[f"{out_prefix}_pattern"] = pd.Series(cats, dtype="string")

    valid = (df_out[f"{out_prefix}_k"] > 1) & df_out[f"{out_prefix}_pattern"].notna()
    counts = df_out.loc[valid, f"{out_prefix}_pattern"].value_counts()
    total = counts.sum() if counts.size else 0
    summary = pd.DataFrame({
        "count": counts,
        "ratio": (counts / total).round(4),
    }) if total > 0 else pd.DataFrame(columns=["count", "ratio"])

    return df_out, summary


# ---------------------------------------------------------------------------
# DBSCAN for k>=4 segmentation
# ---------------------------------------------------------------------------

def _canonize_by_centroid(M):
    """Reindex period labels by spatial centroid order (label invariance)."""
    M = np.asarray(M, dtype=float).copy()
    mask = ~np.isnan(M)
    labs = np.unique(M[mask]).astype(float)
    if labs.size <= 1:
        return M
    anchors = []
    for lab in labs:
        rr, cc = np.where(M == lab)
        anchors.append((lab, rr.mean(), cc.mean()))
    anchors.sort(key=lambda t: (t[1], t[2]))
    remap = {lab: i for i, (lab, _, _) in enumerate(anchors)}
    out = M.copy()
    for lab in labs:
        out[M == lab] = remap[lab]
    return out


def _flatten_for_hamming(M):
    """Flatten 12x24 matrix; encode NaN as -1."""
    v = np.asarray(M, dtype=float).flatten()
    v[np.isnan(v)] = -1.0
    return v


def _masked_hamming_matrix(X):
    """Compute pairwise masked Hamming distance matrix."""
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        A = X[i]
        maskA = A != -1
        valid = maskA & (X != -1)
        same = (X == A) & valid
        denom = valid.sum(axis=1).astype(float)
        denom[denom == 0] = 1.0
        D[i] = 1.0 - same.sum(axis=1) / denom
    return D


def _cluster_medoid(D, idx):
    """Select medoid from a cluster by minimizing intra-cluster distance."""
    if len(idx) == 1:
        return idx[0]
    sub = D[np.ix_(idx, idx)]
    return idx[int(np.argmin(sub.sum(axis=1)))]


def dbscan_segmentation(
    df,
    schedule_col="energyweekdayschedule",
    pattern_col="energyweekday_pattern",
    target_class="k4plus",
    eps=0.08,
    min_samples=6,
    out_col="k4plus_dbscan",
):
    """
    Apply DBSCAN clustering to high-period (k>=4) TOU segmentation maps.

    Parameters
    ----------
    df : DataFrame (output of classify_tou_patterns)
    schedule_col : column with 12x24 period-map strings
    pattern_col  : column with pattern labels (from classify_tou_patterns)
    target_class : which pattern to cluster (default 'k4plus')
    eps          : DBSCAN neighborhood radius (fraction mismatch tolerance)
    min_samples  : DBSCAN min core-point cardinality

    Returns
    -------
    df_out : DataFrame with cluster labels in out_col
    meta   : dict with cluster sizes, ratios, medoid indices, etc.
    """
    sub = df[df[pattern_col] == target_class].copy().reset_index(drop=False)
    if sub.empty:
        raise ValueError(f"No '{target_class}' samples found.")

    # Parse, canonize, flatten
    mats, flats, keep_rows = [], [], []
    for ridx, s in zip(sub["index"], sub[schedule_col]):
        try:
            M = parse_12x24(s)
            Mc = _canonize_by_centroid(M)
        except Exception:
            continue
        mats.append(Mc)
        flats.append(_flatten_for_hamming(Mc))
        keep_rows.append(ridx)

    if not flats:
        raise ValueError("No valid 12x24 schedules found.")

    X = np.vstack(flats)
    D = _masked_hamming_matrix(X)

    # DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels_raw = db.fit_predict(D)

    # Reorder clusters by size (descending)
    valid = labels_raw >= 0
    uniq, cnt = np.unique(labels_raw[valid], return_counts=True)
    order = uniq[np.argsort(-cnt)]
    remap = {old: new for new, old in enumerate(order)}
    labels_new = np.array([remap.get(l, -1) for l in labels_raw], dtype=int)

    total_kept = int(valid.sum())
    cluster_sizes = {remap[o]: int(c) for o, c in zip(uniq, cnt)}
    cluster_ratios = {k: v / total_kept for k, v in cluster_sizes.items()}

    # Medoids
    medoids = {}
    medoid_mats = {}
    for old in order:
        new_id = remap[old]
        idx = np.where(labels_raw == old)[0].tolist()
        med_idx = _cluster_medoid(D, idx)
        medoids[new_id] = med_idx
        medoid_mats[new_id] = mats[med_idx]

    # Write back
    df_out = df.copy()
    df_out[out_col] = pd.array([pd.NA] * len(df_out), dtype="Int64")
    disp = pd.Series(labels_new).replace(-1, pd.NA).astype("Int64")
    df_out.loc[keep_rows, out_col] = disp.values

    meta = {
        "keep_rows": keep_rows,
        "labels": labels_new,
        "n_clusters": len(cluster_sizes),
        "n_noise": int((labels_raw == -1).sum()),
        "total_clustered": total_kept,
        "cluster_sizes": cluster_sizes,
        "cluster_ratios": {k: round(v, 4) for k, v in cluster_ratios.items()},
        "medoid_indices": medoids,
        "medoid_matrices": medoid_mats,
        "eps": eps,
        "min_samples": min_samples,
    }
    return df_out, meta
