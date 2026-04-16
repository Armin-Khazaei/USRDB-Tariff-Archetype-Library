"""
feature_extraction.py
=====================
Extract per-period pricing features from parsed USRDB rate structures.

This module bridges usrdb_parser (raw tidy tables) and pricing_clustering
(k-means input). For each plan and each period, it computes:
  - c_avg_mean   : mean average cost across a consumption grid (ACATI proxy)
  - growth_ratio : ratio of average cost at q_max vs q_start (GRATI proxy)
  - num_tiers    : number of tiers in the period
  - tier_growth_ratio : last-tier price / first-tier price
  - price_rank   : 0 (low) / 1 (mid) / 2 (high) based on cross-period ordering

Public API
----------
  generate_rate_features(df, rate_col, q_step, q_max) -> DataFrame
"""

import json
import ast
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Rate parsing (duplicated from pricing_clustering for standalone use)
# ---------------------------------------------------------------------------

def _safe_to_rates(detail_str):
    """
    Parse a USRDB rate detail string into per-period tier lists.

    Returns list[period] where each period is list[(price, ceiling)].
    price = base_rate + adjustment; ceiling = tier upper bound (inf if open).
    """
    s = (detail_str.decode("utf-8", errors="ignore").strip()
         if isinstance(detail_str, (bytes, bytearray))
         else str(detail_str).strip())
    v = json.loads(s) if s and s[0] == "[" else ast.literal_eval(s)
    out = []
    for period in v:
        tiers = []
        if isinstance(period, (list, tuple)):
            for t in period:
                if isinstance(t, (list, tuple)) and len(t) >= 3:
                    price = float(t[0]) + float(t[2])
                    ceiling = np.inf if float(t[1]) >= 1e20 / 2 else float(t[1])
                    tiers.append((price, ceiling))
        out.append(tiers)
    return out


def _price_rank_map(price_by_period):
    """
    Map period indices to price ranks: 0=low, 1=mid, 2=high.

    Uses last-tier price for ranking. For k=2: assigns 0 and 2.
    For k>=3: assigns 0, 1, 2 with nearest-pivot assignment.
    """
    pairs = [(pid, tiers[-1][0])
             for pid, tiers in enumerate(price_by_period) if tiers]
    k = len(pairs)
    if k < 2:
        return {}
    pairs.sort(key=lambda x: (x[1], x[0]))
    m = {}
    if k == 2:
        m[pairs[0][0]] = 0
        m[pairs[1][0]] = 2
    else:
        lo, md, hi = 0, k // 2, k - 1
        piv = {pairs[lo][0]: 0, pairs[md][0]: 1, pairs[hi][0]: 2}
        piv_val = {pid: price for pid, price in pairs if pid in piv}
        m.update(piv)
        for pid, price in pairs:
            if pid in piv:
                continue
            tgt = min(piv_val.items(), key=lambda kv: abs(price - kv[1]))[0]
            m[pid] = piv[tgt]
    return m


def _avg_cost_curve(tiers, q_grid):
    """
    Compute the average cost curve C_avg(q) for a tiered rate structure.

    Parameters
    ----------
    tiers  : list of (price, ceiling) tuples, sorted by ceiling ascending.
             Last tier ceiling should be inf (open-ended).
    q_grid : 1-D array of consumption levels at which to evaluate.

    Returns
    -------
    AC : array of average cost ($/unit) at each q in q_grid.
    """
    # Sort tiers: finite ceilings first (ascending), then infinite
    finite = [(p, c) for p, c in tiers if np.isfinite(c)]
    infs = [(p, c) for p, c in tiers if not np.isfinite(c)]
    finite.sort(key=lambda x: x[1])
    tiers_sorted = finite + infs

    prices = [p for p, _ in tiers_sorted]
    bounds = [c if np.isfinite(c) else np.inf for _, c in tiers_sorted]

    C = np.zeros_like(q_grid, dtype=float)
    rem = q_grid.copy()
    start = 0.0

    for p, end in zip(prices, bounds):
        seg_len = np.minimum(rem, end - start)
        seg_len[seg_len < 0] = 0
        C += np.clip(seg_len, 0, None) * p
        rem -= np.clip(seg_len, 0, None)
        start = end
        if np.all(rem <= 1e-12):
            break

    with np.errstate(divide="ignore", invalid="ignore"):
        AC = np.where(q_grid > 0, C / q_grid, 0.0)
    return AC


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_rate_features(
    df,
    rate_col,
    q_step=50.0,
    q_max=5000.0,
):
    """
    Extract per-period pricing features for every plan in df.

    For each plan (row in df) and each period within its rate structure,
    computes a feature vector suitable for downstream k-means clustering.

    Parameters
    ----------
    df       : DataFrame indexed by plan_id, with rate detail column.
    rate_col : column name containing the rate structure string
               (e.g. 'energyratestructure_detail' or 'demandratestructure_detail').
    q_step   : consumption grid step size.
    q_max    : maximum consumption level for cost curve evaluation.

    Returns
    -------
    DataFrame with one row per (plan_id, period_id), columns:
        plan_id, rate_col, period_id, price_rank,
        c_avg_mean, c_avg_median, c_total_end,
        growth_ratio, tier_growth_ratio, c_avg_std,
        num_tiers, c_avg_start, c_avg_end, p0_price, last_tier_price
    """
    n_steps = int(np.round(q_max / q_step)) + 1
    q_grid = np.linspace(0, q_max, n_steps)

    rows = []

    for plan_id, row in df.iterrows():
        rate_str = row[rate_col]
        if pd.isna(rate_str):
            continue

        try:
            periods = _safe_to_rates(rate_str)
            if not periods:
                continue
            ranks_map = _price_rank_map(periods)
        except Exception:
            continue

        for period_id, tiers in enumerate(periods):
            if not tiers:
                continue

            # Cost curve over consumption grid
            c_avg_vector = _avg_cost_curve(tiers, q_grid)
            c_avg_pos = c_avg_vector[1:]  # exclude q=0

            # Level features
            c_avg_mean = float(np.nanmean(c_avg_pos))
            c_avg_median = float(np.nanmedian(c_avg_pos))
            c_avg_start = float(c_avg_pos[0]) if len(c_avg_pos) > 0 else np.nan
            c_avg_end = float(c_avg_pos[-1]) if len(c_avg_pos) > 0 else np.nan

            # Trend features
            growth_ratio = (c_avg_end / c_avg_start
                            if c_avg_start > 1e-6 else np.nan)
            c_avg_std = float(np.nanstd(c_avg_pos))

            # Total cost at q_max
            c_total_end = c_avg_end * q_max if not np.isnan(c_avg_end) else np.nan

            # Tier structure features
            num_tiers = len(tiers)
            p0_price = tiers[0][0]
            last_tier_price = tiers[-1][0]
            tier_growth_ratio = (last_tier_price / p0_price
                                 if p0_price > 1e-6 else np.nan)

            rows.append({
                "plan_id": plan_id,
                "rate_col": rate_col,
                "period_id": period_id,
                "price_rank": ranks_map.get(period_id, np.nan),
                "c_avg_mean": c_avg_mean,
                "c_avg_median": c_avg_median,
                "c_total_end": c_total_end,
                "growth_ratio": growth_ratio,
                "tier_growth_ratio": tier_growth_ratio,
                "c_avg_std": c_avg_std,
                "num_tiers": num_tiers,
                "c_avg_start": c_avg_start,
                "c_avg_end": c_avg_end,
                "p0_price": p0_price,
                "last_tier_price": last_tier_price,
            })

    df_features = pd.DataFrame(rows)
    return df_features
