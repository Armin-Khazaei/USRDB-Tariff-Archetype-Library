"""
pricing_clustering.py
=====================
K-means clustering of tariff pricing structures.

Supports period-count-specific workflows:
  - 1-period plans: 2 features (avg cost, growth ratio)
  - 2-period plans: 4 features (low/high × avg cost, growth ratio)
  - 3-period plans: 6 features (low/mid/high × avg cost, growth ratio)
  - 4+ period plans: 4 features per period (avg cost, growth ratio,
    num tiers, tier growth ratio)

Public API
----------
  cluster_k1_plans(df_features, best_k, ...)  -> df_result, meta
  cluster_k2_plans(df_features, best_k, ...)  -> df_result, meta
  cluster_k3_plans(df_features, best_k, ...)  -> df_result, meta
  cluster_k4plus_plans(df_features, best_k, ...)  -> df_result, meta
  compute_elbow(X_scaled, k_range)            -> wcss_list
"""

import json
import ast
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Rate parsing helpers
# ---------------------------------------------------------------------------

def parse_rate_structure(detail_str):
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


def price_rank_map(price_by_period):
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


# ---------------------------------------------------------------------------
# Common clustering utilities
# ---------------------------------------------------------------------------

def compute_elbow(X_scaled, k_range=range(2, 11)):
    """Compute WCSS for a range of k values (elbow method)."""
    wcss = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10,
                    max_iter=300, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
    return list(k_range), wcss


def _filter_outliers(df, features, lower_q=0.01, upper_q=0.99):
    """Remove outliers by quantile filtering on feature columns."""
    q = df[features].quantile([lower_q, upper_q])
    mask = pd.Series(True, index=df.index)
    for col in features:
        mask &= (df[col] >= q.loc[lower_q, col]) & (df[col] <= q.loc[upper_q, col])
    return df[mask].copy()


def _run_kmeans(df, features, best_k, k_range_max=10):
    """
    Run k-means clustering pipeline.

    Returns
    -------
    df_result : DataFrame with 'cluster' column (1-indexed)
    meta : dict with scaler, kmeans model, centers, wcss, etc.
    """
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow
    k_range, wcss = compute_elbow(X_scaled, range(2, k_range_max + 1))

    # Final clustering
    km = KMeans(n_clusters=best_k, init="k-means++", n_init=10,
                max_iter=300, random_state=42)
    clusters = km.fit_predict(X_scaled) + 1  # 1-indexed

    df_result = df.copy()
    df_result["cluster"] = clusters

    centers_orig = scaler.inverse_transform(km.cluster_centers_)

    # Cluster statistics
    counts = pd.Series(clusters).value_counts().sort_index()
    total = len(clusters)

    meta = {
        "scaler": scaler,
        "kmeans": km,
        "centers_scaled": km.cluster_centers_,
        "centers_original": centers_orig,
        "k_range": k_range,
        "wcss": wcss,
        "best_k": best_k,
        "cluster_sizes": counts.to_dict(),
        "cluster_shares": {k: round(v / total, 4) for k, v in counts.items()},
        "n_total": total,
        "features": features,
    }
    return df_result, meta


def _find_nearest_plan(df_clustered, meta, cluster_num):
    """Find the plan closest to a cluster center in scaled space."""
    features = meta["features"]
    scaler = meta["scaler"]
    X_scaled = scaler.transform(df_clustered[features])
    center = meta["centers_scaled"][cluster_num - 1]

    mask = df_clustered["cluster"] == cluster_num
    X_cluster = X_scaled[mask.values]
    df_cluster = df_clustered[mask]

    if len(X_cluster) == 0:
        return None
    dist = np.linalg.norm(X_cluster - center, axis=1)
    idx = np.argmin(dist)
    return df_cluster.iloc[idx]


# ---------------------------------------------------------------------------
# 1-period plans
# ---------------------------------------------------------------------------

def cluster_k1_plans(
    df_features,
    best_k,
    rate_col_name="energyratestructure_detail",
    features=("c_avg_mean", "growth_ratio"),
    lower_quantile=0.005,
    upper_quantile=0.995,
    k_range_max=10,
):
    """
    Cluster 1-period energy rate plans.

    Parameters
    ----------
    df_features : DataFrame with per-period feature rows
        Must have columns: plan_id, price_rank, c_avg_mean, growth_ratio
    best_k : number of clusters
    """
    features = list(features)
    df_k1 = df_features[df_features["price_rank"].isna()].copy()
    df_k1 = df_k1.dropna(subset=features)
    if df_k1.empty:
        raise ValueError("No K=1 plan data found.")

    df_filtered = _filter_outliers(df_k1, features, lower_quantile, upper_quantile)
    if df_filtered.empty:
        raise ValueError("No data remaining after outlier filtering.")

    return _run_kmeans(df_filtered, features, best_k, k_range_max)


# ---------------------------------------------------------------------------
# 2-period plans
# ---------------------------------------------------------------------------

def cluster_k2_plans(
    df_features,
    best_k,
    rate_col_name="energyratestructure_detail",
    features=("c_avg_mean_low", "growth_ratio_low",
              "c_avg_mean_high", "growth_ratio_high"),
    lower_quantile=0.01,
    upper_quantile=0.99,
    k_range_max=10,
):
    """
    Cluster 2-period energy rate plans.

    Merges low-price and high-price period features into plan-level vectors.
    """
    features = list(features)
    source_cols = ["c_avg_mean", "growth_ratio"]

    df_low = df_features[
        (df_features["rate_col"] == rate_col_name) &
        (df_features["price_rank"] == 0.0)
    ].set_index("plan_id")[source_cols]

    df_high = df_features[
        (df_features["rate_col"] == rate_col_name) &
        (df_features["price_rank"] == 2.0)
    ].set_index("plan_id")[source_cols]

    df_plans = pd.merge(df_low, df_high, left_index=True, right_index=True,
                        suffixes=("_low", "_high"))
    df_plans.columns = features

    df_cleaned = df_plans.dropna(subset=features)
    if df_cleaned.empty:
        raise ValueError("No K=2 plan data found.")

    df_filtered = _filter_outliers(df_cleaned, features, lower_quantile, upper_quantile)
    if df_filtered.empty:
        raise ValueError("No data remaining after outlier filtering.")

    return _run_kmeans(df_filtered, features, best_k, k_range_max)


# ---------------------------------------------------------------------------
# 3-period plans
# ---------------------------------------------------------------------------

def cluster_k3_plans(
    df_features,
    best_k,
    rate_col_name="energyratestructure_detail",
    features=("c_avg_mean_low", "growth_ratio_low",
              "c_avg_mean_mid", "growth_ratio_mid",
              "c_avg_mean_high", "growth_ratio_high"),
    lower_quantile=0.01,
    upper_quantile=0.99,
    k_range_max=10,
):
    """
    Cluster 3-period energy rate plans.

    Merges low/mid/high period features into plan-level vectors.
    """
    features = list(features)
    source_cols = ["c_avg_mean", "growth_ratio"]

    df_low = df_features[
        (df_features["rate_col"] == rate_col_name) &
        (df_features["price_rank"] == 0.0)
    ].set_index("plan_id")[source_cols]

    df_mid = df_features[
        (df_features["rate_col"] == rate_col_name) &
        (df_features["price_rank"] == 1.0)
    ].set_index("plan_id")[source_cols]

    df_high = df_features[
        (df_features["rate_col"] == rate_col_name) &
        (df_features["price_rank"] == 2.0)
    ].set_index("plan_id")[source_cols]

    df_plans = pd.merge(df_low, df_mid, left_index=True, right_index=True,
                        suffixes=("_low", "_mid"))
    df_plans = pd.merge(df_plans, df_high, left_index=True, right_index=True,
                        suffixes=("", "_high"))
    df_plans.columns = features

    df_cleaned = df_plans.dropna(subset=features)
    if df_cleaned.empty:
        raise ValueError("No K=3 plan data found.")

    df_filtered = _filter_outliers(df_cleaned, features, lower_quantile, upper_quantile)
    if df_filtered.empty:
        raise ValueError("No data remaining after outlier filtering.")

    return _run_kmeans(df_filtered, features, best_k, k_range_max)


# ---------------------------------------------------------------------------
# 4+ period plans
# ---------------------------------------------------------------------------

def cluster_k4plus_plans(
    df_features,
    best_k,
    rate_col_name="energyratestructure_detail",
    features=("c_avg_mean", "growth_ratio", "num_tiers", "tier_growth_ratio"),
    lower_quantile=0.01,
    upper_quantile=0.99,
    k_range_max=10,
):
    """
    Cluster plans with 4+ periods (period-level clustering).

    Unlike K=1/2/3 which cluster at the plan level, this clusters
    individual periods from plans that have 4+ periods total.
    """
    features = list(features)

    plan_period_counts = df_features.groupby("plan_id")["period_id"].nunique()
    k4_plan_ids = plan_period_counts[plan_period_counts >= 4].index

    df_periods = df_features[
        (df_features["plan_id"].isin(k4_plan_ids)) &
        (df_features["rate_col"] == rate_col_name)
    ].copy()

    df_cleaned = df_periods.dropna(subset=features)
    if df_cleaned.empty:
        raise ValueError("No K>=4 period data found.")

    df_filtered = _filter_outliers(df_cleaned, features, lower_quantile, upper_quantile)
    if df_filtered.empty:
        raise ValueError("No data remaining after outlier filtering.")

    return _run_kmeans(df_filtered, features, best_k, k_range_max)
