"""
usrdb_parser.py
===============
Parse raw USRDB wide-format CSV into tidy long tables of
(section, period, tier, rate, max, adj, unit).

Three structural sections are supported:
  - energyratestructure   (energy charges, $/kWh)
  - demandratestructure   (TOU demand charges, $/kW or $/kVA)
  - flatdemandstructure   (flat demand charges, $/kW or $/kVA)

Public API
----------
  tidy_urdb_tiers(df)          -> long table of all tiers
  extract_flat_demand_months(df) -> month-level enablement for flat demand
  summarize_tier_brackets(tidy)  -> human-readable interval summary
  parse_usrdb_all(df)          -> (tiers_long, flat_months, summary)
"""

import re
import json
import ast
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SECTIONS = [
    "energyratestructure",
    "demandratestructure",
    "flatdemandstructure",
]

# (has_sell_field, fallback_unit_column)
_SECTION_META = {
    "energyratestructure":    (True,  "energyrateunit"),
    "demandratestructure":    (False, "demandrateunit"),
    "flatdemandstructure":    (False, "demandrateunit"),
    "coincidentratestructure":(False, "coincidentrateunit"),
}

_COL_RE = re.compile(
    r"^(?P<section>energyratestructure|demandratestructure"
    r"|flatdemandstructure|coincidentratestructure)"
    r"/period(?P<period>\d+)"
    r"/tier(?P<tier>\d+)"
    r"(?P<field>max|rate|adj|unit|sell)$"
)

# Fields used to decide whether a tier/period is "present"
PRESENCE_FIELDS = {"rate", "adj", "max"}

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5,  "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _key_col(df: pd.DataFrame) -> str:
    """Pick a row-identifier column: label > name > index."""
    if "label" in df.columns:
        return "label"
    if "name" in df.columns:
        return "name"
    df["row_id"] = df.index.astype(str)
    return "row_id"


def _safe_to_mat_12x24(x):
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


# ---------------------------------------------------------------------------
# 1. Melt wide-format rate structures into long table
# ---------------------------------------------------------------------------

def _melt_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Melt all *ratestructure columns into (id, section, period, tier, field, value)."""
    id_col = "label" if "label" in df.columns else (
        "name" if "name" in df.columns else None)

    value_vars = [c for c in df.columns if _COL_RE.match(c)]
    if not value_vars:
        return pd.DataFrame(
            columns=[id_col or "row_id", "section", "period", "tier", "field", "value"])

    melted = df[value_vars].copy()
    if id_col:
        melted.insert(0, id_col, df[id_col])
    else:
        melted.insert(0, "row_id", df.index.astype(str))

    long = melted.melt(
        id_vars=[id_col] if id_col else ["row_id"],
        var_name="key",
        value_name="value",
    )
    meta = pd.DataFrame(
        list(long["key"].apply(
            lambda k: _COL_RE.match(k).groupdict() if _COL_RE.match(k) else {})))
    out = pd.concat([long.drop(columns=["key"]), meta], axis=1)
    out["period"] = out["period"].astype("Int64")
    out["tier"] = out["tier"].astype("Int64")
    return out


def _pivot_fields(struct_long: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Pivot field dimension into columns and handle unit fallback."""
    if struct_long.empty:
        return struct_long

    idx_cols = [c for c in struct_long.columns
                if c in ("label", "name", "row_id", "section", "period", "tier")]
    wide = (struct_long
            .pivot_table(index=idx_cols, columns="field",
                         values="value", aggfunc="first")
            .reset_index())

    for col in ["max", "rate", "adj", "sell"]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce")

    # Unit fallback: tier-level -> section-level global column
    if "unit" in wide.columns:
        key = "label" if "label" in wide.columns else "name"
        for section, (_, unit_fb) in _SECTION_META.items():
            if unit_fb not in df_raw.columns:
                continue
            mask = wide["section"].eq(section)
            fb_map = dict(zip(df_raw[key], df_raw[unit_fb]))
            wide.loc[mask, "unit"] = wide.loc[mask, "unit"].fillna(
                wide.loc[mask, key].map(fb_map))

    sort_keys = [c for c in ["label", "name", "row_id", "section", "period", "tier"]
                 if c in wide.columns]
    wide = wide.sort_values(sort_keys).reset_index(drop=True)

    desired = [c for c in ["label", "name", "row_id", "section", "period", "tier",
                           "max", "rate", "adj", "sell", "unit"]
               if c in wide.columns]
    rest = [c for c in wide.columns if c not in desired]
    return wide[desired + rest]


# ---------------------------------------------------------------------------
# 2. Public: tidy long table
# ---------------------------------------------------------------------------

def tidy_urdb_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse all four rate-structure sections into a unified tidy long table.

    Returns
    -------
    DataFrame with columns:
        [label|name|row_id, section, period, tier, max, rate, adj, sell?, unit]
    """
    long = _melt_structure(df)
    if long.empty:
        return long
    long = long[long["section"].isin(_SECTION_META.keys())].copy()
    out = _pivot_fields(long, df)

    # Sell column only meaningful for energy
    for sec, (has_sell, _) in _SECTION_META.items():
        if not has_sell and "sell" in out.columns:
            out.loc[out["section"].eq(sec), "sell"] = np.nan

    # Drop rows where rate/max/adj are all NaN (unless unit is present)
    core_na = out[["rate", "max", "adj"]].isna().all(axis=1)
    only_unit = out["unit"].notna() & core_na
    out = out[~(core_na & ~only_unit)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 3. Flat demand month enablement
# ---------------------------------------------------------------------------

def extract_flat_demand_months(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract flatDemandMonth_* columns into a long table
    [label|name, month (1-12), enabled (bool)].
    """
    month_cols = [c for c in df.columns if c.startswith("flatDemandMonth_")]
    if not month_cols:
        return pd.DataFrame(columns=["key", "month", "enabled"])

    key = ("label" if "label" in df.columns
           else ("name" if "name" in df.columns else None))
    id_series = df[key] if key else pd.Series(df.index.astype(str), name="row_id")

    sub = (df[[key] + month_cols] if key
           else pd.concat([id_series, df[month_cols]], axis=1))
    long = sub.melt(
        id_vars=[key] if key else ["row_id"],
        var_name="month_col", value_name="enabled")
    long["month"] = (long["month_col"]
                     .str.split("_", n=1).str[1]
                     .str.lower().map(_MONTH_MAP).astype("Int64"))
    long = long.drop(columns=["month_col"]).sort_values("month").reset_index(drop=True)

    def _to_bool(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return bool(int(x))
        s = str(x).strip().lower()
        if s in {"1", "true", "t", "yes", "y"}:
            return True
        if s in {"0", "false", "f", "no", "n"}:
            return False
        return np.nan

    long["enabled"] = long["enabled"].map(_to_bool)
    return long


# ---------------------------------------------------------------------------
# 4. Tier bracket summary
# ---------------------------------------------------------------------------

def summarize_tier_brackets(tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Convert tidy tier table into human-readable interval brackets.

    For each [label, section, period], outputs:
        [start, end, rate, adj, sell, unit, note]
    Last tier with no max is treated as open-ended [prev_max, +inf).
    """
    key_cols = [c for c in ["label", "name", "row_id"] if c in tidy.columns]
    req = key_cols + ["section", "period", "tier", "max", "rate", "adj", "sell", "unit"]
    df = tidy[[c for c in req if c in tidy.columns]].copy()

    rows = []
    for group_keys, g in df.groupby(key_cols + ["section", "period"], dropna=False):
        g = g.sort_values("tier")
        prev = 0.0
        for _, row in g.iterrows():
            end = row["max"] if not pd.isna(row["max"]) else np.inf
            rows.append({
                **{k: v for k, v in zip(key_cols + ["section", "period"], group_keys)},
                "start": prev,
                "end": end,
                "rate": row["rate"],
                "adj": row.get("adj", np.nan),
                "sell": row.get("sell", np.nan),
                "unit": row.get("unit", np.nan),
                "tier": row["tier"],
                "note": "open-ended" if np.isinf(end) else None,
            })
            if not pd.isna(row["max"]):
                prev = float(row["max"])

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["label", "name", "row_id", "section", "period", "start", "tier"]
                 if c in out.columns]
    if len(out):
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 5. Period and tier counting
# ---------------------------------------------------------------------------

def count_periods_and_tiers(df: pd.DataFrame, section: str):
    """
    Count the number of active periods and tiers per utility for a given section.

    Returns
    -------
    periods_summary : DataFrame  [key, {section}_periods]
    tiers_detail    : DataFrame  [key, section, period, n_tiers]
    """
    key = _key_col(df)

    # Collect column maps
    per_period_cols = {}
    per_tier_cols = {}
    for col in df.columns:
        m = _COL_RE.match(col)
        if not m:
            continue
        gd = m.groupdict()
        if gd["section"] != section or gd["field"] not in PRESENCE_FIELDS:
            continue
        p, t = int(gd["period"]), int(gd["tier"])
        per_period_cols.setdefault(p, []).append(col)
        per_tier_cols.setdefault((p, t), []).append(col)

    key_series = df[key] if key in df.columns else pd.Series(df.index.astype(str), name=key)

    if not per_period_cols:
        return (
            pd.DataFrame({key: key_series, f"{section}_periods": 0}),
            pd.DataFrame(columns=[key, "section", "period", "n_tiers"]),
        )

    # Which periods are used per row
    period_used = pd.DataFrame({key: key_series})
    used_cols = []
    for p, cols in sorted(per_period_cols.items()):
        cname = f"__used_p{p}"
        period_used[cname] = df[cols].notna().any(axis=1)
        used_cols.append(cname)

    periods_summary = period_used[[key]].copy()
    periods_summary[f"{section}_periods"] = period_used[used_cols].sum(axis=1).astype(int)

    # Tier counts per period
    period_masks = {
        p: df[per_period_cols[p]].notna().any(axis=1).values
        for p in per_period_cols
    }
    key_vals = key_series.values
    rows = []
    for p in sorted(per_period_cols):
        mask_p = period_masks[p]
        if not mask_p.any():
            continue
        tiers_in_p = sorted({t for (pp, t) in per_tier_cols if pp == p})
        n_tiers = np.zeros(len(df), dtype=int)
        for t in tiers_in_p:
            exists = df[per_tier_cols[(p, t)]].notna().any(axis=1).values
            n_tiers += (exists & mask_p).astype(int)
        for i, (k, used) in enumerate(zip(key_vals, mask_p)):
            if used:
                rows.append({key: k, "section": section, "period": p,
                             "n_tiers": int(n_tiers[i])})

    tiers_detail = pd.DataFrame(rows, columns=[key, "section", "period", "n_tiers"])
    return periods_summary, tiers_detail


def run_counts(df: pd.DataFrame, sections=None):
    """
    Count periods and tiers for all sections.

    Returns
    -------
    periods_summary : DataFrame  [key, energy_periods, demand_periods, flat_periods]
    tiers_detail    : DataFrame  [key, section, period, n_tiers]
    """
    if sections is None:
        sections = SECTIONS
    key = _key_col(df)
    out_periods = None
    out_tiers = []

    for sec in sections:
        ps, td = count_periods_and_tiers(df, sec)
        out_tiers.append(td)
        if out_periods is None:
            out_periods = ps
        else:
            out_periods = out_periods.merge(ps, on=key, how="left")

    for c in out_periods.columns:
        if c != key:
            out_periods[c] = out_periods[c].fillna(0).astype(int)

    tiers_all = pd.concat(out_tiers, ignore_index=True) if out_tiers else pd.DataFrame()
    return out_periods, tiers_all


# ---------------------------------------------------------------------------
# 6. Detailed rate extraction with validation warnings
# ---------------------------------------------------------------------------

def _collect_field_maps(df: pd.DataFrame, section: str):
    """Build column-name lookup for (period, tier, field) tuples."""
    field_col = {}
    tiers_by_period = {}
    unit_cols = []
    for col in df.columns:
        m = _COL_RE.match(col)
        if not m:
            continue
        gd = m.groupdict()
        if gd["section"] != section:
            continue
        p, t, field = int(gd["period"]), int(gd["tier"]), gd["field"]
        field_col[(p, t, field)] = col
        tiers_by_period.setdefault(p, set()).add(t)
        if field == "unit":
            unit_cols.append(col)
    return field_col, tiers_by_period, unit_cols


def build_rate_detail(
    df: pd.DataFrame,
    tiers_per_period: pd.DataFrame,
    sections_cfg=None,
):
    """
    Build per-utility rate detail arrays with validation warnings.

    Parameters
    ----------
    df : raw USRDB DataFrame
    tiers_per_period : output of run_counts(...)[1]
    sections_cfg : dict mapping section name to
        {"unit_fallback_col": str, "default_unit": str}

    Returns
    -------
    DataFrame with columns:
        [key, {section}_detail, {section}_unit, {section}_warning]
    where {section}_detail is a nested list:
        [[tier0=[rate, max, adj], tier1=...], ...]  per period
    """
    if sections_cfg is None:
        sections_cfg = {
            "energyratestructure": {"unit_fallback_col": "energyrateunit",
                                   "default_unit": "kWh"},
            "demandratestructure": {"unit_fallback_col": "demandrateunit",
                                   "default_unit": "kW"},
            "flatdemandstructure": {"unit_fallback_col": "demandrateunit",
                                   "default_unit": "kW"},
        }

    key = _key_col(df)
    out = pd.DataFrame({key: df[key] if key in df.columns else df.index.astype(str)})

    # Pre-aggregate tier specs per utility
    per_util = {sec: {} for sec in sections_cfg}
    for sec in sections_cfg:
        sub = tiers_per_period[tiers_per_period["section"] == sec].sort_values("period")
        for k, g in sub.groupby(key, sort=False):
            per_util[sec][k] = list(zip(g["period"].tolist(), g["n_tiers"].tolist()))

    for section, cfg in sections_cfg.items():
        unit_fb = cfg["unit_fallback_col"]
        default_unit = cfg["default_unit"]
        field_col, tiers_by_period, unit_cols = _collect_field_maps(df, section)

        details, units, warns = [], [], []

        for _, row in df.iterrows():
            util_key = row[key] if key in row.index else str(row.name)
            spec = per_util.get(section, {}).get(util_key)
            warnings = []

            # Resolve unit
            unit_val = np.nan
            for c in unit_cols:
                if c in row.index and pd.notna(row[c]):
                    unit_val = row[c]
                    break
            if pd.isna(unit_val) and unit_fb in row.index and pd.notna(row[unit_fb]):
                unit_val = row[unit_fb]
            if pd.isna(unit_val):
                unit_val = default_unit
                warnings.append(f"{section}: unit missing, using default '{default_unit}'")

            if not spec:
                details.append([])
                units.append(unit_val)
                warns.append("; ".join(warnings))
                continue

            detail_periods = []
            for p, n_expected in spec:
                candidates = sorted(tiers_by_period.get(p, set()))
                triples = []
                read = 0
                for t in candidates:
                    present = any(
                        field_col.get((p, t, f)) and
                        pd.notna(row.get(field_col.get((p, t, f)), np.nan))
                        for f in PRESENCE_FIELDS
                    )
                    if not present:
                        continue
                    rate = row.get(field_col.get((p, t, "rate")), np.nan)
                    adj = row.get(field_col.get((p, t, "adj")), np.nan)
                    mx = row.get(field_col.get((p, t, "max")), np.nan)
                    adj = 0.0 if pd.isna(adj) else adj
                    triples.append([rate, mx, adj])
                    read += 1
                    if n_expected and read >= int(n_expected):
                        break

                if n_expected and read != int(n_expected):
                    warnings.append(
                        f"{section} period{p}: found {read} tiers, expected {n_expected}")

                # Fill missing max values
                for j in range(len(triples)):
                    if pd.isna(triples[j][1]):
                        triples[j][1] = 1e20
                        if j < len(triples) - 1:
                            warnings.append(
                                f"{section} period{p} tier{j}: missing max (non-last)")

                # Monotonicity check
                for j in range(1, len(triples)):
                    if triples[j - 1][1] >= triples[j][1]:
                        warnings.append(
                            f"{section} period{p}: non-increasing max at tier {j-1}->{j}")

                detail_periods.append(triples)

            details.append(detail_periods)
            units.append(unit_val)
            warns.append("; ".join(sorted(set(warnings))) if warnings else "")

        out[f"{section}_detail"] = details
        out[f"{section}_unit"] = units
        out[f"{section}_warning"] = warns

    return out


# ---------------------------------------------------------------------------
# 7. Convenience: parse everything at once
# ---------------------------------------------------------------------------

def parse_usrdb_all(df: pd.DataFrame):
    """
    One-call entry point.

    Returns
    -------
    tiers_long  : tidy long table of all tiers
    flat_months : flat demand month enablement
    summary     : tier bracket summary
    """
    tiers_long = tidy_urdb_tiers(df)
    flat_months = extract_flat_demand_months(df)
    summary = summarize_tier_brackets(tiers_long)
    return tiers_long, flat_months, summary
