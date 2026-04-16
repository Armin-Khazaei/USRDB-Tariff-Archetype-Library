"""
load_and_merge.py
=================
Load the raw USRDB CSV and merge with EIA utility metadata
(county, state, utility name) to produce an analysis-ready DataFrame.

Usage
-----
    from pipeline.load_and_merge import load_usrdb, merge_eia_utilities

    df = load_usrdb("data/usurdb.csv")
    df_merged = merge_eia_utilities(df, "data/eia_utilities.csv")
"""

import numpy as np
import pandas as pd


def load_usrdb(path: str) -> pd.DataFrame:
    """Load raw USRDB CSV and apply minimal cleaning."""
    df = pd.read_csv(path, low_memory=False)
    return df


def _normalize_text(s):
    """Strip whitespace; return NaN for missing values."""
    if pd.isna(s):
        return np.nan
    return str(s).strip()


def build_eia_utility_table(df_eia: pd.DataFrame,
                            group_key: str = "Utility Number") -> pd.DataFrame:
    """
    Aggregate EIA utility records into one row per utility,
    collecting states, counties, and a stable county-code mapping.

    Parameters
    ----------
    df_eia : DataFrame with columns including
        'Utility Number', 'Utility Name', 'Short Form',
        'Data Year', 'State', 'County'
    group_key : column to group on (default 'Utility Number')

    Returns
    -------
    DataFrame with one row per utility, columns:
        Utility Number, Utility Name, Short Form,
        Data_Years, States, States_List,
        Counties, Counties_List, County_Codes, N_Counties
    """
    df_t = df_eia.copy()

    # Normalize text
    df_t["County_clean"] = df_t["County"].map(_normalize_text).str.title()
    df_t["State_clean"] = df_t["State"].map(_normalize_text).str.upper()

    # Combined county-state key
    mask = df_t["County_clean"].notna() & df_t["State_clean"].notna()
    df_t["county_state"] = np.where(
        mask,
        df_t["County_clean"] + ", " + df_t["State_clean"],
        np.nan,
    )

    # Stable integer codes for all counties
    all_counties = sorted(df_t["county_state"].dropna().unique())
    county2code = {c: i + 1 for i, c in enumerate(all_counties)}

    # Aggregation helpers
    def _unique_sorted(s: pd.Series):
        return sorted(s.dropna().unique().tolist())

    utilities = (
        df_t.groupby(group_key, dropna=False)
        .agg(
            **{
                "Utility Name": ("Utility Name", "first"),
                "Short Form": ("Short Form", "first"),
                "Data_Years": ("Data Year", lambda s: sorted(set(s.dropna()))),
                "States_List": ("State_clean", _unique_sorted),
                "Counties_List": ("county_state", _unique_sorted),
            }
        )
        .reset_index()
    )

    # Derived columns
    utilities["States"] = utilities["States_List"].apply(
        lambda L: "; ".join(L) if isinstance(L, list) and L else ""
    )
    utilities["Counties"] = utilities["Counties_List"].apply(
        lambda L: "; ".join(L) if isinstance(L, list) and L else ""
    )
    utilities["County_Codes"] = utilities["Counties_List"].apply(
        lambda L: [county2code[c] for c in L] if isinstance(L, list) else []
    )
    utilities["N_Counties"] = utilities["Counties_List"].apply(
        lambda L: len(L) if isinstance(L, list) else 0
    )

    return utilities


def merge_eia_utilities(
    df_usrdb: pd.DataFrame,
    df_eia: pd.DataFrame,
    usrdb_key: str = "eiaid",
    eia_key: str = "Utility Number",
) -> pd.DataFrame:
    """
    Left-join USRDB rates with aggregated EIA utility metadata.

    Parameters
    ----------
    df_usrdb : USRDB rate table (from load_usrdb)
    df_eia   : raw EIA utility table
    usrdb_key : join key in df_usrdb
    eia_key   : join key in EIA table

    Returns
    -------
    Merged DataFrame with all USRDB columns plus utility metadata.
    """
    utilities = build_eia_utility_table(df_eia, group_key=eia_key)
    df_merged = df_usrdb.merge(
        utilities,
        how="left",
        left_on=usrdb_key,
        right_on=eia_key,
        validate="m:1",
    )
    return df_merged
