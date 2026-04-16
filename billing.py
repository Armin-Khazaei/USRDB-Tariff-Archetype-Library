"""
billing.py
==========
Reference billing engine for computing commercial electricity bills
from archetype JSON files and hourly/sub-hourly load profiles.

Usage
-----
    from billing_engine.billing import compute_annual_bill

    bill = compute_annual_bill(
        load_profile,          # numpy array, kW at each timestep
        energy_archetype,      # dict (from JSON)
        demand_archetype,      # dict (from JSON), optional
        flat_demand_pattern,   # dict (from JSON), optional
        timestep_minutes=60,
    )
"""

import json
import numpy as np
from pathlib import Path


def load_archetype(path: str) -> dict:
    """Load an archetype JSON file."""
    with open(path) as f:
        return json.load(f)


def _get_period_index(period_map, month, hour):
    """
    Look up the period index for a given month (0-11) and hour (0-23)
    from a 12x24 period map.
    """
    if period_map is None:
        return 0
    return int(period_map[month][hour])


def compute_energy_cost(
    load_kw: np.ndarray,
    energy_archetype: dict,
    timestep_minutes: int = 60,
    is_weekday: np.ndarray = None,
) -> dict:
    """
    Compute annual energy charges from load profile and energy archetype.

    Parameters
    ----------
    load_kw : array of shape (N_timesteps,), power in kW
    energy_archetype : archetype dict with 'period_map' and 'periods'
    timestep_minutes : resolution of load_kw (default 60 = hourly)
    is_weekday : boolean array, True for weekday timesteps

    Returns
    -------
    dict with 'total_energy_cost', 'monthly_breakdown', 'kwh_by_period'
    """
    n = len(load_kw)
    hours_per_step = timestep_minutes / 60.0
    kwh_per_step = load_kw * hours_per_step

    # Build time index
    steps_per_hour = 60 // timestep_minutes
    steps_per_day = 24 * steps_per_hour
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    expected_steps = sum(d * steps_per_day for d in days_per_month)

    if n != expected_steps:
        raise ValueError(
            f"Expected {expected_steps} timesteps for 8760h at {timestep_minutes}-min "
            f"resolution, got {n}")

    # Get period map
    weekday_map = energy_archetype.get("period_map", {}).get("weekday")
    weekend_map = energy_archetype.get("period_map", {}).get("weekend")
    periods = energy_archetype["periods"]

    total_cost = 0.0
    monthly_cost = [0.0] * 12
    kwh_by_period = {}

    step = 0
    for month_idx, days in enumerate(days_per_month):
        for day in range(days):
            for hour in range(24):
                for sub in range(steps_per_hour):
                    # Determine period
                    use_weekend = (is_weekday is not None and
                                   not is_weekday[step] and
                                   weekend_map is not None)
                    pmap = weekend_map if use_weekend else weekday_map
                    period_idx = _get_period_index(pmap, month_idx, hour)

                    # Look up rate
                    period = periods[period_idx]
                    kwh = kwh_per_step[step]
                    rate = _tiered_rate(kwh, period["tiers"])

                    cost = kwh * rate
                    total_cost += cost
                    monthly_cost[month_idx] += cost
                    kwh_by_period[period_idx] = kwh_by_period.get(period_idx, 0) + kwh

                    step += 1

    return {
        "total_energy_cost": total_cost,
        "monthly_energy_cost": monthly_cost,
        "kwh_by_period": kwh_by_period,
    }


def compute_demand_cost(
    load_kw: np.ndarray,
    demand_archetype: dict,
    timestep_minutes: int = 60,
    is_weekday: np.ndarray = None,
) -> dict:
    """
    Compute annual demand charges from load profile and demand archetype.

    Demand is the maximum kW within each period for each billing month.

    Parameters
    ----------
    load_kw : array of shape (N_timesteps,), power in kW
    demand_archetype : archetype dict with 'period_map' and 'periods'
    timestep_minutes : resolution
    is_weekday : boolean array

    Returns
    -------
    dict with 'total_demand_cost', 'monthly_breakdown',
              'peak_kw_by_month_period'
    """
    steps_per_hour = 60 // timestep_minutes
    steps_per_day = 24 * steps_per_hour
    days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    weekday_map = demand_archetype.get("period_map", {}).get("weekday")
    weekend_map = demand_archetype.get("period_map", {}).get("weekend")
    periods = demand_archetype["periods"]
    n_periods = len(periods)

    # Track monthly peak per period
    # monthly_peak[month][period] = max kW
    monthly_peak = [[0.0] * n_periods for _ in range(12)]

    step = 0
    for month_idx, days in enumerate(days_per_month):
        for day in range(days):
            for hour in range(24):
                for sub in range(steps_per_hour):
                    use_weekend = (is_weekday is not None and
                                   not is_weekday[step] and
                                   weekend_map is not None)
                    pmap = weekend_map if use_weekend else weekday_map
                    period_idx = _get_period_index(pmap, month_idx, hour)

                    kw = load_kw[step]
                    if kw > monthly_peak[month_idx][period_idx]:
                        monthly_peak[month_idx][period_idx] = kw

                    step += 1

    # Compute cost
    total_cost = 0.0
    monthly_cost = [0.0] * 12
    for month_idx in range(12):
        for period_idx in range(n_periods):
            peak = monthly_peak[month_idx][period_idx]
            rate = _tiered_rate(peak, periods[period_idx]["tiers"])
            cost = peak * rate
            total_cost += cost
            monthly_cost[month_idx] += cost

    return {
        "total_demand_cost": total_cost,
        "monthly_demand_cost": monthly_cost,
        "peak_kw_by_month_period": monthly_peak,
    }


def _tiered_rate(quantity: float, tiers: list) -> float:
    """
    Look up the marginal rate for a given quantity under a tiered structure.

    For simplicity, returns the rate of the tier whose max bracket
    the quantity falls into (block rate, not marginal accumulation).
    """
    for tier in tiers:
        mx = tier.get("max")
        if mx is None or quantity <= mx:
            rate = tier.get("rate", 0.0) or 0.0
            adj = tier.get("adj", 0.0) or 0.0
            return rate + adj
    # Beyond all tiers: use last tier rate
    last = tiers[-1]
    return (last.get("rate", 0.0) or 0.0) + (last.get("adj", 0.0) or 0.0)


def compute_annual_bill(
    load_kw: np.ndarray,
    energy_archetype: dict = None,
    demand_archetype: dict = None,
    timestep_minutes: int = 60,
    is_weekday: np.ndarray = None,
) -> dict:
    """
    Compute full annual bill (energy + demand).

    Parameters
    ----------
    load_kw : power profile in kW
    energy_archetype : energy rate archetype (dict from JSON)
    demand_archetype : demand rate archetype (dict from JSON)
    timestep_minutes : resolution (15 or 60 typical)
    is_weekday : boolean array marking weekday timesteps

    Returns
    -------
    dict with total_bill, energy_cost, demand_cost,
         demand_share, monthly breakdowns
    """
    result = {"total_bill": 0.0}

    if energy_archetype:
        e = compute_energy_cost(load_kw, energy_archetype,
                                timestep_minutes, is_weekday)
        result["energy_cost"] = e["total_energy_cost"]
        result["monthly_energy_cost"] = e["monthly_energy_cost"]
        result["total_bill"] += e["total_energy_cost"]
    else:
        result["energy_cost"] = 0.0

    if demand_archetype:
        d = compute_demand_cost(load_kw, demand_archetype,
                                timestep_minutes, is_weekday)
        result["demand_cost"] = d["total_demand_cost"]
        result["monthly_demand_cost"] = d["monthly_demand_cost"]
        result["peak_kw_by_month_period"] = d["peak_kw_by_month_period"]
        result["total_bill"] += d["total_demand_cost"]
    else:
        result["demand_cost"] = 0.0

    if result["total_bill"] > 0:
        result["demand_share"] = result["demand_cost"] / result["total_bill"]
    else:
        result["demand_share"] = 0.0

    return result
