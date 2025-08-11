# utils/charging_curves.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


# -----------------------------
# EV spec container (strict schema)
# -----------------------------
@dataclass
class EVSpec:
    model: str
    usable_battery_kwh: float          # from EV_Metadata.battery_kWh
    max_dc_kw: float                   # from EV_Metadata.dc_max_power_kW
    max_ac_kw: float                   # from EV_Metadata.ac_max_power_kW
    efficiency_wh_per_km: float        # from EV_Metadata.avg_consumption_Wh_per_km

    @property
    def capacity(self) -> float:
        return float(self.usable_battery_kwh or 0.0)


# -----------------------------
# Public API (strict to your headers)
# -----------------------------
def build_ev_specs(ev_metadata: pd.DataFrame) -> Dict[str, EVSpec]:
    """
    Build EV specifications strictly from EV_Metadata.csv.

    REQUIRED columns (exact):
      - model
      - battery_kWh
      - dc_max_power_kW
      - ac_max_power_kW
      - avg_consumption_Wh_per_km
    """
    required = [
        "model",
        "battery_kWh",
        "dc_max_power_kW",
        "ac_max_power_kW",
        "avg_consumption_Wh_per_km",
    ]
    missing = [c for c in required if c not in ev_metadata.columns]
    if missing:
        raise KeyError(f"EV_Metadata missing required columns: {missing}")

    df = ev_metadata.copy()

    # enforce types
    df["battery_kWh"] = pd.to_numeric(df["battery_kWh"], errors="raise")
    df["dc_max_power_kW"] = pd.to_numeric(df["dc_max_power_kW"], errors="raise")
    df["ac_max_power_kW"] = pd.to_numeric(df["ac_max_power_kW"], errors="raise")
    df["avg_consumption_Wh_per_km"] = pd.to_numeric(df["avg_consumption_Wh_per_km"], errors="raise")

    specs: Dict[str, EVSpec] = {}
    for _, r in df.iterrows():
        m = str(r["model"])
        specs[m] = EVSpec(
            model=m,
            usable_battery_kwh=float(r["battery_kWh"]),
            max_dc_kw=float(r["dc_max_power_kW"]),
            max_ac_kw=float(r["ac_max_power_kW"]),
            efficiency_wh_per_km=float(r["avg_consumption_Wh_per_km"]),
        )
    return specs


def build_power_curve(curves_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Build a per‑SOC power curve for `model` from EV_Charging_Curve_Data.csv.

    REQUIRED columns (exact):
      - model
      - soc_percent
      - charging_power_kW

    Returns
    -------
    DataFrame with columns: ["soc", "power_kw"] at integer SOC 0..100.
    """
    required = ["model", "soc_percent", "charging_power_kW"]
    missing = [c for c in required if c not in curves_df.columns]
    if missing:
        raise KeyError(f"EV_Charging_Curve_Data missing required columns: {missing}")

    df = curves_df.loc[curves_df["model"].astype(str) == str(model)].copy()
    if df.empty:
        raise ValueError(f"No charging-curve rows for model='{model}'")

    df = df[["soc_percent", "charging_power_kW"]].dropna()
    df.rename(columns={"soc_percent": "soc", "charging_power_kW": "power_kw"}, inplace=True)

    df["soc"] = pd.to_numeric(df["soc"], errors="raise").clip(0, 100)
    df["power_kw"] = pd.to_numeric(df["power_kw"], errors="raise").clip(lower=0)
    df = df.sort_values("soc").groupby("soc", as_index=False)["power_kw"].mean()

    grid = np.arange(0, 101, 1, dtype=float)
    interp = np.interp(grid, df["soc"].values, df["power_kw"].values)
    out = pd.DataFrame({"soc": grid, "power_kw": np.maximum(interp, 0)})
    return out


def integrate_charge_session(
    ev_spec: EVSpec,
    power_curve: pd.DataFrame,
    start_soc: float,
    target_soc: float,
    station_power_kw: float,
    is_dc: bool = True,
    soc_step: float = 1.0,
) -> Dict:
    """
    Numerically integrate charge from start_soc -> target_soc using the power curve,
    capped by station power and EV max power (DC/AC).

    Parameters
    ----------
    ev_spec : EVSpec
    power_curve : DataFrame with columns ["soc", "power_kw"] for 0..100
    start_soc : float (0..100)
    target_soc : float (0..100)
    station_power_kw : float
    is_dc : bool
    soc_step : float  # SOC step in percentage points (1.0 is fine)

    Returns
    -------
    dict: {
      delivered_kwh, minutes, avg_power_kw, start_soc, target_soc,
      steps, caps
    }
    """
    start_soc = float(np.clip(start_soc, 0, 100))
    target_soc = float(np.clip(target_soc, 0, 100))
    if target_soc <= start_soc:
        return {
            "delivered_kwh": 0.0,
            "minutes": 0.0,
            "avg_power_kw": 0.0,
            "start_soc": start_soc,
            "target_soc": target_soc,
            "steps": [],
            "caps": {"station_cap_hits": 0, "ev_cap_hits": 0, "curve_zero_hits": 0},
        }

    # EV power cap (strictly from EV metadata)
    ev_cap_kw = float(ev_spec.max_dc_kw if is_dc else ev_spec.max_ac_kw)
    if not np.isfinite(ev_cap_kw) or ev_cap_kw <= 0:
        ev_cap_kw = 1e9  # if missing, let curve/station cap

    # prepare lookup arrays
    pc = power_curve[["soc", "power_kw"]].copy()
    # auto‑regrid if needed (saves headaches)
    pc = pc.set_index("soc")["power_kw"].reindex(range(0, 101)).interpolate().fillna(method="bfill").fillna(method="ffill").reset_index()

    power_grid = pc.set_index("soc")["power_kw"].reindex(range(0, 101)).to_numpy()

    def curve_power_at(s: float) -> float:
        idx = int(np.clip(round(s), 0, 100))
        return float(power_grid[idx])

    delivered_kwh = 0.0
    minutes = 0.0
    station_hits = 0
    ev_hits = 0
    curve_zeros = 0
    trace: List[Dict] = []

    soc = start_soc
    station_kw = float(max(0.0, station_power_kw))

    while soc < target_soc:
        nxt = min(soc + soc_step, target_soc)
        d_soc = max(1e-9, nxt - soc)

        slice_kwh = ev_spec.capacity * (d_soc / 100.0)
        p_curve = max(0.0, curve_power_at(soc))
        if p_curve <= 1e-9:
            curve_zeros += 1

        p_eff = min(p_curve, station_kw, ev_cap_kw)
        p_eff = max(p_eff, 1e-6)

        hours = slice_kwh / p_eff
        mins = hours * 60.0

        delivered_kwh += slice_kwh
        minutes += mins

        if p_eff + 1e-9 < p_curve:
            if station_kw <= ev_cap_kw + 1e-9 and station_kw <= p_curve + 1e-9:
                station_hits += 1
            elif ev_cap_kw <= p_curve + 1e-9:
                ev_hits += 1

        if len(trace) == 0 or ((nxt - start_soc) % max(1.0, 5.0) < 1e-9) or nxt == target_soc:
            trace.append({
                "soc_from": round(soc, 3),
                "soc_to": round(nxt, 3),
                "slice_kwh": round(slice_kwh, 4),
                "p_curve_kw": round(p_curve, 3),
                "p_effective_kw": round(p_eff, 3),
                "minutes": round(mins, 3),
            })

        soc = nxt

    avg_power_kw = (delivered_kwh / (minutes / 60.0)) if minutes > 0 else 0.0
    return {
        "delivered_kwh": round(delivered_kwh, 4),
        "minutes": round(minutes, 2),
        "avg_power_kw": round(avg_power_kw, 3),
        "start_soc": start_soc,
        "target_soc": target_soc,
        "steps": trace,
        "caps": {
            "station_cap_hits": int(station_hits),
            "ev_cap_hits": int(ev_hits),
            "curve_zero_hits": int(curve_zeros),
        },
    }
