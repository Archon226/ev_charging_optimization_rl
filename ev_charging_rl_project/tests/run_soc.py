# tests/run_soc.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import sys
import os

# repo path hack for tests
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_all_data
from utils.charging_curves import build_ev_specs, build_power_curve, integrate_charge_session, EVSpec
from env.charger import charger_from_row

DATA_DIR = Path("data")

AVG_SPEED_KPH = 28.0  # inner London fixed speed for now
DEFAULT_WH_PER_KM = 180.0
RESERVE_SOC = 10.0    # safety buffer if not provided by user row
MIN_TRIP_KM = 15.0  # keep near the top with your constants

def sample_far_destination(stations_merged: pd.DataFrame, s_lat: float, s_lon: float, min_km: float) -> tuple[float,float]:
    df = stations_merged.dropna(subset=["latitude","longitude"]).copy()
    if df.empty:
        raise RuntimeError("No stations with coordinates to sample a destination from.")

    # vectorized distance from (s_lat, s_lon)
    lats = pd.to_numeric(df["latitude"], errors="coerce").to_numpy()
    lons = pd.to_numeric(df["longitude"], errors="coerce").to_numpy()
    dists = haversine_km_vec(s_lat, s_lon, lats, lons)
    df = df.assign(dist_km=dists)

    # try to sample among those >= min_km
    cand = df[df["dist_km"] >= float(min_km)]
    if cand.empty:
        # fallback: farthest available
        d = df.sort_values("dist_km", ascending=False).iloc[0]
        return float(d["latitude"]), float(d["longitude"])

    d = cand.sample(1).iloc[0]
    return float(d["latitude"]), float(d["longitude"])


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def haversine_km_vec(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    R = 6371.0088
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    latsr = np.radians(lats.astype(float))
    lonsr = np.radians(lons.astype(float))
    dphi = latsr - lat1r
    dl = lonsr - lon1r
    a = np.sin(dphi/2)**2 + np.cos(lat1r) * np.cos(latsr) * np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def can_reach(initial_soc: float, usable_kwh: float, wh_per_km: float, distance_km: float, reserve_soc: float) -> bool:
    kwh_available = usable_kwh * max(0.0, (initial_soc - reserve_soc)) / 100.0
    kwh_needed = (distance_km * wh_per_km) / 1000.0
    return kwh_available >= kwh_needed


def soc_needed_for_km(usable_kwh: float, wh_per_km: float, km: float) -> float:
    kwh_needed = (km * wh_per_km) / 1000.0
    return 100.0 * (kwh_needed / max(usable_kwh, 1e-9))


def pick_nearest_priced_charger(stations_merged: pd.DataFrame, lat: float, lon: float) -> pd.Series | None:
    df = stations_merged.copy()
    df = df[df["company_id"].notna()].copy()  # must be priceable
    df = df.dropna(subset=["latitude", "longitude"])
    if df.empty:
        return None
    df["dist_km"] = haversine_km(lat, lon, df["latitude"], df["longitude"])
    # prefer nearer & higher power
    df = df.sort_values(["dist_km", "rated_power_kw"], ascending=[True, False])
    return df.iloc[0]


def _as_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _ev_fields(evm) -> tuple[float, float, float]:
    """Return (usable_kwh, max_dc_kw, max_ac_kw) from either an EVSpec or a dict-like spec."""
    if isinstance(evm, EVSpec):
        usable_kwh = _as_float(evm.usable_battery_kwh, 0.0)
        max_dc_kw = _as_float(getattr(evm, "max_dc_kw", None), 120.0)
        max_ac_kw = _as_float(getattr(evm, "max_ac_kw", None), 11.0)
        return usable_kwh, max_dc_kw, max_ac_kw
    # dict-like fallback
    usable_kwh = _as_float(evm.get("usable_battery_kwh", evm.get("battery_kWh", 0.0)), 0.0)
    max_dc_kw = _as_float(evm.get("max_dc_kw", evm.get("max_dc_power_kw", 120.0)), 120.0)
    max_ac_kw = _as_float(evm.get("max_ac_kw", 11.0), 11.0)
    return usable_kwh, max_dc_kw, max_ac_kw


def _ev_wh_per_km(evm) -> float:
    """
    Return Wh/km from either an EVSpec (with optional efficiency_wh_per_km)
    or dict-like spec (consumption_wh_per_km or efficiency_mi_per_kwh).
    """
    # EVSpec path
    if isinstance(evm, EVSpec):
        v = getattr(evm, "efficiency_wh_per_km", None)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            return _as_float(v, DEFAULT_WH_PER_KM)

    # dict-like path
    if isinstance(evm, dict):
        if "consumption_wh_per_km" in evm:
            return _as_float(evm["consumption_wh_per_km"], DEFAULT_WH_PER_KM)
        if "efficiency_mi_per_kwh" in evm:
            mi_per_kwh = _as_float(evm["efficiency_mi_per_kwh"], 3.0)
            km_per_kwh = mi_per_kwh * 1.60934
            if km_per_kwh > 0:
                return 1000.0 / km_per_kwh

    return DEFAULT_WH_PER_KM


def main(limit: int = 20):
    ds = load_all_data(DATA_DIR)

    # Users (generate first if missing)
    users_path = DATA_DIR / "simulated_users.csv"
    if not users_path.exists():
        from utils.generate_users import UserConfig, generate_users
        dfu = generate_users(ds.stations_merged, ds.ev_metadata, UserConfig(n_users=50))
        dfu.to_csv(users_path, index=False)
    users = pd.read_csv(users_path)

    # EV specs map (model -> EVSpec or dict depending on your build)
    ev_specs_map = build_ev_specs(ds.ev_metadata)

    # Cache curves per model
    curve_cache: dict[str, pd.DataFrame] = {}

    results = []
    users = users.head(limit).copy()

    for _, u in users.iterrows():
        agent_id = str(u["agent_id"])
        model = str(u["model"])
        user_type = str(u.get("user_type", "Payg"))
        objective = str(u.get("objective", "cost"))
        initial_soc = _as_float(u.get("initial_soc", 50.0), 50.0)
        target_soc = _as_float(u.get("target_soc", 80.0), 80.0)
        reserve_soc = _as_float(u.get("reserve_soc", RESERVE_SOC), RESERVE_SOC)
        depart_dt = datetime.fromisoformat(str(u["depart_dt"]))
        s_lat, s_lon = _as_float(u["start_lat"]), _as_float(u["start_lon"])
        d_lat, d_lon = _as_float(u["dest_lat"]), _as_float(u["dest_lon"])

        if model not in ev_specs_map:
            results.append({"agent_id": agent_id, "status": "no_ev_spec", "model": model})
            continue

        evm = ev_specs_map[model]
        usable_kwh, max_dc_kw, max_ac_kw = _ev_fields(evm)
        wh_per_km = _ev_wh_per_km(evm)

        # Build or reuse curve
        if model not in curve_cache:
            curve_cache[model] = build_power_curve(ds.charging_curves, model)
        power_curve = curve_cache[model]

        # Leg: start -> dest
        leg_km = haversine_km(s_lat, s_lon, d_lat, d_lon)
        if leg_km < MIN_TRIP_KM:
            d_lat, d_lon = sample_far_destination(ds.stations_merged, s_lat, s_lon, MIN_TRIP_KM)
        leg_minutes = (leg_km / max(AVG_SPEED_KPH, 1e-6)) * 60.0

        # If reachable with current SOC (keeping reserve), just drive
        if can_reach(initial_soc, usable_kwh, wh_per_km, leg_km, reserve_soc):
            final_soc = initial_soc - soc_needed_for_km(usable_kwh, wh_per_km, leg_km)
            results.append({
                "agent_id": agent_id, "status": "no_charge_needed",
                "model": model, "user_type": user_type, "objective": objective,
                "operator": "n/a",
                "company_id": None,
                "charger_type": "n/a",
                "distance_km": round(leg_km, 3),
                "drive_minutes": round(leg_minutes, 2),
                "initial_soc": round(initial_soc, 1),
                "final_soc": round(final_soc, 2),
                "kwh_delivered": 0.0,
                "charge_minutes": 0.0,
                "unit_price": 0.0,
                "price_source": "n/a",
                "energy_cost": 0.0,
                "fees_connection": 0.0,
                "fees_infra": 0.0,
                "fees_idle": 0.0,
                "fees_overstay": 0.0,
                "min_charge_topup": 0.0,
                "total_cost": 0.0,
            })
            continue

        # Need to charge: pick nearest priced charger from START
        cand = pick_nearest_priced_charger(ds.stations_merged, s_lat, s_lon)
        if cand is None:
            results.append({
                "agent_id": agent_id, "status": "no_priced_charger_near_start",
                "model": model, "user_type": user_type, "objective": objective,
                "operator": "n/a", "company_id": None, "charger_type": "n/a",
                "distance_km": round(leg_km, 3),
                "drive_minutes": round(leg_minutes, 2),
                "initial_soc": round(initial_soc, 1),
                "final_soc": round(initial_soc, 2),
                "kwh_delivered": 0.0,
                "charge_minutes": 0.0,
                "unit_price": 0.0,
                "price_source": "n/a",
                "energy_cost": 0.0,
                "fees_connection": 0.0,
                "fees_infra": 0.0,
                "fees_idle": 0.0,
                "fees_overstay": 0.0,
                "min_charge_topup": 0.0,
                "total_cost": float("nan"),
            })
            continue

        # SOC required to reach destination + reserve
        soc_for_leg = soc_needed_for_km(usable_kwh, wh_per_km, leg_km) + reserve_soc
        desired_soc = float(min(100.0, max(target_soc, soc_for_leg)))

        # Station power & DC/AC heuristic
        station_power_kw = _as_float(cand.get("rated_power_kw", 50.0), 50.0)
        ctype = str(cand.get("charger_type", "Rapid")).strip().title()
        is_dc = ctype in ("Rapid", "Ultra")

        # Integrate real charging session
        ev_spec_obj = EVSpec(
            model=model,
            usable_battery_kwh=usable_kwh,
            max_dc_kw=max_dc_kw,
            max_ac_kw=max_ac_kw,
        )
        integ = integrate_charge_session(
            ev_spec=ev_spec_obj,
            power_curve=power_curve,
            start_soc=initial_soc,
            target_soc=desired_soc,
            station_power_kw=station_power_kw,
            is_dc=is_dc,
        )

        # Price it
        charger = charger_from_row(cand, ds.pricing_catalog)
        breakdown = charger.estimate_session_cost(
            kwh=integ["delivered_kwh"],
            start_dt=depart_dt,
            user_type=user_type,
            session_minutes=integ["minutes"],
            include_subscription=False,
        )

        final_soc = desired_soc - soc_needed_for_km(usable_kwh, wh_per_km, leg_km)

        results.append({
            "agent_id": agent_id, "status": "charged_then_drive",
            "model": model, "user_type": user_type, "objective": objective,
            "operator": breakdown.get("operator", "n/a"),
            "company_id": breakdown.get("company_id"),
            "charger_type": breakdown.get("charger_type", "n/a"),
            "distance_km": round(leg_km, 3),
            "drive_minutes": round(leg_minutes, 2),
            "initial_soc": round(initial_soc, 1),
            "target_soc": round(desired_soc, 1),
            "final_soc": round(final_soc, 2),
            "kwh_delivered": float(integ["delivered_kwh"]),
            "charge_minutes": float(integ["minutes"]),
            "unit_price": float(breakdown.get("unit_price", 0.0)),
            "price_source": breakdown.get("unit_source", breakdown.get("price_source", "")),
            "energy_cost": float(breakdown.get("energy_cost", 0.0)),
            "fees_connection": float(breakdown.get("connection_fee", 0.0)),
            "fees_infra": float(breakdown.get("infrastructure_fee", 0.0)),
            "fees_idle": float(breakdown.get("idle_fee", breakdown.get("idle_cost", 0.0))),
            "fees_overstay": float(breakdown.get("overstay_fee", 0.0)),
            "min_charge_topup": float(breakdown.get("min_charge_topup", breakdown.get("minimum_applied", 0.0))),
            "total_cost": float(breakdown.get("total_cost", np.nan)),
        })

    out = pd.DataFrame(results)
    out_path = Path("tests") / "soc_based_pricing_results.csv"
    out.to_csv(out_path, index=False)
    ok = (out["status"].isin(["no_charge_needed", "charged_then_drive"])).sum()
    print(f"[done] users processed={len(users)}  successful={ok}  → {out_path}")
    if ok:
        cols = [
            "agent_id","status","model","operator","charger_type",
            "distance_km","kwh_delivered","charge_minutes","unit_price","total_cost","price_source"
        ]
        # ensure columns exist even if future edits change dict population
        for c in cols:
            if c not in out.columns:
                out[c] = np.nan
        print(out.loc[out["status"].isin(['no_charge_needed','charged_then_drive']), cols]
              .head(10).to_string(index=False))


if __name__ == "__main__":
    main(limit=20)
