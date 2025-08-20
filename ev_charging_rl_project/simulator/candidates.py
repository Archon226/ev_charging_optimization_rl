import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass

@dataclass
class EVSpecLite:
    battery_kwh: float
    eff_kwh_per_km: float
    allowed_connectors: tuple
    max_dc_kw: float
    max_ac_kw: float

@dataclass
class Candidate:
    station_id: str
    charger_type: str
    rated_power_kw: float
    detour_km: float
    detour_time_s: float
    reachable_with_current_soc: bool
    connector_ok: bool
    lat: float
    lon: float
    company_id: int | None = None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def _finite_or(x, fallback=0.0):
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.fillna(fallback)

def find_charger_candidates(
    sim,
    origin: tuple,
    dest: tuple,
    stations: pd.DataFrame,
    connectors: pd.DataFrame,
    ev: EVSpecLite,
    current_soc: float,
    top_k: int = 5,
    max_detour_km: float = 5.0  # kept for API compatibility; no longer used as a hard cutoff
) -> List[Candidate]:
    """
    Build charger candidates 'on the route' by extra-distance heuristics:
      1) Merge metadata.
      2) Rank by UB ≈ hav(O,C)+hav(C,D)-hav(O,D) (smaller ~ closer to OD path).
      3) Keep best K (no hard cap), then compute real SUMO detours.
    """
    o_lat, o_lon = origin
    d_lat, d_lon = dest

    merged = connectors.merge(stations, on="station_id", how="inner")

    if "rated_power_kw" not in merged.columns and "rated_power_kW" in merged.columns:
        merged["rated_power_kw"] = merged["rated_power_kW"]

    need_cols = ["company_id", "lat", "lon", "connector_type", "charger_type", "rated_power_kw"]
    for c in need_cols:
        if c not in merged.columns:
            merged[c] = np.nan

    # Keep only rows we can price and that have coords & power
    merged = merged[merged["company_id"].notna()].copy()
    merged = merged[merged["rated_power_kw"].notna()].copy()
    for c in ["rated_power_kw", "lat", "lon"]:
        merged[c] = _finite_or(merged[c], 0.0)

    if merged.empty:
        # Fallback: nearest station to origin (guarantee ≥1)
        merged = connectors.merge(stations, on="station_id", how="inner").copy()
        merged["so"] = haversine_km(merged["lat"], merged["lon"], o_lat, o_lon)
        merged = merged.nsmallest(1, "so")
        merged["detour_km"] = 0.0
        merged["detour_time_s"] = 0.0
        merged["connector_ok"] = True
        merged["reachable_with_current_soc"] = True

    else:
        # ---- “On-route” prefilter via detour upper bound (no hard <= 5 km gate) ----
        od = haversine_km(o_lat, o_lon, d_lat, d_lon)
        so = haversine_km(merged["lat"], merged["lon"], o_lat, o_lon)
        sd = haversine_km(merged["lat"], merged["lon"], d_lat, d_lon)
        est_detour = _finite_or((so + sd) - od, np.inf)

        merged = merged.loc[est_detour >= 0].copy()
        merged["est_detour"] = est_detour.loc[merged.index]

        # Take the best UB_TOP_K
        UB_TOP_K = 60
        merged = merged.nsmallest(UB_TOP_K, "est_detour")
        if merged.empty:
            merged = connectors.merge(stations, on="station_id", how="inner").copy()
            merged["so"] = haversine_km(merged["lat"], merged["lon"], o_lat, o_lon)
            merged = merged.nsmallest(1, "so")

        # ---- Compute true detours with SUMO ----
        detour_km_list, detour_s_list = [], []
        total = len(merged)
        for i, row in enumerate(merged.itertuples(index=False), 1):
            if i % 10 == 0:
                print(f"[candidate routing] {i}/{total}", flush=True)
            dk, dt = sim.detour_via_cached(o_lat, o_lon, float(row.lat), float(row.lon), d_lat, d_lon)
            if not np.isfinite(dk): dk = 0.0
            if not np.isfinite(dt): dt = 0.0
            detour_km_list.append(float(dk))
            detour_s_list.append(float(dt))
        merged["detour_km"] = detour_km_list
        merged["detour_time_s"] = detour_s_list

        # ---- Compatibility and reachability ----
        merged["connector_ok"] = merged["connector_type"].isin(ev.allowed_connectors)
        so_direct = haversine_km(merged["lat"], merged["lon"], o_lat, o_lon)
        max_range_km = float(ev.battery_kwh) * float(max(0.0, min(1.0, current_soc))) / float(ev.eff_kwh_per_km)
        merged["reachable_with_current_soc"] = (so_direct <= max_range_km)

        # Final selection: top_k by detour time
        merged = merged.nsmallest(int(max(1, top_k)), "detour_time_s")
        if merged.empty:
            # last resort guarantee
            merged = connectors.merge(stations, on="station_id", how="inner").copy()
            merged["so"] = haversine_km(merged["lat"], merged["lon"], o_lat, o_lon)
            merged = merged.nsmallest(1, "so")
            merged["detour_km"] = 0.0
            merged["detour_time_s"] = 0.0
            merged["connector_ok"] = True
            merged["reachable_with_current_soc"] = True

    out: List[Candidate] = []
    for r in merged.itertuples(index=False):
        cid = None
        if getattr(r, "company_id", None) is not None and str(r.company_id).lower() not in ("nan", "none"):
            try:
                cid = int(r.company_id)
            except Exception:
                cid = None

        out.append(
            Candidate(
                station_id=str(r.station_id),
                charger_type=str(getattr(r, "charger_type", "AC")),
                rated_power_kw=float(r.rated_power_kw) if np.isfinite(r.rated_power_kw) else 0.0,
                detour_km=float(r.detour_km),
                detour_time_s=float(r.detour_time_s),
                reachable_with_current_soc=bool(getattr(r, "reachable_with_current_soc", True)),
                connector_ok=bool(getattr(r, "connector_ok", True)),
                lat=float(r.lat),
                lon=float(r.lon),
                company_id=cid
            )
        )

    return out
