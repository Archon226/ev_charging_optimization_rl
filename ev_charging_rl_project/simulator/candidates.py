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

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def find_charger_candidates(
    sim,
    origin: tuple,
    dest: tuple,
    stations: pd.DataFrame,
    connectors: pd.DataFrame,
    ev: EVSpecLite,
    current_soc: float,
    top_k: int = 5,
    max_detour_km: float = 5.0
) -> List[Candidate]:
    """
    Vectorized candidate filtering + cached SUMO detour metrics.
    """
    o_lat, o_lon = origin
    d_lat, d_lon = dest

    # Merge station metadata + connectors
    merged = connectors.merge(stations, on="station_id", how="left")

    # Prefilter by rough detour estimate
    od = haversine_km(o_lat, o_lon, d_lat, d_lon)
    so = haversine_km(merged.lat, merged.lon, o_lat, o_lon)
    sd = haversine_km(merged.lat, merged.lon, d_lat, d_lon)
    est_detour = (so + sd) - od
    merged = merged.loc[(est_detour >= 0) & (est_detour <= max_detour_km)].copy()
    merged["est_detour"] = est_detour[(est_detour >= 0) & (est_detour <= max_detour_km)]

    if merged.empty:
        return []

    # Detailed SUMO detour metrics (cached inside sim)
    detour_data = []
    for row in merged.itertuples(index=False):
        dk, dt = sim.detour_via_cached(o_lat, o_lon, row.lat, row.lon, d_lat, d_lon)
        detour_data.append((dk, dt))
    merged["detour_km"], merged["detour_time_s"] = zip(*detour_data)

    # Connector compatibility
    merged["connector_ok"] = merged["connector_type"].isin(ev.allowed_connectors)

    # Range check
    max_range_km = ev.battery_kwh * current_soc / ev.eff_kwh_per_km
    merged["reachable_with_current_soc"] = (so <= max_range_km)

    # Select best top_k by detour time
    merged = merged.nsmallest(top_k, "detour_time_s")

    return [
        Candidate(
            station_id=r.station_id,
            charger_type=r.charger_type,
            rated_power_kw=r.rated_power_kw,
            detour_km=r.detour_km,
            detour_time_s=r.detour_time_s,
            reachable_with_current_soc=r.reachable_with_current_soc,
            connector_ok=r.connector_ok
        )
        for r in merged.itertuples(index=False)
    ]
