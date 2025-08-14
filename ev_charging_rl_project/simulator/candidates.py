"""
Charger candidate generator (fast path with prefilter)

- Accepts already-merged stations (connector_type/rated_power_kw/station_connectors) OR raw+connectors.
- Applies a cheap Haversine corridor prefilter to avoid calling SUMO on thousands of stations.
- Calls SUMO detour routing only for a capped subset (max_evals).
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Iterable
import math
import pandas as pd
import numpy as np

from simulator.sumo_adapter import SumoSim

# ---- column mapping for internal use ----
COLS = {
    "station_id": "station_id",
    "lat": "latitude",
    "lon": "longitude",
    "company_id": "company_id",
    "charger_type": "charger_type",       # "Rapid" | "Fast" | "Slow"
    "rated_power_kw": "rated_power_kw",
    "connector_station_id": "station_id",
    "connector_type": "connector_type",   # e.g., "CCS2","CHAdeMO","Type2"
}

DC_TYPES = {"Rapid"}  # adjust if your metadata uses different naming

@dataclass
class EVSpecLite:
    battery_kwh: float
    eff_kwh_per_km: float               # energy consumption on road
    allowed_connectors: Tuple[str, ...] # e.g. ("CCS2","Type2")
    max_dc_kw: float = 150.0
    max_ac_kw: float = 11.0

@dataclass
class Candidate:
    station_id: str
    company_id: int
    charger_type: str
    rated_power_kw: float
    lat: float
    lon: float
    detour_km: float
    detour_time_s: float
    reachable_with_current_soc: bool
    connector_ok: bool
    is_dc: bool

    def to_dict(self):
        return asdict(self)

def _compatible_connector(allowed: Iterable[str], station_conns: Iterable[str]) -> bool:
    allowed_set = {str(c).strip().upper() for c in allowed}
    station_set = {str(c).strip().upper() for c in station_conns}
    return len(allowed_set & station_set) > 0

def _merge_connectors(stations: pd.DataFrame, connectors: pd.DataFrame) -> pd.DataFrame:
    """Merge raw stations + connectors into a table with station_connectors tuple."""
    sid = COLS["station_id"]
    csid = COLS["connector_station_id"]
    ctype = COLS["connector_type"]

    conns = (
        connectors[[csid, ctype]]
        .groupby(csid)[ctype]
        .apply(lambda s: tuple(sorted(set(x for x in s if pd.notna(x) and str(x).strip()))))
        .rename("station_connectors")
        .reset_index()
    )
    out = stations.merge(conns, left_on=sid, right_on=csid, how="left")
    out["station_connectors"] = out["station_connectors"].apply(
        lambda x: x if isinstance(x, tuple) else tuple()
    )
    return out.drop(columns=[csid])

# ---------- fast Haversine helpers ----------
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _dist_to_segment_km(lat, lon, A, B) -> float:
    """Approx distance from (lat,lon) to segment A->B using local equirectangular projection."""
    (lat1, lon1), (lat2, lon2) = A, B
    # project
    lat0 = (lat1 + lat2) * 0.5
    x1 = (lon1 - lon) * math.cos(math.radians(lat0)) * 111.32
    y1 = (lat1 - lat) * 110.57
    x2 = (lon2 - lon) * math.cos(math.radians(lat0)) * 111.32
    y2 = (lat2 - lat) * 110.57
    # distance from point to segment via projection
    vx, vy = (x2 - x1), (y2 - y1)
    wx, wy = (-x1), (-y1)
    c1 = vx*wx + vy*wy
    c2 = vx*vx + vy*vy
    t = 0.0 if c2 == 0 else max(0.0, min(1.0, c1/c2))
    px, py = (x1 + t*vx), (y1 + t*vy)
    # km
    return math.hypot(px, py)

def _prefilter_corridor(st: pd.DataFrame, A: Tuple[float,float], B: Tuple[float,float],
                        corridor_km: float, max_take: int) -> pd.DataFrame:
    """Cheaply keep stations near the straight line A->B, then take top-N nearest."""
    lat = st[COLS["lat"]].to_numpy(dtype=float)
    lon = st[COLS["lon"]].to_numpy(dtype=float)
    # vectorized approx distance to line
    lat0 = (A[0] + B[0]) * 0.5
    # project to km grid
    xA = (A[1] - lon) * math.cos(math.radians(lat0)) * 111.32
    yA = (A[0] - lat) * 110.57
    xB = (B[1] - lon) * math.cos(math.radians(lat0)) * 111.32
    yB = (B[0] - lat) * 110.57
    vx = xB - xA; vy = yB - yA
    wx = -xA;    wy = -yA
    c1 = vx*wx + vy*wy
    c2 = vx*vx + vy*vy
    t = np.where(c2 == 0, 0.0, np.clip(c1/c2, 0.0, 1.0))
    px = xA + t*vx; py = yA + t*vy
    dist_km = np.hypot(px, py)
    keep = dist_km <= corridor_km
    cand = st.loc[keep].copy()
    if len(cand) > max_take:
        cand["__d"] = dist_km[keep]
        cand = cand.nsmallest(max_take, "__d").drop(columns="__d")
    return cand

# ---------- main ----------
def find_charger_candidates(
    sim: SumoSim,
    current_latlon: Tuple[float, float],
    dest_latlon: Tuple[float, float],
    stations_df: pd.DataFrame,
    connectors_df: pd.DataFrame | None,
    ev: EVSpecLite,
    current_soc: float,
    top_k: int = 5,
    max_detour_km: float = 5.0,
    require_connector_ok: bool = True,
    min_power_kw: float | None = None,
    prefilter_km: float = 3.0,
    max_evals: int = 300,
) -> List[Candidate]:
    """
    Returns up to top_k candidates sorted by detour_time_s.
    Fast path: Haversine corridor prefilter (prefilter_km) and evaluation cap (max_evals).
    """
    sid = COLS["station_id"]; latc = COLS["lat"]; lonc = COLS["lon"]
    comp = COLS["company_id"]; ctyp = COLS["charger_type"]; pkw = COLS["rated_power_kw"]

    # Use merged table if already merged; otherwise merge with connectors_df
    already_merged = (
        {"company_id", "charger_type", "rated_power_kw"}.issubset(stations_df.columns)
        and ("connector_type" in stations_df.columns or "station_connectors" in stations_df.columns)
    )
    if already_merged:
        stations = stations_df.copy()
        if "station_connectors" not in stations.columns:
            if "connector_type" in stations.columns:
                stations["station_connectors"] = stations["connector_type"].apply(
                    lambda x: (str(x),) if pd.notna(x) and str(x).strip() else tuple()
                )
            else:
                stations["station_connectors"] = tuple()
    else:
        if connectors_df is None or connectors_df.empty:
            raise ValueError(
                "connectors_df is required when stations_df is not already merged "
                "(expected connector_type/station_connectors and rated_power_kw in stations_df)."
            )
        stations = _merge_connectors(stations_df, connectors_df).copy()

    # Optional power filter
    if min_power_kw is not None:
        stations = stations[stations[pkw].fillna(0) >= float(min_power_kw)]

    # ---- cheap spatial prefilter: corridor around straight line current->dest
    if len(stations) > max_evals:
        stations = _prefilter_corridor(stations, current_latlon, dest_latlon,
                                       corridor_km=max(prefilter_km, max_detour_km*0.8),
                                       max_take=max_evals)

    cand_list: List[Candidate] = []

    for _, row in stations.iterrows():
        s_lat, s_lon = float(row[latc]), float(row[lonc])

        det = sim.detour_metrics(current_latlon, dest_latlon, (s_lat, s_lon))
        if det["detour_dist_km"] > max_detour_km:
            continue  # too far off-route

        # Distance from current to station (first leg)
        leg_to_station = sim.route_between(current_latlon, (s_lat, s_lon))

        energy_needed_kwh = ev.eff_kwh_per_km * leg_to_station.dist_km
        energy_available_kwh = max(0.0, min(1.0, current_soc)) * ev.battery_kwh
        reachable = energy_needed_kwh <= energy_available_kwh

        # Connector compatibility
        station_conns = row.get("station_connectors", tuple())
        conn_ok = _compatible_connector(ev.allowed_connectors, station_conns) if station_conns else False
        if require_connector_ok and not conn_ok:
            continue

        charger_type = str(row[ctyp]).strip().title()
        power_kw = float(row[pkw]) if pd.notna(row[pkw]) else 0.0
        is_dc = charger_type in DC_TYPES

        cand_list.append(Candidate(
            station_id=str(row[sid]),
            company_id=int(row[comp]),
            charger_type=charger_type,
            rated_power_kw=power_kw,
            lat=s_lat, lon=s_lon,
            detour_km=float(det["detour_dist_km"]),
            detour_time_s=float(det["detour_time_s"]),
            reachable_with_current_soc=bool(reachable),
            connector_ok=bool(conn_ok),
            is_dc=is_dc
        ))

    cand_list.sort(key=lambda c: (c.detour_time_s, c.detour_km))
    return cand_list[:top_k]
