# rl/candidate_providers.py
from __future__ import annotations
import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
from env.sumo_runner import SumoRunner, SumoConfig
# ---- Phase 4: masking helpers ----
from typing import Iterable

@dataclass
class Candidate:
    station_id: str
    rep_type: str                 # Fast/Rapid/Ultra
    detour_km: float
    approx_eta_min: float
    price_per_kwh: float

def _cooldown_active(now_min: Optional[float], last_charge_end_min: Optional[float], min_gap_min: float) -> bool:
    if now_min is None or last_charge_end_min is None:
        return False
    try:
        return float(now_min) - float(last_charge_end_min) < float(min_gap_min)
    except Exception:
        return False

def mask_candidates_phase4(
    candidates: List[Candidate],
    *,
    visited_station_ids: Optional[Iterable[str]] = None,
    now_min: Optional[float] = None,
    last_charge_end_min: Optional[float] = None,
    min_charge_gap_min: float = 0.0,
    emergency: bool = False,
    keep_nearest_n_on_emergency: int = 1,
) -> List[Candidate]:
    """
    Source-side masking for Phase 4.
    - Removes repeat stations and cooldown-violating stations.
    - If emergency=True, we still allow a small set of nearest candidates
      even if they would otherwise be masked (to avoid trapping the agent).
    """
    visited = set(str(s) for s in (visited_station_ids or []))
    cd_active = _cooldown_active(now_min, last_charge_end_min, min_charge_gap_min)

    if not visited and not cd_active:
        return candidates[:]  # no masking needed

    kept: List[Candidate] = []
    masked_emergency_pool: List[Candidate] = []

    for c in candidates:
        is_repeat = c.station_id in visited
        violates_cd = cd_active  # cooldown is global for all stations until gap passes

        if (is_repeat or violates_cd):
            # In emergency we keep a small nearest subset even if masked
            masked_emergency_pool.append(c)
            continue
        kept.append(c)

    if emergency and not kept:
        # allow a tiny window (nearest few by ETA) so we don't deadlock
        masked_emergency_pool.sort(key=lambda x: (x.approx_eta_min, x.price_per_kwh))
        kept = masked_emergency_pool[:max(1, int(keep_nearest_n_on_emergency))]

    # Final sort stays the same policy as before
    kept.sort(key=lambda x: (x.approx_eta_min, x.price_per_kwh))

    # ---- SAFETY GUARANTEE ----
    # Never block Drive (action 0)
    # If no stations survive masking, return an *empty station list*,
    # but the env must still keep Drive = legal.
    return kept  # env will always add Drive separately


class SumoRouteProvider:
    """
    Candidate provider using SUMO network travel times.
    Computes detour time for stopping at a station vs. direct trip.
    """

    def __init__(self, sumo_runner: SumoRunner, max_detour_min: float = 15.0):
        self.sumo = sumo_runner
        self.max_detour_min = float(max_detour_min)
        # Station → edge cache (filled lazily on demand)
        self._station_edge_map: dict[str, str] = {}
        # ---- lightweight caches to avoid per-call recompute ----
        self._coords_xy_cache = {}        # station_id -> (x, y)
        self._coords_latlon_cache = {}    # station_id -> (lat, lon)
        self._station_edge_map = {}       # station_id -> SUMO edge id
        self._coords_cached = False


    def generate(
        self,
        trip,
        station_index,
        connectors_df,
        price_lookup,
        top_k: int,
        *,
        # ---- Phase 4 (optional; if omitted, no masking is applied) ----
        visited_station_ids: Optional[Iterable[str]] = None,
        now_min: Optional[float] = None,
        last_charge_end_min: Optional[float] = None,
        min_charge_gap_min: float = 0.0,
        emergency: bool = False,
        keep_nearest_n_on_emergency: int = 1,
    ):
        """
        Corridor-only candidate search with Phase 4 masking:
          - Build O→D corridor once.
          - **Pre-mask** repeats & cooldown before routing (saves SUMO calls).
          - Route only the surviving set O→S and S→D (hard-capped), early-stopping at top_k*3.
          - **Post-mask** again as a safety net; return final top_k.
        """
        # -------- constants (tune lightly if needed) ----------
        MAX_STATIONS_TO_EVAL = max(24, top_k * 12)     # hard cap on SUMO routing
        SEARCH_RADIUS_M = 200                          # station edge snap radius
        MAX_DETOUR_MIN = self.max_detour_min           # time-based detour budget
        corridor_width_m = float(getattr(trip, "max_detour_km", 2.0)) * 1000.0

        # 1) snap origin/dest; bail early if outside the net
        try:
            o_lat, o_lon = trip.origin[0], trip.origin[1]
            d_lat, d_lon = trip.dest[0], trip.dest[1]
            origin_edge = self.sumo.snap_to_edge(o_lat, o_lon)
            dest_edge   = self.sumo.snap_to_edge(d_lat, d_lon)
        except Exception:
            return []

        # 2) baseline O→D and route geometry (polyline in XY meters)
        try:
            edges, base_len, base_time = self.sumo.find_route_edges(origin_edge, dest_edge)
        except Exception:
            return []
        base_min = base_time / 60.0

        # Build route polyline from edge shapes; SUMO net coords are meters
        net = self.sumo.net
        route_xy: list[tuple[float, float]] = []
        for eid in edges:
            try:
                e = net.getEdge(eid)
                shape = e.getShape()  # list[(x,y)]
                if shape:
                    if not route_xy:
                        route_xy.extend(shape)
                    else:
                        route_xy.extend(shape[1:])
            except Exception:
                continue
        if len(route_xy) < 2:
            return []

        # 3) collect station coords and convert to XY; filter by corridor distance
        coords_xy: dict[str, tuple[float, float]] = {}
        coords_latlon: dict[str, tuple[float, float]] = {}
        for row in connectors_df.itertuples(index=False):
            sid = getattr(row, "station_id", None)
            lat = getattr(row, "lat", None) if hasattr(row, "lat") else getattr(row, "latitude", None)
            lon = getattr(row, "lon", None) if hasattr(row, "lon") else getattr(row, "longitude", None)
            if sid and (lat is not None) and (lon is not None) and sid not in coords_xy:
                try:
                    x, y = self.sumo._to_xy(float(lat), float(lon))
                    coords_xy[str(sid)] = (float(x), float(y))
                    coords_latlon[str(sid)] = (float(lat), float(lon))
                except Exception:
                    continue

        # distance from point to polyline (in meters)
        def _dist_point_polyline_m(px: float, py: float, poly: list[tuple[float,float]]) -> float:
            import math
            best = float("inf")
            for i in range(len(poly) - 1):
                x1, y1 = poly[i]
                x2, y2 = poly[i+1]
                vx, vy = x2 - x1, y2 - y1
                wx, wy = px - x1, py - y1
                denom = vx*vx + vy*vy
                if denom <= 1e-12:
                    d = math.hypot(px - x1, py - y1)
                else:
                    t = max(0.0, min(1.0, (wx*vx + wy*vy) / denom))
                    qx, qy = x1 + t * vx, y1 + t * vy
                    d = math.hypot(px - qx, py - qy)
                if d < best:
                    best = d
            return best

        # 4) corridor filter → list of station_ids to consider
        corridor_sids: list[str] = []
        for sid, (sx, sy) in coords_xy.items():
            d = _dist_point_polyline_m(sx, sy, route_xy)
            if d <= corridor_width_m:
                corridor_sids.append(str(sid))

        # rank by proximity to corridor; cap set size for routing
        corridor_sids.sort(key=lambda s: _dist_point_polyline_m(coords_xy[s][0], coords_xy[s][1], route_xy))
        corridor_sids = corridor_sids[:MAX_STATIONS_TO_EVAL]

        # ---- Phase 4: PRE-MASK (cheap, before routing) ----
        visited = set(str(s) for s in (visited_station_ids or []))
        cd_active = _cooldown_active(now_min, last_charge_end_min, min_charge_gap_min)

        # If cooldown is active and not in emergency → return [] (Drive remains available in the env)
        if cd_active and not emergency:
            return []

        # If repeats exist and not in emergency → drop them before routing
        if visited and not emergency:
            corridor_sids = [sid for sid in corridor_sids if sid not in visited]
            if not corridor_sids:
                return []

        # 5) route O→S + S→D for the (pre-masked) corridor set; early-stop at ~top_k*3
        candidates: list[Candidate] = []
        for sid in corridor_sids:
            st = station_index.get(sid)
            if not st:
                continue
            rep = st["rep_type"]
            comp = st["company_id"]

            try:
                slat, slon = coords_latlon[sid]
                st_edge = self.sumo.snap_to_edge(slat, slon, search_radius=SEARCH_RADIUS_M)
                self._station_edge_map[str(sid)] = st_edge
                _, len1, t1 = self.sumo.find_route_edges(origin_edge, st_edge)
                _, len2, t2 = self.sumo.find_route_edges(st_edge, dest_edge)
            except Exception:
                continue

            detour_min = (t1 + t2 - base_time) / 60.0
            if detour_min < 0:
                detour_min = 0.0
            if detour_min <= MAX_DETOUR_MIN:
                price = price_lookup.get((str(comp), rep)) or {"Fast":0.30, "Rapid":0.45, "Ultra":0.65}[rep]
                candidates.append(Candidate(
                    station_id=sid,
                    rep_type=rep,
                    detour_km=(len1 + len2 - base_len) / 1000.0,
                    approx_eta_min=detour_min,
                    price_per_kwh=float(price),
                ))
                if len(candidates) >= top_k * 3:
                    break  # stop early; we’ll sort & mask

        # Sort pre-mask so emergency fallback uses nearest by ETA
        candidates.sort(key=lambda c: (c.approx_eta_min, c.price_per_kwh))

        # ---- Phase 4: POST-MASK (safety net; also handles emergency keep) ----
        candidates = mask_candidates_phase4(
            candidates,
            visited_station_ids=visited_station_ids,
            now_min=now_min,
            last_charge_end_min=last_charge_end_min,
            min_charge_gap_min=min_charge_gap_min,
            emergency=emergency,
            keep_nearest_n_on_emergency=keep_nearest_n_on_emergency,
        )

        # Return the final top_k after masking
        return candidates[:top_k]



    def _ensure_station_coords_cached(self, connectors_df):
        if self._coords_cached:
            return
        for row in connectors_df.itertuples(index=False):
            sid = getattr(row, "station_id", None)
            lat = getattr(row, "lat", None) if hasattr(row, "lat") else getattr(row, "latitude", None)
            lon = getattr(row, "lon", None) if hasattr(row, "lon") else getattr(row, "longitude", None)
            if not sid or lat is None or lon is None or sid in self._coords_xy_cache:
                continue
            try:
                x, y = self.sumo._to_xy(float(lat), float(lon))
                self._coords_xy_cache[str(sid)] = (float(x), float(y))
                self._coords_latlon_cache[str(sid)] = (float(lat), float(lon))
            except Exception:
                continue
        self._coords_cached = True

    def edge_for_station(self, station_id: str):
        eid = self._station_edge_map.get(str(station_id))
        if eid:
            return eid
        latlon = self._coords_latlon_cache.get(str(station_id))
        if not latlon:
            return None
        try:
            eid = self.sumo.snap_to_edge(latlon[0], latlon[1])
            if eid:
                self._station_edge_map[str(station_id)] = eid
        except Exception:
            eid = None
        return eid


    def edge_near_xy(self, x: float, y: float) -> Optional[str]:
        """
        Return nearest edge to XY coordinates (in SUMO net meters).
        """
        try:
            return self.sumo.nearest_edge_by_xy(float(x), float(y))
        except Exception:
            return None



# tiny helper without importing pandas at module top
def pd_isnan(x: Any) -> bool:
    try:
        return x != x  # NaN != NaN
    except Exception:
        return False
