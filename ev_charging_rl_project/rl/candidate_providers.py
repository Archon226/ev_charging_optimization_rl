# rl/candidate_providers.py
from __future__ import annotations
import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

@dataclass
class Candidate:
    station_id: str
    rep_type: str                 # Fast/Rapid/Ultra
    detour_km: float
    approx_eta_min: float
    price_per_kwh: float


from env.sumo_runner import SumoRunner, SumoConfig

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


    def generate(self, trip, station_index, connectors_df, price_lookup, top_k: int):
        """
        Corridor-only candidate search:
          - Get SUMO O→D route once.
          - Build a polyline corridor (± max_detour_km) around that route in XY meters.
          - Keep only stations inside the corridor.
          - Route only that small set O→S and S→D (hard-capped), early-stopping at top_k.
        """
        # -------- constants (tune lightly if needed) ----------
        MAX_STATIONS_TO_EVAL = max(24, top_k * 12)     # hard cap on SUMO routing
        SEARCH_RADIUS_M = 200                          # station edge snap radius
        MAX_DETOUR_MIN = self.max_detour_min           # time-based detour budget
        # distance-based corridor width from TripPlan (km → m); fallback 2km
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

        # Build route polyline from edge shapes; sumo net coords are meters
        net = self.sumo.net
        route_xy: list[tuple[float, float]] = []
        for eid in edges:
            try:
                e = net.getEdge(eid)
                shape = e.getShape()  # list[(x,y)]
                if shape:
                    # append with continuity (avoid duplicating the first point of each edge)
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
                    coords_xy[sid] = (float(x), float(y))
                    coords_latlon[sid] = (float(lat), float(lon))
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
                    # degenerate segment
                    d = math.hypot(px - x1, py - y1)
                else:
                    t = max(0.0, min(1.0, (wx*vx + wy*vy) / denom))
                    qx, qy = x1 + t * vx, y1 + t * vy
                    d = math.hypot(px - qx, py - qy)
                if d < best:
                    best = d
            return best

        # 4) corridor filter
        corridor_sids: list[str] = []
        for sid, (sx, sy) in coords_xy.items():
            d = _dist_point_polyline_m(sx, sy, route_xy)
            if d <= corridor_width_m:
                corridor_sids.append(sid)

        # cap how many we’ll route with SUMO (cheapest first: prox to line)
        corridor_sids.sort(key=lambda s: _dist_point_polyline_m(coords_xy[s][0], coords_xy[s][1], route_xy))
        corridor_sids = corridor_sids[:MAX_STATIONS_TO_EVAL]

        # 5) route O→S + S→D for the corridor set; early-stop at top_k
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
                if len(candidates) >= top_k:
                    break

        candidates.sort(key=lambda c: (c.approx_eta_min, c.price_per_kwh))
        return candidates[:top_k]

    def edge_for_station(self, station_id: str) -> Optional[str]:
        """
        Return nearest SUMO edge id for a given station_id.
        Caches results so we don’t snap repeatedly.
        """
        sid = str(station_id)
        if sid in self._station_edge_map:
            return self._station_edge_map[sid]
        try:
            # You already store lat/lon in connectors_df when generate() is called,
            # but here we may be called outside generate → so best-effort only.
            # If you have a global station index with lat/lon, hook it here.
            return self._station_edge_map.get(sid)
        except Exception:
            return None

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
