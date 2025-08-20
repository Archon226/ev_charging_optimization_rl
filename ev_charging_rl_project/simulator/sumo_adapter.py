# simulator/sumo_adapter.py
from __future__ import annotations
import math
from functools import lru_cache
from typing import Tuple, Optional, List

import numpy as np
import sumolib


# ---------- geometry helpers ----------
def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


class SumoSim:
    """
    Minimal routing adapter:
      - Accepts (lat, lon) from callers.
      - Snaps each (lat, lon) to the nearest drivable edge using net.convertLonLat2XY(lon, lat).
      - Tries getShortestPath by travel time (or length).
      - If snapping fails or SUMO returns a degenerate route while Haversine is non-trivial,
        falls back to Haversine distance and an assumed speed.

    Public methods (kept for compatibility):
      * route_between_cached(o_lat, o_lon, d_lat, d_lon) -> (km, sec)
      * detour_via_cached(o_lat, o_lon, c_lat, c_lon, d_lat, d_lon) -> (dkm, dsec)
    """

    def __init__(self, net_path: str, prefer_time: bool = True, fallback_speed_kph: float = 30.0):
        self.net = sumolib.net.readNet(net_path, withInternal=False, withPedestrians=False, withBicycles=False)
        self.prefer_time = bool(prefer_time)
        self.fallback_speed_kph = float(fallback_speed_kph)

    # ---------- low-level snapping ----------
    def _xy(self, lat: float, lon: float) -> Tuple[float, float]:
        # SUMO expects (lon, lat) here
        return self.net.convertLonLat2XY(lon, lat)

    @lru_cache(maxsize=500_000)
    def _nearest_edge_cached(self, qlat: float, qlon: float) -> Optional["sumolib.net.edge.Edge"]:
        """Cached snap to nearest drivable edge for quantized (lat,lon)."""
        lat = round(float(qlat), 6)
        lon = round(float(qlon), 6)
        x, y = self._xy(lat, lon)
        # radius kept modest; you can raise to 250 if points are sometimes off-net
        neigh = self.net.getNeighboringEdges(x, y, r=150.0)
        if not neigh:
            return None
        def score(item):
            e, dist = item
            drivable = ("highway" in (e.getFunction() or "")) or (e.allows("passenger"))
            return (0 if drivable else 1, dist)
        neigh.sort(key=score)
        return neigh[0][0]

    # keep the old name for internal calls
    def _nearest_edge(self, lat: float, lon: float):
        return self._nearest_edge_cached(lat, lon)

    def _route_cost(self, edge: "sumolib.net.edge.Edge") -> Tuple[float, float]:
        """Return (length_m, time_s) for one edge."""
        L = float(edge.getLength() or 0.0)
        v = float(edge.getSpeed() or 0.0)  # m/s
        return L, (L / v if v > 0 else 0.0)

    def _sum_path_cost(self, path: List["sumolib.net.edge.Edge"]) -> Tuple[float, float]:
        total_len_m = 0.0
        total_time_s = 0.0
        for e in path or []:
            L, t = self._route_cost(e)
            total_len_m += L
            total_time_s += t
        return total_len_m, total_time_s

    def _shortest_path(self, e_from, e_to, prefer_time: Optional[bool] = None):
        """
        Try several getShortestPath variants for maximum compatibility across SUMO builds.
        Returns (path_edges, cost) or (None, inf) if not found.
        """
        prefer_time = self.prefer_time if prefer_time is None else prefer_time

        # Attempt 1: named weight 'travelTime'
        try:
            w = "travelTime" if prefer_time else "length"
            path, cost = self.net.getShortestPath(e_from, e_to, weight=w)
            if path:
                return path, float(cost if cost is not None else np.inf)
        except Exception:
            pass

        # Attempt 2: boolean preferTime signature
        try:
            path, cost = self.net.getShortestPath(e_from, e_to, preferTime=prefer_time)
            if path:
                return path, float(cost if cost is not None else np.inf)
        except Exception:
            pass

        # Attempt 3: legacy signature (undocumented fallbacks may exist)
        try:
            path, cost = self.net.getShortestPath(e_from, e_to)
            if path:
                return path, float(cost if cost is not None else np.inf)
        except Exception:
            pass

        return None, float("inf")
    
    def cheap_detour_upperbound_km(self, o_lat, o_lon, c_lat, c_lon, d_lat, d_lon) -> float:
        """
        Fast upper bound on extra km for detouring via C:
          UB ≈ hav(O,C) + hav(C,D) - hav(O,D), clipped at 0.
        Use this to prefilter candidates before calling expensive routing.
        """
        hav = _haversine_km
        ub = (hav(o_lat, o_lon, c_lat, c_lon) +
              hav(c_lat, c_lon, d_lat, d_lon) - 
              hav(o_lat, o_lon, d_lat, d_lon))
        return max(0.0, ub)

    # ---------- public API ----------
    @lru_cache(maxsize=200_000)
    def route_between_cached(self, o_lat: float, o_lon: float, d_lat: float, d_lon: float) -> Tuple[float, float]:
        """
        Return (km, sec) from origin to dest.
        Robustly falls back to Haversine if snapping/pathing fails or returns degenerate distances.
        """
        # Haversine baseline
        hav_km = _haversine_km(o_lat, o_lon, d_lat, d_lon)

        # Snap both endpoints
        e_from = self._nearest_edge(o_lat, o_lon)
        e_to = self._nearest_edge(d_lat, d_lon)

        if (e_from is None) or (e_to is None):
            # Out of net → fallback
            sec = (hav_km / max(self.fallback_speed_kph, 1e-3)) * 3600.0
            return float(hav_km), float(sec)

        # Get shortest path
        path, _ = self._shortest_path(e_from, e_to, prefer_time=self.prefer_time)

        if not path:
            # No path → fallback
            sec = (hav_km / max(self.fallback_speed_kph, 1e-3)) * 3600.0
            return float(hav_km), float(sec)

        length_m, time_s = self._sum_path_cost(path)

        # Final robustness guard:
        # If SUMO path is (almost) zero but Haversine says O(>300 m), trust Haversine.
        if (length_m < 20.0 and hav_km > 0.3):
            sec = (hav_km / max(self.fallback_speed_kph, 1e-3)) * 3600.0
            return float(hav_km), float(sec)

        return float(length_m / 1000.0), float(time_s)

    @lru_cache(maxsize=100_000)
    def detour_via_cached(self, o_lat, o_lon, c_lat, c_lon, d_lat, d_lon) -> Tuple[float, float]:
        """Returns additional detour (km, sec) when going O->C->D vs O->D."""
        bd_km, bd_s = self.route_between_cached(o_lat, o_lon, d_lat, d_lon)
        oc_km, oc_s = self.route_between_cached(o_lat, o_lon, c_lat, c_lon)
        cd_km, cd_s = self.route_between_cached(c_lat, c_lon, d_lat, d_lon)

        # Extra guard: if a leg collapses but straight-line says otherwise, trust Haversine
        if (oc_km < 0.02) and (_haversine_km(o_lat, o_lon, c_lat, c_lon) > 0.3):
            oc_km = _haversine_km(o_lat, o_lon, c_lat, c_lon)
            oc_s  = (oc_km / max(self.fallback_speed_kph, 1e-3)) * 3600.0
        if (cd_km < 0.02) and (_haversine_km(c_lat, c_lon, d_lat, d_lon) > 0.3):
            cd_km = _haversine_km(c_lat, c_lon, d_lat, d_lon)
            cd_s  = (cd_km / max(self.fallback_speed_kph, 1e-3)) * 3600.0

        dk = max(0.0, (oc_km + cd_km) - bd_km)
        dt = max(0.0, (oc_s + cd_s) - bd_s)

        if not (np.isfinite(dk) and np.isfinite(dt)):
            return 0.0, 0.0
        return dk, dt
