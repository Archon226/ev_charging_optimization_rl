# simulator/sumo_adapter.py
from __future__ import annotations
import math
from functools import lru_cache
from typing import Tuple, Optional, List

import numpy as np
import sumolib


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


class SumoSim:
    """
    Minimal routing adapter:
      - Prefers sumolib.route.mapMatch when available.
      - Falls back to manual nearest-edge snap + getShortestPath (length/time).
      - If snapping fails (point outside the net), falls back to great-circle distance + assumed speed.
    Public methods:
      * route_between_cached(o_lat, o_lon, d_lat, d_lon) -> (km, sec)
      * detour_via_cached(o_lat, o_lon, c_lat, c_lon, d_lat, d_lon) -> (dkm, dsec)
    """

    def __init__(self, net_path: str, prefer_time: bool = True, fallback_speed_kph: float = 30.0):
        # net_path should be absolute or a path SUMO can open
        self.net = sumolib.net.readNet(net_path)
        self._has_mapMatch = hasattr(sumolib.route, "mapMatch")
        self.prefer_time = prefer_time  # when computing shortest path, prefer time over length
        self.fallback_speed_kph = max(5.0, float(fallback_speed_kph))
        self._route_cache = {}

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _xy(self, lat: float, lon: float) -> Tuple[float, float]:
        # SUMO order: convertLonLat2XY(lon, lat)
        return self.net.convertLonLat2XY(lon, lat)

    def _closest_edge(self, lat: float, lon: float, radii: List[float] = (100.0, 250.0, 500.0, 1000.0)):
        x, y = self._xy(lat, lon)
        for r in radii:
            cand = self.net.getNeighboringEdges(x, y, r)
            if cand:
                # Prefer edges that allow passenger cars, then by distance
                car_ok = [(e, d) for (e, d) in cand if self._edge_allows_passenger(e)]
                pool = car_ok if car_ok else cand
                edge, _dist = min(pool, key=lambda t: t[1])
                return edge
        raise RuntimeError(f"No edges found near lat={lat}, lon={lon}")

    @staticmethod
    def _edge_allows_passenger(edge) -> bool:
        try:
            return edge.allows("passenger")
        except Exception:
            # Some builds may not expose allows(); assume edge is okay
            return True

    @staticmethod
    def _edge_len_m(edge) -> float:
        try:
            return edge.getLength()
        except Exception:
            return getattr(edge, "length", float("inf"))

    @staticmethod
    def _edge_speed_ms(edge) -> float:
        try:
            s = edge.getSpeed()
        except Exception:
            s = getattr(edge, "speed", 0.0)
        return max(0.1, s or 0.0)  # avoid divide-by-zero

    def _route_cost(self, edge) -> Tuple[float, float]:
        """Return (length_m, time_s) contribution for an edge."""
        L = self._edge_len_m(edge)
        t = L / self._edge_speed_ms(edge)
        return L, t

    def _sum_path_metrics(self, path) -> Tuple[float, float]:
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

        # Attempt 1: named weight 'travelTime' / 'length'
        try:
            weight_key = "travelTime" if prefer_time else "length"
            path, cost = self.net.getShortestPath(e_from, e_to, weight=weight_key)
            if path is not None:
                return path, cost
        except Exception:
            pass

        # Attempt 2: callable weight (edge -> cost)
        try:
            if prefer_time:
                path, cost = self.net.getShortestPath(
                    e_from, e_to, weight=lambda e: self._edge_len_m(e) / self._edge_speed_ms(e)
                )
            else:
                path, cost = self.net.getShortestPath(e_from, e_to, weight=lambda e: self._edge_len_m(e))
            if path is not None:
                return path, cost
        except Exception:
            pass

        # Not found
        return None, float("inf")

    # ---------------------------------------------------------------------
    # Map-matching strategies
    # ---------------------------------------------------------------------
    def _route_mapmatch_api(self, o_lat, o_lon, d_lat, d_lon) -> Tuple[float, float]:
        """
        Use sumolib.route.mapMatch if present (newer SUMO).
        For O/D pairs, manual shortest path after snapping is robust and fast,
        so we keep using manual for now. Left as a placeholder.
        """
        raise NotImplementedError

    def _route_manual(self, o_lat, o_lon, d_lat, d_lon) -> Tuple[float, float]:
        """
        Manual: snap O/D to nearest edges, then shortest path; returns (km, sec).
        If snapping fails (point outside net), fallback to great-circle + assumed speed.
        """
        try:
            e_o = self._closest_edge(o_lat, o_lon)
            e_d = self._closest_edge(d_lat, d_lon)
            path, _ = self._shortest_path(e_o, e_d)
            if path is None:
                # No path in the graph between these snapped edges
                raise RuntimeError("No path between snapped edges")
            L_m, T_s = self._sum_path_metrics(path)
            return L_m / 1000.0, T_s
        except Exception:
            # Fallback: haversine + assumed traffic speed
            dk = _haversine_km(o_lat, o_lon, d_lat, d_lon)
            dt = (dk / self.fallback_speed_kph) * 3600.0
            return float(dk), float(dt)

    # ---------------------------------------------------------------------
    # Public cached API
    # ---------------------------------------------------------------------
    @lru_cache(maxsize=100_000)
    def route_between_cached(self, o_lat, o_lon, d_lat, d_lon) -> Tuple[float, float]:
        """Returns (baseline_distance_km, baseline_time_s)."""
        if self._has_mapMatch:
            try:
                return self._route_manual(o_lat, o_lon, d_lat, d_lon)
            except Exception:
                return self._route_manual(o_lat, o_lon, d_lat, d_lon)
        else:
            return self._route_manual(o_lat, o_lon, d_lat, d_lon)

    @lru_cache(maxsize=100_000)
    def detour_via_cached(self, o_lat, o_lon, c_lat, c_lon, d_lat, d_lon) -> Tuple[float, float]:
        """Returns additional detour (km, sec) when going O->C->D vs O->D."""
        bd_km, bd_s = self.route_between_cached(o_lat, o_lon, d_lat, d_lon)
        oc_km, oc_s = self.route_between_cached(o_lat, o_lon, c_lat, c_lon)
        cd_km, cd_s = self.route_between_cached(c_lat, c_lon, d_lat, d_lon)

        dk = max(0.0, (oc_km + cd_km) - bd_km)
        dt = max(0.0, (oc_s + cd_s) - bd_s)
        # force-finite
        if not (np.isfinite(dk) and np.isfinite(dt)):
            return 0.0, 0.0
        return dk, dt
