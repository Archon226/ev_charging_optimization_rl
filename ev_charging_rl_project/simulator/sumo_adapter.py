"""
SUMO adapter (Phase 1, Windows‑ready)

What it does (static, free‑flow):
  - Load a SUMO .net.xml
  - Snap (lat, lon) -> nearest edge
  - Compute a shortest path (sumolib Dijkstra) between two points
  - Advance along a route by a time horizon (seconds)
  - Compute detour (origin -> POI -> dest) vs baseline (origin -> dest)

Windows notes:
  - Make sure SUMO is installed and SUMO_HOME is set, e.g.:
      setx SUMO_HOME "C:\\SUMO" /M
  - The folder %SUMO_HOME% must contain subfolders: bin, tools, docs, ...
  - This module appends %SUMO_HOME%\\tools to sys.path so sumolib imports work.
"""

from __future__ import annotations
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Dict

# --- Ensure SUMO tools on sys.path (Windows safe) ---
SUMO_HOME = os.environ.get("SUMO_HOME")
if not SUMO_HOME:
    raise RuntimeError(
        "SUMO_HOME is not set.\n"
        "Install SUMO and set the environment variable, e.g. (PowerShell as Admin):\n"
        '  setx SUMO_HOME "C:\\SUMO" /M\n'
        "Where C:\\SUMO contains 'bin', 'tools', 'docs', ..."
    )
TOOLS_DIR = os.path.join(SUMO_HOME, "tools")
if not os.path.isdir(TOOLS_DIR):
    raise RuntimeError(
        f"SUMO tools folder not found: {TOOLS_DIR}\n"
        "Check that SUMO_HOME points to the SUMO install root."
    )
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)

import sumolib  # now available because we added ...\\tools


@dataclass(frozen=True)
class EdgeInfo:
    edge_id: str
    dist_km: float
    freeflow_s: float


@dataclass(frozen=True)
class SumoRoute:
    edges: Tuple[str, ...]
    per_edge: Tuple[EdgeInfo, ...]
    dist_km: float
    time_s: float


class SumoSim:
    def __init__(self, net_file: str):
        if not os.path.exists(net_file):
            raise FileNotFoundError(
                f"SUMO net not found: {net_file}\n"
                "Create one with netconvert or point to your existing .net.xml."
            )
        # Load network once (fast)
        self.net: sumolib.net.Net = sumolib.net.readNet(net_file)

    # ---------------------------
    # Internal helpers
    # ---------------------------
    def _xy(self, lat: float, lon: float) -> Tuple[float, float]:
        # SUMO expects lon,lat order; returns local (x,y)
        x, y = self.net.convertLonLat2XY(float(lon), float(lat))
        return float(x), float(y)

    def _snap_edge(self, lat: float, lon: float, search_m: float = 120.0) -> sumolib.net.edge.Edge:
        """Find closest edge to a point; try a radius first, then full search."""
        x, y = self._xy(lat, lon)
        candidates = self.net.getNeighboringEdges(x, y, r=search_m)
        if candidates:
            # list of (edge, distance); pick nearest
            return candidates[0][0]
        return self.net.getClosestEdge(x, y)

    def _edges_path(self, e_from: sumolib.net.edge.Edge, e_to: sumolib.net.edge.Edge):
        """Shortest path using free-flow travel time."""
        # SUMO >= 1.17 has sumolib.route.computePath
        if hasattr(sumolib, "route") and hasattr(sumolib.route, "computePath"):
            path, _ = sumolib.route.computePath(e_from, e_to, self.net)
            return path or [e_from]
        else:
            # Older SUMO: use built-in shortest path from net object
            path_edges, _ = self.net.getShortestPath(e_from, e_to)
            return path_edges or [e_from]


    def _edgeinfo(self, e: sumolib.net.edge.Edge) -> EdgeInfo:
        length_m = max(0.0, float(e.getLength()))
        speed_mps = max(0.1, float(e.getSpeed()))  # guard against zero
        return EdgeInfo(
            edge_id=e.getID(),
            dist_km=length_m / 1000.0,
            freeflow_s=length_m / speed_mps,
        )

    def _route_from_edges(self, edges) -> SumoRoute:
        infos = tuple(self._edgeinfo(e) for e in edges)
        return SumoRoute(
            edges=tuple(e.getID() for e in edges),
            per_edge=infos,
            dist_km=sum(i.dist_km for i in infos),
            time_s=sum(i.freeflow_s for i in infos),
        )

    # ---------------------------
    # Public API
    # ---------------------------
    def route_between(self, origin_latlon: Tuple[float, float], dest_latlon: Tuple[float, float]) -> SumoRoute:
        """Compute free‑flow route between two lat/lon points."""
        o_lat, o_lon = float(origin_latlon[0]), float(origin_latlon[1])
        d_lat, d_lon = float(dest_latlon[0]), float(dest_latlon[1])
        e_from = self._snap_edge(o_lat, o_lon)
        e_to   = self._snap_edge(d_lat, d_lon)
        path_edges = self._edges_path(e_from, e_to)
        return self._route_from_edges(path_edges)

    def advance(self, route: SumoRoute, start_idx: int, horizon_s: float) -> Dict[str, float | int]:
        """Walk forward along route edges up to horizon_s (free‑flow).
        Returns: {'end_idx', 'dist_km', 'time_s'}
        """
        if not (0 <= start_idx <= len(route.per_edge)):
            raise IndexError("start_idx out of bounds")
        t = 0.0
        d = 0.0
        idx = start_idx
        n = len(route.per_edge)
        while idx < n:
            e = route.per_edge[idx]
            if t + e.freeflow_s > horizon_s:
                break
            t += e.freeflow_s
            d += e.dist_km
            idx += 1
        return {"end_idx": idx, "dist_km": d, "time_s": t}

    def detour_metrics(
        self,
        origin_latlon: Tuple[float, float],
        dest_latlon: Tuple[float, float],
        poi_latlon: Tuple[float, float],
    ) -> Dict[str, float]:
        """Extra distance/time if we visit a POI vs going direct (free‑flow routing)."""
        base = self.route_between(origin_latlon, dest_latlon)
        leg1 = self.route_between(origin_latlon, poi_latlon)
        leg2 = self.route_between(poi_latlon, dest_latlon)

        detour_dist = (leg1.dist_km + leg2.dist_km) - base.dist_km
        detour_time = (leg1.time_s  + leg2.time_s ) - base.time_s

        return {
            "baseline_dist_km": base.dist_km,
            "baseline_time_s": base.time_s,
            "detour_dist_km": max(0.0, detour_dist),
            "detour_time_s": max(0.0, detour_time),
        }
