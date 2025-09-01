# env/sumo_runner.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import sumolib
import traci


@dataclass
class SumoConfig:
    net_path: str = "london_inner.net.xml"
    sumo_bin: str = "sumo"          # "sumo" or "sumo-gui"
    gui: bool = False
    step_length: float = 1.0
    seed: int = 0
    quit_on_end: bool = False       # keep window open by default


class SumoRunner:
    """
    SUMO/TraCI helper:
      - robust lon/lat -> XY conversion
      - snap to drivable edges (lane-level 'allowed')
      - route + spawn vehicle (auto-zoom, track, red color)
      - GUI helpers (delay/zoom/track)
    """

    def __init__(self, cfg: SumoConfig):
        self.cfg = cfg
        self.net: Optional[sumolib.net.Net] = None
        self._running = False
        self._conn = None
        self._geo2xy: Optional[Callable[[float, float], Tuple[float, float]]] = None

    # ---------- lifecycle ----------
    def start(self):
        if self._running:
            return
        net_path = Path(self.cfg.net_path)
        if not net_path.exists():
            raise FileNotFoundError(f"SUMO net not found: {net_path}")

        self.net = sumolib.net.readNet(str(net_path))
        self._geo2xy = self._build_geo2xy(self.net)

        sumo_bin = self.cfg.sumo_bin if not self.cfg.gui else "sumo-gui"
        cmd = [
            sumo_bin,
            "-n", str(net_path),
            "--step-length", str(self.cfg.step_length),
            "--seed", str(self.cfg.seed),
        ]
        if self.cfg.quit_on_end:
            cmd += ["--quit-on-end", "true"]

        traci.start(cmd)
        self._conn = traci
        self._running = True

    def stop(self):
        if not self._running:
            return
        try:
            self._conn.close()
        finally:
            self._running = False
            self._conn = None

    # ---------- geo conversion ----------
    @staticmethod
    def _build_geo2xy(net: sumolib.net.Net) -> Optional[Callable[[float, float], Tuple[float, float]]]:
        try:
            lc = net.getLocationConverter()
            if lc is not None:
                def _lc(lat: float, lon: float) -> Tuple[float, float]:
                    x, y = lc.convertLonLat2XY(float(lon), float(lat))
                    return float(x), float(y)
                _ = _lc(51.5, -0.12)
                return _lc
        except Exception:
            pass

        for meth in ("convertLonLat2XY", "convertGeo"):
            if hasattr(net, meth):
                fn = getattr(net, meth)
                def _wrap(lat: float, lon: float, fn=fn):
                    x, y = fn(float(lon), float(lat))
                    return float(x), float(y)
                try:
                    _ = _wrap(51.5, -0.12)
                    return _wrap
                except Exception:
                    continue
        return None

    def _to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        if self.net is None:
            raise RuntimeError("SUMO net not loaded")
        if self._geo2xy is None:
            return float(lon), float(lat)
        return self._geo2xy(float(lat), float(lon))

    # ---------- GUI helpers ----------
    def gui_set_delay_ms(self, ms: int = 100, zoom: float = 600.0):
        if self._running and self.cfg.gui:
            try:
                view = self._conn.gui.getIDList()[0]
                self._conn.gui.setDelay(ms)
                self._conn.gui.setZoom(view, zoom)
                self._conn.gui.setVisible(view, True)
                try:
                    self._conn.gui.setSchema(view, "real world")
                except Exception:
                    pass
            except Exception:
                pass

    def gui_track_vehicle(self, veh_id: str):
        if self._running and self.cfg.gui:
            try:
                view = self._conn.gui.getIDList()[0]
                self._conn.gui.trackVehicle(view, veh_id)
            except Exception:
                pass

    # ---------- mapping ----------
    def _lane_allowed_classes(self, edge) -> set:
        allowed = set()
        try:
            for ln in edge.getLanes():
                try:
                    a = ln.getAllowed()
                    if a:
                        allowed.update(a)
                except Exception:
                    continue
        except Exception:
            pass
        return allowed

    def snap_to_edge(self, lat: float, lon: float, search_radius: float = 50.0, zoom_gui: bool = True) -> str:
        if self.net is None:
            raise RuntimeError("SUMO net not loaded")
        x, y = self._to_xy(lat, lon)

        radii = [search_radius, 100, 200, 500, 1000, 2000, 5000]
        edge_id: Optional[str] = None
        for r in radii:
            neighbors = self.net.getNeighboringEdges(x, y, r=r)
            if not neighbors:
                continue
            neighbors.sort(key=lambda t: t[1])
            fallback = None
            for edge, _ in neighbors:
                if edge.getFunction() == "internal":
                    continue
                allowed = self._lane_allowed_classes(edge)
                if "passenger" in allowed or not allowed:
                    edge_id = edge.getID()
                    break
                if fallback is None:
                    fallback = edge.getID()
            if edge_id is None and fallback is not None:
                edge_id = fallback
            if edge_id:
                break

        if edge_id is None:
            try:
                edge = self.net.getClosestEdge((x, y))
                edge_id = edge.getID()
            except Exception:
                pass

        if edge_id is None:
            raise ValueError(f"No edges near lat={lat}, lon={lon} (XY={x:.1f},{y:.1f}).")

        if self._running and self.cfg.gui and zoom_gui:
            try:
                view = self._conn.gui.getIDList()[0]
                self._conn.gui.setZoom(view, 1000)
                self._conn.gui.setOffset(view, x, y)
            except Exception:
                pass

        return edge_id

    # ---------- routing ----------
    def find_route_edges(self, from_edge: str, to_edge: str) -> Tuple[List[str], float, float]:
        if not self._running:
            raise RuntimeError("SUMO not running")
        route = self._conn.simulation.findRoute(from_edge, to_edge)
        return route.edges, float(route.length), float(route.travelTime)

    # ---------- vehicle ----------
    def _ensure_vehicle_type(self, preferred: str = "passenger") -> str:
        if not self._running:
            raise RuntimeError("SUMO not running")
        types = set(self._conn.vehicletype.getIDList())
        if preferred in types:
            return preferred
        if "DEFAULT_VEHTYPE" in types:
            return "DEFAULT_VEHTYPE"
        vt_id = "ev_car"
        if vt_id not in types:
            self._conn.vehicletype.add(vt_id, accel=2.6, decel=4.5, sigma=0.5,
                                       length=4.4, minGap=2.5, maxSpeed=33.33,
                                       speedFactor=1.0, speedDev=0.1, impatience=0.0)
            try:
                self._conn.vehicletype.setEmissionClass(vt_id, "Zero")
                self._conn.vehicletype.setGuiShape(vt_id, "passenger")
            except Exception:
                pass
        return vt_id

    def add_vehicle_with_route(self, veh_id: str, edge_from: str, edge_to: str,
                               depart_s: float = 0.0, route_id: Optional[str] = None,
                               type_id: Optional[str] = None) -> tuple[list[str], float, float]:
        """
        Adds a vehicle, assigns a route, colors it red, zooms & tracks it, and slows the GUI.
        Returns (edges, length_m, duration_s).
        """
        if not self._running:
            raise RuntimeError("SUMO not running")
        edges, length_m, duration_s = self.find_route_edges(edge_from, edge_to)
        if not edges:
            raise RuntimeError(f"Cannot route from {edge_from} to {edge_to}")

        if route_id is None:
            route_id = f"r_{veh_id}"
        self._conn.route.add(routeID=route_id, edges=edges)

        vt = type_id or self._ensure_vehicle_type("passenger")
        self._conn.vehicle.add(vehID=veh_id, routeID=route_id, typeID=vt, depart=str(depart_s))

        # make visible + camera
        try:
            self._conn.vehicle.setColor(veh_id, (255, 0, 0, 255))
        except Exception:
            pass
        if self.cfg.gui:
            self.gui_set_delay_ms(ms=120, zoom=500.0)  # slower & closer
            self.gui_track_vehicle(veh_id)

        return edges, length_m, duration_s

    # ---------- stepping ----------
    def step(self):
        if not self._running:
            raise RuntimeError("SUMO not running")
        self._conn.simulationStep()

    def get_vehicle_state(self, veh_id: str) -> dict:
        if not self._running:
            raise RuntimeError("SUMO not running")
        x, y = self._conn.vehicle.getPosition(veh_id)
        try:
            lon, lat = (self.net.convertXY2LonLat(x, y) if hasattr(self.net, "convertXY2LonLat") else (x, y))
        except Exception:
            lon, lat = x, y
        return {
            "edge_id": self._conn.vehicle.getRoadID(veh_id),
            "speed_mps": self._conn.vehicle.getSpeed(veh_id),
            "pos_xy": (x, y),
            "pos_latlon": (lat, lon),
        }
