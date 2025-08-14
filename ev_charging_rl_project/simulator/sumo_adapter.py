from functools import lru_cache
import sumolib

class SumoSim:
    def __init__(self, net_path: str):
        self.net = sumolib.net.readNet(net_path)

    @lru_cache(maxsize=100_000)
    def snap_edge_cached(self, lat: float, lon: float):
        return self.net.getNeighboringEdges(lat, lon, 50)[0][0]

    @lru_cache(maxsize=200_000)
    def route_between_cached(self, a_lat, a_lon, b_lat, b_lon):
        edge_a = self.snap_edge_cached(a_lat, a_lon)
        edge_b = self.snap_edge_cached(b_lat, b_lon)
        route = sumolib.route.mapMatch([edge_a, edge_b])
        dist_km = route.getLength() / 1000
        time_s = dist_km / 13.89  # ~50 km/h avg
        return dist_km, time_s

    def detour_via_cached(self, o_lat, o_lon, p_lat, p_lon, d_lat, d_lon):
        bd_km, bd_s = self.route_between_cached(o_lat, o_lon, d_lat, d_lon)
        d1_km, d1_s = self.route_between_cached(o_lat, o_lon, p_lat, p_lon)
        d2_km, d2_s = self.route_between_cached(p_lat, p_lon, d_lat, d_lon)
        return max(0, (d1_km + d2_km) - bd_km), max(0, (d1_s + d2_s) - bd_s)
