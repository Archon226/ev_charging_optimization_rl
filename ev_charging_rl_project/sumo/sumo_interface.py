import traci
import traci.constants as tc

class SumoInterface:
    def __init__(self, sumo_binary="sumo", net_file=None, fast_mode=False):
        self.sumo_binary = sumo_binary
        self.net_file = net_file
        self.fast_mode = fast_mode
        self.vehicles = {}

    def start(self, gui=False):
        import subprocess
        cmd = [self.sumo_binary + ("-gui" if gui else ""), "-n", self.net_file]
        traci.start(cmd)

    def close(self):
        traci.close()

    def add_vehicle(self, veh_id, from_edge, to_edge, depart_time=0):
        route_id = f"route_{veh_id}"
        if not self.fast_mode:
            edges = self._shortest_path_edges(from_edge, to_edge)
        else:
            edges = [from_edge, to_edge]  # fast mode shortcut
        traci.route.add(route_id, edges)
        traci.vehicle.add(veh_id, route_id, depart=depart_time)
        self.vehicles[veh_id] = {"from": from_edge, "to": to_edge}

    def step(self):
        traci.simulationStep()

    def multi_step(self, steps=1):
        for _ in range(steps):
            traci.simulationStep()

    def get_vehicle_position(self, veh_id):
        x, y = traci.vehicle.getPosition(veh_id)
        return self._to_geo(x, y)

    def _to_geo(self, x, y):
        return traci.simulation.convertGeo(x, y, fromGeo=False)

    def _shortest_path_edges(self, from_edge, to_edge):
        """
        Use SUMO's inbuilt routing to get the shortest path as a list of edges.
        """
        traci.simulation.findRoute(from_edge, to_edge)
        route = traci.simulation.findRoute(from_edge, to_edge).edges
        return route if route else [from_edge, to_edge]

    def remove_vehicle(self, veh_id):
        if veh_id in self.vehicles:
            traci.vehicle.remove(veh_id)
            del self.vehicles[veh_id]
