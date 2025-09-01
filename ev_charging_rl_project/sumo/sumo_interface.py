import traci
import traci.constants as tc

class SumoInterface:
    def __init__(self, sumo_binary="sumo", net_file=None, fast_mode=False):
        self.sumo_binary = sumo_binary
        self.net_file = net_file
        self.fast_mode = fast_mode
        self.vehicles = {}
        self._last_cmd = None  # for debugging/tests

    def start(self, gui=False, extra_args=None):
        """
        Launch SUMO or SUMO-GUI safely.
        - If sumo_binary="sumo" and gui=True -> use "sumo-gui"
        - If sumo_binary="sumo-gui" and gui=False -> use "sumo"
        - extra_args: list of additional SUMO CLI args (e.g., ["--step-length", "0.2"])
        """
        bin_name = self.sumo_binary
        if gui and not bin_name.endswith("-gui"):
            bin_name = bin_name + "-gui"
        if not gui and bin_name.endswith("-gui"):
            bin_name = bin_name[:-4]  # strip "-gui"

        cmd = [bin_name, "-n", self.net_file]
        if self.fast_mode:
            # conservative quality-of-life options that don't change your logic
            cmd += ["--start", "--no-warnings"]
        if extra_args:
            cmd += list(extra_args)

        self._last_cmd = cmd
        traci.start(cmd)

    def close(self):
        traci.close()

    def add_vehicle(self, veh_id, from_edge, to_edge, depart_time=0):
        route_id = f"route_{veh_id}"
        edges = [from_edge, to_edge] if self.fast_mode else self._shortest_path_edges(from_edge, to_edge)
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
        # Use SUMO's routing to get a sequence of edges
        route = traci.simulation.findRoute(from_edge, to_edge).edges
        return route if route else [from_edge, to_edge]

    def remove_vehicle(self, veh_id):
        if veh_id in self.vehicles:
            traci.vehicle.remove(veh_id)
            del self.vehicles[veh_id]
