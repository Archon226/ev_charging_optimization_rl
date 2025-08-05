import traci

class SumoInterface:
    def __init__(self, config):
        self.config = config
        self.sumo_binary = config.get("sumo_binary", "sumo")  # or "sumo-gui"
        self.sumo_cfg_file = config.get("sumo_config", "sumo_config.sumocfg")
        self.simulation_time = 0

    def reset(self):
        """Start or restart the SUMO simulation"""
        if traci.isLoaded():
            traci.close()

        sumo_cmd = [self.sumo_binary, "-c", self.sumo_cfg_file, "--start"]
        traci.start(sumo_cmd)
        self.simulation_time = 0

    def step(self):
        """Advance one simulation step"""
        traci.simulationStep()
        self.simulation_time += 1

    def route_agent_to_charger(self, agent, charger):
        """
        Route an EVAgent to a specific charger.
        Requires: agent.id, charger.location
        """
        # Assume charger.location is a road edge ID
        route = self._generate_route_to_charger(agent.current_edge, charger.edge_id)
        traci.vehicle.setRoute(agent.traci_id, route)

    def get_time_of_day_norm(self):
        """Returns time of day as normalized float [0, 1]"""
        total_seconds_in_day = 24 * 60 * 60
        return (self.simulation_time % total_seconds_in_day) / total_seconds_in_day

    def _generate_route_to_charger(self, from_edge, to_edge):
        """
        Placeholder route function. Use TraCI or Dijkstra routing here.
        """
        return [from_edge, to_edge]  # mock; replace with real route logic