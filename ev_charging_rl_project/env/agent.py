import uuid

class EVAgent:
    def __init__(self, user_id, ev_model, battery_kWh, efficiency_Wh_per_km,
                 ac_power_kW, dc_power_kW, start_soc_percent, origin, destination,
                 objective='cost', max_budget=None, max_time_slack=None):

        self.user_id = user_id or str(uuid.uuid4())
        self.ev_model = ev_model

        # Specs
        self.battery_kWh = battery_kWh
        self.efficiency_Wh_per_km = efficiency_Wh_per_km
        self.ac_power_kW = ac_power_kW
        self.dc_power_kW = dc_power_kW

        # Energy state
        self.max_soc = battery_kWh
        self.soc_percent = start_soc_percent
        self.current_energy_kWh = (start_soc_percent / 100) * battery_kWh
        self.soc = self.current_energy_kWh

        # Trip
        self.origin = origin  # (lat, lon) or SUMO node
        self.destination = destination
        self.route_history = [origin]
        self.objective = objective
        self.max_budget = max_budget
        self.max_time_slack = max_time_slack

        # SUMO integration attributes
        self.traci_id = None
        self.current_edge = None
        self.location = origin  # updated from SUMO if needed

        # Logs
        self.total_cost = 0.0
        self.total_time = 0.0
        self.charging_sessions = []

    def estimate_range_km(self):
        """Estimate how far the EV can go with current energy."""
        return (self.current_energy_kWh * 1000) / self.efficiency_Wh_per_km

    def update_soc(self, energy_used_kWh):
        self.current_energy_kWh -= energy_used_kWh
        self.soc_percent = (self.current_energy_kWh / self.battery_kWh) * 100
        self.soc = self.current_energy_kWh

    def charge(self, energy_kWh, time_spent_min, cost):
        """Add charge to the EV, log session."""
        self.current_energy_kWh += energy_kWh
        self.current_energy_kWh = min(self.current_energy_kWh, self.battery_kWh)
        self.soc_percent = (self.current_energy_kWh / self.battery_kWh) * 100
        self.soc = self.current_energy_kWh
        self.total_cost += cost
        self.total_time += time_spent_min
        self.charging_sessions.append({
            "energy_kWh": energy_kWh,
            "cost": cost,
            "time_min": time_spent_min
        })

    def drive_to(self, location, distance_km, travel_time_min):
        """Simulate driving and consuming energy."""
        energy_used = (distance_km * self.efficiency_Wh_per_km) / 1000
        self.update_soc(energy_used)
        self.total_time += travel_time_min
        self.route_history.append(location)
        self.location = location

    def needs_charging(self, required_range_km):
        """Check if the EV needs charging to reach a certain range."""
        return self.estimate_range_km() < required_range_km

    def update_state(self, sumo_interface):
        """Update agent state from SUMO."""
        self.current_edge = sumo_interface.get_edge_id(self.traci_id)
        self.location = sumo_interface.get_position(self.traci_id)

    def is_fully_charged(self):
        return self.current_energy_kWh >= self.battery_kWh

    def reset(self):
        """Reset agent to initial energy state and clear logs."""
        self.soc_percent = 30.0
        self.current_energy_kWh = 0.3 * self.battery_kWh
        self.soc = self.current_energy_kWh
        self.total_cost = 0.0
        self.total_time = 0.0
        self.route_history = [self.origin]
        self.charging_sessions = []
        self.current_edge = None
        self.traci_id = None
        self.location = self.origin
