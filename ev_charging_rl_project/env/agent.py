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
        self.soc_percent = start_soc_percent
        self.current_energy_kWh = (start_soc_percent / 100) * battery_kWh

        # Trip
        self.origin = origin  # (lat, lon) or SUMO node
        self.destination = destination
        self.route_history = [origin]
        self.objective = objective
        self.max_budget = max_budget
        self.max_time_slack = max_time_slack

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

    def charge(self, energy_kWh, time_spent_min, cost):
        """Add charge to the EV, log session."""
        self.current_energy_kWh += energy_kWh
        self.current_energy_kWh = min(self.current_energy_kWh, self.battery_kWh)
        self.soc_percent = (self.current_energy_kWh / self.battery_kWh) * 100
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

    def needs_charging(self, required_range_km):
        """Check if the EV needs charging to reach a certain range."""
        return self.estimate_range_km() < required_range_km
