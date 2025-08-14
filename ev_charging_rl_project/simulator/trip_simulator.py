class TripSimulator:
    def __init__(self, sim, pricing_catalog):
        self.sim = sim
        self.pricing = pricing_catalog

    def reset(self, origin, dest, ev_spec, current_soc):
        self.origin = origin
        self.dest = dest
        self.ev_spec = ev_spec
        self.current_soc = current_soc
        self.position = origin
        self.done = False
        return self._get_state()

    def step(self, action_station):
        """
        Action: station_id to stop at next
        """
        # Travel to station
        dk, dt = self.sim.detour_via_cached(
            self.position[0], self.position[1],
            action_station.lat, action_station.lon,
            self.dest[0], self.dest[1]
        )
        self.current_soc -= dk * self.ev_spec.eff_kwh_per_km / self.ev_spec.battery_kwh
        cost = self.pricing.compute_price(action_station, self.ev_spec)

        # Check if reached destination
        if self.position == self.dest:
            self.done = True
            reward = -cost
        else:
            reward = -cost - dt / 3600  # penalize time

        self.position = (action_station.lat, action_station.lon)
        return self._get_state(), reward, self.done, {"cost": cost, "time": dt}

    def _get_state(self):
        return {
            "position": self.position,
            "soc": self.current_soc,
            "dest": self.dest
        }
