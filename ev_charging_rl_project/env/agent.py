import math

class EVAgent:
    def __init__(self, ev_spec, normalize_rewards=False, reward_norm_factor=1.0):
        self.ev_spec = ev_spec
        self.normalize_rewards = normalize_rewards
        self.reward_norm_factor = reward_norm_factor
        self.reset()

    def reset(self):
        self.soc = 1.0  # state of charge (fraction)
        self.position = None
        self.total_cost = 0.0
        self.total_time = 0.0

    def drive(self, distance_km, speed_kmh=50.0):
        self.soc -= (distance_km * self.ev_spec["eff_kwh_per_km"]) / self.ev_spec["battery_kwh"]
        self.total_time += (distance_km / speed_kmh) * 3600  # seconds

    def charge(self, added_kwh, cost):
        self.soc = min(1.0, self.soc + added_kwh / self.ev_spec["battery_kwh"])
        self.total_cost += cost

    def session_reward(self, delta_time_s, delta_cost):
        reward = -(delta_time_s + delta_cost)  # penalize time & cost
        if self.normalize_rewards:
            reward /= self.reward_norm_factor
        return reward
