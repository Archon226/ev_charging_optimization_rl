import gymnasium as gym
import numpy as np
from simulator.candidates import find_charger_candidates
from simulator.trip_simulator import TripSimulator

class EVChargingEnv(gym.Env):
    def __init__(self, sim, stations, connectors, pricing, ev_spec,
                 normalize_obs=True, normalize_rewards=True, reward_norm_factor=100.0,
                 fast_mode=False):
        super().__init__()
        self.sim = sim
        self.stations = stations
        self.connectors = connectors
        self.pricing = pricing
        self.ev_spec = ev_spec
        self.simulator = TripSimulator(sim, pricing)
        self.normalize_obs = normalize_obs
        self.normalize_rewards = normalize_rewards
        self.reward_norm_factor = reward_norm_factor
        self.fast_mode = fast_mode

        self.action_space = gym.spaces.Discrete(10)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.trip = options["trip"]
        self.state = self.simulator.reset(
            self.trip.origin,
            self.trip.dest,
            self.ev_spec,
            self.trip.start_soc
        )
        self.candidates = find_charger_candidates(
            self.sim,
            self.trip.origin,
            self.trip.dest,
            self.stations,
            self.connectors,
            self.ev_spec,
            self.trip.start_soc,
            top_k=10,
            max_detour_km=10.0
        )
        return self._make_obs(), {}

    def step(self, action):
        station = self.candidates[action]
        state, reward, done, info = self.simulator.step(station)
        if self.normalize_rewards:
            reward /= self.reward_norm_factor
        self.state = state
        return self._make_obs(), reward, done, False, info

    def _make_obs(self):
        soc = self.state["soc"]
        lat, lon = self.state["position"]
        dest_lat, dest_lon = self.state["dest"]
        dist_to_dest = np.linalg.norm([lat - dest_lat, lon - dest_lon])
        obs = np.array([soc, lat, lon, dist_to_dest, 0], dtype=np.float32)
        if self.normalize_obs:
            obs = (obs - obs.min()) / (obs.max() - obs.min() + 1e-6)
        return obs

    def render(self):
        print(f"SOC: {self.state['soc']}, Pos: {self.state['position']}")
