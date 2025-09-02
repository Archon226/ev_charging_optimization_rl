import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulator.trip_simulator import TripSimulator
from simulator.candidates import find_charger_candidates

class EVChargingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, sim, stations, connectors, pricing, ev_spec):
        super().__init__()
        self.sim = sim
        self.stations = stations
        self.connectors = connectors
        self.pricing = pricing
        self.ev_spec = ev_spec
        self.simulator = TripSimulator(sim, pricing)

        # Discrete action = candidate index
        self.action_space = spaces.Discrete(10)  # top 10 chargers max
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

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
            self.trip.start_soc
        )
        return self._make_obs(), {}

    def step(self, action):
        station = self.candidates[action]
        state, reward, done, info = self.simulator.step(station)
        self.state = state
        return self._make_obs(), reward, done, False, info

    def _make_obs(self):
        soc = self.state["soc"]
        dist_to_dest = 0.0  # TODO: compute from SUMO
        return np.array([soc, dist_to_dest, 0, 0, 0], dtype=np.float32)

    def render(self):
        print(f"SOC: {self.state['soc']}, Pos: {self.state['position']}")
