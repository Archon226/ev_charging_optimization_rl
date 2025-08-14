import numpy as np
import gymnasium as gym
from gymnasium import spaces
from datetime import timedelta
from simulator.trip_simulator import TripSimulator, TripPlan

class EVChargeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sim: TripSimulator, plan: TripPlan):
        super().__init__()
        self.sim = sim
        self.base_plan = plan
        # Observation: [soc_pct, remaining_km, tod_min] + K candidates [detour_km, power_kw] (K=5)
        self.K = plan.top_k_candidates
        obs_low  = np.array([0.0, 0.0, 0.0] + [0.0, 0.0]*self.K, dtype=np.float32)
        obs_high = np.array([100.0, 200.0, 24*60.0] + [50.0, 400.0]*self.K, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(self.K + 1)  # 0=continue, 1..K=stop at cand i
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Rebuild a fresh simulator state by copying the plan (we’ll piggyback TripSimulator’s internals)
        self.plan = self.base_plan
        # Kick off by generating an initial observation using a one-step “peek”
        obs = self._make_obs(peek_only=True)
        return obs, {}

    def _make_obs(self, peek_only=False):
        # Cheat: construct remaining_km from a short internal route calc
        # We reuse the simulator’s SUMO route stored in run_single_trip logic
        # Here we keep it simple:  pretend remaining_km ~ haversine between plan start/dest on reset
        remaining_km = 5.0  # conservative placeholder; training will still learn
        soc_pct = self.plan.init_soc_pct
        tod_min = self.plan.depart_dt.hour*60 + self.plan.depart_dt.minute
        # Candidate placeholders (detour_km, power_kw): naive bins to begin training quickly
        cands = [0.5, 50,  0.8, 150,  1.2, 7,  2.0, 22,  3.0, 11]  # 5 candidates
        obs = np.array([soc_pct, remaining_km, float(tod_min)] + cands, dtype=np.float32)
        self._cands_cache = [(0.5,50),(0.8,150),(1.2,7),(2.0,22),(3.0,11)]
        return obs

    def step(self, action: int):
        # Minimal dynamics to start training:
        # If action==0: continue → -time penalty; soc drops a bit
        # If action>0: simulate a charge using TripSimulator end-to-end once, return terminal
        if action == 0:
            # drive for a small horizon
            reward = -1.0  # time penalty
            terminated, truncated = False, False
            # Nudge SOC down
            o = self._make_obs()  # keep it simple for first training runs
            o[0] = max(0.0, o[0] - 2.0)  # soc -2%
            return o, reward, terminated, truncated, {}

        # Stop at candidate i → call the real sim once and finish the episode
        res = self.sim.run_single_trip(self.plan)
        # Reward: negative hybrid cost/time (normalize lightly)
        time_pen = (res.total_drive_min + res.total_wait_min + res.total_charge_min) / 60.0
        cost_pen = res.total_cost / 20.0
        reward = - (0.5*time_pen + 0.5*cost_pen)
        obs = self._make_obs()
        terminated, truncated = True, False
        info = {"charges": [c.__dict__ for c in res.charges], "arrived": res.arrived, "total_cost": res.total_cost}
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass
