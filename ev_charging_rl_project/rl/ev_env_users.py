# rl/ev_env_users.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from utils.charging_curves import build_ev_specs, build_power_curve
from simulator.candidates import find_charger_candidates, EVSpecLite
from rl.candidate_eval import evaluate_candidates

class EVUsersEnv(gym.Env):
    def __init__(self, sim, stations_df, catalog, ev_meta, curves, plans):
        super().__init__()
        self.sim = sim
        self.stations = stations_df
        self.catalog = catalog
        self.ev_meta = ev_meta
        self.curves = curves
        self.plans = plans
        self.ev_specs = build_ev_specs(ev_meta)

        self.K = 5
        self.observation_space = spaces.Box(
            low=np.array([0,0,0] + [0,0,0,0]*self.K, dtype=np.float32),
            high=np.array([100,200,24*60] + [50,400,200,5]*self.K, dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.K + 1)  # 0=continue, 1..K=choose candidate

    def reset(self, seed=None, options=None):
        self.idx = np.random.randint(0, len(self.plans)) if not options or "idx" not in options else options["idx"]
        self.plan = self.plans[self.idx]
        self._build_context()
        obs = self._obs_from_candidates()
        return obs, {}

    def _build_context(self):
        self.spec = self.ev_specs[self.plan.model]
        self.ev_curve = build_power_curve(self.curves, self.plan.model)
        eff = float(self.ev_meta.loc[self.ev_meta["model"]==self.plan.model].iloc[0]["avg_consumption_Wh_per_km"])/1000.0
        batt = float(getattr(self.spec,"usable_battery_kWh", getattr(self.spec,"battery_kWh", 0.0)))
        self.ev_lite = EVSpecLite(
            battery_kwh=batt, eff_kwh_per_km=eff, allowed_connectors=("CCS2","Type2"),
            max_dc_kw=getattr(self.spec,"dc_max_power_kW",120), max_ac_kw=getattr(self.spec,"ac_max_power_kW",11)
        )
        self.eff_kwh_per_km = eff
        self.batt_kwh = batt

        # candidates
        self.cands = find_charger_candidates(
            self.sim, (self.plan.start_lat,self.plan.start_lon),(self.plan.dest_lat,self.plan.dest_lon),
            self.stations, None, self.ev_lite, current_soc=self.plan.init_soc_pct/100.0,
            top_k=self.plan.top_k_candidates, max_detour_km=self.plan.max_detour_km, require_connector_ok=False
        )
        self.evals = evaluate_candidates(self.sim, self.stations, self.cands, self.plan, self.spec,
                                         self.ev_curve, self.eff_kwh_per_km, self.batt_kwh,
                                         self.catalog, self.plan.depart_dt)

    def _obs_from_candidates(self):
        # Obs: [init_soc, est_remaining_km, tod_min] + K * [detour_min, power_kw, charge_min, unit_price]
        tod = self.plan.depart_dt.hour*60 + self.plan.depart_dt.minute
        remain_km = self.sim.route_between((self.plan.start_lat,self.plan.start_lon),(self.plan.dest_lat,self.plan.dest_lon)).dist_km
        base = [float(self.plan.init_soc_pct), float(remain_km), float(tod)]
        feat = []
        for i in range(self.K):
            if i < len(self.evals):
                e = self.evals[i]
                feat += [e.detour_min, e.rated_power_kw, e.charge_min, e.unit_price]
            else:
                feat += [0,0,0,0]
        return np.array(base+feat, dtype=np.float32)

    def step(self, action):
        # 0=continue (penalize time); >0 choose candidate i → reward = -objective score, terminal
        if action == 0 or len(self.evals)==0:
            obs = self._obs_from_candidates()
            reward = -1.0  # time tick
            return obs, reward, False, False, {"skipped": True}

        i = min(action-1, len(self.evals)-1)
        e = self.evals[i]
        # score already matches objective
        reward = -float(e.score)
        # terminal (we treat one decision per user as an episode); you can extend to multi-step later
        obs = self._obs_from_candidates()
        info = {"picked": e.station_id, "ctype": e.charger_type, "kW": e.rated_power_kw,
                "detour_min": e.detour_min, "wait_min": e.wait_min, "charge_min": e.charge_min,
                "kwh": e.delivered_kwh, "unit_price": e.unit_price, "cost": e.total_cost}
        return obs, reward, True, False, info
