import gymnasium as gym
import numpy as np
import random
from typing import List, Optional

from simulator.candidates import find_charger_candidates
from simulator.trip_simulator import TripSimulator
from simulator.candidates import EVSpecLite


class EVChargingEnv(gym.Env):
    """
    Single-agent EV charging environment.

    - Objectives: "time", "cost", "hybrid" (with hybrid_alpha).
    - Samples a trip on reset from `episodes` (or accepts options={"trip": ...}).
    - Gymnasium API: returns (obs, reward, terminated, truncated, info).
    - Fixed Discrete(top_k_default) action space; candidates are padded to this size.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sim,
        stations,
        connectors,
        pricing,
        ev_spec,
        episodes: Optional[List] = None,
        objective: str = "hybrid",
        hybrid_alpha: float = 0.5,
        top_k_default: int = 10,
        max_detour_km_default: float = 10.0,
        normalize_obs: bool = True,
        normalize_rewards: bool = True,
        reward_norm_factor: float = 100.0,
        fast_mode: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        # Backends / data
        self.sim = sim
        self.stations = stations
        self.connectors = connectors
        self.pricing = pricing
        self.ev_spec = ev_spec
        self.simulator = TripSimulator(sim, pricing)

        # Episodes
        self.episodes = episodes or []
        self.rng = np.random.default_rng(seed)

        # Objective & scaling
        assert objective in {"time", "cost", "hybrid"}
        self.objective = objective
        self.hybrid_alpha = float(hybrid_alpha)

        # Candidate settings
        self.top_k_default = int(top_k_default)
        self.max_detour_km_default = float(max_detour_km_default)

        # Normalization
        self.normalize_obs = normalize_obs
        self.normalize_rewards = normalize_rewards
        self.reward_norm_factor = float(reward_norm_factor)

        # Misc
        self.fast_mode = fast_mode

        # Spaces
        self.action_space = gym.spaces.Discrete(self.top_k_default)
        # obs = [soc, lat, lon, dist_to_dest, last_action_ok]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # Runtime state
        self.state = None
        self.trip = None
        self.candidates = []

    # -------------- Gymnasium API --------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # --- pick trip ---
        if options is not None and "trip" in options:
            self.trip = options["trip"]
        else:
            if not self.episodes:
                raise RuntimeError(
                    "EVChargingEnv.reset() needs either options['trip'] or a non-empty `episodes` list."
                )
            self.trip = self.rng.choice(self.episodes)

        # --- start SOC in [0,1] ---
        start_soc = getattr(self.trip, "start_soc", None)
        if start_soc is None:
            start_soc_pct = float(getattr(self.trip, "start_soc_pct", getattr(self.trip, "init_soc_pct", 80.0)))
            start_soc = start_soc_pct / 100.0

        # --- episode-specific knobs ---
        ep_top_k = int(getattr(self.trip, "top_k", getattr(self.trip, "top_k_candidates", self.top_k_default)))
        ep_top_k = max(1, min(ep_top_k, self.top_k_default))
        ep_max_detour = float(getattr(self.trip, "max_detour_km", self.max_detour_km_default))

        # =========================
        # 1) Build per-trip EVSpecLite (BEFORE using it)
        # =========================
        model = getattr(self.trip, "ev_model", None) or ""
        battery_kwh = float(self.ev_spec.battery_kwh(model))
        eff_kwh_per_km = float(self.ev_spec.kwh_per_km(model))

        spec = self.ev_spec.specs.get(model.strip())
        if spec is None:
            spec = next(iter(self.ev_spec.specs.values()))
        max_dc_kw = float(getattr(spec, "max_dc_kw", 0.0) or 0.0)
        max_ac_kw = float(getattr(spec, "max_ac_kw", 0.0) or 0.0)

        # allowed connectors (dataset-driven)
        all_conn = tuple(sorted(self.connectors["connector_type"].dropna().unique()))
        ac_only = tuple([c for c in all_conn if ("DC" not in c and "CCS" not in c and "CHAdeMO" not in c)])
        allowed_connectors = all_conn if max_dc_kw > 0 else ac_only

        ev_lite = EVSpecLite(
            battery_kwh=battery_kwh,
            eff_kwh_per_km=eff_kwh_per_km,
            allowed_connectors=allowed_connectors,
            max_dc_kw=max_dc_kw,
            max_ac_kw=max_ac_kw,
        )
        self._active_ev = ev_lite  # optional: for debugging

        # =========================
        # 2) Reset trip simulator WITH ev_lite
        # =========================
        self.state = self.simulator.reset(
        self.trip.origin,
        self.trip.dest,
        ev_lite,                      # ← per-trip EV spec (we fixed earlier)
        start_soc,
        user_type=getattr(self.trip, "user_type", "Payg"),
        include_subscription=bool(getattr(self.trip, "include_subscription", False)),
        sessions_per_month=int(getattr(self.trip, "sessions_per_month", 0)),
)


        # =========================
        # 3) Prepare connectors_aug and compute candidates
        # =========================
        connectors_aug = self.connectors.copy()
        if "charger_type" not in connectors_aug.columns:
            _ac = {"Type 2 Mennekes (IEC62196)", "3-pin Type G (BS1363)"}
            _dc = {"CCS Type 2 Combo (IEC62196)", "JEVS G105 (CHAdeMO) DC"}

            def _to_bucket(name: str) -> str:
                if not isinstance(name, str):
                    return "AC"
                n = name.strip()
                if n in _dc:
                    return "DC"
                if n in _ac:
                    return "AC"
                if "DC" in n or "CCS" in n or "CHAdeMO" in n:
                    return "DC"
                return "AC"

            connectors_aug["charger_type"] = connectors_aug["connector_type"].map(_to_bucket)

        # normalize power column if needed
        if "max_power_kw" in connectors_aug.columns and "max_power_kW" not in connectors_aug.columns:
            connectors_aug["max_power_kW"] = connectors_aug["max_power_kw"]

        # --- build candidate list ---
        self.candidates = find_charger_candidates(
            self.sim,
            self.trip.origin,
            self.trip.dest,
            self.stations,
            connectors_aug,
            ev_lite,                # pass the same per-trip EV spec
            start_soc,
            top_k=ep_top_k,
            max_detour_km=ep_max_detour
        )
        self._pad_candidates(self.action_space.n)

        obs = self._make_obs(last_action_ok=1.0)
        info = {}
        return obs, info


    def step(self, action: int):
        # Bound action to available candidates (list is padded if shorter)
        action = int(np.clip(action, 0, len(self.candidates) - 1))
        station = self.candidates[action]

        # Step simulator: returns (state, reward, done, info)
        next_state, sim_reward, done, info = self.simulator.step(station)

        # Compute reward by objective using info (fallback to sim_reward if missing)
        reward = self._compute_reward_from_info(info, default=sim_reward)

        if self.normalize_rewards:
            denom = self.reward_norm_factor if self.reward_norm_factor > 0 else 1.0
            reward = float(reward) / denom

        self.state = next_state

        terminated = bool(done)
        truncated = False

        if terminated or truncated:
            info.setdefault("episode_end", True)
            if "total_cost_gbp" in info or "session_cost_gbp" in info:
                info.setdefault("episode_cost", float(info.get("total_cost_gbp", info.get("session_cost_gbp", 0.0))))
            total_detour_s = float(info.get("total_detour_seconds", info.get("detour_seconds", 0.0)))
            total_charge_min = float(info.get("total_charge_minutes", info.get("charge_minutes", 0.0)))
            info.setdefault("episode_detour_s", total_detour_s)
            info.setdefault("episode_minutes", total_charge_min + total_detour_s / 60.0)
            info.setdefault("success", 1.0 if info.get("reached", False) else 0.0)

        obs = self._make_obs(last_action_ok=1.0 if not (terminated or truncated) else 0.0)
        return obs, float(reward), terminated, truncated, info

    # -------------- Helpers --------------

    def _pad_candidates(self, target_len: int):
        """Pad candidate list to fixed length by repeating the last element."""
        if len(self.candidates) == 0:
            self.candidates = [None] * target_len
            return
        if len(self.candidates) >= target_len:
            self.candidates = self.candidates[:target_len]
            return
        last = self.candidates[-1]
        self.candidates = self.candidates + [last] * (target_len - len(self.candidates))

    def _compute_reward_from_info(self, info: dict, default: float) -> float:
        """
        - time objective: negative total time (detour_seconds + charge_minutes*60)
        - cost objective: negative GBP cost
        - hybrid: weighted sum (alpha*time_norm + (1-alpha)*cost_norm), negative
        """
        # Cost
        cost = None
        if "total_cost_gbp" in info:
            cost = float(info["total_cost_gbp"])
        elif "session_cost_gbp" in info:
            cost = float(info["session_cost_gbp"])

        # Time (seconds)
        detour_s = float(info.get("total_detour_seconds", info.get("detour_seconds", 0.0)))
        charge_min = float(info.get("total_charge_minutes", info.get("charge_minutes", 0.0)))
        total_time_s = detour_s + 60.0 * charge_min

        # If no metrics, fall back
        if cost is None and (detour_s == 0.0 and charge_min == 0.0):
            return float(default)

        if self.objective == "time":
            return -float(total_time_s)
        elif self.objective == "cost":
            return -float(cost if cost is not None else 0.0)
        else:
            # Simple normalization constants
            time_term = (total_time_s / 60.0) / 30.0   # minutes scaled by ~30
            cost_term = (cost if cost is not None else 0.0) / 10.0  # £ scaled by ~10
            return -(self.hybrid_alpha * time_term + (1.0 - self.hybrid_alpha) * cost_term)

    def _make_obs(self, last_action_ok: float = 1.0):
        """
        Observation: [soc, lat, lon, dist_to_dest, last_action_ok]
        """
        soc = float(self.state["soc"])
        lat, lon = self.state["position"]
        dest_lat, dest_lon = self.state["dest"]

        # naive euclidean in degrees as a proxy; SUMO handles actual routing
        dist_to_dest = float(np.linalg.norm([lat - dest_lat, lon - dest_lon]))

        obs = np.array([soc, lat, lon, dist_to_dest, float(last_action_ok)], dtype=np.float32)

        if self.normalize_obs:
            soc_n = soc
            lat_n = (lat - 51.3) / (51.7 - 51.3)
            lon_n = (lon + 0.6) / (0.4)  # maps ~[-0.6, -0.2] to [0,1]
            dist_n = min(1.0, dist_to_dest / 0.1)  # rough scale
            ok_n = float(last_action_ok)
            obs = np.array([soc_n, lat_n, lon_n, dist_n, ok_n], dtype=np.float32)
            obs = np.clip(obs, 0.0, 1.0)

        return obs

    def render(self):
        print(f"SOC: {self.state.get('soc', None)}, Pos: {self.state.get('position', None)}")
