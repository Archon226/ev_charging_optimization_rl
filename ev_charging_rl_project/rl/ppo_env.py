# rl/ppo_env.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rl.episodes import TripPlan   # NEW: episode injection
from rl.candidate_providers import  SumoRouteProvider, Candidate
from env.sumo_runner import SumoRunner, SumoConfig


# ---------------------------
# Config dataclass
# ---------------------------
@dataclass
class PPOEnvConfig:
    # Observation/action
    obs_top_k: int = 5                    # how many candidate stations to expose
    max_steps: int = 240                  # 240 * 0.5min = 120 min if dt_minutes=0.5, etc.
    dt_minutes: float = 5.0               # decision interval
    # Trip + EV defaults (used ONLY if not provided by utils bundle)
    start_soc_range: Tuple[float, float] = (0.10, 0.30)
    trip_distance_km_range: Tuple[float, float] = (12.0, 25.0)
    default_battery_kwh: float = 60.0
    default_kwh_per_km: float = 0.18
    # Economics
    value_of_time_per_min: float = 0.05   # £/min (time penalty)
    charge_efficiency: float = 0.92       # battery_kWh = grid_kWh * efficiency
    # Objectives
    prefer: str = "hybrid"                # "time" | "cost" | "hybrid"
    success_bonus: float = 50.0
    strand_penalty: float = 200.0
    invalid_action_penalty: float = 2.0
    # Sampling
    rng_seed: Optional[int] = None


class PPOChargingEnv(gym.Env):
    """
    Gymnasium env that **only** consumes pre-indexed utils outputs:
      bundle = load_all_ready(...)
    and (optionally) a PricingCatalog from utils.pricing.

    No CSV reads inside the env. Everything comes from the bundle.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: PPOEnvConfig,
        data_bundle: Dict[str, Any],
        pricing_catalog: Optional[Any] = None,
        rng_seed: Optional[int] = None,
        candidate_provider=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.bundle = data_bundle
        self.pricing_catalog = pricing_catalog

        self.rng = np.random.default_rng(rng_seed if rng_seed is not None else cfg.rng_seed)

        # ---------- Unpack bundle (strictly from utils) ----------
        self.station_caps: Dict[str, Any] = data_bundle["station_capabilities"]
        self.connectors_df = data_bundle["station_connectors_enriched"]
        self.ev_caps: Dict[str, Any] = data_bundle["ev_capabilities"]
        self.pricing_index: Dict[str, Any] = data_bundle["pricing_index"]
        self.ev_curves_index: Dict[str, Any] = data_bundle["ev_curves_index"]
        self.power_model = data_bundle.get("ev_power_model", None)
        self.pricing_catalog = data_bundle.get("pricing_catalog", None)

        # Fast price lookup (company_id, charger_type) -> price_per_kwh
        self.price_lookup = self._build_price_lookup(self.pricing_index)

        # Pre-index station → representative category + cap per category
        self.station_index = self._build_station_index(self.station_caps)
        # default: use SUMO provider
        if candidate_provider is not None:
            self.candidate_provider = candidate_provider
        else:
            sumo_cfg = SumoConfig(net_path="london_inner.net.xml", gui=False)
            self._sumo_runner = SumoRunner(sumo_cfg)
            self._sumo_runner.start()
            self.candidate_provider = SumoRouteProvider(self._sumo_runner, max_detour_min=15.0)

        # ---------- Spaces ----------
        K = int(cfg.obs_top_k)
        # obs = [soc, remaining_km] + K * [dist_km, price_per_kwh, 3 one-hots for type]
        self._feat_per_cand = 5
        low = np.array([0.0, 0.0] + K * [0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1e3] + K * [1e3, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # action 0 = keep driving; 1..K = go charge at candidate i during dt_minutes
        self.action_space = spaces.Discrete(K + 1)

        # ---------- Episode state ----------
        self._reset_state()

    # ---------------------------
    # Core helpers (bundle-only)
    # ---------------------------
    def _build_price_lookup(self, pricing_index: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        out: Dict[Tuple[str, str], float] = {}
        by_type = pricing_index.get("by_type", {})
        for (company_id, charger_type), row in by_type.items():
            price = row.get("price_per_kwh", None)
            if price is None:
                continue
            out[(str(company_id), str(charger_type))] = float(price)
        return out

    def _build_station_index(self, station_caps: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        For each station:
          - categories_available (set)
          - power_by_category (dict)
          - representative_type (Fast/Rapid/Ultra) (for candidate generation)
          - company_id
        """
        idx: Dict[str, Dict[str, Any]] = {}
        for sid, caps in station_caps.items():
            cats = set(caps.categories_available)
            pwr = dict(caps.power_stats_by_category or {})
            rep = self._representative_type(cats)
            idx[sid] = {
                "categories": cats,
                "power_by_category": pwr,
                "rep_type": rep,
                "company_id": str(caps.company_id),
            }
        return idx

    @staticmethod
    def _representative_type(categories: Sequence[str]) -> str:
        # prefer higher categories if present
        order = ["Ultra", "Rapid", "Fast"]
        s = set(categories)
        for ct in order:
            if ct in s:
                return ct
        return "Fast"
    
    def _resolve_ev_from_trip(self, ev_model_str: str) -> Tuple[str, Any]:
        """
        TripPlan gives something like "Model X" / "EQS" / etc.
        Our ev_caps keys look like "Brand|Model|Year".
        Strategy:
        1) exact model-name match against the middle token
        2) fallback: substring match (case-insensitive)
        3) final fallback: random EV
        """
        target = (ev_model_str or "").strip().lower()
        if not target:
            return self._pick_random_ev()

        # Parse keys once
        for ev_id, caps in self.ev_caps.items():
            parts = ev_id.split("|")
            model = parts[1].strip().lower() if len(parts) >= 2 else ev_id.lower()
            if model == target:
                return ev_id, caps

        # substring fallback
        for ev_id, caps in self.ev_caps.items():
            parts = ev_id.split("|")
            model = parts[1].strip().lower() if len(parts) >= 2 else ev_id.lower()
            if target in model or model in target:
                return ev_id, caps

        # last resort
        return self._pick_random_ev()

    # ---------------------------
    # Episode lifecycle
    # ---------------------------
    def _reset_state(self, trip: Optional[TripPlan] = None) -> None:
        self.step_count = 0

        # -------- episode config from TripPlan (if provided) --------
        self._trip: Optional[TripPlan] = trip
        if trip is not None:
            # prefer / objective
            obj = (trip.objective or "hybrid").strip().lower()
            if obj in ("time", "cost", "hybrid"):
                self.cfg.prefer = obj

            # EV selection: map TripPlan.ev_model -> ev_caps key
            self.ev_id, self.ev = self._resolve_ev_from_trip(trip.ev_model)

            # Battery size & efficiency (best effort, using columns the generator put in the CSV)
            # kWh/km (grid) – use if present; else keep cfg default
            self.kwh_per_km = float(trip.kwh_per_km) if trip.kwh_per_km else float(self.cfg.default_kwh_per_km)

            # Battery kWh: try to back-calc from available_kwh_at_start and SoC/reserve if present
            self.battery_kwh = float(self.cfg.default_battery_kwh)
            try:
                start = float(trip.start_soc_pct) / 100.0
                reserve = float(trip.reserve_soc_pct) / 100.0
                avail = float(trip.available_kwh_at_start) if trip.available_kwh_at_start is not None else None
                if avail is not None:
                    denom = max(1e-6, (start - reserve))
                    est_pack = avail / denom
                    if np.isfinite(est_pack) and est_pack > 5.0:  # sanity
                        self.battery_kwh = float(est_pack)
            except Exception:
                pass

            # SoC and remaining km
            self.soc = float(trip.start_soc_pct) / 100.0
            # If trip_km is present, use it; else fall back to previous range
            if trip.trip_km is not None and np.isfinite(trip.trip_km):
                self.remaining_km = float(trip.trip_km)
            else:
                # keep old stochastic fallback
                d0, d1 = self.cfg.trip_distance_km_range
                self.remaining_km = float(self.rng.uniform(d0, d1))

            # # Per-episode knobs (optional)
            # if getattr(trip, "top_k_candidates", None):
            #     self.cfg.obs_top_k = int(trip.top_k_candidates)
            # # note: we’ll use max_detour_km when we wire in real candidate finding

        else:
            # -------- original random init (no TripPlan) --------
            self.ev_id, self.ev = self._pick_random_ev()
            self.battery_kwh = float(self.cfg.default_battery_kwh)
            self.kwh_per_km = float(self.cfg.default_kwh_per_km)
            s0, s1 = self.cfg.start_soc_range
            self.soc = float(self.rng.uniform(s0, s1))
            d0, d1 = self.cfg.trip_distance_km_range
            self.remaining_km = float(self.rng.uniform(d0, d1))

        # Candidate list priming
        self.all_station_ids: List[str] = list(self.station_index.keys())
        self._refresh_candidates()

        self.total_minutes = 0.0
        self.total_cost = 0.0
        self.charge_events = 0
        self._done = False


    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if not options or "trip" not in options or options["trip"] is None:
            raise ValueError(
                "PPOChargingEnv.reset(...) requires options={'trip': TripPlan}. "
                "Pass in a TripPlan from rl.episodes.iter_episodes(...)."
            )

        trip = options["trip"]
        self._reset_state(trip=trip)
        self._trip = trip
        
        # sanity assertions (fail early if user gen is off)
        assert 0.0 <= self.soc <= 1.0, f"start SoC out of range: {self.soc}"
        assert self.remaining_km > 0, f"trip distance invalid: {self.remaining_km}"

        obs = self._build_observation()
        info = {"ev_id": self.ev_id, "episode_source": "trip"}
        return obs, info



    # ---------------------------
    # Transition dynamics
    # ---------------------------
    def step(self, action: int):
        if self._done:
            raise RuntimeError("Call reset() before step() after episode terminated.")
        self.step_count += 1

        dt_min = float(self.cfg.dt_minutes)
        reward = 0.0
        info: Dict[str, Any] = {}

        if action == 0:
            # DRIVE
            reward += self._apply_drive(dt_min, info)
        else:
            idx = action - 1
            if idx >= len(self.candidates):
                reward -= self.cfg.invalid_action_penalty
            else:
                station_id = self.candidates[idx].get("station_id", "")
                if not station_id:
                    reward -= self.cfg.invalid_action_penalty
                else:
                    reward += self._apply_charge(station_id, dt_min, info)


        # Termination conditions
        terminated = False
        if self.remaining_km <= 0.0:
            terminated = True
            self._done = True
            # success bonus
            reward += self.cfg.success_bonus
        elif self.soc <= 0.0:
            terminated = True
            self._done = True
            reward -= self.cfg.strand_penalty

        truncated = self.step_count >= self.cfg.max_steps
        if truncated:
            self._done = True

        obs = self._build_observation()
        return obs, float(reward), bool(terminated), bool(truncated), info
    
    def close(self):
        try:
            if hasattr(self, "_sumo_runner") and self._sumo_runner is not None:
                self._sumo_runner.stop()
        except Exception:
            pass
        super().close()

    # ---------------------------
    # Driving & Charging models
    # ---------------------------
    def _apply_drive(self, dt_min: float, info: Dict[str, Any]) -> float:
        # Simple kinematics; later hook to SUMO
        # Assume 25 km/h average inner-London → distance in this interval:
        avg_speed_kmh = 25.0
        dist = avg_speed_kmh * (dt_min / 60.0)
        self.remaining_km = max(0.0, self.remaining_km - dist)

        # Energy use
        grid_kwh = self.kwh_per_km * dist
        batt_kwh = grid_kwh * self.cfg.charge_efficiency  # keep convention consistent

        # SoC drop
        self.soc = max(0.0, self.soc - batt_kwh / max(self.battery_kwh, 1e-6))

        # Time cost only (driving energy is not billed here)
        self.total_minutes += dt_min
        time_cost = dt_min * self.cfg.value_of_time_per_min

        return -time_cost if self.cfg.prefer in ("time", "hybrid") else 0.0

    def _apply_charge(self, station_id: str, dt_min: float, info: Dict[str, Any]) -> float:
        st = self.station_index[station_id]
        company_id = st["company_id"]

        # Pick best available category that the EV supports
        ev_cats = set(self.ev.categories_supported)
        cats = list(st["categories"] & ev_cats)
        if not cats:
            # no compatible connector → penalize
            return -self.cfg.invalid_action_penalty

        # Prefer higher class
        cats.sort(key=lambda c: ["Fast", "Rapid", "Ultra"].index(c))
        cats = cats[::-1]
        chosen = cats[0]

        station_cap_kw = float(st["power_by_category"].get(chosen, 0.0))
        is_dc = True if chosen in ("Rapid", "Ultra") else False

        # Prefer curve-based power if model is available
        eff_kw = None
        if self.power_model is not None and hasattr(self._trip, "ev_model") and self._trip.ev_model:
            try:
                # EVPowerModel.power_at_soc expects SOC in percent (0..100)
                eff_kw = float(self.power_model.power_at_soc(
                    model=str(self._trip.ev_model),
                    soc=float(self.soc * 100.0),
                    station_cap_kw=station_cap_kw,
                    is_dc=is_dc
                ))
            except Exception:
                eff_kw = None

        # Fallback to caps if curves unavailable
        if eff_kw is None or not np.isfinite(eff_kw):
            ev_side_cap_kw = float((self.ev.dc_max_power_kW if is_dc else self.ev.ac_max_power_kW) or station_cap_kw)
            eff_kw = max(0.0, min(station_cap_kw, ev_side_cap_kw))

        # Energy delivered (battery) in this interval
        batt_kwh = eff_kw * (dt_min / 60.0)
        # If we consider efficiency<1, grid_kwh > batt_kwh; pricing typically bills grid kWh
        grid_kwh = batt_kwh / max(self.cfg.charge_efficiency, 1e-9)

        # Update SoC (cap at 100%)
        new_soc = min(1.0, self.soc + batt_kwh / max(self.battery_kwh, 1e-6))
        soc_gain = new_soc - self.soc
        self.soc = new_soc

        # --- Billing (catalog first, fallback to per-kWh) ---
        unit_price = None
        energy_cost = 0.0

        if self.pricing_catalog is not None and self._trip is not None:
            try:
                energy_cost = float(self.pricing_catalog.compute_price(
                    company_id=int(company_id),
                    charger_type=str(chosen),
                    user_type=str(getattr(self._trip, "user_type", "Payg")),
                    start_dt=getattr(self._trip, "depart_datetime", None),
                    kwh=float(grid_kwh),
                    session_minutes=float(dt_min),
                    idle_minutes=0.0,
                    include_subscription=bool(getattr(self._trip, "include_subscription", 0)),
                    sessions_per_month=int(getattr(self._trip, "sessions_per_month", 20)),
                ))
            except Exception:
                # fallback to simple per-kWh
                unit_price = self._price_for(company_id, chosen)
                energy_cost = float(unit_price) * float(grid_kwh)
        else:
            unit_price = self._price_for(company_id, chosen)
            energy_cost = float(unit_price) * float(grid_kwh)

        # Tally episode totals
        self.total_minutes += dt_min
        self.total_cost += energy_cost

        # Reward = -(time + (cost if hybrid/cost))
        time_cost = dt_min * self.cfg.value_of_time_per_min if self.cfg.prefer in ("time", "hybrid") else 0.0
        cost_cost = energy_cost if self.cfg.prefer in ("cost", "hybrid") else 0.0

        # Small shaping reward if charging increases reach and we haven't arrived
        reach_km_gain = (soc_gain * self.battery_kwh) / max(self.kwh_per_km, 1e-9)
        shaping = 0.05 * reach_km_gain if self.remaining_km > 0 else 0.0

        info.update({
            "station_id": station_id,
            "category": chosen,
            "eff_kw": eff_kw,
            "grid_kwh": grid_kwh,
            "batt_kwh": batt_kwh,
            "price_per_kwh": unit_price,   # may be None if catalog used
            "energy_cost": energy_cost,
        })
        
        self.charge_events += 1   # <--- ADD THIS
        
        return -(time_cost + cost_cost) + shaping


    def _price_for(self, company_id: Optional[str], charger_type: str) -> float:
        if company_id is not None:
            key = (str(company_id), str(charger_type))
            if key in self.price_lookup:
                return float(self.price_lookup[key])
        # safe defaults if no tariff present
        return {"Fast": 0.30, "Rapid": 0.45, "Ultra": 0.65}.get(charger_type, 0.45)

    # ---------------------------
    # Candidates & observation
    # ---------------------------
    def _refresh_candidates(self) -> None:
        K = int(self.cfg.obs_top_k)
        if hasattr(self, "_trip") and self._trip is not None:
            # Use provider
            cand_list = self.candidate_provider.generate(
                trip=self._trip,
                station_index=self.station_index,
                connectors_df=self.connectors_df,
                price_lookup=self.price_lookup,
                top_k=K,
            )
            # Convert to the env’s internal dict format used by _build_observation()
            self.candidates = []
            for c in cand_list:
                self.candidates.append({
                    "station_id": c.station_id,
                    "rep_type": c.rep_type,
                    "price": float(c.price_per_kwh),
                    # Distance signal for the obs: use detour_km + a small fraction of remaining_km
                    "dist_km": float(c.detour_km + 0.1 * max(self.remaining_km, 0.0)),
                    "eta_min": float(c.approx_eta_min),
                })
            # pad if fewer than K below (in _build_observation)
        else:
            # Fallback (shouldn't happen once TripPlan is mandatory)
            self.candidates = []

    def _build_observation(self) -> np.ndarray:
        # refresh candidate set every timestep to avoid stale options

        feats: List[float] = [float(self.soc), float(self.remaining_km)]
        for c in self.candidates:
            onehot = self._type_onehot(c["rep_type"])
            feats.extend([c["dist_km"], c["price"], *onehot])
        # pad if fewer than K
        K = int(self.cfg.obs_top_k)
        # pad features only (do NOT append fake candidates)
        pad = K - len(self.candidates)
        for _ in range(max(0, pad)):
            feats.extend([1e3, 1.0, 0.0, 0.0, 0.0])
        return np.asarray(feats, dtype=np.float32)



    @staticmethod
    def _type_onehot(charger_type: str) -> Tuple[float, float, float]:
        return (
            1.0 if charger_type == "Fast" else 0.0,
            1.0 if charger_type == "Rapid" else 0.0,
            1.0 if charger_type == "Ultra" else 0.0,
        )

    # ---------------------------
    # EV selection
    # ---------------------------
    def _pick_random_ev(self) -> Tuple[str, Any]:
        ev_ids = list(self.ev_caps.keys())
        ev_id = self.rng.choice(ev_ids)
        return ev_id, self.ev_caps[ev_id]

    
    def close(self):
        try:
            if hasattr(self, "_sumo_runner") and (self._sumo_runner is not None):
                self._sumo_runner.stop()
        except Exception:
            pass
        super().close()
