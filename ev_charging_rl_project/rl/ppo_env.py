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
    # Charging UX
    charge_session_overhead_min: float = 3.0  # fixed per-plug overhead to discourage many tiny charges
    
    # Traffic modeling (lightweight scalar congestion)
    traffic_mode: str = "none"          # "none" | "light"
    traffic_peak_factor_am: float = 1.6 # ~60% slower during AM peak
    traffic_peak_factor_pm: float = 1.5 # ~50% slower during PM peak
    traffic_offpeak_factor: float = 1.0 # baseline

    # Objectives
    prefer: str = "hybrid"                # "time" | "cost" | "hybrid"
    respect_trip_objective: bool = False  # NEW: if True, override prefer from TripPlan.objective
    success_bonus: float = 50.0
    strand_penalty: float = 200.0
    invalid_action_penalty: float = 2.0
    # Sampling
    rng_seed: Optional[int] = None
    
    # --- SUMO driving integration ---
    use_sumo_drive: bool = True          # turn ON SUMO-backed driving
    sumo_net_path: str = "london_inner.net.xml"
    sumo_gui: bool = False
    sumo_step_length_s: float = 1.0
    sumo_mode: str = "route_time"        # "route_time" | "microsim"
    sumo_vehicle_type: str = "passenger" # vType must exist in SUMO net

    # === Phase 3: policy constraints & penalties ===
    disallow_repeat_station: bool = True
    max_charges_per_trip: int = 2
    min_charge_gap_min: float = 12.0      # 10–15 typical
    penalty_repeat: float = -5.0          # negative values
    penalty_overlimit: float = -20.0
    penalty_cooldown: float = -5.0
    terminate_on_overlimit: bool = True
    # Penalty when a chosen station is unroutable in SUMO (episode continues)
    penalty_unreachable: float = -3.0

    # === Phase 5: reward shaping ===
    enable_shaping: bool = True
    # Use gamma=1.0 to keep shaping neutral on non-progress steps (esp. during charge).
    # If you want potential-based invariance w.r.t. PPO's gamma, set this to the same value (e.g., 0.997).
    shaping_gamma: float = 1.0
    # Potential φ(s) = - value_of_time_per_min * ETA_left_min where ETA_left_min = remaining_km / v_ref_kmh * 60
    enable_potential_time: bool = True
    potential_vref_kmh: float = 25.0   # reference cruise speed for ETA estimate

    # Anti-dither (tiny negatives to discourage no-progress drive or micro-charges)
    idle_penalty_per_step: float = 0.0       # e.g., 0.05 ⇒ -0.05 when drive progress < epsilon
    idle_progress_epsilon_km: float = 0.15
    micro_charge_penalty: float = 0.0        # e.g., 0.5 ⇒ -0.5 when the charge slice is tiny
    micro_charge_min_kwh: float = 1.0
    micro_charge_min_minutes: float = 6.0

class PPOChargingEnv(gym.Env):
    """
    Gymnasium env that **only** consumes pre-indexed utils outputs:
      bundle = load_all_ready(...)
    and (optionally) a PricingCatalog from utils.pricing.

    No CSV reads inside the env. Everything comes from the bundle.
    """

    metadata = {"render_modes": []}
    # Print the time-mode banner only once per process
    _time_mode_banner_printed = False


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
        
        # SUMO runner (only if enabled)
        self._sumo_runner = None
        if self.cfg.use_sumo_drive:
            sumo_cfg = SumoConfig(
                net_path=self.cfg.sumo_net_path,
                sumo_bin="sumo-gui" if self.cfg.sumo_gui else "sumo",
                gui=self.cfg.sumo_gui,
                step_length=self.cfg.sumo_step_length_s,
            )
            self._sumo_runner = SumoRunner(sumo_cfg)

        # fields for active SUMO segment
        self._seg_eta_s = 0.0       # travel time of active segment
        self._seg_left_s = 0.0      # time left
        self._seg_len_m = 0.0       # <--- cached length of current segment (meters)
        self._seg_from_edge = None
        self._seg_to_edge = None
        self._veh_id = None         # vehicle id (microsim mode)


        # default: provider; only start SUMO if enabled
        if candidate_provider is not None:
            self.candidate_provider = candidate_provider
        else:
            if self.cfg.use_sumo_drive:
                if self._sumo_runner is None:
                    sumo_cfg = SumoConfig(
                        net_path=self.cfg.sumo_net_path,
                        sumo_bin="sumo-gui" if self.cfg.sumo_gui else "sumo",
                        gui=self.cfg.sumo_gui,
                        step_length=self.cfg.sumo_step_length_s,
                    )
                    self._sumo_runner = SumoRunner(sumo_cfg)
                self._sumo_runner.start()
                self.candidate_provider = SumoRouteProvider(self._sumo_runner, max_detour_min=15.0)
            else:
                # Lightweight no-SUMO provider for smoke tests; returns no candidates (env pads obs)
                class _NoopProvider:
                    def generate(self, **kwargs):
                        return []
                self.candidate_provider = _NoopProvider()

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
    
    def _traffic_factor(self, now_dt):
        """Return a time-of-day congestion factor for 'light' traffic mode."""
        if self.cfg.traffic_mode != "light" or now_dt is None:
            return 1.0
        wd = now_dt.weekday()  # 0=Mon
        h = now_dt.hour + now_dt.minute / 60.0
        is_weekday = wd < 5
        if is_weekday and 7.0 <= h <= 10.0:   # AM peak
            return float(self.cfg.traffic_peak_factor_am)
        if is_weekday and 16.0 <= h <= 19.0:  # PM peak
            return float(self.cfg.traffic_peak_factor_pm)
        return float(self.cfg.traffic_offpeak_factor)

    # ---------------------------
    # Episode lifecycle
    # ---------------------------
    def _reset_state(self, trip: Optional[TripPlan] = None) -> None:
        """
        Initialise all per-episode state.
        NOTE: Objective handling:
        - If cfg.respect_trip_objective == True and TripPlan.objective is provided,
            self.cfg.prefer is overridden per episode (time|cost|hybrid).
        - Otherwise, we LEAVE self.cfg.prefer as set by the trainer.
        """
        self.step_count = 0

        # -------- episode config from TripPlan (if provided) --------
        self._trip: Optional[TripPlan] = trip
        if trip is not None:
            # --- Prefer / objective (now gated by config flag) ---
            if getattr(self.cfg, "respect_trip_objective", False):
                obj = (trip.objective or "hybrid").strip().lower()
                if obj in ("time", "cost", "hybrid"):
                    self.cfg.prefer = obj
            # else: leave self.cfg.prefer as set by the trainer
            # --- Time-optimisation stabilisers (episode-scoped effective params) ---
            # Defaults mirror the config, but we soften time-only penalties and boost the
            # positive signal (success + shaping) when prefer == "time".
            self._vot_eff = float(self.cfg.value_of_time_per_min)          # effective value-of-time (penalty scale)
            self._success_bonus_eff = float(self.cfg.success_bonus)        # effective success bonus
            self._phi_scale = 1.0                                          # shaping multiplier
            self._charge_overhead_eff = float(self.cfg.charge_session_overhead_min)  # effective per-session overhead (min)

            if self.cfg.prefer == "time":
                # 1) shrink per-minute penalty so negatives don't swamp the value function
                self._vot_eff *= 0.35
                # 2) make finishing decisively rewarding vs. accumulating time costs
                self._success_bonus_eff = max(self._success_bonus_eff, 500.0)
                # 3) amplify progress-based shaping so PPO "feels" forward motion
                self._phi_scale = 3.0
                # 4) reduce fixed charge-overhead to not over-punish necessary pit-stops
                self._charge_overhead_eff *= 0.5
                if not PPOChargingEnv._time_mode_banner_printed:
                    print("[env] Time-mode stabilisers ACTIVE "
                        f"(vot_eff={self._vot_eff}, success_bonus_eff={self._success_bonus_eff}, "
                        f"phi_scale={self._phi_scale}, charge_overhead_eff={self._charge_overhead_eff})")
                    PPOChargingEnv._time_mode_banner_printed = True

            # EV selection: map TripPlan.ev_model -> ev_caps key
            self.ev_id, self.ev = self._resolve_ev_from_trip(trip.ev_model)

            # Energy model: kWh/km (grid) if provided; else cfg default
            try:
                self.kwh_per_km = float(trip.kwh_per_km) if trip.kwh_per_km is not None else float(self.cfg.default_kwh_per_km)
            except Exception:
                self.kwh_per_km = float(self.cfg.default_kwh_per_km)

            # Battery capacity estimate:
            # Try to back-calc pack size from available_kwh_at_start and (start - reserve) if present
            self.battery_kwh = float(self.cfg.default_battery_kwh)
            try:
                start = float(trip.start_soc_pct) / 100.0
                reserve = float(trip.reserve_soc_pct) / 100.0
                avail = float(trip.available_kwh_at_start) if trip.available_kwh_at_start is not None else None
                if avail is not None:
                    denom = max(1e-6, (start - reserve))
                    est_pack = avail / denom
                    if np.isfinite(est_pack) and est_pack > 5.0:
                        self.battery_kwh = float(est_pack)
            except Exception:
                # keep default_battery_kwh on any parsing issue
                pass

            # SoC and remaining distance
            try:
                self.soc = float(trip.start_soc_pct) / 100.0
            except Exception:
                s0, s1 = self.cfg.start_soc_range
                self.soc = float(self.rng.uniform(s0, s1))

            if getattr(trip, "trip_km", None) is not None and np.isfinite(trip.trip_km):
                self.remaining_km = float(trip.trip_km)
            else:
                d0, d1 = self.cfg.trip_distance_km_range
                self.remaining_km = float(self.rng.uniform(d0, d1))

        else:
            # -------- random init (no TripPlan; keeps smoke-tests working) --------
            self.ev_id, self.ev = self._pick_random_ev()
            self.battery_kwh = float(self.cfg.default_battery_kwh)
            self.kwh_per_km = float(self.cfg.default_kwh_per_km)
            s0, s1 = self.cfg.start_soc_range
            self.soc = float(self.rng.uniform(s0, s1))
            d0, d1 = self.cfg.trip_distance_km_range
            self.remaining_km = float(self.rng.uniform(d0, d1))

        # ---------- Candidate priming ----------
        self.all_station_ids: List[str] = list(self.station_index.keys())
        self._refresh_candidates()

        # ---------- Episode running tallies ----------
        self.total_minutes = 0.0
        self.total_cost = 0.0
        self.charge_events = 0
        self._done = False

        # --- Telemetry (Phase 1.5) ---
        self.drive_steps = 0
        self.charge_steps = 0
        self._last_step_type = None       # "drive" | "charge"
        self._arrival_via = None          # "drive" | "charge"

        # --- Constraints state (Phase 3) ---
        self.visited_station_ids: set = set()
        self.last_charge_end_min: Optional[float] = None
        self.violations_repeat: int = 0
        self.violations_overlimit: int = 0
        self.violations_cooldown: int = 0
        self.unreachable_station_ids: set = set()  # stations found unroutable this episode

        # --- Episode clock for traffic shaping ---
        self._clock_dt = getattr(trip, "depart_datetime", None) if trip is not None else None

        # Fallback synthetic clock if traffic mode is active but no depart time provided
        if getattr(self.cfg, "traffic_mode", "none") != "none" and self._clock_dt is None:
            from datetime import datetime, timedelta
            weekday = int(self.rng.integers(0, 7))     # 0..6 (Mon..Sun)
            hour    = int(self.rng.integers(0, 24))    # 0..23
            minute  = int(self.rng.integers(0, 60))    # 0..59
            base = datetime(2024, 1, 1, 0, 0, 0)       # known Monday anchor
            self._clock_dt = base + timedelta(days=weekday, hours=hour, minutes=minute)
            # keep a trace for KPI/info in reset()
            self._depart_dt_fallback_iso = self._clock_dt.isoformat()

    # ---------------------------
    # Phase 3 helpers
    # ---------------------------
    def _cooldown_active(self, now_min: float) -> bool:
        if self.last_charge_end_min is None:
            return False
        return (now_min - float(self.last_charge_end_min)) < float(self.cfg.min_charge_gap_min)

    @staticmethod
    def _apply_penalty(base_reward: float, penalty: float) -> float:
        # penalties are negative; just add them
        return float(base_reward) + float(penalty)


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
        # --- SUMO integration ---
        if self.cfg.use_sumo_drive and self._sumo_runner is not None:
            self._sumo_runner.start()

            # pick a start edge (from trip explicit edge, else origin lat/lon, else nearest, else any)
            start_edge = None
            try:
                # 1) explicit precomputed edge id on the trip (if generator put it there)
                if getattr(self._trip, "start_edge_id", None):
                    start_edge = self._trip.start_edge_id

                # 2) use trip.origin (lat, lon)
                if start_edge is None:
                    if hasattr(self._trip, "origin") and self._trip.origin and len(self._trip.origin) >= 2:
                        o_lat, o_lon = float(self._trip.origin[0]), float(self._trip.origin[1])
                        start_edge = self._sumo_runner.snap_to_edge(o_lat, o_lon)

                # 3) legacy fields (start_lat/lon)
                if start_edge is None and getattr(self._trip, "start_lat", None) is not None and getattr(self._trip, "start_lon", None) is not None:
                    start_edge = self._sumo_runner.nearest_edge_by_latlon(self._trip.start_lat, self._trip.start_lon)

            except Exception:
                start_edge = None

            # 4) last resort: any edge from the network (keeps training alive)
            if start_edge is None:
                try:
                    start_edge = self._sumo_runner.any_edge_id()
                except Exception:
                    start_edge = None  # will be caught on segment begin

            self._seg_from_edge = start_edge

            # reset segment timers
            self._seg_eta_s = 0.0
            self._seg_left_s = 0.0
            self._seg_to_edge = None
            self._seg_len_m = 0.0   # <— add this

            # Determine destination edge for OD routing
            self._dest_edge = None
            try:
                # Prefer a precomputed edge id if your generator provided it
                if getattr(self._trip, "end_edge_id", None):
                    self._dest_edge = self._trip.end_edge_id

                # Else try trip.destination (lat, lon)
                if self._dest_edge is None:
                    if hasattr(self._trip, "destination") and self._trip.destination and len(self._trip.destination) >= 2:
                        d_lat, d_lon = float(self._trip.destination[0]), float(self._trip.destination[1])
                        self._dest_edge = self._sumo_runner.snap_to_edge(d_lat, d_lon)

                # Legacy fields (end_lat/lon)
                if self._dest_edge is None and getattr(self._trip, "end_lat", None) is not None and getattr(self._trip, "end_lon", None) is not None:
                    self._dest_edge = self._sumo_runner.nearest_edge_by_latlon(self._trip.end_lat, self._trip.end_lon)
            except Exception:
                self._dest_edge = None

            # microsim: add vehicle
            if self.cfg.sumo_mode == "microsim":
                self._veh_id = f"veh_{np.random.randint(1e9)}"
                try:
                    self._sumo_runner.add_vehicle_on_edge(
                        veh_id=self._veh_id,
                        edge_id=self._seg_from_edge or self._sumo_runner.any_edge_id(),
                        type_id=self.cfg.sumo_vehicle_type,
                        depart_s=0.0,
                    )
                except Exception:
                    self._veh_id = None

        self._trip = trip
        
        # sanity assertions (fail early if user gen is off)
        assert 0.0 <= self.soc <= 1.0, f"start SoC out of range: {self.soc}"
        assert self.remaining_km > 0, f"trip distance invalid: {self.remaining_km}"

        obs = self._build_observation()
        info = {"ev_id": self.ev_id, "episode_source": "trip"}
        # If we synthesized a depart time, expose it for logging/inspection
        if hasattr(self, "_depart_dt_fallback_iso"):
            info["depart_datetime_fallback"] = self._depart_dt_fallback_iso
            # optional: clean up so it only appears once
            del self._depart_dt_fallback_iso
        return obs, info



    # ---------------------------
    # Transition dynamics
    # ---------------------------
    def step(self, action: int):
        if self._done:
            raise RuntimeError("Call reset() before step() after episode terminated.")
        self.step_count += 1
        # Phase 5: capture distance before the transition
        prev_remaining_km = float(self.remaining_km)
        
        # ---- Lazy-init Phase 3 state (safe if already set in _reset_state) ----
        if not hasattr(self, "visited_station_ids") or self.visited_station_ids is None:
            self.visited_station_ids = set()
        if not hasattr(self, "last_charge_end_min"):
            self.last_charge_end_min = None
        if not hasattr(self, "violations_repeat"):
            self.violations_repeat = 0
        if not hasattr(self, "violations_overlimit"):
            self.violations_overlimit = 0
        if not hasattr(self, "violations_cooldown"):
            self.violations_cooldown = 0

        # Convenience: current episode time in minutes
        now_min = float(self.total_minutes)

        dt_min = float(self.cfg.dt_minutes)
        reward = 0.0
        info: Dict[str, Any] = {}

        # --- identify step type, apply transition, and count it ---
        # Default: not arrived on this step; we will set True after drive if we cross 0
        self._arrived_this_step = False

        terminated = False
        truncated = False
        termination_reason = None

        if action == 0:
            # =======================
            # DRIVE
            # =======================
            info["step_type"] = "drive"  # PHASE 1.5
            step_type = "drive"
            reward += self._apply_drive(dt_min, info)
            self.drive_steps += 1  # PHASE 1.5

            # Arrival can only be credited on a DRIVE step
            if self.remaining_km <= 0.0:
                self._arrived_this_step = True  # PHASE 1.5

        else:
            # =======================
            # CHARGE
            # =======================
            info["step_type"] = "charge"  # PHASE 1.5
            step_type = "charge"
            idx = action - 1

            if idx >= len(self.candidates):
                reward -= self.cfg.invalid_action_penalty
            else:
                cand = self.candidates[idx]
                station_id = cand.get("station_id", "")

                begin_ok = True
                if self.cfg.use_sumo_drive and self._sumo_runner is not None and station_id:
                    try:
                        self._begin_sumo_segment(to_station_id=station_id)
                    except RuntimeError as e:
                        msg = str(e)
                        # Soft-handle only the routing gap; re-raise other errors
                        if "cannot route" in msg or "No SUMO edge" in msg:
                            # Remember & penalize once, then mask in future candidate refreshes
                            if not hasattr(self, "unreachable_station_ids"):
                                self.unreachable_station_ids = set()
                            self.unreachable_station_ids.add(station_id)
                            info["unreachable_station_id"] = station_id
                            # apply small penalty but do NOT also apply invalid_action_penalty below
                            reward += float(getattr(self.cfg, "penalty_unreachable", -3.0))
                            begin_ok = False
                        else:
                            raise


                if (not station_id) or (not begin_ok):
                    # if unroutable, we already added penalty_unreachable above; avoid double-penalizing
                    if not begin_ok:
                        pass
                    else:
                        reward -= self.cfg.invalid_action_penalty
                else:
                    # Phase 3 constraints..
                    # ---------- Phase 3: constraints BEFORE executing the charge ----------
                    # A) repeat station
                    if getattr(self.cfg, "disallow_repeat_station", True) and station_id in self.visited_station_ids:
                        # apply repeat penalty (do not block; Phase 4 will mask)
                        repeat_pen = getattr(self.cfg, "penalty_repeat", -5.0)
                        reward += float(repeat_pen)
                        self.violations_repeat += 1

                    # C) cooldown between charges
                    if not terminated:
                        min_gap = float(getattr(self.cfg, "min_charge_gap_min", 12.0))
                        if self.last_charge_end_min is not None and (now_min - float(self.last_charge_end_min)) < min_gap:
                            cd_pen = getattr(self.cfg, "penalty_cooldown", -5.0)
                            reward += float(cd_pen)
                            self.violations_cooldown += 1

                    # ---------- Execute charge if not terminated early ----------
                    if not terminated:
                        detour_min = float(cand.get("eta_min", 0.0))  # extra O→S + S→D minutes (already congestion-scaled later)
                        reward += self._apply_charge(station_id, dt_min, info, detour_min=detour_min)
                        # After a completed charge, enforce over-limit
                        max_ch = int(getattr(self.cfg, "max_charges_per_trip", 2))
                        if self.charge_events > max_ch:
                            over_pen = getattr(self.cfg, "penalty_overlimit", -20.0)
                            reward += float(over_pen)
                            self.violations_overlimit += 1
                            if bool(getattr(self.cfg, "terminate_on_overlimit", True)):
                                terminated = True
                                termination_reason = "overlimit"
                        self.charge_steps += 1  # PHASE 1.5

                        # Update Phase 3 state only if a *real* charge happened
                        lc = info.get("last_charge", {}) or {}
                        kwh = float(lc.get("grid_kwh") or 0.0)
                        mins = float(lc.get("minutes_used") or 0.0)
                        if (kwh > 0.0) and (mins > 0.0):
                            self.visited_station_ids.add(station_id)
                            # end-of-charge time is current total_minutes (apply_charge already advanced clock)
                            self.last_charge_end_min = float(self.total_minutes)
                            info["charge_end_min"] = self.last_charge_end_min
                    else:
                        # if we terminated due to overlimit, still count the step as a charge attempt
                        self.charge_steps += 1

        # ------------------------
        # Termination conditions
        # ------------------------
        arrival_via = None

        # Success ONLY if we arrived on THIS (drive) step
        if not terminated:
            if self._arrived_this_step:
                terminated = True
                self._done = True
                reward += getattr(self, "_success_bonus_eff", self.cfg.success_bonus)
                arrival_via = "drive"
                termination_reason = "success"
            elif self.soc <= 0.0:
                terminated = True
                self._done = True
                reward -= self.cfg.strand_penalty
                arrival_via = "none"
                termination_reason = "stranded"
            else:
                arrival_via = "none"

        # ------------------------
        # Phase 5: Reward shaping (time-potential + anti-dither)
        # ------------------------
        if getattr(self.cfg, "enable_shaping", True):
            # Make prev distance visible for logs
            info["prev_remaining_km"] = float(prev_remaining_km)

            # Progress in km (always >= 0)
            prev_km = float(prev_remaining_km)
            curr_km = float(self.remaining_km)
            progress_km = max(0.0, prev_km - curr_km)
            info["progress_km"] = float(progress_km)

            # (A) Potential-based time shaping (only for time/hybrid objectives)
            if getattr(self.cfg, "enable_potential_time", True) and (self.cfg.prefer in ("time", "hybrid")):
                vref = max(1e-3, float(self.cfg.potential_vref_kmh))
                eta_old_min = max(0.0, prev_km) / vref * 60.0
                eta_new_min = max(0.0, curr_km) / vref * 60.0
                vot = float(getattr(self, "_vot_eff", self.cfg.value_of_time_per_min))
                gamma_phi = float(getattr(self.cfg, "shaping_gamma", 1.0))
                phi_scale = float(getattr(self, "_phi_scale", 1.0))
                phi_old = - vot * eta_old_min
                phi_new = - vot * eta_new_min
                r_phi = phi_scale * (gamma_phi * phi_new - phi_old)
                reward += float(r_phi)
                info["shaping_time"] = float(r_phi)


            # (B) Anti-idle on DRIVE steps: tiny negative if no real progress
            idle_pen = float(getattr(self.cfg, "idle_penalty_per_step", 0.0))
            if idle_pen > 0.0:
                eps_km = float(getattr(self.cfg, "idle_progress_epsilon_km", 0.15))
                if info.get("step_type") == "drive" and (progress_km < eps_km) and not (terminated or truncated):
                    reward -= idle_pen
                    info["penalty_idle"] = float(idle_pen)

            # (C) Micro-charge penalty: tiny negative if the charge slice is too small
            micro_pen = float(getattr(self.cfg, "micro_charge_penalty", 0.0))
            if micro_pen > 0.0 and info.get("step_type") == "charge":
                lc = info.get("last_charge", {}) or {}
                kwh = float(lc.get("grid_kwh") or 0.0)
                mins = float(lc.get("minutes_used") or 0.0)
                if (kwh < float(self.cfg.micro_charge_min_kwh)) or (mins < float(self.cfg.micro_charge_min_minutes)):
                    reward -= micro_pen
                    info["penalty_micro_charge"] = float(micro_pen)


        # Time-limit truncation
        truncated = truncated or (self.step_count >= self.cfg.max_steps)
        if truncated and not self._done:
            self._done = True
            if termination_reason is None:
                termination_reason = "time_limit"

        # --- PHASE 1.5: KPI snapshot at episode end (so callbacks can read it from 'info') ---
        if terminated or truncated:
            reason = termination_reason or (
                "success" if self._arrived_this_step else
                "stranded" if self.soc <= 0.0 else
                "time_limit" if truncated else "other"
            )
            info.update({
                "episode_minutes":     float(self.total_minutes),
                "episode_cost_gbp":    float(self.total_cost),
                "episode_steps":       int(self.step_count),
                "charge_events":       int(self.charge_events),
                "drive_steps":         int(self.drive_steps),     # PHASE 1.5
                "charge_steps":        int(self.charge_steps),    # PHASE 1.5
                "soc_final":           float(self.soc),
                "remaining_km":        float(self.remaining_km),
                "termination_reason":  reason,
                "arrival_via":         arrival_via,               # PHASE 1.5
                # ---- Phase 3 observability ----
                "violations_repeat":   int(self.violations_repeat),
                "violations_overlimit":int(self.violations_overlimit),
                "violations_cooldown": int(self.violations_cooldown),
                "visited_station_ids": list(self.visited_station_ids),
                "last_charge_end_min": self.last_charge_end_min,
            })

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
    def _apply_drive_fallback(self, dt_min: float, info: Dict[str, Any]) -> float:
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

        # Traffic factor for this slice
        tf = self._traffic_factor(self._clock_dt)
        minutes_used = float(dt_min) * tf

        # Tally time and advance clock
        self.total_minutes += minutes_used
        if self._clock_dt is not None:
            from datetime import timedelta
            self._clock_dt = self._clock_dt + timedelta(minutes=minutes_used)

        vot = getattr(self, "_vot_eff", self.cfg.value_of_time_per_min)
        time_cost = minutes_used * vot
        return -time_cost if self.cfg.prefer in ("time", "hybrid") else 0.0

    
    def _current_xy_guess(self) -> Tuple[float, float]:
        """
        Best-effort guess of current XY on the map.
        If microsim is running and we have a vehicle, ask SUMO; otherwise return (nan, nan)
        and let caller fall back to an edge.
        """
        try:
            if self.cfg.sumo_mode == "microsim" and self._veh_id:
                st = self._sumo_runner.get_vehicle_state(self._veh_id)
                return float(st.get("x", float("nan"))), float(st.get("y", float("nan")))
        except Exception:
            pass
        return float("nan"), float("nan")

    def _begin_sumo_segment(self, to_station_id: str):
        to_edge = self.candidate_provider.edge_for_station(to_station_id)
        if not to_edge:
            raise RuntimeError(f"No SUMO edge for station {to_station_id}")

        # choose from_edge robustly
        from_edge = self._seg_from_edge
        if not from_edge:
            # try a best-effort XY → edge, then fallback to origin edge, then any edge
            xy = self._current_xy_guess()
            if all(np.isfinite(v) for v in xy):
                from_edge = self.candidate_provider.edge_near_xy(*xy)
        if not from_edge and hasattr(self._trip, "origin") and self._trip.origin and len(self._trip.origin) >= 2:
            try:
                o_lat, o_lon = float(self._trip.origin[0]), float(self._trip.origin[1])
                from_edge = self._sumo_runner.snap_to_edge(o_lat, o_lon)
            except Exception:
                from_edge = None
        if not from_edge:
            # absolute last resort
            from_edge = self._sumo_runner.any_edge_id()

        edges, length_m, travel_s = self._sumo_runner.find_route_edges(from_edge, to_edge)
        if not edges:
            raise RuntimeError(f"SUMO cannot route from {from_edge} to {to_edge}")

        self._seg_eta_s = float(travel_s)
        self._seg_left_s = self._seg_eta_s
        self._seg_from_edge = from_edge
        self._seg_to_edge = to_edge
        self._seg_len_m = float(length_m)  # <--- cache length now, avoid re-querying

        if self.cfg.sumo_mode == "microsim" and self._veh_id:
            self._sumo_runner.route_vehicle(self._veh_id, edge_from=from_edge, edge_to=to_edge, depart_s=0.0)

    def _begin_od_segment(self, to_edge: Optional[str]) -> None:
        """
        Start a SUMO route-time segment from the current edge toward 'to_edge' (destination).
        Robustly chooses a 'from_edge' (current -> origin -> any) and configures:
        self._seg_eta_s, self._seg_left_s, self._seg_from_edge, self._seg_to_edge, self._seg_len_m
        Raises RuntimeError only if SUMO is available but cannot route.
        """
        # Must have SUMO runner for route_time
        if not getattr(self, "_sumo_runner", None):
            raise RuntimeError("SUMO runner not available; cannot start OD segment")

        if not to_edge:
            raise RuntimeError("No destination edge available for OD segment")

        # 1) Prefer the currently known edge if any
        from_edge = getattr(self, "_seg_from_edge", None)

        # 2) Else try snapping our current XY guess to an edge via the provider
        if not from_edge:
            try:
                xy = self._current_xy_guess()
                if all(np.isfinite(v) for v in xy):
                    from_edge = self.candidate_provider.edge_near_xy(*xy)
            except Exception:
                from_edge = None

        # 3) Else snap the trip origin (lat/lon) if available
        if not from_edge and hasattr(self._trip, "origin") and self._trip.origin and len(self._trip.origin) >= 2:
            try:
                o_lat, o_lon = float(self._trip.origin[0]), float(self._trip.origin[1])
                from_edge = self._sumo_runner.snap_to_edge(o_lat, o_lon)
            except Exception:
                from_edge = None

        # 4) Absolute last resort: pick any valid edge from the network
        if not from_edge:
            from_edge = self._sumo_runner.any_edge_id()

        # Compute route using SUMO (edges list, total length [m], travel time [s])
        edges, length_m, travel_s = self._sumo_runner.find_route_edges(from_edge, to_edge)
        if not edges:
            raise RuntimeError(f"SUMO cannot route from {from_edge} to dest {to_edge}")

        # Configure the active segment
        self._seg_eta_s   = float(travel_s)
        self._seg_left_s  = self._seg_eta_s
        self._seg_from_edge = from_edge
        self._seg_to_edge   = to_edge
        self._seg_len_m   = float(length_m)
        # (In route_time mode we don't need to touch the microsim vehicle here.)
        
    def _apply_drive(self, dt_min: float, info: Dict[str, Any]) -> float:
        if not self.cfg.use_sumo_drive or self._sumo_runner is None:
            return self._apply_drive_fallback(dt_min, info)

        dt_s = dt_min * 60.0

        def _finish_time_and_reward(raw_minutes: float) -> float:
            # apply traffic factor even in SUMO mode to preserve your scalar peak-hour shaping (optional)
            tf = self._traffic_factor(self._clock_dt)
            minutes_used = float(raw_minutes) * tf
            self.total_minutes += minutes_used
            if self._clock_dt is not None:
                from datetime import timedelta
                self._clock_dt = self._clock_dt + timedelta(minutes=minutes_used)
            vot = getattr(self, "_vot_eff", self.cfg.value_of_time_per_min)
            return -(minutes_used * vot) if self.cfg.prefer in ("time", "hybrid") else 0.0

        if self.cfg.sumo_mode == "route_time":
            # If we don't currently have an active segment, try to start one to the destination
            if self._seg_left_s <= 0.0:
                try:
                    if self._dest_edge:
                        self._begin_od_segment(self._dest_edge)
                except Exception:
                    # No route available right now; fall back to kinematics for this slice
                    return self._apply_drive_fallback(dt_min, info)

            # Consume the active segment if we have one; else fallback
            if self._seg_left_s > 0:
                prev_left = self._seg_left_s
                self._seg_left_s = max(0.0, self._seg_left_s - dt_s)
                delta_progress = (prev_left - self._seg_left_s) / max(self._seg_eta_s, 1e-6)
                dist_km = (self._seg_len_m / 1000.0) * max(0.0, min(1.0, delta_progress))
                self.remaining_km = max(0.0, self.remaining_km - dist_km)

                grid_kwh = self.kwh_per_km * dist_km
                batt_kwh = grid_kwh * self.cfg.charge_efficiency
                self.soc = max(0.0, self.soc - batt_kwh / max(self.battery_kwh, 1e-6))

                if self._seg_left_s == 0.0:
                    self._seg_from_edge = self._seg_to_edge
                    self._seg_to_edge = None
                    self._seg_len_m = 0.0

                # Use SUMO-consumed time for KPI minutes and time cost
                consumed_min = (prev_left - self._seg_left_s) / 60.0
                return _finish_time_and_reward(consumed_min if consumed_min > 0.0 else dt_min)
            else:
                # Still no segment (e.g., no dest edge resolved): fall back this slice
                return self._apply_drive_fallback(dt_min, info)



        elif self.cfg.sumo_mode == "microsim":
            dist_km = 0.0
            if self._veh_id:
                steps = int(round(dt_s / max(self.cfg.sumo_step_length_s, 1e-3)))
                for _ in range(max(1, steps)):
                    self._sumo_runner.step()
                try:
                    st = self._sumo_runner.get_vehicle_state(self._veh_id)
                    speed_mps = float(st.get("speed_mps", 0.0))
                    dist_km = (speed_mps * steps * self.cfg.sumo_step_length_s) / 1000.0
                except Exception:
                    dist_km = 0.0

            self.remaining_km = max(0.0, self.remaining_km - dist_km)
            grid_kwh = self.kwh_per_km * dist_km
            batt_kwh = grid_kwh * self.cfg.charge_efficiency
            self.soc = max(0.0, self.soc - batt_kwh / max(self.battery_kwh, 1e-6))

            return _finish_time_and_reward(dt_min)

        else:
            return self._apply_drive_fallback(dt_min, info)

    def _apply_charge(self, station_id: str, dt_min: float, info: Dict[str, Any], detour_min: float = 0.0) -> float:
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

        # Minutes used this decision: charge duration + fixed session overhead + route detour (traffic-adjusted)
        overhead = float(getattr(self, "_charge_overhead_eff", self.cfg.charge_session_overhead_min))
        tf = self._traffic_factor(self._clock_dt)
        extra_detour = max(0.0, float(detour_min)) * tf
        minutes_used = float(dt_min) + overhead + extra_detour

        # Tally episode totals and advance clock
        self.total_minutes += minutes_used
        self.total_cost += energy_cost
        if self._clock_dt is not None:
            from datetime import timedelta
            self._clock_dt = self._clock_dt + timedelta(minutes=minutes_used)

        # --- PHASE 1.5: charge must be strictly net-negative, regardless of 'prefer'
        # Always pay BOTH time and energy costs on charge; no positive shaping.
        vot = getattr(self, "_vot_eff", self.cfg.value_of_time_per_min)
        time_cost = minutes_used * vot
        cost_cost = energy_cost

        info.update({
            "station_id": station_id,
            "category": chosen,
            "eff_kw": eff_kw,
            "grid_kwh": float(grid_kwh),
            "batt_kwh": float(batt_kwh),
            "price_per_kwh": unit_price,   # may be None if catalog used
            "energy_cost": energy_cost,
            "detour_min": extra_detour,
            "overhead_min": overhead,
            "minutes_used": minutes_used,
        })
        
        # Phase-5 friendly: also provide a nested 'last_charge' dict that callers use
        info["last_charge"] = {
            "station_id": station_id,
            "category": chosen,
            "eff_kw": eff_kw,
            "grid_kwh": float(grid_kwh),
            "batt_kwh": float(batt_kwh),
            "price_per_kwh": unit_price,
            "energy_cost": energy_cost,
            "detour_min": extra_detour,
            "overhead_min": overhead,
            "minutes_used": minutes_used,
        }


        # Count only if this slice delivered meaningful energy
        if batt_kwh > 0.0 and minutes_used > 0.0:
            self.charge_events += 1

        return -(time_cost + cost_cost)


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
                # ---- Phase 4 masking metadata ----
                visited_station_ids=self.visited_station_ids,
                now_min=float(self.total_minutes),
                last_charge_end_min=self.last_charge_end_min,
                min_charge_gap_min=float(self.cfg.min_charge_gap_min),
                emergency=(self.soc < 0.08),   # allow exception if SoC < 8%
                keep_nearest_n_on_emergency=1,
            )
            # Drop stations that were flagged as unroutable earlier in the episode
            cand_list = [c for c in cand_list if c.station_id not in getattr(self, "unreachable_station_ids", set())]

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
        self._refresh_candidates()
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

    # ---------------------------
    # Action mask (Phase 4 safety)
    # ---------------------------
    def get_action_mask(self) -> np.ndarray:
        """
        Build a boolean mask for legal actions.
        Action 0 = Drive (always legal).
        Actions 1..K correspond to self.candidates.
        """
        n_actions = self.action_space.n
        mask = np.ones(n_actions, dtype=bool)

        # If there are fewer candidates than obs_top_k, mask out the extras
        K = int(self.cfg.obs_top_k)
        valid_station_count = len(self.candidates)
        for i in range(valid_station_count, K):
            mask[i + 1] = False

        # ---- SAFETY GUARANTEE ----
        mask[0] = True  # Drive is always legal
        return mask


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

    

