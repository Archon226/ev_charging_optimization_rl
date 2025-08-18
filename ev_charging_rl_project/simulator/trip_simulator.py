from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import numpy as np

class TripSimulator:
    def __init__(self, sim, pricing_catalog):
        self.sim = sim
        self.pricing = pricing_catalog

    def reset(self, origin, dest, ev_spec, current_soc,
              user_type: str = "Payg",
              include_subscription: bool = False,
              sessions_per_month: int = 0):
        self.origin = origin
        self.dest = dest
        self.ev_spec = ev_spec
        self.current_soc = float(current_soc)
        self.position = origin
        self.done = False

        # store user economics for pricing
        self.user_type = (user_type or "Payg").title()
        self.include_subscription = bool(include_subscription)
        self.sessions_per_month = int(sessions_per_month or 0)

        return self._get_state()

    def _compute_session_cost(self, station, now=None):
        if station.company_id is None:
            raise ValueError("Candidate.company_id is missing; cannot price without company_id.")
        company_id = int(station.company_id)

        # AC/DC and power caps
        is_dc = str(station.charger_type or "").strip().upper() == "DC"
        p_station_kw = float(station.rated_power_kw or 0.0)
        p_ev_kw = float(self.ev_spec.max_dc_kw if is_dc else self.ev_spec.max_ac_kw)
        charge_power_kw = max(0.1, min(p_station_kw, p_ev_kw))  # avoid zero

        # Remaining distance from station to destination
        rem_km, _ = self.sim.route_between_cached(
            float(station.lat), float(station.lon),
            float(self.dest[0]), float(self.dest[1])
        )
        # Force finite
        if not np.isfinite(rem_km):
            rem_km = 0.0

        # Energy on board & energy needed (add 10% buffer)
        energy_now_kwh = float(self.ev_spec.battery_kwh) * float(max(0.0, min(1.0, self.current_soc)))
        need_kwh = float(rem_km) * float(self.ev_spec.eff_kwh_per_km) * 1.10
        charge_kwh = max(0.0, need_kwh - energy_now_kwh)

        session_minutes = (charge_kwh / charge_power_kw) * 60.0 if charge_power_kw > 0 else 0.0

        # Pricing call (your API)
        now = now or datetime.now()
        pb = self.pricing.estimate_session(
            company_id=company_id,
            charger_type=("DC" if is_dc else "AC"),
            user_type=self.user_type,
            start_dt=now,
            kwh=charge_kwh,
            session_minutes=session_minutes,
            include_subscription=self.include_subscription,
            sessions_per_month=self.sessions_per_month if self.include_subscription else 0,
        )

        total_cost = float(pb.total_cost) if hasattr(pb, "total_cost") else float(pb)
        if not np.isfinite(total_cost):
            total_cost = 0.0

        return total_cost, pb, {
            "charge_kwh": float(charge_kwh),
            "session_minutes": float(session_minutes),
            "charge_power_kw": float(charge_power_kw),
            "remaining_km_from_station": float(rem_km),
        }

    def step(self, action_station):
        # Detour (O->C->D minus O->D)
        dk, dt = self.sim.detour_via_cached(
            float(self.position[0]), float(self.position[1]),
            float(action_station.lat), float(action_station.lon),
            float(self.dest[0]), float(self.dest[1])
        )
        if not np.isfinite(dk): dk = 0.0
        if not np.isfinite(dt): dt = 0.0

        # Consume energy for the detour to the station
        self.current_soc -= float(dk) * float(self.ev_spec.eff_kwh_per_km) / float(self.ev_spec.battery_kwh)
        self.current_soc = max(0.0, self.current_soc)

        # Price session
        cost_gbp, pb, extras = self._compute_session_cost(action_station)

        # Reward: cost + small time penalty (you can tweak later)
        reward = -float(cost_gbp) - float(dt) / 3600.0
        if not np.isfinite(reward):
            reward = -1e3  # strong penalty instead of NaN

        # Move to station
        self.position = (float(action_station.lat), float(action_station.lon))

        info = {
            "total_cost_gbp": float(cost_gbp),
            "pricing_breakdown": pb.__dict__ if hasattr(pb, "__dict__") else pb,
            "detour_seconds": float(dt),
            **extras
        }
        return self._get_state(), float(reward), self.done, info

    def _get_state(self):
        # you can expand this later; keep it finite
        state = {"position": (float(self.position[0]), float(self.position[1])),
                 "soc": float(self.current_soc),
                 "dest": (float(self.dest[0]), float(self.dest[1]))}
        return state
