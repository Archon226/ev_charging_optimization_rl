# simulator/trip_simulator.py
from __future__ import annotations
from datetime import datetime
import numpy as np
import json  # for pricing_breakdown_json
import os

# --- small haversine helper for a reset-time sanity check ---
from math import radians, sin, cos, asin, sqrt
def _hav_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))


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
        self.total_cost = 0.0  # cumulative cost for info
        self.total_time_s = 0.0  # accumulate drive + charge seconds

        # store user economics for pricing
        self.user_type = (user_type or "Payg").title()
        self.include_subscription = bool(include_subscription)
        self.sessions_per_month = int(sessions_per_month or 0)

        # compute OD route once and guard against degenerate zero-distance
        try:
            od_km, od_s = self.sim.route_between_cached(
                float(self.origin[0]), float(self.origin[1]),
                float(self.dest[0]), float(self.dest[1])
            )
        except Exception:
            od_km, od_s = 0.0, 0.0

        # keep as optional attrs for downstream consumers
        self.od_km = float(od_km)
        self.od_time_s = float(od_s)

        # If SUMO path is ~0 but Haversine says it's a real trip, do NOT allow instant arrival
        try:
            hav = _hav_km(float(self.origin[0]), float(self.origin[1]),
                          float(self.dest[0]), float(self.dest[1]))
            if (self.od_km < 0.02) and (hav > 0.3):  # <20 m vs >300 m straight-line
                self.done = False
        except Exception:
            pass

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

        # --- Map to pricing buckets expected by pricing_by_charger_type.csv ---
        # Tariff table uses: Fast | Rapid | Ultra (not "AC"/"DC")
        # Heuristic:
        #   AC -> "Fast"
        #   DC & <100 kW -> "Rapid"
        #   DC & >=100 kW -> "Ultra"
        bucket = "Fast"
        if is_dc:
            bucket = "Ultra" if p_station_kw >= 100.0 else "Rapid"

        # Remaining distance from station to destination
        rem_km, _ = self.sim.route_between_cached(
            float(station.lat), float(station.lon),
            float(self.dest[0]), float(self.dest[1])
        )
        if not np.isfinite(rem_km):
            rem_km = 0.0

        # Energy on board & energy needed (add 10% buffer)
        energy_now_kwh = float(self.ev_spec.battery_kwh) * float(max(0.0, min(1.0, self.current_soc)))
        need_kwh = float(rem_km) * float(self.ev_spec.eff_kwh_per_km) * 1.10
        charge_kwh = max(0.0, need_kwh - energy_now_kwh)

        session_minutes = (charge_kwh / charge_power_kw) * 60.0 if charge_power_kw > 0 else 0.0

        # Pricing call (your API)
        now = now or datetime.now()

        # Optional debug: print pricing keys & the row we'd expect to match
        if os.getenv("EV_PRICING_DEBUG", "0") == "1":
            try:
                bt = getattr(self.pricing, "by_type", None)
                row_preview = None
                if bt is not None:
                    # try to show a matching by_type row for visibility
                    sel = bt[
                        (bt["company_id"] == company_id)
                        & (bt["charger_type"] == bucket)
                        & (bt["user_type"].str.title() == str(self.user_type).title())
                    ]
                    if not sel.empty:
                        row_preview = sel.iloc[0].to_dict()
                print(
                    f"[EV_PRICING_DEBUG] station={getattr(station,'station_id',None)} "
                    f"company_id={company_id} charger_type_raw={'DC' if is_dc else 'AC'} "
                    f"bucket={bucket} user_type={self.user_type} "
                    f"dt={(now or datetime.now()).isoformat(timespec='seconds')} "
                    f"by_type_row={row_preview}",
                    flush=True
                )
            except Exception as _e:
                # Do not let debug path crash pricing
                print(f"[EV_PRICING_DEBUG] warn: {type(_e).__name__}: {_e}", flush=True)

        pb = self.pricing.estimate_session(
            company_id=company_id,
            charger_type=bucket,
            user_type=self.user_type,
            start_dt=now,
            kwh=charge_kwh,
            session_minutes=session_minutes,
            include_subscription=self.include_subscription,
            sessions_per_month=self.sessions_per_month if self.include_subscription else 0,
        )

        total_cost = float(getattr(pb, "total_cost", pb))
        if not np.isfinite(total_cost):
            total_cost = 0.0

        return total_cost, pb, {
            "charge_kwh": float(charge_kwh),
            "session_minutes": float(session_minutes),
            "charge_power_kw": float(charge_power_kw),
            "remaining_km_from_station": float(rem_km),
        }

    def step(self, action_station):
        """
        One step = choose a station (action_station), then:
          1) drive Origin/Current -> Station (SOC down),
          2) charge at Station (SOC up, cost),
          3) if enough energy for Station -> Destination, mark ARRIVE (terminal),
             else stay at Station (non-terminal) to pick another charger next step.
        """
        if self.done:
            raise RuntimeError("Step called after episode finished. Reset required.")

        # --- 1) Drive to the chosen station (O->C) using the real leg ---
        oc_km, oc_s = self.sim.route_between_cached(
            float(self.position[0]), float(self.position[1]),
            float(action_station.lat), float(action_station.lon)
        )
        if not np.isfinite(oc_km): oc_km = 0.0
        if not np.isfinite(oc_s):  oc_s  = 0.0

        eff = float(self.ev_spec.eff_kwh_per_km)
        batt_kwh = float(self.ev_spec.battery_kwh)

        energy_kwh_to_station = float(oc_km) * eff
        soc_before_drive = float(self.current_soc)
        self.current_soc -= energy_kwh_to_station / batt_kwh
        self.current_soc = max(0.0, self.current_soc)
        self.total_time_s += float(oc_s)

        # If we ran out before reaching the station and cannot recover (no charge yet)
        if self.current_soc <= 0.0 and energy_kwh_to_station > 0.0:
            # Move to station location anyway for logging consistency
            self.position = (float(action_station.lat), float(action_station.lon))
            self.done = True
            info = {
                "event_type": "dead_battery",
                "reason": "battery_depleted_on_detour",
                "distance_km": float(oc_km),
                "detour_seconds": float(oc_s),
                "total_time_s": float(self.total_time_s),
                "start_soc": soc_before_drive,
                "end_soc": float(self.current_soc),
                "delta_soc": float(self.current_soc - soc_before_drive),
                "energy_kwh": 0.0,
                "session_minutes": 0.0,
                "session_cost_gbp": 0.0,
                "total_cost_gbp": float(self.total_cost),
                "station_id": getattr(action_station, "station_id", None),
                "company_id": getattr(action_station, "company_id", None),
                "connector_type": getattr(action_station, "connector_type", None),
                "charger_type": ("DC" if str(getattr(action_station, "charger_type", "") or "").upper()=="DC" else "AC") if getattr(action_station, "charger_type", None) is not None else None,
                "power_kw": float(getattr(action_station, "rated_power_kw", 0.0) or 0.0),
            }
            return self._get_state(), float(-3600.0 * oc_s), True, info  # strong penalty on death

        # Arrived at station
        self.position = (float(action_station.lat), float(action_station.lon))

        # --- 2) Charge at the station (compute enough-to-destination need) ---
        cost_gbp, pb, extras = self._compute_session_cost(action_station)
        charge_kwh = float(extras.get("charge_kwh", 0.0) or 0.0)
        session_minutes = float(extras.get("session_minutes", 0.0) or 0.0)
        charge_power_kw = float(extras.get("charge_power_kw", 0.0) or 0.0)

        soc_before_charge = float(self.current_soc)
        did_charge = (charge_kwh > 0.0) and (session_minutes > 0.0)
        if did_charge:
            self.current_soc = min(1.0, self.current_soc + (charge_kwh / batt_kwh))
            self.total_cost += float(cost_gbp)
            self.total_time_s += 60.0 * float(session_minutes)
        # else: no charge needed; keep SOC

        # --- 3) Decide if we can reach D from the station ---
        cd_km, cd_s = self.sim.route_between_cached(
            float(self.position[0]), float(self.position[1]),
            float(self.dest[0]), float(self.dest[1])
        )
        if not np.isfinite(cd_km):
            cd_km = float(extras.get("remaining_km_from_station", np.inf))
        if not np.isfinite(cd_km):
            cd_km = 1e9  # keep non-arrival if routing failed

        need_kwh_for_dest = float(cd_km) * eff
        available_kwh = batt_kwh * float(self.current_soc)

        # Common diagnostics
        pricing_json = None
        try:
            if pb is not None:
                pricing_json = json.dumps(getattr(pb, "__dict__", pb), default=float)
        except Exception:
            pricing_json = None

        station_id = getattr(action_station, "station_id", None)
        company_id = getattr(action_station, "company_id", None)
        connector_type = getattr(action_station, "connector_type", None)
        charger_type = getattr(action_station, "charger_type", None)

        base_info = {
            "pricing_breakdown": pb.__dict__ if hasattr(pb, "__dict__") else pb,
            "pricing_breakdown_json": pricing_json,
            "detour_seconds": float(oc_s),      # per-step travel seconds (drive leg)
            "distance_km": float(oc_km),       # per-step travel km (drive leg)
            "station_id": station_id,
            "company_id": company_id,
            "connector_type": connector_type,
            "charger_type": ("DC" if str(charger_type or "").upper()=="DC" else "AC") if charger_type is not None else None,
            "power_kw": charge_power_kw,
            # SOC checkpoints
            "soc_before_drive": soc_before_drive,
            "soc_after_drive": float(soc_before_drive - energy_kwh_to_station / batt_kwh),
            "soc_after_charge": float(self.current_soc),
            # per-step charge metrics
            "energy_kwh": float(charge_kwh),
            "session_minutes": float(session_minutes),
            "session_cost_gbp": float(cost_gbp) if did_charge else 0.0,
            "total_cost_gbp": float(self.total_cost),
            # arrival math
            "dist_to_dest_km": float(cd_km),
            "available_kwh": float(available_kwh),
            "need_kwh_for_dest": float(need_kwh_for_dest),
            # totals so far
            "total_time_s": float(self.total_time_s),
        }

        # If we now have enough energy to reach D, drive C->D and ARRIVE (terminal)
        if available_kwh >= need_kwh_for_dest:
            self.total_time_s += float(cd_s)
            self.done = True
            info = {
                **base_info,
                "event_type": "arrive",
                "reason": None,
                # include the remaining leg in the terminal event row
                "distance_km": float(oc_km + cd_km),
                "detour_seconds": float(oc_s + (cd_s if np.isfinite(cd_s) else 0.0)),
                "total_time_s": float(self.total_time_s),
                "start_soc": float(soc_before_charge),
                "end_soc": float(self.current_soc),
                "delta_soc": float(self.current_soc - soc_before_charge),
            }
            reward = -float(cost_gbp) - float(((oc_s if np.isfinite(oc_s) else 0.0) + (cd_s if np.isfinite(cd_s) else 0.0)) / 3600.0)
            return self._get_state(), float(reward), True, info

        # Otherwise, we stay at this station for the next decision (non-terminal)
        info = {
            **base_info,
            "event_type": "charge" if did_charge else "drive",
            "reason": None if did_charge else "insufficient_energy_continue",
            "start_soc": float(soc_before_charge),
            "end_soc": float(self.current_soc),
            "delta_soc": float(self.current_soc - soc_before_charge),
        }
        reward = -float(cost_gbp) - float((oc_s if np.isfinite(oc_s) else 0.0) / 3600.0)
        return self._get_state(), float(reward), False, info

    def _get_state(self):
        # Keep state shape identical
        state = {
            "position": (float(self.position[0]), float(self.position[1])),
            "soc": float(self.current_soc),
            "dest": (float(self.dest[0]), float(self.dest[1])),
        }
        return state
