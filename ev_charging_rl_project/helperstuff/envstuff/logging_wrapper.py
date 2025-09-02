# env/logging_wrapper.py
from __future__ import annotations

import json
import uuid
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import gymnasium as gym

# project imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logging_hooks import CSVDump, EventRow, EpisodeRow  # keep existing schemas


class LoggingWrapper(gym.Wrapper):
    """
    Generic CSV logging wrapper that aligns legacy columns with the new pipeline.

    - Maps new 'category' -> existing 'charger_type' column (Fast/Rapid/Ultra).
    - Accepts 'total_price' from the new session planner and falls back to 'session_cost_gbp'.
    - Embeds 'unit_price_source' into pricing_breakdown_json to preserve provenance.
    """

    def __init__(
        self,
        env,
        log_dir: str = "logs",
        run_id: Optional[str] = None,
        log_every_n: int = 1,
    ):
        super().__init__(env)
        self.dump = CSVDump(log_dir)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.log_every_n = max(1, int(log_every_n))
        self.episode_id = -1
        self.episode_events: list[EventRow] = []
        self._episode_meta: Dict[str, Any] = {}

    def reset(self, **kwargs):
        self.episode_id += 1
        self.episode_events.clear()
        obs, info = self.env.reset(**kwargs)

        # Pull minimal trip metadata (robust to source differences)
        trip = getattr(self.env, "trip", None)
        self._episode_meta = {
            "trip_id": getattr(trip, "trip_id", str(self.episode_id)),
            "objective": getattr(self.env, "objective", getattr(trip, "objective", "hybrid")),
            "user_type": getattr(trip, "user_type", "Payg"),
            "ev_model": getattr(trip, "ev_model", getattr(self.env, "ev_model", "unknown")),
            "start_soc": (info or {}).get("start_soc", getattr(self.env, "start_soc", None)),
            "start_time": datetime.utcnow().isoformat(timespec="seconds"),
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = info or {}

        # ---------- normalize new-pipeline fields to legacy column names ----------
        # category -> charger_type (Fast/Rapid/Ultra)
        charger_type = info.get("charger_type") or info.get("category")
        # connector_type (keep if provided)
        connector_type = info.get("connector_type")
        # session price (prefer new field name total_price; fallback to legacy)
        session_cost = (
            info.get("total_price")
            if info.get("total_price") is not None
            else info.get("session_cost_gbp")
        )
        # pricing breakdown JSON: include unit_price_source if provided
        pricing_json: Dict[str, Any] = {}
        if isinstance(info.get("pricing_breakdown_json"), str):
            try:
                pricing_json = json.loads(info["pricing_breakdown_json"])
            except Exception:
                pricing_json = {"raw": info.get("pricing_breakdown_json")}
        elif isinstance(info.get("pricing_breakdown_json"), dict):
            pricing_json = dict(info["pricing_breakdown_json"])

        if "unit_price_source" in info:
            pricing_json["unit_price_source"] = info["unit_price_source"]

        # ---------- base fields shared by all step events ----------
        base = dict(
            run_id=self.run_id,
            episode_id=self.episode_id,
            step=len(self.episode_events),
            trip_id=self._episode_meta.get("trip_id", str(self.episode_id)),
            objective=self._episode_meta.get("objective", "hybrid"),
            user_type=self._episode_meta.get("user_type", "Payg"),
            ev_model=self._episode_meta.get("ev_model", "unknown"),
        )

        # ---------- primary step event ----------
        ev = EventRow(
            **base,
            event_type=str(info.get("event_type", "noop")),
            action=str(info.get("action")) if "action" in info else None,
            candidate_rank=info.get("candidate_rank"),

            reason=info.get("reason"),
            station_id=info.get("station_id"),
            company_id=info.get("company_id"),

            # harmonized fields:
            connector_type=connector_type,
            charger_type=charger_type,

            power_kw=info.get("power_kw"),
            start_soc=info.get("start_soc"),
            end_soc=info.get("end_soc"),
            delta_soc=info.get("delta_soc"),
            energy_kwh=info.get("energy_kwh"),
            session_minutes=info.get("session_minutes"),

            distance_km=info.get("distance_km"),
            eta_s=info.get("eta_s"),
            detour_seconds=info.get("detour_seconds"),
            queue_wait_s=info.get("queue_wait_s"),

            session_cost_gbp=session_cost,
            total_cost_gbp=info.get("total_cost_gbp"),

            pricing_breakdown_json=json.dumps(pricing_json) if pricing_json else None,
            notes=info.get("notes"),
        )
        self.episode_events.append(ev)

        # throttle: always log terminal & charge; otherwise every N steps
        if (ev.event_type in ("arrive", "dead_battery", "charge")) or (ev.step % self.log_every_n == 0):
            self.dump.append_event(ev)

        # ---------- episode summary on termination ----------
        if terminated or truncated:
            # find last known end_soc
            end_soc = ev.end_soc if ev.end_soc is not None else info.get("end_soc")
            n_charges = sum(1 for e in self.episode_events if e.event_type == "charge")

            ep = EpisodeRow(
                run_id=self.run_id,
                episode_id=self.episode_id,
                trip_id=self._episode_meta.get("trip_id", str(self.episode_id)),
                objective=self._episode_meta.get("objective", "hybrid"),
                user_type=self._episode_meta.get("user_type", "Payg"),
                ev_model=self._episode_meta.get("ev_model", "unknown"),

                start_soc=float(self._episode_meta.get("start_soc") or info.get("start_soc") or 0.0),
                end_soc=float(end_soc or 0.0),

                total_time_s=float(info.get("total_time_s", 0.0)),
                total_cost_gbp=float(info.get("total_cost_gbp", 0.0)),

                reached_dest=bool(info.get("event_type") == "arrive"),
                ran_out_of_battery=bool(info.get("event_type") == "dead_battery"),
                n_charges=n_charges,

                notes=str(info.get("notes", "")),
            )
            self.dump.append_episode(ep)

        return obs, reward, terminated, truncated, info
