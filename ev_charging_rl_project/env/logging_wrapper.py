# env/logging_wrapper.py
import gymnasium as gym
import uuid
import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logging_hooks import CSVDump, EventRow, EpisodeRow

class LoggingWrapper(gym.Wrapper):
    def __init__(self, env, log_dir: str = "logs", run_id: str | None = None, log_every_n: int = 1):
        super().__init__(env)
        self.dump = CSVDump(log_dir)
        self.run_id = run_id or uuid.uuid4().hex[:8]
        self.log_every_n = max(1, int(log_every_n))
        self.episode_id = -1
        self.episode_events = []
        self._episode_meta = {}

    def reset(self, **kwargs):
        self.episode_id += 1
        self.episode_events.clear()
        obs, info = self.env.reset(**kwargs)
        # Minimal metadata for every episode (try to pick from env or info)
        trip = getattr(self.env, "trip", None)
        self._episode_meta = {
            "trip_id": getattr(trip, "trip_id", str(self.episode_id)),
            "objective": getattr(self.env, "objective", "hybrid"),
            "user_type": getattr(trip, "user_type", "Payg"),
            "ev_model": getattr(trip, "ev_model", getattr(self.env, "ev_model", "unknown")),
            "start_soc": info.get("start_soc", getattr(self, "start_soc", None)) if isinstance(info, dict) else None
        }
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compose one EventRow per step (and one extra if there was a charge)
        base = dict(
            run_id=self.run_id,
            episode_id=self.episode_id,
            step=len(self.episode_events),
            trip_id=self._episode_meta["trip_id"],
            objective=self._episode_meta["objective"],
            user_type=self._episode_meta["user_type"],
            ev_model=self._episode_meta["ev_model"],
        )

        # 1) The step event (drive/arrive/dead_battery/charge)
        ev = EventRow(
            **base,
            event_type=str(info.get("event_type", "noop")),
            action=str(info.get("action")) if "action" in info else None,
            candidate_rank=info.get("candidate_rank"),
            reason=info.get("reason"),
            station_id=info.get("station_id"),
            company_id=info.get("company_id"),
            connector_type=info.get("connector_type"),
            charger_type=info.get("charger_type"),
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
            session_cost_gbp=info.get("session_cost_gbp"),
            total_cost_gbp=info.get("total_cost_gbp"),
            pricing_breakdown_json=info.get("pricing_breakdown_json"),
            notes=info.get("notes"),
        )
        self.episode_events.append(ev)
        # Always keep terminal/charge events; throttle the rest
        if (ev.event_type in ("arrive", "dead_battery", "charge")) or (ev.step % self.log_every_n == 0):
            self.dump.append_event(ev)

        if terminated or truncated:
            # Summarize the episode
            end_soc = ev.end_soc if ev.end_soc is not None else info.get("end_soc")
            ep = EpisodeRow(
                run_id=self.run_id,
                episode_id=self.episode_id,
                trip_id=self._episode_meta["trip_id"],
                objective=self._episode_meta["objective"],
                user_type=self._episode_meta["user_type"],
                ev_model=self._episode_meta["ev_model"],
                start_soc=float(self._episode_meta.get("start_soc") or info.get("start_soc") or 0.0),
                end_soc=float(end_soc or 0.0),
                total_time_s=float(info.get("total_time_s", 0.0)),
                total_cost_gbp=float(info.get("total_cost_gbp", 0.0)),
                reached_dest=bool(info.get("event_type") == "arrive"),
                ran_out_of_battery=bool(info.get("event_type") == "dead_battery"),
                n_charges=sum(1 for e in self.episode_events if e.event_type == "charge"),
                notes=str(info.get("notes", "")),
            )
            self.dump.append_episode(ep)

        return obs, reward, terminated, truncated, info
