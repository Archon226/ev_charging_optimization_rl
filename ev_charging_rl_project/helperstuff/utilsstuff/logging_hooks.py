# utils/logging_hooks.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import os, json, csv, time

@dataclass
class EventRow:
    run_id: str
    episode_id: int
    step: int
    event_type: str  # drive|charge|arrive|dead_battery|noop
    trip_id: str
    objective: str
    user_type: str
    ev_model: str
    # action-level
    action: Optional[str] = None
    candidate_rank: Optional[int] = None
    reason: Optional[str] = None
    # station/charger
    station_id: Optional[str] = None
    company_id: Optional[str] = None
    connector_type: Optional[str] = None
    charger_type: Optional[str] = None  # AC|DC
    power_kw: Optional[float] = None
    # soc/energy
    start_soc: Optional[float] = None
    end_soc: Optional[float] = None
    delta_soc: Optional[float] = None
    energy_kwh: Optional[float] = None
    session_minutes: Optional[float] = None
    # movement/time
    distance_km: Optional[float] = None
    eta_s: Optional[float] = None
    detour_seconds: Optional[float] = None
    queue_wait_s: Optional[float] = None
    # money
    session_cost_gbp: Optional[float] = None
    total_cost_gbp: Optional[float] = None
    pricing_breakdown_json: Optional[str] = None
    # misc
    notes: Optional[str] = None
    ts: float = time.time()

@dataclass
class EpisodeRow:
    run_id: str
    episode_id: int
    trip_id: str
    objective: str
    user_type: str
    ev_model: str
    start_soc: float
    end_soc: float
    total_time_s: float
    total_cost_gbp: float
    reached_dest: bool
    ran_out_of_battery: bool
    n_charges: int
    notes: str = ""

class CSVDump:
    def __init__(self, out_dir: str = "logs"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.events_path = os.path.join(out_dir, "events.csv")
        self.episodes_path = os.path.join(out_dir, "episodes.csv")
        self._ensure_headers()

    def _ensure_headers(self):
        if not os.path.exists(self.events_path):
            with open(self.events_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[f.name for f in EventRow.__dataclass_fields__.values()])
                w.writeheader()
        if not os.path.exists(self.episodes_path):
            with open(self.episodes_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[f.name for f in EpisodeRow.__dataclass_fields__.values()])
                w.writeheader()

    def append_event(self, e: EventRow):
        with open(self.events_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[f.name for f in EventRow.__dataclass_fields__.values()])
            w.writerow(asdict(e))

    def append_episode(self, e: EpisodeRow):
        with open(self.episodes_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[f.name for f in EpisodeRow.__dataclass_fields__.values()])
            w.writerow(asdict(e))
