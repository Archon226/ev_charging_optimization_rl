# rl/episodes.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Iterator, Optional
from pathlib import Path
from datetime import datetime

import pandas as pd


@dataclass
class TripPlan:
    user_id: int
    ev_model: str
    user_type: str                  # "Payg" | "Member"
    include_subscription: int       # 0/1
    sessions_per_month: int

    origin: tuple[float, float]
    dest: tuple[float, float]
    depart_datetime: datetime

    start_soc_pct: float
    reserve_soc_pct: float
    target_soc_pct: float           # env can override default

    objective: str                  # "cost" | "time" | "hybrid"
    max_detour_km: float
    top_k_candidates: int

    # optional analytics
    trip_km: Optional[float] = None
    est_required_kwh: Optional[float] = None
    available_kwh_at_start: Optional[float] = None
    kwh_per_km: Optional[float] = None


def _parse_dt(x) -> datetime:
    try:
        return datetime.fromisoformat(str(x))
    except Exception:
        return datetime.now()


def load_episodes(path: str | Path = "data/simulated_users.csv",
                  default_target_soc: float = 80.0) -> List[TripPlan]:
    df = pd.read_csv(path)
    trips: List[TripPlan] = []
    for r in df.itertuples(index=False):
        trips.append(TripPlan(
            user_id=int(r.user_id),
            ev_model=str(r.ev_model),
            user_type=str(r.user_type),
            include_subscription=int(getattr(r, "include_subscription", 0)),
            sessions_per_month=int(getattr(r, "sessions_per_month", 20)),

            origin=(float(r.start_lat), float(r.start_lon)),
            dest=(float(r.end_lat), float(r.end_lon)),
            depart_datetime=_parse_dt(getattr(r, "depart_datetime", None)),

            start_soc_pct=float(r.start_soc_pct),
            reserve_soc_pct=float(r.reserve_soc_pct),
            target_soc_pct=float(getattr(r, "target_soc_pct", default_target_soc)),

            objective=str(getattr(r, "objective", "hybrid")),
            max_detour_km=float(getattr(r, "max_detour_km", 3.0)),
            top_k_candidates=int(getattr(r, "top_k_candidates", 6)),

            trip_km=float(getattr(r, "trip_km", None)),
            est_required_kwh=float(getattr(r, "est_required_kwh", None)),
            available_kwh_at_start=float(getattr(r, "available_kwh_at_start", None)),
            kwh_per_km=float(getattr(r, "kwh_per_km", None)),
        ))
    return trips


def iter_episodes(path: str | Path = "data/simulated_users.csv") -> Iterator[TripPlan]:
    for t in load_episodes(path):
        yield t


def episodes_as_dataframe(path: str | Path = "data/simulated_users.csv") -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in load_episodes(path)])
