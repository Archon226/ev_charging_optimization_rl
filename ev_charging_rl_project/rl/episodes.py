# rl/episodes.py  (TINY TWEAK)
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

@dataclass
class TripPlan:
    origin: Tuple[float, float]
    dest: Tuple[float, float]
    ev_model: str
    objective: str
    start_soc_pct: float
    reserve_soc_pct: float
    target_soc_pct: float
    user_type: str
    top_k: int
    max_detour_km: float

def load_episodes(path: str) -> List[TripPlan]:
    df = pd.read_csv(path)
    trips: List[TripPlan] = []
    for r in df.itertuples(index=False):
        trips.append(TripPlan(
            origin=(float(r.start_lat), float(r.start_lon)),
            dest=(float(r.dest_lat), float(r.dest_lon)),
            ev_model=(getattr(r, "ev_model", None) or getattr(r, "model", "")),
            objective=str(getattr(r, "objective", "hybrid")),
            start_soc_pct=float(r.start_soc_pct),
            reserve_soc_pct=float(r.reserve_soc_pct),
            target_soc_pct=float(r.target_soc_pct),
            user_type=str(r.user_type),
            top_k=int(getattr(r, "top_k_candidates", 6)),
            max_detour_km=float(getattr(r, "max_detour_km", 3.0)),
        ))
    return trips
