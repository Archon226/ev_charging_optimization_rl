import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class TripPlan:
    origin: tuple
    dest: tuple
    ev_spec: dict
    start_soc: float

def load_episodes(path: str) -> List[TripPlan]:
    df = pd.read_csv(path)
    trips = []
    for row in df.itertuples(index=False):
        trips.append(
            TripPlan(
                origin=(row.origin_lat, row.origin_lon),
                dest=(row.dest_lat, row.dest_lon),
                ev_spec={"model": row.ev_model},
                start_soc=row.start_soc
            )
        )
    return trips
