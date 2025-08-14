# utils/generate_users.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import random
import numpy as np
import pandas as pd

@dataclass
class UserGenConfig:
    n_users: int = 100
    depart_start: datetime = datetime(2025, 8, 12, 7, 0)
    depart_end:   datetime = datetime(2025, 8, 12, 21, 0)
    init_soc_min: float = 8.0
    init_soc_max: float = 55.0
    reserve_soc_pct: float = 12.0
    target_soc_pct: float = 80.0
    max_detour_km: float = 12.0
    top_k_candidates: int = 5
    step_horizon_s: int = 180
    user_type_dist: List[Tuple[str, float]] = (("Payg", 0.70), ("Member", 0.30))
    objective_dist: List[Tuple[str, float]] = (("time", 0.33), ("cost", 0.34), ("hybrid", 0.33))
    member_sessions_choices: List[int] = (4, 8, 12)
    payg_sessions_choices:   List[int] = (0,)
    seed: Optional[int] = 42

def _weighted_choice(pairs: List[Tuple[str, float]]) -> str:
    labels, probs = zip(*pairs)
    return random.choices(labels, weights=probs, k=1)[0]

def _random_dt(start: datetime, end: datetime) -> datetime:
    span = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, span))

def _load_stations_dataframe(data_dir: Path) -> pd.DataFrame:
    for fname in ("stations_in_net.csv", "stations_merged.csv", "charging_station_metadata.csv"):
        p = data_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            if fname == "charging_station_metadata.csv":
                df = df.rename(columns={"chargeDeviceID": "station_id"})
            break
    else:
        raise FileNotFoundError("No stations CSV found.")
    if not {"latitude","longitude"}.subset(df.columns):
        raise KeyError("Stations CSV lacks 'latitude'/'longitude'")
    return df.dropna(subset=["latitude","longitude"]).reset_index(drop=True)

def _load_ev_metadata(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "EV_Metadata.csv"
    if not p.exists(): raise FileNotFoundError(p)
    return pd.read_csv(p)

def _sample_sessions_per_month(user_type: str, cfg: UserGenConfig) -> int:
    return int(random.choice(cfg.member_sessions_choices if user_type=="Member" else cfg.payg_sessions_choices))

def generate_users(data_dir: str | Path, cfg: UserGenConfig) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if cfg.seed is not None:
        random.seed(cfg.seed); np.random.seed(cfg.seed)

    stations = _load_stations_dataframe(data_dir)
    ev_meta  = _load_ev_metadata(data_dir)
    if "model" not in ev_meta.columns: raise KeyError("EV_Metadata.csv needs 'model' column")
    models = ev_meta["model"].dropna().astype(str).unique().tolist()

    if len(stations) < 2: raise ValueError("Need ≥2 stations for start/dest pairs")

    rows = []
    for i in range(cfg.n_users):
        s_idx = np.random.randint(0, len(stations))
        d_idx = (s_idx + np.random.randint(1, len(stations))) % len(stations)
        s, d = stations.iloc[s_idx], stations.iloc[d_idx]

        user_type = _weighted_choice(cfg.user_type_dist)
        objective = _weighted_choice(cfg.objective_dist)
        include_subscription = (user_type == "Member")
        sessions_per_month = _sample_sessions_per_month(user_type, cfg)

        depart_dt = _random_dt(cfg.depart_start, cfg.depart_end)
        init_soc  = round(random.uniform(cfg.init_soc_min, cfg.init_soc_max), 1)

        rows.append({
            "agent_id": f"U{i+1:03d}",
            "ev_model": random.choice(models),

            "user_type": user_type,
            "objective": objective,

            "origin_lat": float(s["latitude"]),  "origin_lon": float(s["longitude"]),
            "dest_lat":   float(d["latitude"]),  "dest_lon":   float(d["longitude"]),
            "depart_dt":  depart_dt.isoformat(timespec="seconds"),

            "init_soc_pct": init_soc,
            "reserve_soc_pct": float(cfg.reserve_soc_pct),
            "target_soc_pct":  float(cfg.target_soc_pct),

            "max_detour_km":    float(cfg.max_detour_km),
            "top_k_candidates": int(cfg.top_k_candidates),
            "step_horizon_s":   int(cfg.step_horizon_s),

            "include_subscription": bool(include_subscription),
            "sessions_per_month":   int(sessions_per_month),
        })

    return pd.DataFrame(rows)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = UserGenConfig(n_users=args.n_users, seed=args.seed)
    df = generate_users(args.data_dir, cfg)
    out = Path(args.data_dir) / "simulated_users.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} users → {out}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
