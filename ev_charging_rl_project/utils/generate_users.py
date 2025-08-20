# utils/generate_users.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import random
import os
import numpy as np
import pandas as pd

@dataclass
class UserGenConfig:
    n_users: int = 100
    depart_start: datetime = datetime(2025, 8, 12, 7, 0)
    depart_end:   datetime = datetime(2025, 8, 12, 21, 0)

    # legacy bounds (kept for safety)
    init_soc_min: float = 8.0
    init_soc_max: float = 55.0

    # ---- NEW: strong low-SoC bias (defaults tuned to force charging) ----
    low_soc_share: float = 0.85
    low_soc_min: float = 5.0
    low_soc_max: float = 15.0
    high_soc_min: float = 20.0
    high_soc_max: float = 90.0

    # ---- NEW: prefer longer O->D trips to require stops ----
    long_trip_share: float = 0.80
    min_trip_km: float = 12.0  # try to ensure OD >= 12 km

    reserve_soc_pct: float = 12.0
    target_soc_pct: float = 80.0
    max_detour_km: float = 12.0
    top_k_candidates: int = 5
    step_horizon_s: int = 60
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
    # pick the richest station file available
    for fname in ("stations_in_net.csv", "stations_merged.csv", "charging_station_metadata.csv"):
        p = data_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            if fname == "charging_station_metadata.csv":
                df = df.rename(columns={"chargeDeviceID": "station_id"})
            break
    else:
        raise FileNotFoundError("No stations CSV found.")
    if not {"latitude","longitude"}.issubset(df.columns):
        raise KeyError("Stations CSV lacks 'latitude'/'longitude'")
    # ✅ fixed: reset_index(drop=True) instead of index=True
    return df.dropna(subset=["latitude","longitude"]).reset_index(drop=True)


def _load_ev_metadata(data_dir: Path) -> pd.DataFrame:
    p = Path(data_dir) / "EV_Metadata.csv"
    return pd.read_csv(p)

def _sample_sessions_per_month(user_type: str, cfg: UserGenConfig) -> int:
    if user_type == "Member":
        return int(random.choice(cfg.member_sessions_choices))
    return int(random.choice(cfg.payg_sessions_choices))

# --- haversine in km ---
def _hav_km(lat1, lon1, lat2, lon2) -> float:
    from math import radians, sin, cos, asin, sqrt
    R = 6371.0088
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def generate_users(data_dir: str | Path, cfg: UserGenConfig) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if cfg.seed is not None:
        random.seed(cfg.seed); np.random.seed(cfg.seed)

    stations = _load_stations_dataframe(data_dir)
    ev_meta  = _load_ev_metadata(data_dir)
    if "model" not in ev_meta.columns:
        raise KeyError("EV_Metadata.csv needs 'model' column")
    models = ev_meta["model"].dropna().astype(str).unique().tolist()

    if len(stations) < 2:
        raise ValueError("Need ≥2 stations for start/dest pairs")

    # Preload arrays for quick distance checks
    lats = stations["latitude"].astype(float).to_numpy()
    lons = stations["longitude"].astype(float).to_numpy()
    n = len(stations)

    rows = []
    for i in range(cfg.n_users):
        # pick origin
        s_idx = np.random.randint(0, n)
        s_lat, s_lon = float(lats[s_idx]), float(lons[s_idx])

        # choose destination with a preference for long trips
        want_long = (random.random() < float(cfg.long_trip_share))
        if want_long:
            # compute Haversine to all, filter by min_trip_km, exclude self
            dists = np.array([_hav_km(s_lat, s_lon, float(lats[j]), float(lons[j])) for j in range(n)])
            candidates = np.where((dists >= float(cfg.min_trip_km)) & (np.arange(n) != s_idx))[0]
            if candidates.size == 0:
                # fallback: farthest different index
                d_idx = int(np.argsort(dists)[-1])
                if d_idx == s_idx and n > 1:
                    d_idx = int(np.argsort(dists)[-2])
            else:
                # pick randomly among the qualifying set
                d_idx = int(np.random.choice(candidates))
        else:
            # random dest ≠ origin
            d_idx = (s_idx + np.random.randint(1, n)) % n

        s, d = stations.iloc[s_idx], stations.iloc[d_idx]

        user_type = _weighted_choice(cfg.user_type_dist)
        objective = _weighted_choice(cfg.objective_dist)
        include_subscription = (user_type == "Member")
        sessions_per_month = _sample_sessions_per_month(user_type, cfg)

        depart_dt = _random_dt(cfg.depart_start, cfg.depart_end)

        # --- strong low-SoC mix: 85% in [5,15), 15% in [20,90] by default ---
        use_low = (random.random() < float(getattr(cfg, "low_soc_share", 0.0)))
        if use_low:
            lo, hi = float(cfg.low_soc_min), float(cfg.low_soc_max)
        else:
            lo, hi = float(cfg.high_soc_min), float(cfg.high_soc_max)
        if not (0.0 <= lo < hi <= 100.0):
            lo, hi = float(cfg.init_soc_min), float(cfg.init_soc_max)
        init_soc  = round(random.uniform(lo, hi), 1)

        rows.append({
            "agent_id": f"U{i+1:03d}",
            "ev_model": random.choice(models),

            "user_type": user_type,
            "objective": objective,
            "include_subscription": include_subscription,
            "sessions_per_month": sessions_per_month,

            "start_lat": float(s["latitude"]),
            "start_lon": float(s["longitude"]),
            "dest_lat": float(d["latitude"]),
            "dest_lon": float(d["longitude"]),
            "depart_datetime": depart_dt.isoformat(timespec="seconds"),

            "start_soc_pct": init_soc,
            "reserve_soc_pct": float(cfg.reserve_soc_pct),
            "target_soc_pct": float(cfg.target_soc_pct),

            "max_detour_km": float(cfg.max_detour_km),
            "top_k_candidates": int(cfg.top_k_candidates),
            "step_horizon_s": int(cfg.step_horizon_s),
        })

    return pd.DataFrame(rows)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-users", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    # SoC mix (CLI flags or env fallbacks)
    ap.add_argument("--low-soc-share", type=float, default=float(os.getenv("EV_LOW_SOC_SHARE", "0.85")))
    ap.add_argument("--low-soc-min",   type=float, default=float(os.getenv("EV_LOW_SOC_MIN", "5")))
    ap.add_argument("--low-soc-max",   type=float, default=float(os.getenv("EV_LOW_SOC_MAX", "15")))
    ap.add_argument("--high-soc-min",  type=float, default=float(os.getenv("EV_HIGH_SOC_MIN", "20")))
    ap.add_argument("--high-soc-max",  type=float, default=float(os.getenv("EV_HIGH_SOC_MAX", "90")))

    # Trip length controls
    ap.add_argument("--long-trip-share", type=float, default=float(os.getenv("EV_LONG_TRIP_SHARE", "0.8")))
    ap.add_argument("--min-trip-km",     type=float, default=float(os.getenv("EV_MIN_TRIP_KM", "12")))

    args = ap.parse_args()
    cfg = UserGenConfig(
        n_users=args.n_users, seed=args.seed,
        low_soc_share=args.low_soc_share,
        low_soc_min=args.low_soc_min, low_soc_max=args.low_soc_max,
        high_soc_min=args.high_soc_min, high_soc_max=args.high_soc_max,
        long_trip_share=args.long_trip_share, min_trip_km=args.min_trip_km,
    )
    df = generate_users(args.data_dir, cfg)
    out = Path(args.data_dir) / "simulated_users.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} users → {out}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()
