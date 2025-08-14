from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple
import random
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class UserGenConfig:
    n_users: int = 100
    # time window to sample departures from
    depart_start: datetime = datetime(2025, 8, 12, 7, 0)
    depart_end:   datetime = datetime(2025, 8, 12, 21, 0)

    # SOC policy
    init_soc_min: float = 8.0
    init_soc_max: float = 55.0
    reserve_soc_pct: float = 12.0
    target_soc_pct: float = 80.0

    # Candidate search knobs (used by TripPlan)
    max_detour_km: float = 12.0
    top_k_candidates: int = 5
    step_horizon_s: int = 180

    # User type / objective distributions
    # Keep types consistent with pricing engine ("Payg", "Member")
    user_type_dist: List[Tuple[str, float]] = (("Payg", 0.70), ("Member", 0.30))
    objective_dist: List[Tuple[str, float]] = (("time", 0.33), ("cost", 0.34), ("hybrid", 0.33))

    # Membership behaviour
    member_sessions_choices: List[int] = (4, 8, 12)   # typical monthly bundles
    payg_sessions_choices:   List[int] = (0,)         # payg has 0 by default

    seed: Optional[int] = 42  # reproducibility

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _weighted_choice(pairs: List[Tuple[str, float]]) -> str:
    labels, probs = zip(*pairs)
    return random.choices(labels, weights=probs, k=1)[0]

def _random_dt(start: datetime, end: datetime) -> datetime:
    span = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, span))

def _pick_model_column(ev_metadata: pd.DataFrame) -> str:
    for cand in ("model", "Model"):
        if cand in ev_metadata.columns:
            return cand
    raise KeyError("EV metadata missing 'model' column (expected 'model' or 'Model').")

def _load_stations_dataframe(data_dir: Path) -> pd.DataFrame:
    """
    Prefer stations filtered to your SUMO net if present,
    else fall back to the canonical merged stations.
    """
    in_net = data_dir / "stations_in_net.csv"
    merged = data_dir / "stations_merged.csv"
    meta   = data_dir / "charging_station_metadata.csv"

    if in_net.exists():
        df = pd.read_csv(in_net)
    elif merged.exists():
        df = pd.read_csv(merged)
    else:
        # last resort: raw metadata (fewer fields, but still gives start/dest coords)
        df = pd.read_csv(meta).rename(columns={"chargeDeviceID": "station_id"})
        if "deviceNetworks" in df.columns and "company_id" not in df.columns:
            df["company_id"] = -1  # placeholder; pricing may not work for these rows

    # require lat/lon
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise KeyError("Stations CSV lacks 'latitude'/'longitude' columns.")
    return df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

def _load_ev_metadata(data_dir: Path) -> pd.DataFrame:
    p = data_dir / "EV_Metadata.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    return pd.read_csv(p)

def _sample_sessions_per_month(user_type: str, cfg: UserGenConfig) -> int:
    if user_type == "Member":
        return int(random.choice(cfg.member_sessions_choices))
    return int(random.choice(cfg.payg_sessions_choices))

# ------------------------------------------------------------------
# Main generator
# ------------------------------------------------------------------

def generate_users(data_dir: str | Path, cfg: UserGenConfig) -> pd.DataFrame:
    """
    Create a simulated_users table aligned with rl/episodes.py expectations and
    the simulator/pricing code paths.
    """
    data_dir = Path(data_dir)
    if cfg.seed is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

    stations = _load_stations_dataframe(data_dir)
    ev_meta  = _load_ev_metadata(data_dir)
    model_col = _pick_model_column(ev_meta)
    models = ev_meta[model_col].dropna().astype(str).unique().tolist()
    if len(models) == 0:
        raise ValueError("No EV models found in EV_Metadata.csv")

    if len(stations) < 2:
        raise ValueError("Need at least 2 stations to form start/destination pairs.")

    rows = []
    for i in range(cfg.n_users):
        # pick distinct start/dest stations
        s_idx = np.random.randint(0, len(stations))
        d_idx = np.random.randint(0, len(stations))
        if d_idx == s_idx:
            d_idx = (d_idx + 1) % len(stations)

        s = stations.iloc[s_idx]
        d = stations.iloc[d_idx]

        user_type = _weighted_choice(cfg.user_type_dist)             # "Payg" | "Member"
        objective = _weighted_choice(cfg.objective_dist)             # "time"|"cost"|"hybrid"
        include_subscription = (user_type == "Member")
        sessions_per_month = _sample_sessions_per_month(user_type, cfg)

        depart_dt = _random_dt(cfg.depart_start, cfg.depart_end)
        init_soc  = round(random.uniform(cfg.init_soc_min, cfg.init_soc_max), 1)

        rows.append({
            "agent_id": f"U{i+1:03d}",
            # Episodes loader accepts either "ev_model" or "model"; we provide both for clarity
            "ev_model": random.choice(models),
            "model":    random.choice(models),

            "user_type": user_type,
            "objective": objective,  # lowercase (episodes.py maps/accepts lower)

            "start_lat": float(s["latitude"]),
            "start_lon": float(s["longitude"]),
            "dest_lat":  float(d["latitude"]),
            "dest_lon":  float(d["longitude"]),

            "depart_dt": depart_dt.isoformat(timespec="seconds"),

            # SOC policy (names matched to rl/episodes.py / TripPlan)
            "init_soc_pct":    init_soc,
            "reserve_soc_pct": float(cfg.reserve_soc_pct),
            "target_soc_pct":  float(cfg.target_soc_pct),

            # Candidate search knobs (TripPlan fields)
            "max_detour_km":   float(cfg.max_detour_km),
            "top_k_candidates": int(cfg.top_k_candidates),
            "step_horizon_s":   int(cfg.step_horizon_s),

            # Pricing membership context used by your pricing engine
            "include_subscription": bool(include_subscription),
            "sessions_per_month":   int(sessions_per_month),
        })

    df = pd.DataFrame(rows)
    return df

# ------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Folder containing stations_*.csv & EV_Metadata.csv")
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
