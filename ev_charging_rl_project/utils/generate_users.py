# utils/generate_users.py
from __future__ import annotations
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_all_data  # use our loader & merged stations

# Distributions (sum to ~1.0)
USER_TYPE_DIST = [("Subscriber", 0.30), ("Payg", 0.60), ("Contactless", 0.10)]
OBJECTIVE_DIST = [("cost", 0.34), ("time", 0.33), ("hybrid", 0.33)]

@dataclass
class UserConfig:
    n_users: int = 50
    depart_start: datetime = datetime(2025, 8, 9, 7, 0)
    depart_end: datetime = datetime(2025, 8, 9, 21, 0)
    min_initial_soc: float = 10.0
    max_initial_soc: float = 70.0
    target_soc: float = 80.0
    reserve_soc: float = 10.0
    seed: Optional[int] = 42  # reproducibility

def _weighted_choice(pairs: List[Tuple[str, float]]) -> str:
    labels, probs = zip(*pairs)
    return random.choices(labels, weights=probs, k=1)[0]

def _random_dt(start: datetime, end: datetime) -> datetime:
    span = (end - start).total_seconds()
    return start + timedelta(seconds=random.uniform(0, span))

def _pick_model_list(ev_metadata: pd.DataFrame) -> List[str]:
    if "model" in ev_metadata.columns:
        col = "model"
    elif "Model" in ev_metadata.columns:
        col = "Model"
    else:
        raise KeyError("EV metadata missing 'model' column (expected 'model' or 'Model').")
    return ev_metadata[col].dropna().astype(str).tolist()

def generate_users(stations_merged: pd.DataFrame, ev_metadata: pd.DataFrame, cfg: UserConfig) -> pd.DataFrame:
    # RNG seed (optional reproducibility)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    models = _pick_model_list(ev_metadata)
    if not models:
        raise ValueError("No EV models found in EV metadata.")

    # Use only valid stations (coords present and company_id present so pricing works later)
    st = stations_merged.copy()
    if "latitude" not in st.columns or "longitude" not in st.columns:
        raise KeyError("stations_merged missing latitude/longitude.")

    st = st.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if "company_id" in st.columns:
        st = st[st["company_id"].notna()].reset_index(drop=True)

    if len(st) < 2:
        raise ValueError("Need at least 2 valid stations to generate trips (start/dest).")

    rows = []
    for i in range(cfg.n_users):
        s_idx = np.random.randint(0, len(st))
        d_idx = np.random.randint(0, len(st))
        if d_idx == s_idx:
            d_idx = (d_idx + 1) % len(st)

        s_row, d_row = st.iloc[s_idx], st.iloc[d_idx]
        depart = _random_dt(cfg.depart_start, cfg.depart_end)

        rows.append({
            "agent_id": f"U{i+1:03d}",
            "model": random.choice(models),
            "user_type": _weighted_choice(USER_TYPE_DIST),
            "objective": _weighted_choice(OBJECTIVE_DIST),
            "initial_soc": round(random.uniform(cfg.min_initial_soc, cfg.max_initial_soc), 1),
            "target_soc": float(cfg.target_soc),
            "reserve_soc": float(cfg.reserve_soc),
            "depart_dt": depart.isoformat(),
            "start_lat": float(s_row["latitude"]),
            "start_lon": float(s_row["longitude"]),
            "dest_lat": float(d_row["latitude"]),
            "dest_lon": float(d_row["longitude"]),
        })

    return pd.DataFrame(rows)

def main():
    data_dir = Path("data")
    ds = load_all_data(data_dir)  # use the same loader the simulator uses

    cfg = UserConfig(n_users=50, seed=42)
    df = generate_users(ds.stations_merged, ds.ev_metadata, cfg)

    out = data_dir / "simulated_users.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} simulated users → {out}")

if __name__ == "__main__":
    main()
