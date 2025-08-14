# rl/episodes.py
import os, pandas as pd
from datetime import datetime
from simulator.trip_simulator import TripPlan

OBJ_MAP = {"time":"time", "cost":"cost", "hybrid":"hybrid"}

def load_user_plans(csv_path: str) -> list[TripPlan]:
    df = pd.read_csv(csv_path)
    plans = []
    for _, r in df.iterrows():
        plans.append(TripPlan(
            agent_id=str(r.get("agent_id", f"U{_}")),
            model=str(r.get("ev_model", r.get("model"))),
            user_type=str(r.get("user_type","Payg")).title(),
            objective=OBJ_MAP.get(str(r.get("objective","hybrid")).lower(),"hybrid"),
            start_lat=float(r["start_lat"]), start_lon=float(r["start_lon"]),
            dest_lat=float(r["dest_lat"]),   dest_lon=float(r["dest_lon"]),
            depart_dt=_parse_dt(r.get("depart_dt", None)),
            init_soc_pct=float(r.get("init_soc_pct", 20)),
            reserve_soc_pct=float(r.get("reserve_soc_pct", 10)),
            target_soc_pct=float(r.get("target_soc_pct", 80)),
            max_detour_km=float(r.get("max_detour_km", 10)),
            top_k_candidates=int(r.get("top_k_candidates", 5)),
            step_horizon_s=int(r.get("step_horizon_s", 180)),
        ))
    return plans

def _parse_dt(x):
    if pd.isna(x) or not x:  # default to “now”
        return datetime.now()
    try:
        return datetime.fromisoformat(str(x))
    except Exception:
        # fallback common fmt "YYYY-MM-DD HH:MM"
        return datetime.strptime(str(x), "%Y-%m-%d %H:%M")
