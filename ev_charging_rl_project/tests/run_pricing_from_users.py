from __future__ import annotations
import math
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_all_data
from utils.generate_users import generate_users, UserConfig
from env.charger import charger_from_row

DATA_DIR = Path("data")

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def load_or_make_users(ds):
    users_path = DATA_DIR / "simulated_users.csv"
    if users_path.exists():
        return pd.read_csv(users_path)
    # make a small set if missing
    cfg = UserConfig(n_users=25, seed=123)
    df = generate_users(ds.stations_merged, ds.ev_metadata, cfg)
    df.to_csv(users_path, index=False)
    print(f"[info] created {len(df)} simulated users → {users_path}")
    return df

def pick_nearest_priced_charger(ds, lat, lon):
    df = ds.stations_merged.copy()
    df = df[df["company_id"].notna()].copy()
    df["dist_km"] = haversine_km(lat, lon, df["latitude"], df["longitude"])
    df = df.sort_values("dist_km")
    return df.iloc[0] if not df.empty else None

def main(limit:int=20, kwh:float=18.0, minutes:float=40.0, include_subscription:bool=False):
    ds = load_all_data(DATA_DIR)
    users = load_or_make_users(ds)

    # sample to limit count
    users = users.head(limit).copy()

    rows = []
    failures = 0
    for _, u in users.iterrows():
        start_dt = datetime.fromisoformat(str(u["depart_dt"]))
        lat, lon = float(u["start_lat"]), float(u["start_lon"])
        user_type = str(u["user_type"])
        obj = str(u.get("objective", "cost"))

        cand = pick_nearest_priced_charger(ds, lat, lon)
        if cand is None:
            failures += 1
            rows.append({
                "agent_id": u["agent_id"], "status": "no_charger",
                "msg": "No priced charger near start", "user_type": user_type, "objective": obj
            })
            continue

        ch = charger_from_row(cand, ds.pricing_catalog)
        try:
            breakdown = ch.estimate_session_cost(
                kwh=kwh,
                start_dt=start_dt,
                user_type=user_type,
                session_minutes=minutes,
                include_subscription=include_subscription,
            )
            rows.append({
                "agent_id": u["agent_id"],
                "status": "ok",
                "company_id": breakdown["company_id"],
                "operator": breakdown["operator"],
                "charger_type": breakdown["charger_type"],
                "unit_price": breakdown["unit_price"],
                "price_source": breakdown.get("unit_source", breakdown.get("price_source", "")),
                "energy_cost": breakdown["energy_cost"],
                "connection_fee": breakdown["connection_fee"],
                "infrastructure_fee": breakdown["infrastructure_fee"],
                "idle_fee": breakdown.get("idle_fee", 0.0) if "idle_fee" in breakdown else breakdown.get("idle_cost", 0.0),
                "overstay_fee": breakdown.get("overstay_fee", 0.0),
                "subscription_cost": breakdown.get("subscription_cost", 0.0),
                "min_charge_topup": breakdown.get("min_charge_topup", breakdown.get("minimum_applied", 0.0)),
                "total_cost": breakdown["total_cost"],
                "dist_km": float(cand["dist_km"]),
                "user_type": user_type,
                "objective": obj,
                "kwh": kwh,
                "minutes": minutes,
                "depart_dt": u["depart_dt"],
            })
        except Exception as e:
            failures += 1
            rows.append({
                "agent_id": u["agent_id"], "status": "error",
                "msg": str(e), "user_type": user_type, "objective": obj
            })

    out = pd.DataFrame(rows)
    out_path = Path("tests") / "pricing_users_results.csv"
    out.to_csv(out_path, index=False)

    ok = (out["status"] == "ok").sum()
    print(f"[done] users processed={len(users)}  ok={ok}  failures={failures}")
    if ok:
        preview_cols = ["agent_id","operator","company_id","charger_type","unit_price","price_source","energy_cost","connection_fee","infrastructure_fee","idle_fee","overstay_fee","subscription_cost","min_charge_topup","total_cost"]
        print(out.loc[out["status"]=="ok", preview_cols].head(10).to_string(index=False))
        print(f"[saved] {out_path}")

if __name__ == "__main__":
    # tweak defaults if you want a different dummy session size
    main(limit=20, kwh=18.0, minutes=40.0, include_subscription=False)
