# test_single_charge.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import math

import pandas as pd

from utils.data_loader import load_all_ready
from utils.pricing import PricingCatalog
from utils.charging_curves import EVPowerModel
from utils.session_planner import get_candidate_pairs, evaluate_session


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))


def main():
    data_dir = Path("data")

    # 1) Load pre-indexed bundle (stations, EV caps, pricing index, curves)
    bundle = load_all_ready(data_dir, strict=True)

    # 2) Pricing + EV power model
    pricing = PricingCatalog.from_index(bundle["pricing_index"], strict=True)
    ev_meta = pd.read_csv(data_dir / "EV_Metadata.csv")
    ev_curve = pd.read_csv(data_dir / "EV_Charging_Curve_Data.csv")
    ev_power = EVPowerModel(ev_meta, ev_curve)

    # 3) Pick one user (or change idx) from generated users
    users = pd.read_csv(data_dir / "simulated_users.csv")
    assert len(users) > 0, "No users found; run utils.generate_users first."
    u = users.iloc[0]

    ev_model = str(u["ev_model"])
    start_lat, start_lon = float(u["start_lat"]), float(u["start_lon"])
    depart_dt = datetime.fromisoformat(str(u["depart_datetime"]))
    start_soc = float(u["start_soc_pct"])
    target_soc = 80.0  # simple fixed target for this sanity test

    print(f"\nUser #{int(u['user_id'])} | EV: {ev_model}")
    print(f"Origin: ({start_lat:.6f}, {start_lon:.6f})  Depart: {depart_dt.isoformat(timespec='minutes')}")
    print(f"SoC: {start_soc:.1f}% -> {target_soc:.1f}%\n")

    # 4) Pick nearest stations (we use the enriched connectors to get lat/lon)
    ce = bundle["station_connectors_enriched"]
    station_pts = ce.groupby("station_id", as_index=False).first()[["station_id", "lat", "lon"]]
    station_pts["dist_km"] = station_pts.apply(
        lambda r: haversine_km(start_lat, start_lon, float(r["lat"]), float(r["lon"])), axis=1
    )
    station_pts = station_pts.sort_values("dist_km").head(60)  # shortlist

    # 5) Build viable (station, category) pairs using EV & station capabilities
    scaps = bundle["station_capabilities"]
    ev_caps = bundle["ev_capabilities"]
    station_ids_near = station_pts["station_id"].astype(str).tolist()
    pairs = get_candidate_pairs(ev_model, station_ids_near, scaps, ev_caps)

    # Keep only pairs whose station is among the N closest (to avoid huge printouts)
    keep_sids = set(station_pts.head(8)["station_id"].astype(str).tolist())
    pairs = [(sid, cat) for (sid, cat) in pairs if sid in keep_sids]

    if not pairs:
        raise RuntimeError("No viable (station, category) pairs near the origin for this EV.")

    print(f"Evaluating {len(pairs)} candidate (station, category) pairs near the origin...\n")

    # 6) Evaluate each candidate with consistent charging-time + pricing
    #    We request a detailed pricing breakdown once here (handy for debugging).
    rows = []
    for sid, cat in pairs:
        res = evaluate_session(
            ev_model=ev_model,
            station_id=sid,
            category=cat,
            start_soc_pct=start_soc,
            target_soc_pct=target_soc,
            when=depart_dt,
            pricing=pricing,
            ev_power=ev_power,
            station_capabilities=scaps,
            user_type=str(u["user_type"]),
            efficiency=1.0,  # price on battery energy in this simple test
            include_subscription=bool(int(u["include_subscription"])),
            sessions_per_month=int(u["sessions_per_month"]),
            idle_minutes=0.0,
            detailed_pricing_breakdown=True,
        )
        rows.append(res)

    # 7) Pretty print results sorted by price then time
    rows_sorted = sorted(rows, key=lambda x: (x.total_price, x.session_minutes))
    print("Top candidates (by price):")
    for r in rows_sorted[:6]:
        print(
            f"  [{r.station_id} | {r.category}] "
            f"£{r.total_price:.2f} | {r.session_minutes:.1f} min | "
            f"{r.delivered_kwh_batt:.2f} kWh | unit={r.unit_price_source}"
        )

    # Also show top by time
    rows_time = sorted(rows, key=lambda x: (x.session_minutes, x.total_price))
    print("\nTop candidates (by time):")
    for r in rows_time[:6]:
        print(
            f"  [{r.station_id} | {r.category}] "
            f"{r.session_minutes:.1f} min | £{r.total_price:.2f} | "
            f"{r.delivered_kwh_batt:.2f} kWh | unit={r.unit_price_source}"
        )

    print("\nOK: single-charge sanity check complete.\n")


if __name__ == "__main__":
    main()
