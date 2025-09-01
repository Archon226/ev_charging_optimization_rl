# utils/generate_users.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# -------------------- geo utils --------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * R * math.asin(math.sqrt(a)))


# -------------------- config --------------------
@dataclass
class UserGenConfig:
    data_dir: str = "data"                        # where EV_Metadata.csv lives
    net_path: str = "london_inner.net.xml"       # SUMO net; constrains to Inner London
    n_users: int = 2000
    seed: int = 42

    # EV/model sampling
    ev_model_column: str = "model"               # EV_Metadata.csv column
    
    # Objective mixing (weights map to cost, time, hybrid); default is balanced
    objective_weights: tuple[float, float, float] = (1/3, 1/3, 1/3)

    # SoC bias (no must-charge enforcement)
    low_soc_share: float = 0.9
    low_soc_min: float = 3.0
    low_soc_max: float = 12.0
    high_soc_min: float = 12.0
    high_soc_max: float = 25.0
    reserve_soc_pct: float = 5.0                 # planning reserve (%), only for analytics

    # Trip length bias (km) – no must-charge check here
    long_trip_share: float = 0.9
    min_trip_km: float = 18.0
    short_trip_min_km: float = 8.0
    short_trip_max_km: float = 16.0

    # Time windows to exercise pricing
    weekend_share: float = 0.30
    peak_share: float = 0.50                     # of the weekday share
    peak_hours: str = "07:30-10:00;16:30-19:30"
    off_peak_hours: str = "00:00-06:30;10:00-16:30;19:30-23:59"

    # Commercial params (passed to pricing)
    user_types: Tuple[str, str] = ("Payg", "Member")
    member_share: float = 0.35
    include_subscription_for_members: bool = True
    sessions_per_month: int = 20

    # Env knobs (pass-through)
    objective_choices: Tuple[str, ...] = ("cost", "time", "hybrid")
    max_detour_km: float = 2.0
    top_k_candidates: int = 8
    step_horizon_s: int = 5


# --- path resolver for SUMO net ---
def _resolve_net_path(net_path: str | Path) -> Path:
    """
    Try common locations so we actually find the SUMO net regardless of CWD.
    """
    net_path = Path(net_path)
    if net_path.exists():
        return net_path
    here = Path(__file__).resolve()
    candidates = [
        net_path,                                     # as given
        here.parent / "london_inner.net.xml",         # utils/../london_inner.net.xml
        here.parent.parent / "london_inner.net.xml",  # project root / london_inner.net.xml
        here.parent.parent / "ev_charging_rl_project" / "london_inner.net.xml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return net_path


# -------------------- SUMO nodes (robust XY→LonLat) --------------------
def _parse_sumo_nodes(net_path: Path) -> np.ndarray:
    """
    Load SUMO net and return an array of (lat, lon) for all nodes, using a robust
    XY->Lon/Lat converter (mirrors env/sumo_runner approach).
    Returns: np.ndarray shape (N, 2) with columns [lat, lon].
    """
    try:
        import sumolib  # type: ignore
    except Exception:
        return np.empty((0, 2), dtype=float)

    if not net_path.exists():
        return np.empty((0, 2), dtype=float)

    # Read the net
    try:
        net = sumolib.net.readNet(str(net_path))
    except Exception:
        return np.empty((0, 2), dtype=float)

    # Build robust XY->LonLat converter
    xy2ll = None
    # Preferred: LocationConverter
    try:
        lc = net.getLocationConverter()
        if lc is not None and hasattr(lc, "convertXY2LonLat"):
            def _xy2ll(x, y):
                lon, lat = lc.convertXY2LonLat(float(x), float(y))
                return float(lat), float(lon)
            _ = _xy2ll(0.0, 0.0)  # smoke
            xy2ll = _xy2ll
    except Exception:
        xy2ll = None

    # Fallback: direct method on net
    if xy2ll is None:
        for meth in ("convertXY2LonLat",):
            if hasattr(net, meth):
                fn = getattr(net, meth)
                try:
                    def _wrap(x, y, fn=fn):
                        lon, lat = fn(float(x), float(y))
                        return float(lat), float(lon)
                    _ = _wrap(0.0, 0.0)
                    xy2ll = _wrap
                    break
                except Exception:
                    continue

    if xy2ll is None:
        return np.empty((0, 2), dtype=float)

    # Collect nodes and convert
    lats: List[float] = []
    lons: List[float] = []
    try:
        for nd in net.getNodes():
            try:
                x, y = nd.getCoord()  # SUMO coords (meters)
                lat, lon = xy2ll(x, y)
                # sanity guard
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    lats.append(lat)
                    lons.append(lon)
            except Exception:
                continue
    except Exception:
        return np.empty((0, 2), dtype=float)

    if not lats:
        return np.empty((0, 2), dtype=float)

    arr = np.column_stack([np.array(lats, dtype=float), np.array(lons, dtype=float)])

    # Optional: subsample if extremely dense for speed
    if len(arr) > 20000:
        idx = np.random.choice(len(arr), size=20000, replace=False)
        arr = arr[idx]

    return arr


# -------------------- helpers --------------------
def _sample_point_from_bbox(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    min_lat, min_lon, max_lat, max_lon = bbox
    return (random.uniform(min_lat, max_lat), random.uniform(min_lon, max_lon))


def _pick_origin_destination(nodes_arr: np.ndarray,
                             bbox: Tuple[float, float, float, float],
                             min_km: float) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """
    Pick two points (lat, lon) at least min_km apart.
    Prefer SUMO nodes (vectorized pairing); no bbox fallback if nodes_arr provided.
    """
    if nodes_arr.size > 0:
        # vectorized: pick A, then sample a batch of Bs and choose any that meets distance
        a = nodes_arr[np.random.randint(len(nodes_arr))]
        for _ in range(20):
            bs = nodes_arr[np.random.randint(len(nodes_arr), size=256)]
            d_lat = np.radians(bs[:, 0] - a[0])
            d_lon = np.radians(bs[:, 1] - a[1])
            p1 = math.radians(a[0])
            p2 = np.radians(bs[:, 0])
            a_h = np.sin(d_lat / 2) ** 2 + np.cos(p1) * np.cos(p2) * (np.sin(d_lon / 2) ** 2)
            dists = 2 * 6371.0088 * np.arcsin(np.sqrt(a_h))
            ok = np.where(dists >= float(min_km))[0]
            if ok.size > 0:
                j = int(np.random.choice(ok))
                b = bs[j]
                return (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), float(dists[j])

        # If we didn’t find a distant-enough pair quickly, just pick two distinct nodes
        # (distance may be below min_km; env/episode can still be valid)
        idxs = np.random.choice(len(nodes_arr), size=2, replace=False)
        a, b = nodes_arr[idxs[0]], nodes_arr[idxs[1]]
        return (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), haversine_km(a[0], a[1], b[0], b[1])

    # No nodes_arr → last resort (shouldn’t happen; we hard-require net)
    for _ in range(200):
        a = _sample_point_from_bbox(bbox)
        b = _sample_point_from_bbox(bbox)
        d = haversine_km(a[0], a[1], b[0], b[1])
        if d >= float(min_km):
            return a, b, d
    a = _sample_point_from_bbox(bbox)
    b = _sample_point_from_bbox(bbox)
    return a, b, haversine_km(a[0], a[1], b[0], b[1])


# -------------------- time windows --------------------
def _parse_windows(txt: str) -> Tuple[Tuple[int, int], ...]:
    if not isinstance(txt, str) or "-" not in txt:
        return tuple()

    def to_min(s: str) -> int:
        hh, mm = s.split(":")
        return int(hh) * 60 + int(mm)

    wins = []
    for chunk in txt.split(";"):
        chunk = chunk.strip()
        if "-" in chunk:
            a, b = [x.strip() for x in chunk.split("-")]
            wins.append((to_min(a), to_min(b)))
    return tuple(wins)


def _force_dt_into_window(dt: datetime, windows: Tuple[Tuple[int, int], ...]) -> datetime:
    if not windows:
        return dt
    w = random.choice(list(windows))
    start_m, end_m = w
    if end_m <= start_m:
        # wrap across midnight
        if random.random() < (1440 - start_m) / (1440 - start_m + end_m):
            minute = random.randint(start_m, 1439)
        else:
            minute = random.randint(0, end_m)
    else:
        minute = random.randint(start_m, max(start_m + 1, end_m) - 1)
    return dt.replace(hour=minute // 60, minute=minute % 60, second=0, microsecond=0)


# -------------------- EV meta --------------------
def _load_ev_meta(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "EV_Metadata.csv")
    need = {"model", "battery_kWh", "avg_consumption_Wh_per_km"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"EV_Metadata.csv missing columns: {sorted(missing)}")
    return df


def _choose_model(ev_meta: pd.DataFrame) -> str:
    return str(random.choice(ev_meta["model"].tolist()))


# -------------------- main API --------------------
def generate_users(cfg: UserGenConfig, out_csv: Optional[str] = None) -> pd.DataFrame:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_dir = Path(cfg.data_dir)
    ev_meta = _load_ev_meta(data_dir)

    # Resolve and load SUMO nodes; REQUIRE a valid net so trips are inside it
    resolved_net = _resolve_net_path(cfg.net_path)
    nodes_arr = _parse_sumo_nodes(resolved_net)
    if nodes_arr.size == 0:
        raise FileNotFoundError(
            f"SUMO net not found or has no geo nodes: {resolved_net}. "
            "Trips must be inside the network; fix --net_path."
        )

    # Derive bbox from the net nodes (only used for last-resort fallbacks)
    min_lat, max_lat = float(nodes_arr[:, 0].min()), float(nodes_arr[:, 0].max())
    min_lon, max_lon = float(nodes_arr[:, 1].min()), float(nodes_arr[:, 1].max())
    bbox = (min_lat, min_lon, max_lat, max_lon)

    # Precompute time windows
    peak_wins = _parse_windows(cfg.peak_hours)
    off_wins = _parse_windows(cfg.off_peak_hours)

    rows: List[Dict] = []
    base_date = datetime.now().replace(hour=9, minute=0, second=0, microsecond=0)

    for uid in range(cfg.n_users):
        ev_model = _choose_model(ev_meta)

        # user_type & subscription
        if random.random() < cfg.member_share:
            user_type = "Member"
            include_subscription = cfg.include_subscription_for_members
        else:
            user_type = "Payg"
            include_subscription = False

        # objective
        # objective (weighted random over cost, time, hybrid)
        _choices = ("cost", "time", "hybrid")
        _weights = getattr(cfg, "objective_weights", (1/3, 1/3, 1/3))
        # guard: normalize and fallback if malformed
        if not isinstance(_weights, (tuple, list)) or len(_weights) != 3:
            _weights = (1/3, 1/3, 1/3)
        _sum = float(_weights[0] + _weights[1] + _weights[2])
        if _sum <= 0:
            _weights = (1/3, 1/3, 1/3)
        else:
            _weights = (float(_weights[0])/_sum, float(_weights[1])/_sum, float(_weights[2])/_sum)
        objective = random.choices(_choices, weights=_weights, k=1)[0]


        # SoC bias (no must-charge enforcement)
        if random.random() < cfg.low_soc_share:
            start_soc = random.uniform(cfg.low_soc_min, cfg.low_soc_max)
        else:
            start_soc = random.uniform(cfg.high_soc_min, cfg.high_soc_max)
        reserve_soc = float(cfg.reserve_soc_pct)

        # time: spread across week & force windows
        dt = base_date + timedelta(days=random.randint(0, 6))
        if random.random() < cfg.weekend_share:
            # shift to Sat/Sun
            dow = dt.weekday()
            shift = (5 - dow) % 7 if random.random() < 0.5 else (6 - dow) % 7
            dt = dt + timedelta(days=shift)
        else:
            if random.random() < cfg.peak_share:
                dt = _force_dt_into_window(dt, peak_wins)
            else:
                dt = _force_dt_into_window(dt, off_wins)

        # Trip length bias (single-pass; no must-charge)
        desired_min_km = cfg.min_trip_km if (random.random() < cfg.long_trip_share) else random.uniform(
            cfg.short_trip_min_km, cfg.short_trip_max_km
        )
        (s_lat, s_lon), (e_lat, e_lon), trip_km = _pick_origin_destination(nodes_arr, bbox, desired_min_km)

        # Analytics helpers (no “must_charge” flag here)
        row = ev_meta.loc[ev_meta["model"].astype(str) == ev_model].iloc[0]
        batt_kwh = float(row["battery_kWh"])
        kwh_per_km = float(row["avg_consumption_Wh_per_km"]) / 1000.0
        available_kwh = batt_kwh * max(0.0, (start_soc - reserve_soc)) / 100.0
        required_kwh = kwh_per_km * trip_km

        rows.append({
            "user_id": uid,
            "ev_model": ev_model,
            "user_type": user_type,  # "Payg" or "Member"
            "include_subscription": int(bool(include_subscription)),
            "sessions_per_month": int(cfg.sessions_per_month),

            "start_lat": round(s_lat, 6),
            "start_lon": round(s_lon, 6),
            "end_lat": round(e_lat, 6),
            "end_lon": round(e_lon, 6),
            "depart_datetime": dt.isoformat(timespec="minutes"),

            "start_soc_pct": round(start_soc, 2),
            "reserve_soc_pct": round(reserve_soc, 2),

            "objective": objective,  # "cost" | "time" | "hybrid"
            "max_detour_km": float(cfg.max_detour_km),
            "top_k_candidates": int(cfg.top_k_candidates),
            "step_horizon_s": int(cfg.step_horizon_s),

            # Analytics (no must-charge decision here)
            "trip_km": round(trip_km, 3),
            "est_required_kwh": round(required_kwh, 3),
            "available_kwh_at_start": round(available_kwh, 3),
            "kwh_per_km": round(kwh_per_km, 4),
        })

    df = pd.DataFrame(rows)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    return df


# -------------------- CLI --------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser("Generate simulated users for RL training (no must-charge enforcement)")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--net_path", type=str, default="london_inner.net.xml")
    p.add_argument("--out", type=str, default="data/simulated_users.csv")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--low_soc_share", type=float, default=0.9)
    p.add_argument("--min_trip_km", type=float, default=18.0)
    p.add_argument("--long_trip_share", type=float, default=0.9)
    p.add_argument("--weekend_share", type=float, default=0.3)
    p.add_argument("--peak_share", type=float, default=0.5)
    args = p.parse_args()

    cfg = UserGenConfig(
        data_dir=args.data_dir,
        net_path=args.net_path,
        n_users=args.n,
        seed=args.seed,
        low_soc_share=args.low_soc_share,
        min_trip_km=args.min_trip_km,
        long_trip_share=args.long_trip_share,
        weekend_share=args.weekend_share,
        peak_share=args.peak_share,
    )
    df = generate_users(cfg, out_csv=args.out)
    print(f"Wrote {len(df)} users → {args.out}")
