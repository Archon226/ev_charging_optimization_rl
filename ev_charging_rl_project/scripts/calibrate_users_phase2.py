# scripts/calibrate_users_phase2.py
"""
Phase 2 — user calibration (Inner London PPO)
- Columns used: trip_km, start_soc_pct, reserve_soc_pct, kwh_per_km,
                available_kwh_at_start, est_required_kwh
- Adds --target (default 0.45) and finishes with a precise top-up step to ensure
  need>=1-charge rate falls within [TARGET_MIN, TARGET_MAX], biasing toward --target.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# Band you want to end up in
TARGET_MIN = 0.30
TARGET_MAX = 0.60

# Defaults
DEFAULT_TARGET = 0.45
RNG_SEED = 42

# Realistic consumption (kWh/km)
KWH_PER_KM_MIN = 0.17
KWH_PER_KM_MAX = 0.20

# Percent ranges (points 0–100)
RESERVE_SOC_PCT_RANGE = (10.0, 15.0)
START_OVER_RESERVE_MIN = 2.0
START_OVER_RESERVE_MAX = 22.0
START_PCT_CAP = 95.0

# Column names
DISTANCE_COL = "trip_km"
START_COL = "start_soc_pct"
RESERVE_COL = "reserve_soc_pct"
AVAIL_COL = "available_kwh_at_start"
REQUIRED_COL = "est_required_kwh"
CONSUMPTION_COL = "kwh_per_km"

def ensure_required_cols(df: pd.DataFrame) -> None:
    required = [DISTANCE_COL, START_COL, RESERVE_COL, AVAIL_COL, REQUIRED_COL, CONSUMPTION_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in users CSV")

def infer_battery_kwh(df: pd.DataFrame) -> pd.Series:
    denom = (df[START_COL].astype(float) - df[RESERVE_COL].astype(float)) / 100.0
    batt = df[AVAIL_COL].astype(float) / denom.replace(0, np.nan)
    batt = batt.replace([np.inf, -np.inf], np.nan)
    median = float(np.nanmedian(batt)) if np.isfinite(np.nanmedian(batt)) else 60.0
    batt = batt.fillna(median).clip(20.0, 120.0)
    return batt

def compute_need_mask(df: pd.DataFrame) -> np.ndarray:
    return (df[REQUIRED_COL].astype(float) > df[AVAIL_COL].astype(float)).to_numpy()

def seed_consumption(df: pd.DataFrame, rng: np.random.Generator) -> None:
    bad = (~np.isfinite(df[CONSUMPTION_COL])) | (df[CONSUMPTION_COL] <= 0.05) | (df[CONSUMPTION_COL] > 0.40)
    df.loc[bad, CONSUMPTION_COL] = rng.uniform(KWH_PER_KM_MIN, KWH_PER_KM_MAX, bad.sum())
    outside = (df[CONSUMPTION_COL] < KWH_PER_KM_MIN) | (df[CONSUMPTION_COL] > KWH_PER_KM_MAX)
    df.loc[outside, CONSUMPTION_COL] = rng.uniform(KWH_PER_KM_MIN, KWH_PER_KM_MAX, outside.sum())

def set_initial_socs(df: pd.DataFrame, rng: np.random.Generator) -> None:
    df[RESERVE_COL] = rng.uniform(*RESERVE_SOC_PCT_RANGE, len(df))
    start_min = df[RESERVE_COL] + START_OVER_RESERVE_MIN
    start_max = df[RESERVE_COL] + START_OVER_RESERVE_MAX
    df[START_COL] = rng.uniform(0.0, 1.0, len(df)) * (start_max - start_min) + start_min
    df[START_COL] = np.minimum(df[START_COL].to_numpy(), START_PCT_CAP)

def recompute_energy(df: pd.DataFrame) -> None:
    df[REQUIRED_COL] = df[DISTANCE_COL].astype(float) * df[CONSUMPTION_COL].astype(float)
    df[AVAIL_COL] = df["_battery_kwh_est"] * (df[START_COL] - df[RESERVE_COL]) / 100.0

def push_rate_toward_target(df: pd.DataFrame, target: float, rng: np.random.Generator) -> None:
    """
    Convert just enough users across the threshold so that the final rate
    lands close to 'target' while staying inside [TARGET_MIN, TARGET_MAX].
    - If low: lower start for the closest no-need users.
    - If high: raise start for the easiest needers to fix.
    """
    n = len(df)
    target = float(np.clip(target, TARGET_MIN, TARGET_MAX))

    recompute_energy(df)
    need = compute_need_mask(df)
    rate = need.mean()

    if rate < target:
        # How many to flip from no-need -> need
        deficit = int(np.ceil(target * n - need.sum()))
        if deficit > 0:
            margin = (df[AVAIL_COL] - df[REQUIRED_COL]).to_numpy()
            candidates = (~need).nonzero()[0]
            if len(candidates) > 0:
                # smallest positive margins first (closest to the boundary)
                order = np.argsort(margin[candidates])
                idxs = candidates[order[:deficit]]
                # set start so available ≈ required * (1 - eps)  (forces need)
                eps = rng.uniform(0.01, 0.03, size=len(idxs))
                req = df.iloc[idxs][REQUIRED_COL].to_numpy()
                batt = df.iloc[idxs]["_battery_kwh_est"].to_numpy()
                reserve = df.iloc[idxs][RESERVE_COL].to_numpy()
                new_gap_pct = 100.0 * (req * (1 - eps)) / batt
                new_start = np.maximum(reserve + 0.5, reserve + new_gap_pct)
                df.loc[df.index[idxs], START_COL] = np.minimum(START_PCT_CAP, new_start)

    elif rate > target:
        # How many to flip from need -> no-need
        excess = int(np.ceil(need.sum() - target * n))
        if excess > 0:
            deficit_energy = (df[REQUIRED_COL] - df[AVAIL_COL]).to_numpy()  # >0 for needers
            need_idx = need.nonzero()[0]
            if len(need_idx) > 0:
                # easiest to fix: smallest deficit first
                order = np.argsort(deficit_energy[need_idx])
                idxs = need_idx[order[:excess]]
                # bump start just enough so available >= required * (1 + eps)
                eps = rng.uniform(0.01, 0.03, size=len(idxs))
                req = df.iloc[idxs][REQUIRED_COL].to_numpy()
                batt = df.iloc[idxs]["_battery_kwh_est"].to_numpy()
                reserve = df.iloc[idxs][RESERVE_COL].to_numpy()
                needed_gap_pct = 100.0 * (req * (1 + eps)) / batt
                new_start = reserve + needed_gap_pct
                df.loc[df.index[idxs], START_COL] = np.minimum(START_PCT_CAP, new_start)

    # Final clamp to stay inside band (handles 29.8% cases)
    recompute_energy(df)
    need = compute_need_mask(df)
    rate = need.mean()

    if rate < TARGET_MIN:
        # flip the minimal number of non-needers
        deficit = int(np.ceil(TARGET_MIN * n - need.sum()))
        if deficit > 0:
            margin = (df[AVAIL_COL] - df[REQUIRED_COL]).to_numpy()
            candidates = (~need).nonzero()[0]
            order = np.argsort(margin[candidates])
            idxs = candidates[order[:deficit]]
            eps = rng.uniform(0.01, 0.02, size=len(idxs))
            req = df.iloc[idxs][REQUIRED_COL].to_numpy()
            batt = df.iloc[idxs]["_battery_kwh_est"].to_numpy()
            reserve = df.iloc[idxs][RESERVE_COL].to_numpy()
            new_gap_pct = 100.0 * (req * (1 - eps)) / batt
            new_start = np.maximum(reserve + 0.5, reserve + new_gap_pct)
            df.loc[df.index[idxs], START_COL] = np.minimum(START_PCT_CAP, new_start)
            recompute_energy(df)

    if rate > TARGET_MAX:
        # flip the minimal number of needers
        excess = int(np.ceil(need.sum() - TARGET_MAX * n))
        if excess > 0:
            deficit_energy = (df[REQUIRED_COL] - df[AVAIL_COL]).to_numpy()
            need_idx = need.nonzero()[0]
            order = np.argsort(deficit_energy[need_idx])
            idxs = need_idx[order[:excess]]
            eps = rng.uniform(0.01, 0.02, size=len(idxs))
            req = df.iloc[idxs][REQUIRED_COL].to_numpy()
            batt = df.iloc[idxs]["_battery_kwh_est"].to_numpy()
            reserve = df.iloc[idxs][RESERVE_COL].to_numpy()
            needed_gap_pct = 100.0 * (req * (1 + eps)) / batt
            new_start = reserve + needed_gap_pct
            df.loc[df.index[idxs], START_COL] = np.minimum(START_PCT_CAP, new_start)
            recompute_energy(df)

def calibrate(df: pd.DataFrame, *, rng: np.random.Generator, target: float) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    ensure_required_cols(df)

    # 1) consumption realism
    seed_consumption(df, rng)

    # 2) infer battery and seed SOCs
    df["_battery_kwh_est"] = infer_battery_kwh(df)
    set_initial_socs(df, rng)

    # 3) recompute energy with seeded SOCs
    recompute_energy(df)

    # 4) push rate toward target, then clamp to [TARGET_MIN, TARGET_MAX]
    push_rate_toward_target(df, target, rng)

    # Final bookkeeping
    df["needs_charge"] = (df[REQUIRED_COL] > df[AVAIL_COL]).astype(int)
    df = df.drop(columns=["_battery_kwh_est"])

    meta = {
        "n_rows": int(len(df)),
        "need_rate": float(df["needs_charge"].mean()),
        "kwh_per_km_range": (float(df[CONSUMPTION_COL].min()), float(df[CONSUMPTION_COL].max())),
        "reserve_pct_range": (float(df[RESERVE_COL].min()), float(df[RESERVE_COL].max())),
        "start_pct_range": (float(df[START_COL].min()), float(df[START_COL].max())),
    }
    return df, meta

def write_calibrated(path: Path, rng: np.random.Generator, target: float) -> tuple[Path, dict]:
    df = pd.read_csv(path)
    df2, meta = calibrate(df, rng=rng, target=target)
    out = path.with_name(path.stem + "_calibrated").with_suffix(path.suffix)
    df2.to_csv(out, index=False)
    return out, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/sim_users_train.csv")
    ap.add_argument("--eval",  default="data/sim_users_eval.csv")
    ap.add_argument("--target", type=float, default=DEFAULT_TARGET,
                    help="Desired fraction of trips needing >=1 charge (will be clamped to 0.30–0.60).")
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    for p in [Path(args.train), Path(args.eval)]:
        if not p.exists():
            raise FileNotFoundError(p)
        print(f"\nCalibrating {p} ...")
        out, meta = write_calibrated(p, rng, args.target)
        need_rate = meta['need_rate']
        print(f"✔ Wrote {out}  | rows={meta['n_rows']}")
        print(f"   need>=1-charge rate: {need_rate:.1%} (target band {int(TARGET_MIN*100)}–{int(TARGET_MAX*100)}%)")
        kmin, kmax = meta["kwh_per_km_range"]
        print(f"   kwh_per_km in [{kmin:.3f}, {kmax:.3f}]")
        rmin, rmax = meta["reserve_pct_range"]
        smin, smax = meta["start_pct_range"]
        print(f"   reserve_soc_pct in [{rmin:.1f}, {rmax:.1f}] | start_soc_pct in [{smin:.1f}, {smax:.1f}]")

if __name__ == "__main__":
    main()
