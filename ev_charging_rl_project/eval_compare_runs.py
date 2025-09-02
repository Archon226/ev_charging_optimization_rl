# eval_compare_runs.py
from __future__ import annotations
import argparse, re, warnings, os
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

# Project imports
from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes, TripPlan
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig

# ----------------------------
# Paths
# ----------------------------
PROJECT = Path(".").resolve()
DATA = PROJECT / "data"
RUNS = PROJECT / "runs"


# ----------------------------
# File helpers (incremental I/O)
# ----------------------------
def _append_rows_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str] | None = None):
    """Append a list of dict rows to CSV, creating file with header if needed."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    if columns:
        # Ensure column order; include any extra columns at the end
        missing = [c for c in df.columns if c not in columns]
        df = df[[c for c in columns if c in df.columns] + missing]
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _append_row_csv(path: Path, row: Dict[str, Any], columns: List[str] | None = None):
    _append_rows_csv(path, [row], columns)


# ----------------------------
# Eval dataset helpers
# ----------------------------
def find_eval_csv() -> Path:
    for name in ("sim_users_eval.csv", "sim_users_train.csv", "simulated_users.csv"):
        p = DATA / name
        if p.exists():
            return p
    raise FileNotFoundError("No eval CSV in ev_charging_rl_project/data "
                            "(looked for sim_users_eval.csv, sim_users_train.csv, simulated_users.csv).")


def list_run_dirs(include: List[str] | None = None) -> List[Path]:
    runs = sorted(RUNS.glob("ppo_ev_*"), key=lambda p: p.stat().st_mtime)
    if include:
        inc = [s.lower() for s in include]
        runs = [r for r in runs if any(s in r.name.lower() for s in inc)]
    return runs


def extract_tag(run_dir: Path) -> str:
    # run name like ppo_ev_YYYYMMDD_HHMMSS_<RUN_TAG>
    m = re.match(r"ppo_ev_\d{8}_\d{6}_(.+)", run_dir.name)
    return m.group(1) if m else run_dir.name


def harden_trip(trip: TripPlan, widen_corridor_km: float = 3.0, force_topk: int = 5,
                low_soc_ceiling_pct: float = 12.0) -> TripPlan:
    """Make charging more likely (so evaluation exercises charging)."""
    if getattr(trip, "start_soc_pct", 15.0) > low_soc_ceiling_pct:
        trip.start_soc_pct = float(low_soc_ceiling_pct - 4.0)  # e.g., 8%
    if getattr(trip, "max_detour_km", 2.0) < widen_corridor_km:
        trip.max_detour_km = float(widen_corridor_km)
    trip.top_k_candidates = int(force_topk)
    return trip


def make_env_cfg(args) -> PPOEnvConfig:
    return PPOEnvConfig(
        obs_top_k=args.obs_top_k,
        dt_minutes=args.dt_minutes,
        max_steps=args.max_steps,
        prefer="hybrid",  # per-episode objective will be set from TripPlan in _reset_state
        value_of_time_per_min=args.vot_per_min,
        charge_efficiency=args.charge_eff,
        charge_session_overhead_min=args.session_overhead_min,
        traffic_mode=args.traffic,  # "none" or "light"
        traffic_peak_factor_am=args.traffic_peak_am,
        traffic_peak_factor_pm=args.traffic_peak_pm,
        traffic_offpeak_factor=args.traffic_offpeak,
    )


def _infer_k_from_model(model, base_feats: int = 2, per_cand_feats: int = 5) -> int | None:
    """
    Read the observation_space saved inside the SB3 model and infer K.
    Layout in your env: dim = 2 + 5*K
    """
    try:
        obs_space = getattr(model, "observation_space", None)
        if obs_space is None or not hasattr(obs_space, "shape"):
            return None
        dim = int(obs_space.shape[0])
        if (dim - base_feats) % per_cand_feats != 0:
            return None
        return (dim - base_feats) // per_cand_feats
    except Exception:
        return None


# ----------------------------
# Evaluation of ONE model (incremental writing)
# ----------------------------
DETAILS_COLUMNS = [
    "run_tag", "run_dir", "episode_idx",
    "steps", "reward", "minutes", "cost_gbp",
    "success", "stranded", "charge_events", "objective",
    "obs_top_k",
]

SKIPPED_COLUMNS = [
    "run_tag", "run_dir", "episode_idx", "reason",
    "objective", "origin_edge", "destination_edge", "obs_top_k",
]


def evaluate_model(model_path: Path,
                   run_dir_name: str,
                   bundle,
                   trips: List[TripPlan],
                   args,
                   tag: str,
                   details_csv: Path,
                   skipped_csv: Path) -> Tuple[pd.DataFrame | None, int, int]:
    """Evaluate one model; append per-episode details and per-skip rows to CSVs as we go."""
    try:
        model = PPO.load(str(model_path))
    except Exception as e:
        warnings.warn(f"[skip] failed to load model at {model_path}: {e}")
        return None, 0, 0

    # Match K to model.
    base_cfg = make_env_cfg(args)
    k_expected = _infer_k_from_model(model)
    if k_expected is None:
        k_expected = int(args.obs_top_k)
        print(f"[warn] {tag}: could not infer obs dim; using CLI obs_top_k={k_expected}")
    else:
        if k_expected != int(args.obs_top_k):
            print(f"[info] {tag}: model expects obs_top_k={k_expected} (CLI={args.obs_top_k}) → using {k_expected}")
    cfg = replace(base_cfg, obs_top_k=int(k_expected))

    # Iterate episodes
    rows_for_memory = []
    valid, invalid = 0, 0
    n = max(1, args.episodes)
    it = iter(trips)

    for i in range(n):
        # Get/prepare trip
        try:
            trip = next(it)
        except StopIteration:
            it = iter(trips)
            trip = next(it)

        trip = harden_trip(trip,
                           widen_corridor_km=args.eval_widen_km,
                           force_topk=cfg.obs_top_k,
                           low_soc_ceiling_pct=args.low_soc_ceiling_pct)

        # Try to reset environment for this trip
        try:
            env = PPOChargingEnv(cfg, data_bundle=bundle)
            obs, info = env.reset(options={"trip": trip})
        except Exception as e:
            # Log skipped trip immediately
            _append_row_csv(
                skipped_csv,
                {
                    "run_tag": tag,
                    "run_dir": run_dir_name,
                    "episode_idx": i + 1,
                    "reason": str(e),
                    "objective": getattr(trip, "objective", None),
                    "origin_edge": getattr(trip, "origin_edge", None),
                    "destination_edge": getattr(trip, "destination_edge", None),
                    "obs_top_k": cfg.obs_top_k,
                },
                SKIPPED_COLUMNS,
            )
            print(f"[skip trip] {tag}: episode {i+1} skipped ({e})")
            invalid += 1
            continue

        # Rollout
        done = False
        total_r = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            obs, r, term, trunc, inf = env.step(int(action))
            total_r += float(r)
            done = bool(term or trunc)
            steps += 1

        # Collect stats
        base = env.unwrapped
        row = {
            "run_tag": tag,
            "run_dir": run_dir_name,
            "episode_idx": i + 1,
            "steps": steps,
            "reward": total_r,
            "minutes": float(getattr(base, "total_minutes", 0.0)),
            "cost_gbp": float(getattr(base, "total_cost", 0.0)),
            "success": int(getattr(base, "remaining_km", 1.0) <= 0.0),
            "stranded": int(getattr(base, "soc", 1.0) <= 0.0),
            "charge_events": int(getattr(base, "charge_events", 0)),
            "objective": getattr(trip, "objective", "unknown"),
            "obs_top_k": cfg.obs_top_k,
        }
        env.close()

        # Append this episode immediately
        _append_row_csv(details_csv, row, DETAILS_COLUMNS)

        rows_for_memory.append(row)
        valid += 1

    return (pd.DataFrame(rows_for_memory) if rows_for_memory else None), valid, invalid


# ----------------------------
# Summaries / Plots
# ----------------------------
def summarize_overall(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("run_tag", as_index=False).agg(
        episodes=("run_tag", "size"),
        success_rate=("success", "mean"),
        stranded_rate=("stranded", "mean"),
        avg_minutes=("minutes", "mean"),
        avg_cost_gbp=("cost_gbp", "mean"),
        avg_charge_events=("charge_events", "mean"),
    )


def summarize_by_objective(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["run_tag", "objective"], as_index=False).agg(
        episodes=("objective", "size"),
        success_rate=("success", "mean"),
        stranded_rate=("stranded", "mean"),
        avg_minutes=("minutes", "mean"),
        avg_cost_gbp=("cost_gbp", "mean"),
        avg_charge_events=("charge_events", "mean"),
    )


def plot_overall_bars(summary: pd.DataFrame, out_dir: Path):
    # Bar: success rate per run_tag
    fig = plt.figure()
    x = np.arange(len(summary))
    plt.bar(x, summary["success_rate"].values)
    plt.xticks(x, summary["run_tag"].tolist(), rotation=30, ha="right")
    plt.ylabel("Success rate")
    plt.title("Overall Success Rate by Run")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_success_rate.png", dpi=160)
    plt.close(fig)

    # Bar: avg minutes per run_tag
    fig = plt.figure()
    plt.bar(x, summary["avg_minutes"].values)
    plt.xticks(x, summary["run_tag"].tolist(), rotation=30, ha="right")
    plt.ylabel("Avg minutes")
    plt.title("Average Minutes by Run")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_avg_minutes.png", dpi=160)
    plt.close(fig)

    # Bar: avg cost £ per run_tag
    fig = plt.figure()
    plt.bar(x, summary["avg_cost_gbp"].values)
    plt.xticks(x, summary["run_tag"].tolist(), rotation=30, ha="right")
    plt.ylabel("Avg cost (£)")
    plt.title("Average Cost (£) by Run")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_avg_cost.png", dpi=160)
    plt.close(fig)


def plot_objective_bars(summary_obj: pd.DataFrame, out_dir: Path):
    objectives = sorted(summary_obj["objective"].dropna().unique().tolist())
    tags = sorted(summary_obj["run_tag"].unique().tolist())
    width = 0.8 / max(1, len(objectives))  # total width 0.8

    for metric, ylabel, fname in [
        ("success_rate", "Success rate", "plot_obj_success_rate.png"),
        ("avg_minutes", "Avg minutes", "plot_obj_avg_minutes.png"),
        ("avg_cost_gbp", "Avg cost (£)", "plot_obj_avg_cost.png"),
        ("avg_charge_events", "Avg charge events", "plot_obj_avg_charge_events.png"),
    ]:
        fig = plt.figure()
        x = np.arange(len(tags))
        for i, obj in enumerate(objectives):
            sub = summary_obj[summary_obj["objective"] == obj]
            sub = sub.set_index("run_tag").reindex(tags)
            vals = sub[metric].values
            plt.bar(x + i * width, vals, width=width, label=obj)
        plt.xticks(x + (len(objectives) - 1) * width / 2, tags, rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} by Run and Objective")
        plt.legend()
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=160)
        plt.close(fig)


def plot_minutes_vs_cost(summary: pd.DataFrame, out_dir: Path):
    # Scatter: avg minutes vs avg cost, label by tag
    fig = plt.figure()
    x = summary["avg_minutes"].values
    y = summary["avg_cost_gbp"].values
    s = 100.0 * (1.0 + summary["avg_charge_events"].values)  # bubble size
    plt.scatter(x, y, s=s)
    for xi, yi, lab in zip(x, y, summary["run_tag"].tolist()):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 6))
    plt.xlabel("Avg minutes")
    plt.ylabel("Avg cost (£)")
    plt.title("Avg Minutes vs Avg Cost (bubble size = charge events)")
    fig.tight_layout()
    fig.savefig(out_dir / "plot_minutes_vs_cost.png", dpi=160)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate all trained runs on a common eval set.")
    ap.add_argument("--episodes", type=int, default=200, help="Episodes per run.")
    ap.add_argument("--deterministic", type=int, default=1, help="Use deterministic policy actions (1 yes / 0 no).")
    ap.add_argument("--include", type=str, default="", help="Comma-separated substrings to filter run names.")
    ap.add_argument("--obs_top_k", type=int, default=5)
    ap.add_argument("--dt_minutes", type=float, default=10.0)
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--vot_per_min", type=float, default=0.05)
    ap.add_argument("--charge_eff", type=float, default=0.92)
    ap.add_argument("--session_overhead_min", type=float, default=3.0)
    ap.add_argument("--traffic", type=str, default="light", choices=["none", "light"])
    ap.add_argument("--traffic_peak_am", type=float, default=1.6)
    ap.add_argument("--traffic_peak_pm", type=float, default=1.5)
    ap.add_argument("--traffic_offpeak", type=float, default=1.0)
    ap.add_argument("--eval_widen_km", type=float, default=3.0)
    ap.add_argument("--low_soc_ceiling_pct", type=float, default=12.0)
    args = ap.parse_args()

    # Data + trips
    eval_csv = find_eval_csv()
    trips = list(iter_episodes(eval_csv))
    if not trips:
        raise RuntimeError(f"No episodes in {eval_csv}")
    bundle = load_all_ready(DATA, strict=True)

    # Runs to evaluate
    filters = [s.strip() for s in args.include.split(",") if s.strip()] if args.include else None
    runs = list_run_dirs(filters)
    if not runs:
        print("No runs found under ./runs (use --include to match specific tags).")
        return

    # Output directory + CSVs (fresh files each run)
    out_dir = RUNS / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    details_csv = out_dir / "eval_details.csv"
    skipped_csv = out_dir / "skipped_trips.csv"

    # Start clean so this run's outputs are self-contained
    if details_csv.exists():
        details_csv.unlink()
    if skipped_csv.exists():
        skipped_csv.unlink()

    # Pre-create CSVs with headers (optional; _append_* handles headers too)
    pd.DataFrame(columns=DETAILS_COLUMNS).to_csv(details_csv, index=False)
    pd.DataFrame(columns=SKIPPED_COLUMNS).to_csv(skipped_csv, index=False)

    # Evaluate all runs
    results = []
    total_valid, total_invalid = 0, 0
    for run in runs:
        model_path = run / "model.zip"
        if not model_path.exists():
            print(f"[skip] {run.name}: no model.zip")
            continue
        tag = extract_tag(run)
        print(f"[eval] {run.name} (tag={tag}) ...")

        df, valid, invalid = evaluate_model(
            model_path=model_path,
            run_dir_name=run.name,
            bundle=bundle,
            trips=trips,
            args=args,
            tag=tag,
            details_csv=details_csv,
            skipped_csv=skipped_csv,
        )
        total_valid += valid
        total_invalid += invalid

        if df is not None and len(df):
            results.append(df)

    if not results:
        print("No successful evaluations.")
        print(f"Trip validity summary:\n - Valid trips: {total_valid}\n - Invalid trips: {total_invalid}")
        return

    # In-memory concat strictly for summaries/plots (details already written incrementally)
    df_all = pd.concat(results, ignore_index=True)

    # Summaries
    overall = summarize_overall(df_all)
    per_obj = summarize_by_objective(df_all)

    # Write summaries (overwrite each time at end)
    overall_csv = out_dir / "eval_summary_overall.csv"
    perobj_csv = out_dir / "eval_summary_by_objective.csv"
    overall.to_csv(overall_csv, index=False)
    per_obj.to_csv(perobj_csv, index=False)

    # Plots
    plot_overall_bars(overall, out_dir)
    plot_objective_bars(per_obj, out_dir)
    plot_minutes_vs_cost(overall, out_dir)

    # Console
    pd.options.display.float_format = "{:.3f}".format
    print("\n=== OVERALL SUMMARY (sorted by success desc, minutes asc) ===")
    print(overall.sort_values(["success_rate", "avg_minutes"], ascending=[False, True]).to_string(index=False))
    print("\n=== PER-OBJECTIVE SUMMARY ===")
    print(per_obj.to_string(index=False))
    print(f"\nWrote (incremental):\n - {details_csv}\n - {skipped_csv}")
    print(f"Wrote (at end):\n - {overall_csv}\n - {perobj_csv}\n - plots → {out_dir}")
    total = total_valid + total_invalid
    frac = (100.0 * total_invalid / total) if total > 0 else 0.0
    print(f"\nTrip validity summary:\n - Valid trips: {total_valid}\n - Invalid trips: {total_invalid}"
          f"\n - Skipped fraction: {total_invalid}/{total} = {frac:.2f}%")

if __name__ == "__main__":
    main()
