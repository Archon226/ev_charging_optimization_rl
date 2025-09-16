# presentation.py
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import csv
import pandas as pd
import numpy as np

from stable_baselines3 import PPO

# --- Project imports (same style as train_ppo / eval_ppo) ---
from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes, TripPlan
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig

PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
CALIBRATED_EVAL_CSV = DATA_DIR / "sim_users_eval_calibrated.csv"


def pick_trip(eval_csv: Path, user_id: int | None, row_index: int | None) -> TripPlan:
    """
    Pick a single TripPlan from sim_users_eval_calibrated.csv
    - Prefer exact user_id match if provided
    - Else use row_index (0-based)
    - Else default to the first row
    """
    trips = list(iter_episodes(eval_csv))
    if not trips:
        raise RuntimeError(f"No episodes found in {eval_csv}")

    if user_id is not None:
        for t in trips:
            if getattr(t, "user_id", None) == user_id:
                return t
        raise ValueError(f"user_id={user_id} not found in {eval_csv}")

    if row_index is not None:
        if not (0 <= row_index < len(trips)):
            raise IndexError(f"row_index {row_index} out of range [0, {len(trips)-1}]")
        return trips[row_index]

    return trips[0]


def build_env(cfg_policy_name: str, bundle: dict, trip: TripPlan) -> PPOChargingEnv:
    """
    Create a single-use env configured like eval_ppo, but for one TripPlan.
    """
    cfg = PPOEnvConfig(
        obs_top_k=5,
        dt_minutes=10.0,
        max_steps=84,
        prefer=cfg_policy_name,          # "cost" | "time" | "hybrid" (for logs/clarity)
        respect_trip_objective=False,    # we want to run the trained policy as-is
        value_of_time_per_min=0.05,
        charge_efficiency=0.92,
        charge_session_overhead_min=3.0,
        traffic_mode="light",
        traffic_peak_factor_am=1.6,
        traffic_peak_factor_pm=1.5,
        traffic_offpeak_factor=1.0,
        use_sumo_drive=True,
        sumo_mode="route_time",
        sumo_net_path="london_inner.net.xml",
        sumo_gui=False,

        # Feasibility / shaping (kept consistent with your eval script)
        max_charges_per_trip=2,
        terminate_on_overlimit=True,
        enable_shaping=True,
        shaping_gamma=1.0,
        enable_potential_time=True,
        potential_vref_kmh=25.0,
        idle_penalty_per_step=0.05,
        idle_progress_epsilon_km=0.15,
        micro_charge_penalty=0.5,
        micro_charge_min_kwh=1.0,
        micro_charge_min_minutes=6.0,
    )

    class _OneTripEnv(PPOChargingEnv):
        def __init__(self, cfg: PPOEnvConfig, data_bundle: dict, trip: TripPlan):
            super().__init__(cfg, data_bundle=data_bundle)
            self._trip_for_demo = trip
            self._used = False

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if self._used:
                # Simple one-shot behaviour: allow only one episode for the demo
                raise StopIteration("Demo episode already consumed. Recreate env to run again.")
            self._used = True
            return super().reset(seed=seed, options={"trip": self._trip_for_demo})

    return _OneTripEnv(cfg, bundle, trip)


def run_demo(model_path: Path, policy_name_hint: str | None,
             user_id: int | None, row_index: int | None,
             out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load bundle (same loader you use for training/eval)
    bundle = load_all_ready(DATA_DIR, strict=True)

    # 2) Pick the calibrated eval trip
    trip = pick_trip(CALIBRATED_EVAL_CSV, user_id=user_id, row_index=row_index)
    user_obj = getattr(trip, "objective", None)  # "cost" | "time" | "hybrid" from CSV

    # 3) Infer policy name (for config/labels only)
    pol = (policy_name_hint or user_obj or "hybrid").lower()
    if pol not in {"cost", "time", "hybrid"}:
        pol = "hybrid"

    # 4) Build env and load model
    env = build_env(pol, bundle, trip)
    model = PPO.load(str(model_path), env=env)

    # 5) Roll out single episode, logging step-by-step
    step_rows = []
    obs, info = env.reset()
    terminated = truncated = False
    step_idx = 0

    # Grab trip metadata for printing later
    trip_meta = asdict(trip)
    ev_id = trip_meta.get("ev_model", "unknown")
    depart_iso = str(trip_meta.get("depart_datetime"))

    print("\n================ DEMO RUN ================")
    print(f"Model:      {model_path}")
    print(f"Policy tag: {pol}")
    print(f"User ID:    {trip_meta.get('user_id')}")
    print(f"EV model:   {ev_id}")
    print(f"Depart:     {depart_iso}")
    print(f"Objective:  {user_obj}")
    print(f"Origin:     {trip_meta.get('origin')}")
    print(f"Dest:       {trip_meta.get('dest')}")
    print(f"Start SoC:  {trip_meta.get('start_soc_pct')}%   Reserve SoC: {trip_meta.get('reserve_soc_pct')}%")
    print("------------------------------------------")

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Access frequently useful fields
        # (Some are from info, others from env internal state — safe for demo/inspection)
        step_type = info.get("step_type", "n/a")
        total_min = float(getattr(env, "total_minutes", np.nan))
        soc = float(getattr(env, "soc", np.nan))
        remain_km = float(getattr(env, "remaining_km", np.nan))
        total_cost = float(getattr(env, "total_cost", np.nan))

        # Optional detail (only if present)
        shaping = info.get("shaping_time")
        last_charge = info.get("last_charge")
        pen_idle = info.get("penalty_idle")
        pen_micro = info.get("penalty_micro_charge")
        progress_km = info.get("progress_km")

        step_rows.append({
            "step": step_idx,
            "step_type": step_type,                 # "drive" or "charge"
            "episode_minutes": total_min,           # cumulative time
            "soc": soc,                             # current SOC (0..1)
            "remaining_km": remain_km,
            "total_cost_gbp": total_cost,           # cumulative £
            "reward": float(reward),
            "progress_km": progress_km,
            "shaping_time": shaping,
            "penalty_idle": pen_idle,
            "penalty_micro_charge": pen_micro,
            "last_charge": last_charge,
        })

        # Friendly narration line
        if step_type == "drive":
            print(f"[{step_idx:03d}] DRIVE  | t={total_min:5.1f} min | SoC={soc:0.3f} | rem={remain_km:5.1f} km | £={total_cost:5.2f}")
        elif step_type == "charge":
            print(f"[{step_idx:03d}] CHARGE | t={total_min:5.1f} min | SoC={soc:0.3f} | rem={remain_km:5.1f} km | £={total_cost:5.2f}")
        else:
            print(f"[{step_idx:03d}] STEP   | t={total_min:5.1f} min | SoC={soc:0.3f} | rem={remain_km:5.1f} km | £={total_cost:5.2f}")

        step_idx += 1

    # 6) Terminal KPIs (your env packs these in `info`)
    term_reason = info.get("termination_reason", "unknown")
    charge_events = int(info.get("charge_events", 0))
    final_soc = float(info.get("soc_final", np.nan))
    ep_minutes = float(info.get("episode_minutes", np.nan))
    ep_cost = float(info.get("episode_cost_gbp", np.nan))
    violations_repeat = int(info.get("violations_repeat", 0))
    violations_overlimit = int(info.get("violations_overlimit", 0))
    violations_cooldown = int(info.get("violations_cooldown", 0))

    print("------------------------------------------")
    print("EPILOGUE")
    print(f"Termination: {term_reason}")
    print(f"Total time:  {ep_minutes:.1f} min")
    print(f"Total cost:  £{ep_cost:.2f}")
    print(f"Charges:     {charge_events}")
    print(f"Final SoC:   {final_soc:0.3f}")
    if any([violations_repeat, violations_overlimit, violations_cooldown]):
        print(f"Violations:  repeat={violations_repeat}, overlimit={violations_overlimit}, cooldown={violations_cooldown}")

    # 7) Save CSV artefacts for the viva
    df_steps = pd.DataFrame(step_rows)
    uid = trip_meta.get("user_id")
    base = f"demo_user{uid}_{pol}"
    df_steps.to_csv(out_dir / f"{base}_steps.csv", index=False)

    # Also dump a tiny one-row episode summary
    summary_row = {
        "user_id": uid,
        "policy": pol,
        "objective_user": user_obj,
        "episode_minutes": ep_minutes,
        "episode_cost_gbp": ep_cost,
        "charge_events": charge_events,
        "soc_final": final_soc,
        "termination_reason": term_reason,
        "violations_repeat": violations_repeat,
        "violations_overlimit": violations_overlimit,
        "violations_cooldown": violations_cooldown,
        "ev_model": trip_meta.get("ev_model"),
        "depart_datetime": str(trip_meta.get("depart_datetime")),
        "origin": trip_meta.get("origin"),
        "dest": trip_meta.get("dest"),
        "start_soc_pct": trip_meta.get("start_soc_pct"),
        "reserve_soc_pct": trip_meta.get("reserve_soc_pct"),
        "needs_charge": trip_meta.get("needs_charge"),
        "trip_km": trip_meta.get("trip_km"),
    }
    pd.DataFrame([summary_row]).to_csv(out_dir / f"{base}_summary.csv", index=False)

    print(f"\nSaved: {out_dir / (base + '_steps.csv')}")
    print(f"Saved: {out_dir / (base + '_summary.csv')}")


def main():
    p = argparse.ArgumentParser(description="Demo a single calibrated eval user through a trained PPO model.")
    p.add_argument("--model", required=True, type=Path, help="Path to trained model zip (e.g., runs/.../model_final.zip)")
    p.add_argument("--user-id", type=int, default=None, help="Exact user_id from sim_users_eval_calibrated.csv")
    p.add_argument("--row-index", type=int, default=None, help="0-based row index from sim_users_eval_calibrated.csv (used if --user-id not given)")
    p.add_argument("--policy-name", type=str, default=None, help='Label to describe model objective in logs: "cost"|"time"|"hybrid" (optional)')
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "demo_outputs", help="Output directory for CSVs")
    args = p.parse_args()

    run_demo(
        model_path=args.model,
        policy_name_hint=args.policy_name,
        user_id=args.user_id,
        row_index=args.row_index,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
