# eval_smoketest.py
from __future__ import annotations
from pathlib import Path
import sys, warnings
import numpy as np
import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import load_all_ready
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig
from rl.episodes import iter_episodes

PROJECT = Path(".").resolve()
DATA = PROJECT / "data"

def latest_run_dir() -> Path | None:
    runs = sorted((PROJECT / "runs").glob("ppo_ev_20250901_012231"), key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

def pick_model_path() -> Path | None:
    run = latest_run_dir()
    if run is None:
        return None
    mp = run / "model.zip"
    return mp if mp.exists() else None

def pick_eval_csv() -> Path:
    # prefer eval split; else train; else original file
    for name in ("sim_users_eval.csv", "sim_users_train.csv", "simulated_users.csv"):
        p = DATA / name
        if p.exists():
            return p
    raise FileNotFoundError("No simulated users CSV found under ev_charging_rl_project/data/")

def harden_trip(trip):
    # encourage charging so we see costs:
    if getattr(trip, "start_soc_pct", 15.0) > 12.0:
        trip.start_soc_pct = 8.0
    if getattr(trip, "max_detour_km", 2.0) < 3.0:
        trip.max_detour_km = 3.0
    trip.top_k_candidates = 5
    # keep per-episode objective from CSV (cost/time/hybrid) so we see behavior across all 3
    return trip

def main():
    bundle = load_all_ready(DATA, strict=True)
    cfg = PPOEnvConfig(
        obs_top_k=5, dt_minutes=5.0, max_steps=140,
        prefer="hybrid", value_of_time_per_min=0.05, charge_efficiency=0.92
    )

    # Load model if present
    policy = None
    model_path = pick_model_path()
    if model_path is not None:
        try:
            from stable_baselines3 import PPO
            policy = PPO.load(str(model_path))
            print(f"[eval] loaded model: {model_path}")
        except Exception as e:
            warnings.warn(f"Could not load model at {model_path}: {e}; using random actions.")
    else:
        print("[eval] no model found; using random actions.")

    # One episode
    csv = pick_eval_csv()
    trip = next(iter_episodes(csv))
    trip = harden_trip(trip)

    env = PPOChargingEnv(cfg, data_bundle=bundle)
    obs, info = env.reset(options={"trip": trip})

    done = False
    total_r = 0.0
    charge_actions = 0
    steps = 0
    while not done:
        if policy is None:
            action = 0 if np.random.rand() < 0.6 else 1  # bias to drive but allow charge
        else:
            action, _ = policy.predict(obs, deterministic=False)
            action = int(action)
        if action != 0:
            charge_actions += 1
        obs, r, term, trunc, inf = env.step(action)
        total_r += float(r)
        done = bool(term or trunc)
        steps += 1

    print("\n=== ONE-EPISODE RESULT ===")
    print("steps:", steps)
    print("reward:", round(total_r, 3))
    print("total_minutes:", getattr(env, "total_minutes", None))
    print("total_cost_gbp:", getattr(env, "total_cost", None))
    print("charge_events:", getattr(env, "charge_events", None))
    print("success:", getattr(env, "remaining_km", 1.0) <= 0.0)
    print("stranded:", getattr(env, "soc", 1.0) <= 0.0)

    env.close()

if __name__ == "__main__":
    main()
