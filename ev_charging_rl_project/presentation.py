# presentation.py
# Viva/demo runner with correct cost reporting:
# - Charging spend (£): what you actually pay for energy
# - Time value (£): value of time (VoT) × total minutes
# - Generalised cost (£): charging spend + time value
#
# Works even if a trip has no charging (previously this showed £0.00).
#
# Usage examples:
#   python presentation.py --model runs/ppo_xxx/best_model.zip --cfg configs/ppo_eval.json --user 123
#   python presentation.py --model runs/ppo_xxx/best_model.zip --autopick
#   python presentation.py --model runs/ppo_xxx/best_model.zip --autopick-off  # to force the exact user id entered
#
# Notes:
# - This script does NOT change your environment or training.
# - It only changes how costs are computed + displayed + saved.
# - It tries multiple common attribute/key names for robustness.

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception:
    gym = None

try:
    from stable_baselines3 import PPO
except Exception:
    PPO = None


# -------------------------------
# Utilities: robust getters
# -------------------------------

def _get(obj: Any, *names, default=None):
    """Return first existing attribute or dict key from names."""
    for name in names:
        if isinstance(obj, dict):
            if name in obj:
                return obj[name]
        else:
            if hasattr(obj, name):
                return getattr(obj, name)
    return default


def _float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _bool(x, default=False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in {"1", "true", "yes", "y"}
    try:
        return bool(x)
    except Exception:
        return default


# -------------------------------
# Environment factory (EDIT HERE if your repo has a helper)
# -------------------------------

from pathlib import Path
from utils.data_loader import load_all_ready
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig
from rl.episodes import iter_episodes, TripPlan

def build_env(cfg_path: str = None, seed: int = 0):
    project_root = Path(".").resolve()
    data_dir = project_root / "data"
    sim_csv = data_dir / "sim_users_train_calibrated.csv"

    # load all the same bundle pieces as training
    bundle = load_all_ready(data_dir, strict=True)

    # use the SAME config values as in train_ppo.py (copy this block from your experiment)
    cfg = PPOEnvConfig(
        obs_top_k=5,
        dt_minutes=10.0,
        max_steps=84,
        prefer="cost",                     # or "hybrid"/"cost" depending on the model you’re demoing
        respect_trip_objective=False,
        value_of_time_per_min=0.05,
        charge_efficiency=0.92,
        charge_session_overhead_min=3.0,
        enable_shaping=True,
        shaping_gamma=1.0,
        enable_potential_time=True,
        potential_vref_kmh=25.0,
        idle_penalty_per_step=0.05,
        idle_progress_epsilon_km=0.15,
        micro_charge_penalty=0.5,
        micro_charge_min_kwh=1.0,
        micro_charge_min_minutes=6.0,
        traffic_mode="light",
        traffic_peak_factor_am=1.6,
        traffic_peak_factor_pm=1.5,
        traffic_offpeak_factor=1.0,
        use_sumo_drive=True,
        sumo_mode="route_time",
        sumo_net_path="london_inner.net.xml",
        sumo_gui=False,
        max_charges_per_trip=2,
        terminate_on_overlimit=True,
    )

    # simple episode loader like training
    episodes = list(iter_episodes(sim_csv))
    if not episodes:
        raise RuntimeError("No episodes found in sim_users_train_calibrated.csv")

    # use first trip for reset if none given (presentation.py will override via user_id)
    env = PPOChargingEnv(cfg, data_bundle=bundle)
    env.reset(seed=seed, options={"trip": episodes[0]})
    return env




# -------------------------------
# Model loader
# -------------------------------

def load_model(model_path: str):
    if PPO is None:
        raise RuntimeError("stable-baselines3 not available. Please install it or run in your project venv.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = PPO.load(model_path)
    return model


# -------------------------------
# Value of Time (VoT) & objective
# -------------------------------

def get_objective(env, info: Dict[str, Any]) -> str:
    """
    Returns one of {"cost", "time", "hybrid"} if possible, else "unknown".
    """
    obj = _get(env, "objective", "preference", "optimise_for", default=None)
    if obj is None:
        obj = _get(info, "objective", "preference", "optimise_for", default="unknown")
    if isinstance(obj, (list, tuple)) and obj:
        obj = obj[0]
    if isinstance(obj, str):
        return obj.lower()
    return str(obj).lower()


def get_vot_gbp_per_min(env, info: Dict[str, Any]) -> float:
    """
    Try to read VoT (£/min) from env or cfg-like dicts.
    Common names: value_of_time_gbp_per_min, vot_gbp_per_min, vot_per_min_gbp.
    """
    cand = (
        _get(env, "value_of_time_gbp_per_min", "vot_gbp_per_min", "vot_per_min_gbp", default=None)
        or _get(getattr(env, "cfg", {}), "value_of_time_gbp_per_min", "vot_gbp_per_min", "vot_per_min_gbp", default=None)
        or _get(getattr(env, "config", {}), "value_of_time_gbp_per_min", "vot_gbp_per_min", "vot_per_min_gbp", default=None)
        or _get(info, "value_of_time_gbp_per_min", "vot_gbp_per_min", "vot_per_min_gbp", default=None)
    )
    return _float(cand, default=0.0)


# -------------------------------
# User picking
# -------------------------------

def pick_user_id(env, explicit_user: int = None, autopick: bool = True, tries: int = 50, rng=None) -> int:
    """
    Pick a user id that leads to a 'doable' trip (>=60% attempts) if autopick==True.
    Assumes env.reset(user_id=...) is supported; if not, falls back to plain reset.
    """
    rng = rng or random.Random()
    # If user explicitly set and autopick is disabled -> return it
    if explicit_user is not None and not autopick:
        return int(explicit_user)

    # If explicit provided but autopick allowed, try explicit first; if fails, then search
    candidates = []
    if explicit_user is not None:
        candidates.append(int(explicit_user))

    # Add a spread of candidates
    candidates += [rng.randint(0, 99999) for _ in range(max(10, tries // 2))]

    # Quick probe loop
    doable_scores: Dict[int, float] = {}

    for u in candidates:
        ok = 0
        attempts = 3  # small probe
        for _ in range(attempts):
            try:
                obs, info = env.reset(options={"user_id": u}), {}
            except TypeError:
                try:
                    obs, info = env.reset(user_id=u), {}
                except Exception:
                    try:
                        res = env.reset()
                        if isinstance(res, tuple) and len(res) == 2:
                            obs, info = res
                        else:
                            obs, info = res, {}
                    except Exception:
                        continue

            # one no-op step to see if it explodes immediately
            try:
                # If action space is continuous/Box, sample() will be fine; for Discrete also fine
                ac = env.action_space.sample()
                step_res = env.step(ac)
                if isinstance(step_res, tuple) and len(step_res) >= 4:
                    _, _, terminated, truncated, step_info = step_res[0:5] if len(step_res) >= 5 else (*step_res, {})
                else:
                    terminated = False
                    truncated = False
                    step_info = {}
                ok += 1  # it stepped fine
            except Exception:
                pass

        doable_scores[u] = ok / attempts

    # Sort by success rate desc
    ranked = sorted(doable_scores.items(), key=lambda kv: kv[1], reverse=True)
    if ranked and ranked[0][1] >= 0.6:
        return ranked[0][0]

    # Fallbacks
    if explicit_user is not None:
        return int(explicit_user)
    return candidates[0] if candidates else rng.randint(0, 99999)


# -------------------------------
# Step loop (with correct costs)
# -------------------------------

def run_episode(env, model, user_id: int, outdir: str) -> Tuple[str, str, Dict[str, Any]]:
    """
    Runs one episode and writes:
      - steps CSV:     <outdir>/demo_user{user_id}_steps.csv
      - summary CSV:   <outdir>/demo_user{user_id}_summary.csv (1 row)
    Returns (steps_csv_path, summary_csv_path, summary_dict)
    """
    os.makedirs(outdir, exist_ok=True)

    # Reset with user
    try:
        obs, info = env.reset(options={"user_id": user_id})
    except TypeError:
        try:
            obs, info = env.reset(user_id=user_id)
        except Exception:
            res = env.reset()
            if isinstance(res, tuple) and len(res) == 2:
                obs, info = res
            else:
                obs, info = res, {}

    objective = get_objective(env, info)
    vot_per_min = get_vot_gbp_per_min(env, info)

    # Running totals (energy spend is env.total_cost in many repos)
    running_energy_spend = _float(_get(env, "total_cost", default=0.0))
    running_minutes = _float(_get(env, "total_minutes", default=0.0))
    last_minutes_seen = running_minutes

    # CSVs
    steps_path = os.path.join(outdir, f"demo_user{user_id}_steps.csv")
    summary_path = os.path.join(outdir, f"demo_user{user_id}_summary.csv")

    steps_fields = [
        "step",
        "action",
        "reward",
        "terminated",
        "truncated",
        "soc",
        "remaining_km",
        "minute_delta",
        "total_minutes",
        "energy_kwh",
        "energy_unit_price_gbp_per_kwh",
        "energy_cost_gbp",
        "running_energy_spend_gbp",
        "running_time_value_gbp",
        "running_generalised_cost_gbp",
        "charge_events",
        "station_id",
        "step_type",
        "note",
    ]

    with open(steps_path, "w", newline="", encoding="utf-8") as f_steps:
        writer = csv.DictWriter(f_steps, fieldnames=steps_fields)
        writer.writeheader()

        step_idx = 0
        total_charge_events = 0
        last_charge_station = None
        done = False

        print("\n-------- Viva Demo --------")
        print(f"User ID: {user_id}")
        print(f"Objective: {objective} | VoT £/min: {vot_per_min:.2f}\n")

        while not done:
            # Policy action
            if hasattr(model, "predict"):
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            step_res = env.step(action)

            # Unpack step tuple robustly across gym versions
            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
            else:
                # (obs, reward, done, info) older API
                obs, reward, done_flag, info = step_res
                terminated, truncated = done_flag, False

            # Minutes progression
            minute_delta = _float(_get(info, "minute_delta", "delta_minutes", default=None))
            if minute_delta is None:
                # fallback: compute from cumulative if available
                tm_now = _float(_get(info, "total_minutes", default=_get(env, "total_minutes", default=last_minutes_seen)))
                minute_delta = max(0.0, tm_now - last_minutes_seen)
                running_minutes = tm_now
            else:
                running_minutes += minute_delta
            last_minutes_seen = running_minutes

            # Energy spend progression
            # Step energy cost if provided in info:
            step_energy_cost = _float(_get(info, "energy_cost_gbp", "delta_energy_cost_gbp", default=0.0))
            if step_energy_cost == 0.0:
                # If info doesn't provide, rely on env.total_cost (most repos update it on charge)
                running_energy_spend = _float(_get(env, "total_cost", default=running_energy_spend))
            else:
                running_energy_spend += step_energy_cost

            # Count charge events if indicated
            is_charge = False
            step_type = _get(info, "step_type", default="")
            if isinstance(step_type, str) and step_type.lower() == "charge":
                is_charge = True
            # Alternative flags:
            if _bool(_get(info, "charge_event", default=False)):
                is_charge = True

            if is_charge:
                total_charge_events += 1
                last_charge_station = _get(info, "station_id", "charger_id", default=last_charge_station)

            # Costs
            running_time_value = vot_per_min * running_minutes
            running_generalised = running_energy_spend + running_time_value

            # Telemetry
            soc = _float(_get(info, "soc", "soc_now", "soc_final", default=_get(env, "soc", default=np.nan)))
            remaining_km = _float(_get(info, "remaining_km", "dist_remaining_km", default=np.nan))
            unit_price = _float(_get(info, "energy_unit_price_gbp_per_kwh", "kwh_price_gbp", default=np.nan))

            # Terminal?
            done = bool(terminated or truncated)

            # Pretty terminal log (single line per step)
            print(
                f"t={running_minutes:6.2f} min | "
                f"£chg={running_energy_spend:6.2f} | £time={running_time_value:6.2f} | £gen={running_generalised:6.2f} | "
                f"SoC={soc:5.1f}% | Rem={remaining_km:6.2f} km"
                + (" | CHARGE" if is_charge else "")
            )

            # Write step row
            writer.writerow({
                "step": step_idx,
                "action": str(action),
                "reward": _float(reward, 0.0),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "soc": soc,
                "remaining_km": remaining_km,
                "minute_delta": minute_delta,
                "total_minutes": running_minutes,
                "energy_kwh": _float(_get(info, "energy_kwh", default=np.nan)),
                "energy_unit_price_gbp_per_kwh": unit_price,
                "energy_cost_gbp": step_energy_cost,
                "running_energy_spend_gbp": running_energy_spend,
                "running_time_value_gbp": running_time_value,
                "running_generalised_cost_gbp": running_generalised,
                "charge_events": total_charge_events,
                "station_id": last_charge_station,
                "step_type": step_type,
                "note": _get(info, "note", default=""),
            })

            step_idx += 1

    # Episode summary
    # Prefer info-provided episode-level numbers, else fall back to env accumulators
    # Energy spend:
    ep_energy_cost = _float(_get(info, "episode_cost_gbp", "episode_energy_cost_gbp", default=_get(env, "total_cost", default=0.0)))
    # Minutes:
    ep_minutes = _float(_get(info, "episode_minutes", default=_get(env, "total_minutes", default=running_minutes)))
    # Time value and generalised:
    ep_time_value = vot_per_min * ep_minutes
    ep_generalised = ep_energy_cost + ep_time_value

    success = _bool(_get(info, "success", default=_get(env, "success", default=False)))
    stranded = _bool(_get(info, "stranded", default=_get(env, "stranded", default=False)))
    term_reason = _get(info, "termination_reason", default="")
    soc_final = _float(_get(info, "soc_final", default=_get(env, "soc", default=np.nan)))
    remaining_km = _float(_get(info, "remaining_km", default=np.nan))
    charge_events = _int(_get(info, "charge_events", default=total_charge_events))

    summary_fields = [
        "timestamp",
        "user_id",
        "objective",
        "vot_gbp_per_min",
        "episode_minutes",
        "episode_energy_cost_gbp",
        "episode_time_value_gbp",
        "episode_generalised_cost_gbp",
        "success",
        "stranded",
        "soc_final",
        "remaining_km",
        "charge_events",
        "termination_reason",
    ]
    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user_id": user_id,
        "objective": objective,
        "vot_gbp_per_min": vot_per_min,
        "episode_minutes": ep_minutes,
        "episode_energy_cost_gbp": ep_energy_cost,
        "episode_time_value_gbp": ep_time_value,
        "episode_generalised_cost_gbp": ep_generalised,
        "success": success,
        "stranded": stranded,
        "soc_final": soc_final,
        "remaining_km": remaining_km,
        "charge_events": charge_events,
        "termination_reason": term_reason,
    }

    # Print summary nicely for viva
    print("\n----- Episode Summary -----")
    print(f"Objective: {objective}")
    print(f"User ID: {user_id}")
    print(f"Duration: {ep_minutes:.2f} min")
    print(f"Charging spend: £{ep_energy_cost:.2f}")
    print(f"Time value:     £{ep_time_value:.2f}  (VoT £/min={vot_per_min:.2f})")
    print(f"Generalised:    £{ep_generalised:.2f}")
    print(f"Charge events:  {charge_events}")
    print(f"SoC final:      {soc_final:.1f}%   | Remaining: {remaining_km:.2f} km")
    print(f"Outcome:        {'SUCCESS' if success else 'FAIL'}"
          f"{' | STRANDED' if stranded else ''}"
          f"{' | ' + term_reason if term_reason else ''}")
    print("---------------------------\n")

    # Write 1-row summary CSV
    with open(summary_path, "w", newline="", encoding="utf-8") as f_sum:
        sw = csv.DictWriter(f_sum, fieldnames=summary_fields)
        sw.writeheader()
        sw.writerow(summary)

    return steps_path, summary_path, summary


# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Demo a trained PPO model with correct cost reporting.")
    parser.add_argument("--model", required=True, help="Path to PPO .zip (e.g., runs/.../best_model.zip)")
    parser.add_argument("--cfg", default=None, help="Optional config path for the environment")
    parser.add_argument("--user", type=int, default=None, help="User ID to simulate")
    parser.add_argument("--autopick", action="store_true", help="Autopick a doable user (>=60% quick-probe success)")
    parser.add_argument("--autopick-off", dest="autopick_off", action="store_true", help="Disable autopick; force --user")
    parser.add_argument("--outdir", default="demo_outputs", help="Directory for CSV outputs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Validate SB3 model present
    model = load_model(args.model)

    # Build env
    env = build_env(cfg_path=args.cfg, seed=args.seed)

    # Decide user
    autopick = args.autopick and not args.autopick_off
    user_id = pick_user_id(env, explicit_user=args.user, autopick=autopick, rng=random.Random(args.seed))

    # Run episode
    steps_csv, summary_csv, summary = run_episode(env, model, user_id=user_id, outdir=args.outdir)

    # Final helpful paths
    print(f"Steps CSV:   {steps_csv}")
    print(f"Summary CSV: {summary_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Fail clearly with context, since this is for a viva
        print("\n[ERROR] presentation.py failed with the following message:\n")
        print(f"  {type(e).__name__}: {e}\n")
        print("Common fixes:")
        print("  • Ensure your venv has stable-baselines3 installed.")
        print("  • Pass the correct --model path to PPO .zip.")
        print("  • Wire up build_env() to your repo’s environment factory if your env isn’t gym-registered.")
        print("  • If your env needs a config file, pass --cfg path/to/cfg.json")
        sys.exit(1)
