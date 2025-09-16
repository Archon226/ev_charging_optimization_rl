# presentation.py
# Viva/demo runner (TripPlan-aware) with correct cost reporting:
# - Charging spend (£): energy you pay for
# - Time value (£): VoT × total minutes
# - Generalised cost (£): charging + time
#
# This version matches train_ppo.py:
#   * Uses PPOEnvConfig inline (no external cfg file)
#   * Builds PPOChargingEnv
#   * Loads TripPlans from data/sim_users_train_calibrated.csv
#   * Resets env with options={"trip": TripPlan}
#
# Usage (viva one-liner below):
#   python presentation.py --model runs/<run>/model.zip --autopick
#   python presentation.py --model runs/<run>/model.zip --user 123
#
# CSV outputs in demo_outputs/: demo_user<id>_steps.csv, demo_user<id>_summary.csv

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
from stable_baselines3 import PPO

# ---- project imports (same stack as training) ----
from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes, TripPlan
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig


# -------------------------------
# Small utils
# -------------------------------

def _get(obj: Any, *names, default=None):
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
# Build env + load trips (mirrors training)
# -------------------------------

def make_config_for_demo(prefer: Optional[str] = None) -> PPOEnvConfig:
    """
    Inline config that mirrors train_ppo.py (EXPERIMENT C block).
    You can tweak 'prefer' here based on the model you’re demoing.
    """
    return PPOEnvConfig(
        # --- observation & horizon ---
        obs_top_k=5,
        dt_minutes=10.0,
        max_steps=84,

        # --- objective & costs ---
        prefer=(prefer or "time"),   # "time" / "cost" / "hybrid"
        respect_trip_objective=False,
        value_of_time_per_min=0.05,
        charge_efficiency=0.92,
        charge_session_overhead_min=3.0,

        # --- shaping ---
        enable_shaping=True,
        shaping_gamma=1.0,
        enable_potential_time=True,
        potential_vref_kmh=25.0,
        idle_penalty_per_step=0.05,
        idle_progress_epsilon_km=0.15,
        micro_charge_penalty=0.5,
        micro_charge_min_kwh=1.0,
        micro_charge_min_minutes=6.0,

        # --- traffic + SUMO ---
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

def infer_prefer_from_model_path(model_path: str) -> Optional[str]:
    name = Path(model_path).as_posix().lower()
    if "cost" in name:
        return "cost"
    if "time" in name:
        return "time"
    if "hybrid" in name:
        return "hybrid"
    return None

def build_env_and_trips(model_path: str, seed: int = 0):
    project_root = Path(".").resolve()
    data_dir = project_root / "data"
    sim_csv = data_dir / "sim_users_eval_calibrated.csv"

    # Bundle (pricing_catalog, power_model, etc.)
    bundle = load_all_ready(data_dir, strict=True)

    # Config (match training)
    prefer = infer_prefer_from_model_path(model_path) or "cost"
    cfg = make_config_for_demo(prefer=prefer)

    # Env
    env = PPOChargingEnv(cfg, data_bundle=bundle)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    # Trips
    trips: List[TripPlan] = list(iter_episodes(sim_csv))
    if not trips:
        raise RuntimeError(f"No episodes found in {sim_csv}")

    return env, trips, cfg


# -------------------------------
# Trip selection (by user or autopick)
# -------------------------------

def get_trip_user_id(tp: TripPlan) -> Optional[int]:
    # Try common field names on TripPlan
    for name in ("user_id", "user", "id"):
        if hasattr(tp, name):
            val = getattr(tp, name)
            try:
                return int(val)
            except Exception:
                return None
    return None

def find_trip_by_user(trips: List[TripPlan], user_id: int) -> Optional[TripPlan]:
    # Exact match by attribute if present
    for tp in trips:
        uid = get_trip_user_id(tp)
        if uid is not None and uid == user_id:
            return tp
    # If not found, allow indexing by position
    if 0 <= user_id < len(trips):
        return trips[user_id]
    return None

def probe_trip(env: PPOChargingEnv, trip: TripPlan, probes: int = 2) -> float:
    """
    Quick health check: reset with the trip and take a random action.
    Returns fraction of successful probes (0.0—1.0).
    """
    ok = 0
    for _ in range(probes):
        try:
            obs, info = env.reset(options={"trip": trip})
            ac = env.action_space.sample()
            step_res = env.step(ac)
            # If we can step without exception, count it
            ok += 1
        except Exception:
            pass
    return ok / max(1, probes)

def select_trip(env: PPOChargingEnv, trips: List[TripPlan], user_id: Optional[int], autopick: bool, rng: random.Random) -> Tuple[TripPlan, int]:
    """
    Returns (TripPlan, display_user_id)
    """
    # If explicit and autopick off → honor exactly
    if user_id is not None and not autopick:
        tp = find_trip_by_user(trips, user_id)
        if tp is None:
            raise ValueError(f"--user {user_id} not found as TripPlan attribute or index")
        uid = get_trip_user_id(tp)
        return tp, (uid if uid is not None else user_id)

    # If explicit but autopick allowed → try explicit, else search
    candidates: List[TripPlan] = []
    if user_id is not None:
        tp = find_trip_by_user(trips, user_id)
        if tp is not None:
            candidates.append(tp)

    # Add random candidates
    idxs = list(range(len(trips)))
    rng.shuffle(idxs)
    for i in idxs[:40]:  # sample up to 40 trips to keep it quick
        candidates.append(trips[i])

    # Probe for a “doable” trip (>=60% probe success)
    best = None
    best_score = -1.0
    for tp in candidates:
        score = probe_trip(env, tp, probes=3)
        if score > best_score:
            best, best_score = tp, score
            if score >= 0.60:
                break

    if best is None:
        best = candidates[0]

    uid = get_trip_user_id(best)
    return best, (uid if uid is not None else -1)


# -------------------------------
# Model loader
# -------------------------------

def load_model(model_path: str) -> PPO:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return PPO.load(model_path)


# -------------------------------
# Episode runner (correct costs)
# -------------------------------

def get_objective(env, info: Dict[str, Any]) -> str:
    obj = _get(env, "objective", "preference", "optimise_for", default=None)
    if obj is None:
        obj = _get(info, "objective", "preference", "optimise_for", default="unknown")
    if isinstance(obj, (list, tuple)) and obj:
        obj = obj[0]
    if isinstance(obj, str):
        return obj.lower()
    return str(obj).lower()

def get_vot_gbp_per_min(env, info: Dict[str, Any]) -> float:
    cand = (
        _get(env, "value_of_time_gbp_per_min", "vot_gbp_per_min", "vot_per_min_gbp", default=None)
        or _get(getattr(env, "cfg", {}), "value_of_time_gbp_per_min", "vot_gbp_per_min", "vot_per_min_gbp", default=None)
        or _get(getattr(env, "config", {}), "value_of_time_gbp_per_min", "vot_gbp_per_min", "vot_per_min_gbp", default=None)
        or _get(info, "value_of_time_gbp_per_min", "vot_gbp_per_min", "vot_per_min_gbp", default=None)
    )
    return _float(cand, default=0.0)

def run_episode(env: PPOChargingEnv, model: PPO, trip: TripPlan, display_user_id: int, outdir: str) -> Tuple[str, str, Dict[str, Any]]:
    os.makedirs(outdir, exist_ok=True)

    # Reset with TripPlan (required by PPOChargingEnv)
    obs, info = env.reset(options={"trip": trip})

    objective = get_objective(env, info)
    vot_per_min = get_vot_gbp_per_min(env, info)

    running_energy_spend = _float(_get(env, "total_cost", default=0.0))
    running_minutes = _float(_get(env, "total_minutes", default=0.0))
    last_minutes_seen = running_minutes

    steps_path = os.path.join(outdir, f"demo_user{display_user_id}_steps.csv")
    summary_path = os.path.join(outdir, f"demo_user{display_user_id}_summary.csv")

    steps_fields = [
        "step","action","reward","terminated","truncated",
        "soc","remaining_km","minute_delta","total_minutes",
        "energy_kwh","energy_unit_price_gbp_per_kwh","energy_cost_gbp",
        "running_energy_spend_gbp","running_time_value_gbp","running_generalised_cost_gbp",
        "charge_events","station_id","step_type","note",
    ]

    with open(steps_path, "w", newline="", encoding="utf-8") as f_steps:
        writer = csv.DictWriter(f_steps, fieldnames=steps_fields)
        writer.writeheader()

        step_idx = 0
        total_charge_events = 0
        last_charge_station = None
        done = False

        print("\n-------- Viva Demo --------")
        print(f"User (from TripPlan): {display_user_id}")
        print(f"Objective: {objective} | VoT £/min: {vot_per_min:.2f}\n")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_res = env.step(action)

            if len(step_res) == 5:
                obs, reward, terminated, truncated, info = step_res
            else:
                obs, reward, done_flag, info = step_res
                terminated, truncated = done_flag, False

            minute_delta = _float(_get(info, "minute_delta", "delta_minutes", default=None))
            if minute_delta is None:
                tm_now = _float(_get(info, "total_minutes", default=_get(env, "total_minutes", default=last_minutes_seen)))
                minute_delta = max(0.0, tm_now - last_minutes_seen)
                running_minutes = tm_now
            else:
                running_minutes += minute_delta
            last_minutes_seen = running_minutes

            step_energy_cost = _float(_get(info, "energy_cost_gbp", "delta_energy_cost_gbp", default=0.0))
            if step_energy_cost == 0.0:
                running_energy_spend = _float(_get(env, "total_cost", default=running_energy_spend))
            else:
                running_energy_spend += step_energy_cost

            is_charge = False
            step_type = _get(info, "step_type", default="")
            if isinstance(step_type, str) and step_type.lower() == "charge":
                is_charge = True
            if _bool(_get(info, "charge_event", default=False)):
                is_charge = True
            if is_charge:
                total_charge_events += 1
                last_charge_station = _get(info, "station_id", "charger_id", default=last_charge_station)

            running_time_value = vot_per_min * running_minutes
            running_generalised = running_energy_spend + running_time_value

            soc = _float(_get(info, "soc", "soc_now", "soc_final", default=_get(env, "soc", default=np.nan)))
            remaining_km = _float(_get(info, "remaining_km", "dist_remaining_km", default=np.nan))
            unit_price = _float(_get(info, "energy_unit_price_gbp_per_kwh", "kwh_price_gbp", default=np.nan))

            done = bool(terminated or truncated)

            print(
                f"t={running_minutes:6.2f} min | "
                f"£chg={running_energy_spend:6.2f} | £time={running_time_value:6.2f} | £gen={running_generalised:6.2f} | "
                f"SoC={soc:5.1f}% | Rem={remaining_km:6.2f} km" + (" | CHARGE" if is_charge else "")
            )

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
    ep_energy_cost = _float(_get(info, "episode_cost_gbp", "episode_energy_cost_gbp", default=_get(env, "total_cost", default=0.0)))
    ep_minutes = _float(_get(info, "episode_minutes", default=_get(env, "total_minutes", default=running_minutes)))
    ep_time_value = vot_per_min * ep_minutes
    ep_generalised = ep_energy_cost + ep_time_value

    success = _bool(_get(info, "success", default=_get(env, "success", default=False)))
    stranded = _bool(_get(info, "stranded", default=_get(env, "stranded", default=False)))
    term_reason = _get(info, "termination_reason", default="")
    soc_final = _float(_get(info, "soc_final", default=_get(env, "soc", default=np.nan)))
    remaining_km = _float(_get(info, "remaining_km", default=np.nan))
    charge_events = _int(_get(info, "charge_events", default=total_charge_events))

    summary_fields = [
        "timestamp","user_id","objective","vot_gbp_per_min",
        "episode_minutes","episode_energy_cost_gbp","episode_time_value_gbp","episode_generalised_cost_gbp",
        "success","stranded","soc_final","remaining_km","charge_events","termination_reason",
    ]
    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "user_id": display_user_id,
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

    print("\n----- Episode Summary -----")
    print(f"Objective: {objective}")
    print(f"User ID: {display_user_id}")
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

    # write CSVs
    with open(summary_path, "w", newline="", encoding="utf-8") as f_sum:
        sw = csv.DictWriter(f_sum, fieldnames=summary_fields)
        sw.writeheader()
        sw.writerow(summary)

    return steps_path, summary_path, summary


# -------------------------------
# CLI
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="TripPlan demo of a trained PPO model with correct cost reporting.")
    parser.add_argument("--model", required=True, help="Path to PPO .zip (e.g., runs/.../model.zip or best_model.zip)")
    parser.add_argument("--user", type=int, default=None, help="User id to select (matches TripPlan.user_id if present; else used as index)")
    parser.add_argument("--autopick", action="store_true", help="Probe and pick a ‘doable’ trip (≥60% step success)")
    parser.add_argument("--autopick-off", dest="autopick_off", action="store_true", help="Disable autopick; force --user exactly")
    parser.add_argument("--outdir", default="demo_outputs", help="Directory for CSV outputs")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    model = load_model(args.model)
    env, trips, cfg = build_env_and_trips(args.model, seed=args.seed)

    rng = random.Random(args.seed)
    autopick = args.autopick and not args.autopick_off
    trip, display_user_id = select_trip(env, trips, args.user, autopick, rng)

    steps_csv, summary_csv, _ = run_episode(env, model, trip, display_user_id, outdir=args.outdir)
    print(f"Steps CSV:   {steps_csv}")
    print(f"Summary CSV: {summary_csv}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERROR] presentation.py failed with the following message:\n")
        print(f"  {type(e).__name__}: {e}\n")
        print("Common fixes:")
        print("  • SUMO must be installed & SUMO_HOME set if use_sumo_drive=True.")
        print("  • Model path is wrong or incompatible with current code.")
        print("  • data/sim_users_train_calibrated.csv must exist and parse via rl.episodes.iter_episodes.")
        print("  • If a specific user id doesn’t exist in TripPlans, use --autopick or a valid index.")
        sys.exit(1)
