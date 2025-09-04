# evaluate_random.py
from __future__ import annotations
import argparse, json, time, random
from pathlib import Path
from typing import Optional, Dict, Any, Iterator, List

import numpy as np
import pandas as pd

# --- project imports ---
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig
from utils.data_loader import load_all_ready
from rl.episodes import TripPlan, load_episodes


# ---------- helpers ----------
def load_cfg_from_run(run_dir: Path, overrides: Dict[str, Any]) -> PPOEnvConfig:
    cfg_json = run_dir / "train_config.json"
    base = {}
    if cfg_json.exists():
        with cfg_json.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            base = obj.get("ppo_env_config", {})
    # Phase-5 safe defaults (won't break older runs)
    phase5_defaults = dict(
        enable_shaping=base.get("enable_shaping", True),
        shaping_gamma=base.get("shaping_gamma", 1.0),
        enable_potential_time=base.get("enable_potential_time", True),
        potential_vref_kmh=base.get("potential_vref_kmh", 25.0),
        idle_penalty_per_step=base.get("idle_penalty_per_step", 0.05),
        idle_progress_epsilon_km=base.get("idle_progress_epsilon_km", 0.15),
        micro_charge_penalty=base.get("micro_charge_penalty", 0.5),
        micro_charge_min_kwh=base.get("micro_charge_min_kwh", 1.0),
        micro_charge_min_minutes=base.get("micro_charge_min_minutes", 6.0),
    )
    merged = {**base, **phase5_defaults, **overrides}
    return PPOEnvConfig(**merged)


def make_trip_iterator(sim_csv: Path, seed: int) -> Iterator[TripPlan]:
    rng = np.random.default_rng(seed)
    episodes: List[TripPlan] = load_episodes(sim_csv, default_target_soc=80.0)
    if not episodes:
        raise RuntimeError(f"No episodes found in {sim_csv}")
    idx = np.arange(len(episodes))
    while True:
        rng.shuffle(idx)
        for i in idx:
            yield episodes[int(i)]


def pick_random_legal_action(env: PPOChargingEnv, info: Dict[str, Any] | None) -> int:
    """
    Tries to get a boolean mask of valid actions; falls back to full space.
    Assumes discrete action space (Drive + N stations).
    """
    mask = None

    # 1) env.get_action_mask() if available
    if hasattr(env, "get_action_mask"):
        try:
            m = env.get_action_mask()
            if m is not None:
                mask = np.asarray(m, dtype=bool)
        except Exception:
            mask = None

    # 2) info["action_mask"] from last step/reset
    if mask is None and info is not None:
        m = info.get("action_mask") or info.get("mask") or None
        if m is not None:
            mask = np.asarray(m, dtype=bool)

    # 3) env.available_actions() returning indices
    if mask is None and hasattr(env, "available_actions"):
        try:
            idxs = list(env.available_actions())
            if idxs:
                mask = np.zeros(env.action_space.n, dtype=bool)
                mask[idxs] = True
        except Exception:
            mask = None

    # 4) fallback: all actions legal
    if mask is None:
        mask = np.ones(env.action_space.n, dtype=bool)

    legal = np.flatnonzero(mask)
    if len(legal) == 0:
        # If nothing is legal, just return 0 as a safe default
        return 0
    return int(random.choice(legal))


# ---------- main eval ----------
def evaluate_random(
    run_dir: Path,
    data_dir: Path,
    sim_csv: Path,
    out_dir: Path,
    episodes: int,
    seed: int,
    sumo_mode: str,
    sumo_gui: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # SUMO sanity (warn if missing)
    if sumo_mode in ("route_time", "microsim"):
        try:
            import traci, sumolib  # noqa: F401
        except Exception as e:
            print(f"[warn] SUMO libs not importable: {e}. If use_sumo_drive=True, steps may fail.")

    # data + cfg
    bundle = load_all_ready(data_dir, strict=True)
    cfg = load_cfg_from_run(
        run_dir,
        overrides=dict(
            sumo_mode=sumo_mode,
            sumo_gui=bool(sumo_gui),
            use_sumo_drive=True,
        ),
    )

    # env + trip iterator
    trip_iter = make_trip_iterator(sim_csv, seed)
    env = PPOChargingEnv(cfg, data_bundle=bundle)
    ep_writer = (out_dir / "eval_random_kpi_episodes.csv").open("w", encoding="utf-8")
    header = [
        "episode","episode_steps","total_minutes","total_cost_gbp",
        "success","stranded","soc_final","remaining_km","charge_events",
        "violations_repeat","violations_overlimit","violations_cooldown",
        "termination_reason","action_mask_size"
    ]
    ep_writer.write(",".join(header) + "\n")
    ep_writer.flush()

    rng = np.random.default_rng(seed)
    random.seed(seed)

    # loop
    done_eps = 0
    obs, info = env.reset(options={"trip": next(trip_iter)})
    while done_eps < episodes:
        a = pick_random_legal_action(env, info)
        obs, reward, terminated, truncated, info = env.step(a)

        if terminated or truncated:
            k = info.get("terminal_kpis", {}) or {}
            row = dict(
                episode=done_eps,
                episode_steps=int(k.get("steps", info.get("episode_steps", 0))),
                total_minutes=float(k.get("minutes", info.get("total_minutes", 0.0))),
                total_cost_gbp=float(k.get("cost", info.get("total_cost_gbp", 0.0))),
                success=int(k.get("success", info.get("success", False))) or 0,
                stranded=int(k.get("stranded", info.get("stranded", False))) or 0,
                soc_final=float(k.get("soc_final", info.get("soc_final", 0.0))),
                remaining_km=float(k.get("remaining_km", info.get("remaining_km", 0.0))),
                charge_events=int(k.get("charge_events", info.get("charge_events", 0))),
                violations_repeat=int(info.get("violations_repeat", 0)),
                violations_overlimit=int(info.get("violations_overlimit", 0)),
                violations_cooldown=int(info.get("violations_cooldown", 0)),
                termination_reason=str(k.get("termination_reason", info.get("termination_reason", ""))),
                action_mask_size=int(info.get("action_mask_size", 0)),
            )
            ep_writer.write(",".join(str(row[h]) for h in header) + "\n")
            ep_writer.flush()
            done_eps += 1

            # next episode
            obs, info = env.reset(options={"trip": next(trip_iter)})

    ep_writer.close()
    env.close()

    # quick summary
    df = pd.read_csv(out_dir / "eval_random_kpi_episodes.csv")
    if len(df):
        summary = {
            "episodes": int(len(df)),
            "success_rate": float(df["success"].mean()),
            "avg_minutes": float(df["total_minutes"].mean()),
            "avg_cost_gbp": float(df["total_cost_gbp"].mean()),
            "avg_charges": float(df["charge_events"].mean()),
            "violations_repeat_total": int(df["violations_repeat"].sum()),
            "violations_overlimit_total": int(df["violations_overlimit"].sum()),
            "violations_cooldown_total": int(df["violations_cooldown"].sum()),
        }
        with (out_dir / "eval_random_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("[random-eval] summary:", json.dumps(summary, indent=2))
    print("[random-eval] wrote", out_dir / "eval_random_kpi_episodes.csv")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Sanity-check env with a Random policy.")
    p.add_argument("--run", type=Path, required=True,
                   help="Path to run dir with train_config.json (e.g., runs/ppo_ev_20250903)")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--sim-csv", type=Path, default=Path("data")/"sim_users_eval_calibrated.csv")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Defaults to <run>/eval_random/")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sumo-mode", choices=["route_time","microsim"], default="route_time")
    p.add_argument("--sumo-gui", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = args.run.resolve()
    out_dir = args.out_dir or (run_dir / "eval_random")
    evaluate_random(
        run_dir=run_dir,
        data_dir=args.data_dir.resolve(),
        sim_csv=args.sim_csv.resolve(),
        out_dir=Path(out_dir).resolve(),
        episodes=args.episodes,
        seed=args.seed,
        sumo_mode=args.sumo_mode,
        sumo_gui=args.sumo_gui,
    )

if __name__ == "__main__":
    main()
