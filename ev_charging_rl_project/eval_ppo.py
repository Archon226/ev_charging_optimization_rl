# eval_ppo.py
from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path
from dataclasses import asdict
from typing import Iterator, Optional, Dict, Any

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# ---- project imports (assumes you run from project root) ----
from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes, TripPlan
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig


def make_trip_iterator(sim_csv: Path, seed: int) -> Iterator[TripPlan]:
    """Infinite generator of TripPlans from a CSV, shuffling each pass."""
    rng = np.random.default_rng(seed)
    while True:
        episodes = list(iter_episodes(sim_csv))
        if not episodes:
            raise RuntimeError(f"No episodes found in {sim_csv}")
        idx = np.arange(len(episodes))
        rng.shuffle(idx)
        for i in idx:
            yield episodes[i]


class PPOTripEnv(PPOChargingEnv):
    """Env wrapper that always injects a TripPlan on reset() for SB3 compatibility."""
    def __init__(self, cfg: PPOEnvConfig, bundle: Dict[str, Any], trip_iter: Iterator[TripPlan]):
        super().__init__(cfg, data_bundle=bundle)
        self._trip_iter = trip_iter

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        trip = next(self._trip_iter)
        return super().reset(seed=seed, options={"trip": trip})


def make_env(bundle, trip_iter, cfg: PPOEnvConfig, log_dir: Path, seed: int):
    def _thunk():
        env = PPOTripEnv(cfg, bundle, trip_iter)
        env = Monitor(env, filename=str(log_dir / "monitor.csv"), allow_early_resets=True)
        env.reset(seed=seed)
        return env
    return _thunk


def load_cfg_from_train(model_path: Path, overrides: Dict[str, Any]) -> PPOEnvConfig:
    """
    Try to load PPOEnvConfig from the training run's train_config.json next to the model.
    Apply 'overrides' (e.g., sumo_mode='microsim') afterwards.
    Fallback to a sensible default if not found.
    """
    run_dir = model_path.parent
    cfg_json = run_dir / "train_config.json"
    base = {}
    if cfg_json.exists():
        with cfg_json.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            base = obj.get("ppo_env_config", {})
    # Construct config (handles missing keys via dataclass defaults)
    cfg = PPOEnvConfig(**{**base, **overrides})
    return cfg


def evaluate(
    model_path: Path,
    data_dir: Path,
    sim_csv: Path,
    out_dir: Path,
    sumo_mode: str,
    episodes: int,
    seed: int,
    sumo_gui: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- seeds ----
    set_random_seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # ---- SUMO sanity (optional) ----
    if not os.environ.get("SUMO_HOME"):
        print("[warn] SUMO_HOME is not set. SUMO must be installed and SUMO_HOME exported.")
    try:
        import traci, sumolib  # noqa: F401
    except Exception as e:
        print(f"[warn] SUMO Python libs not importable: {e}. If use_sumo_drive=True, eval will fail.")

    # ---- load bundle ----
    bundle = load_all_ready(data_dir, strict=True)

    # ---- config: load from train_config.json next to model, override sumo mode/gui ----
    cfg = load_cfg_from_train(model_path, overrides={
        "sumo_mode": sumo_mode,
        "sumo_gui": bool(sumo_gui),
        # in case a saved config had a relative path, ensure common defaults:
        # try typical location for the net if missing
        "sumo_net_path": str((Path(".") / "london_inner.net.xml").resolve()),
        "use_sumo_drive": True,
    })

    # ---- build env ----
    trip_iter = make_trip_iterator(sim_csv, seed)
    env_fn = make_env(bundle, trip_iter, cfg, out_dir, seed)
    vec_env = DummyVecEnv([env_fn])

    # ---- load model ----
    print(f"[eval] loading model: {model_path}")
    model = PPO.load(str(model_path), env=vec_env, device="auto")

    # ---- run episodes & log KPIs ----
    kpi_path = out_dir / "eval_kpi_episodes.csv"
    header = [
        "time_s","episode","episode_steps","total_minutes","total_cost_gbp",
        "success","stranded","soc_final","remaining_km","charge_events"
    ]
    with kpi_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

    # Roll N episodes manually so we can collect env KPIs each time
    done_eps = 0
    total_steps = 0
    while done_eps < episodes:
        obs = vec_env.reset()
        ep_steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, dones, infos = vec_env.step(action)
            ep_steps += 1
            total_steps += 1
            if dones[0]:
                # pull attributes from the unwrapped env 0
                env0 = vec_env.envs[0].unwrapped
                total_minutes = getattr(env0, "total_minutes", 0.0) or 0.0
                total_cost    = getattr(env0, "total_cost", 0.0) or 0.0
                remaining_km  = getattr(env0, "remaining_km", -1.0)
                soc_final     = getattr(env0, "soc", -1.0)
                charge_events = getattr(env0, "charge_events", 0)

                success  = int((remaining_km is not None) and (remaining_km <= 0.0))
                stranded = int((soc_final is not None) and (soc_final <= 0.0))

                row = [
                    str(int(time.time())),
                    str(done_eps + 1),
                    str(ep_steps),
                    f"{float(total_minutes):.6f}",
                    f"{float(total_cost):.6f}",
                    str(success),
                    str(stranded),
                    f"{float(soc_final if soc_final is not None else -1.0):.6f}",
                    f"{float(remaining_km if remaining_km is not None else -1.0):.6f}",
                    str(int(charge_events if charge_events is not None else 0)),
                ]
                with kpi_path.open("a", encoding="utf-8") as f:
                    f.write(",".join(row) + "\n")

                print(f"[eval] episode {done_eps+1}/{episodes}: "
                      f"minutes={total_minutes:.1f}, cost={total_cost:.2f}, "
                      f"success={success}, stranded={stranded}, charges={charge_events}")
                done_eps += 1
                break

    # ---- clean shutdown ----
    try:
        vec_env.envs[0].unwrapped.close()
    except Exception:
        pass
    vec_env.close()
    print(f"[eval] done. KPIs → {kpi_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on EV SUMO env.")
    p.add_argument("--model", required=True, type=Path,
                   help="Path to model.zip from /runs/<run>/model.zip")
    p.add_argument("--episodes", type=int, default=50, help="Number of eval episodes")
    p.add_argument("--sumo-mode", choices=["route_time","microsim"], default="route_time",
                   help="Which SUMO integration to use during eval")
    p.add_argument("--sumo-gui", action="store_true", help="Show SUMO GUI during eval")
    p.add_argument("--sim-csv", type=Path, default=Path("data") / "sim_users_eval.csv",
                   help="Path to evaluation episode CSV")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write eval logs (default: alongside model)")
    p.add_argument("--seed", type=int, default=202, help="RNG seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = args.model.resolve()
    if args.out_dir is None:
        # default next to the model: runs/<run>/eval_<mode>/
        args.out_dir = model_path.parent / f"eval_{args.sumo_mode}"
    evaluate(
        model_path=model_path,
        data_dir=args.data_dir.resolve(),
        sim_csv=args.sim_csv.resolve(),
        out_dir=args.out_dir.resolve(),
        sumo_mode=args.sumo_mode,
        episodes=args.episodes,
        seed=args.seed,
        sumo_gui=args.sumo_gui,
    )
