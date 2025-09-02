#!/usr/bin/env python3
"""
Single-agent PPO trainer for EV charging route optimization over SUMO.

- Uses your existing modules (no edits required):
  * env.ev_env.EVChargingEnv
  * simulator.sumo_adapter.SumoSim
  * rl.episodes.load_episodes
  * utils.data_loader.load_all_data (returns PricingCatalog inside)
  * utils.charging_curves.EVPowerModel

- Windows-friendly by default: DummyVecEnv, no subprocess unless requested.

Example:
python -m rl.train ^
  --objective cost ^
  --n-envs 1 --no-subproc ^
  --data-dir "./data" ^
  --sumo-net "./london_inner.net.xml" ^
  --timesteps 50000 ^
  --log-dir "./runs/ppo_ev_cost_test"
"""

from __future__ import annotations
import os
import json
import argparse
import inspect
from functools import partial
from typing import Any, Dict, Callable
from datetime import datetime
from pathlib import Path
os.environ["EV_PRICING_DEBUG"] = "1"
os.environ["EV_REQUIRE_CHARGE"] = "1"
# --- third-party ---
# (Install: pip install stable-baselines3[extra] gymnasium sumolib)
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# --- project modules ---
from env.ev_env import EVChargingEnv
from rl.episodes import load_episodes
from simulator.sumo_adapter import SumoSim
from utils.data_loader import load_all_data
from utils.charging_curves import EVPowerModel
from env.logging_wrapper import LoggingWrapper


# -------------------------
# Helpers
# -------------------------
def filter_kwargs_for_ctor(ctor: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Pass only arguments that the constructor actually accepts."""
    sig = inspect.signature(ctor)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def make_env_factory(
    env_ctor: Callable,
    env_kwargs: Dict[str, Any],
    seed: int | None = None,
    wrapper_factory: Callable[[Any], Any] | None = None,
) -> Callable[[], Monitor]:
    """Return a thunk that creates a fresh, monitored env (for SB3)."""
    def _init():
        # Filter again at instantiation time (safe even if already filtered)
        ctor_kwargs = filter_kwargs_for_ctor(env_ctor, dict(env_kwargs))
        env = env_ctor(**ctor_kwargs)
        if seed is not None:
            # Gymnasium-compatible seeding on reset call
            env.reset(seed=seed)
            
         # <-- NEW: wrap with LoggingWrapper if provided
        if wrapper_factory is not None:
            env = wrapper_factory(env)
        # Wrap with Monitor to record episode stats for callbacks/eval
        log_dir = env_kwargs.get("log_dir", None)
        return Monitor(env, filename=None if log_dir is None else os.path.join(log_dir, "monitor"))
    return _init


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train PPO agent for EV charging optimization")

    # Objective / reward mixing
    p.add_argument("--objective", type=str, default="hybrid", choices=["time", "cost", "hybrid"],
                   help="Primary training objective")
    p.add_argument("--hybrid-alpha", type=float, default=0.5,
                   help="When objective=hybrid, weight on time (cost gets 1-alpha)")

    # Data / net paths
    p.add_argument("--data-dir", type=str, default="./data",
                   help="Directory with CSVs (EV_Metadata.csv, pricing_*.csv, charging_station_*.csv, simulated_users.csv)")
    p.add_argument("--sumo-net", type=str, default="./london_inner.net.xml",
                   help="SUMO .net.xml path")

    # Vec env & runtime
    p.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs")
    p.add_argument("--no-subproc", action="store_true", help="Force DummyVecEnv (Windows-safe)")
    p.add_argument("--seed", type=int, default=42)

    # Env knobs (only passed if accepted by EVChargingEnv.__init__)
    p.add_argument("--top-k-default", type=int, default=10)
    p.add_argument("--max-detour-km-default", type=float, default=10.0)
    p.add_argument("--normalize-obs", action="store_true", default=False)
    p.add_argument("--normalize-rewards", action="store_true", default=False)
    p.add_argument("--reward-norm-factor", type=float, default=100.0)
    p.add_argument("--fast-mode", action="store_true", default=False)

    # PPO
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--n-epochs", type=int, default=10)

    # Logging / outputs
    p.add_argument("--log-dir", type=str, default="./runs/ppo_ev")
    p.add_argument("--save-freq", type=int, default=50_000, help="Steps between checkpoints")
    p.add_argument("--eval-freq", type=int, default=50_000, help="Steps between evals")
    
    #CLI args
    p.add_argument("--run-id", type=str, default=None, help="Optional run id to nest logs under")
    p.add_argument("--log-every-n", type=int, default=1, help="Write 1 in N step events to CSV (1 = all)")

    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- paths ---
    log_dir = ensure_dir(args.log_dir)
    tb_dir = ensure_dir(os.path.join(log_dir, "tb"))
    models_dir = ensure_dir(os.path.join(log_dir, "models"))

    # ---- Logging wrapper config (events end up under <log_dir>/events) ----
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = args.run_id or f"{args.objective}_{run_stamp}"
    events_dir = ensure_dir(os.path.join(log_dir, "events", rid))
    wrapper_factory = lambda env: LoggingWrapper(env, log_dir=events_dir, run_id=rid, log_every_n=args.log_every_n)

    
    # --- data bundle (stations, connectors, EV metadata/curves, users, pricing) ---
    bundle = load_all_data(args.data_dir)
    stations = bundle.stations
    connectors = bundle.connectors
    pricing = bundle.pricing_catalog

    # Episodes (required)
    users_csv = os.path.join(args.data_dir, "simulated_users.csv")
    episodes = load_episodes(users_csv)

    # SUMO
    sumo_net = os.path.abspath(args.sumo_net)
    sim = SumoSim(sumo_net)

    # EV power/curves catalog (single object; env can use per-episode model info)
    ev_power = EVPowerModel(bundle.ev_metadata, bundle.charging_curves)

    # --- Build base kwargs for EVChargingEnv and filter by ctor signature ---
    base_env_kwargs: Dict[str, Any] = dict(
        sim=sim,
        stations=stations,
        connectors=connectors,
        pricing=pricing,
        ev_spec=ev_power,        # catalog-like; env can use trip.ev_model
        episodes=episodes,
        objective=args.objective,
        hybrid_alpha=args.hybrid_alpha,
        top_k_default=args.top_k_default,
        max_detour_km_default=args.max_detour_km_default,
        normalize_obs=args.normalize_obs,
        normalize_rewards=args.normalize_rewards,
        reward_norm_factor=args.reward_norm_factor,
        fast_mode=args.fast_mode,
        seed=args.seed,
        # not part of env ctor; only for Monitor path if you want per-env files
        log_dir=log_dir,
    )
    filtered_env_kwargs = filter_kwargs_for_ctor(EVChargingEnv.__init__, base_env_kwargs)

    # --- Create vectorized env(s) ---
    make_thunk = partial(
        make_env_factory,
        EVChargingEnv,
        filtered_env_kwargs,
        seed=args.seed,
        wrapper_factory=wrapper_factory,   # <-- added
)

    if args.n_envs == 1 or args.no_subproc:
        vec = DummyVecEnv([make_thunk()])
    else:
        vec = SubprocVecEnv([make_thunk() for _ in range(args.n_envs)])

    # Optional: normalize obs/rewards externally (independent of env's internal normalizers)
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # --- Eval env (single, deterministic) ---
    eval_env = DummyVecEnv([make_thunk()])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # --- PPO model ---
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        policy="MlpPolicy",
        env=vec,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        tensorboard_log=tb_dir,
        verbose=1,
        policy_kwargs=policy_kwargs,
        device="auto",
        seed=args.seed,
    )

    # --- Callbacks: checkpoints + eval ---
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // max(1, args.n_envs)),
        save_path=models_dir,
        name_prefix=f"ppo_ev_{args.objective}"
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # --- Persist config ---
    with open(os.path.join(log_dir, "run_config.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "sumo_net": sumo_net,  # already absolute above
                "data_dir": os.path.abspath(args.data_dir),
                "env_init_kwargs": sorted(list(filtered_env_kwargs.keys())),
            },
            f,
            indent=2,
        )

    # --- Train ---
    model.learn(total_timesteps=args.timesteps, callback=[ckpt_cb, eval_cb], progress_bar=True)

    # --- Save final model + vec norm stats ---
    model.save(os.path.join(models_dir, f"ppo_ev_{args.objective}_final"))
    vec.save(os.path.join(models_dir, "vecnormalize.pkl"))

    print("✅ Training complete. Outputs in:", os.path.abspath(log_dir))


if __name__ == "__main__":
    # Required for Windows when using SubprocVecEnv
    main()
