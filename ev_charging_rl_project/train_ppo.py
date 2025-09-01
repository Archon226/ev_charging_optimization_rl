# train_ppo.py
from __future__ import annotations

import os
import json
import time
import random
import signal
from pathlib import Path
from dataclasses import asdict
from typing import Iterator, Optional, Dict, Any

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

# ---- project imports (assumes you run from project root) ----
from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes, TripPlan
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig


# -----------------------------
# Trip iterator (infinite)
# -----------------------------
def make_trip_iterator(sim_csv: Path, seed: int) -> Iterator[TripPlan]:
    """
    Infinite generator of TripPlans, shuffling each pass to avoid overfitting order.
    """
    rng = np.random.default_rng(seed)
    while True:
        episodes = list(iter_episodes(sim_csv))
        if not episodes:
            raise RuntimeError(f"No episodes found in {sim_csv}")
        idx = np.arange(len(episodes))
        rng.shuffle(idx)
        for i in idx:
            yield episodes[i]


# -----------------------------
# Env factory that always uses a TripPlan
# -----------------------------
class PPOTripEnv(PPOChargingEnv):
    def __init__(self, cfg: PPOEnvConfig, bundle: Dict[str, Any], trip_iter: Iterator[TripPlan]):
        # pricing_catalog & power_model are pulled from bundle (you patched data_loader to include them)
        super().__init__(cfg, data_bundle=bundle)
        self._trip_iter = trip_iter

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        trip = next(self._trip_iter)
        # delegate to base class with our TripPlan
        return super().reset(seed=seed, options={"trip": trip})


# -----------------------------
# Callback: robust KPI logger
# -----------------------------
class EpisodeKpiLogger(BaseCallback):
    """
    Logs per-episode KPIs to a CSV (and TensorBoard scalar) without relying on env 'info' structure.
    Reads unwrapped env attributes at done.
    """
    def __init__(self, log_csv: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_csv = Path(log_csv)
        self._header_written = False

    def _on_training_start(self) -> None:
        self.log_csv.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """
        Robust per-episode KPI logging:
        - Works through DummyVecEnv/Monitor wrappers by using get_attr(indices=0)
        - Writes a CSV row once per episode termination
        - Mirrors key KPIs to TensorBoard
        """
        import time
        import numpy as np

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        if not isinstance(dones, (list, tuple, np.ndarray)) or not dones:
            return True

        # Helper: fetch attribute from the unwrapped env 0
        def GA(name, default=None):
            try:
                return self.training_env.get_attr(name, indices=0)[0]
            except Exception:
                return default

        for done, _ in zip(dones, infos):
            if not done:
                continue

            total_minutes = GA("total_minutes", 0.0) or 0.0
            total_cost    = GA("total_cost", 0.0) or 0.0
            remaining_km  = GA("remaining_km", -1.0)
            soc_final     = GA("soc", -1.0)
            episode_steps = GA("step_count", -1)
            charge_events = GA("charge_events", 0)   # <--- NEW
            
            # Flags
            success  = int((remaining_km is not None) and (remaining_km <= 0.0))
            stranded = int((soc_final is not None) and (soc_final <= 0.0))

            row = {
                "time_s":        int(time.time()),
                "timesteps":     int(self.num_timesteps),
                "episode_steps": int(episode_steps if episode_steps is not None else -1),
                "total_minutes": float(total_minutes),
                "total_cost_gbp":float(total_cost),
                "success":       success,
                "stranded":      stranded,
                "soc_final":     float(soc_final if soc_final is not None else -1.0),
                "remaining_km":  float(remaining_km if remaining_km is not None else -1.0),
                "charge_events": int(charge_events if charge_events is not None else 0),
            }

            # CSV write (header on first write)
            write_header = (not self._header_written) or (not self.log_csv.exists())
            with self.log_csv.open("a", encoding="utf-8") as f:
                if write_header:
                    f.write(",".join(row.keys()) + "\n")
                    self._header_written = True
                f.write(",".join(str(v) for v in row.values()) + "\n")

            # TensorBoard mirrors
            if self.logger is not None:
                self.logger.record("kpi/total_minutes", row["total_minutes"])
                self.logger.record("kpi/total_cost_gbp", row["total_cost_gbp"])
                self.logger.record("kpi/success", row["success"])
                self.logger.record("kpi/stranded", row["stranded"])
                self.logger.record("kpi/charge_events", row["charge_events"])

        return True


# -----------------------------
# Utility: make single-env with Monitor
# -----------------------------
def make_env(bundle, trip_iter, cfg: PPOEnvConfig, log_dir: Path, seed: int):
    def _thunk():
        env = PPOTripEnv(cfg, bundle, trip_iter)
        env = Monitor(env, filename=str(log_dir / "monitor.csv"), allow_early_resets=True)
        env.reset(seed=seed)
        return env
    return _thunk


# -----------------------------
# Main training entrypoint
# -----------------------------
def main():
    # ---- paths & config ----
    project_root = Path(".").resolve()
    data_dir = project_root / "data"
    sim_csv = data_dir / "sim_users_train.csv"

    run_name = time.strftime("ppo_ev_%Y%m%d_%H%M%S")
    out_dir = project_root / "runs" / run_name
    tb_log_dir = out_dir / "tb"
    kpi_csv = out_dir / "kpi_episodes.csv"
    model_path = out_dir / "model.zip"
    config_path = out_dir / "train_config.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- seeds ----
    seed = 42
    set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch seeding (optional)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # ---- load bundle & sanity checks ----
    bundle = load_all_ready(data_dir, strict=True)
    need = ["station_capabilities", "station_connectors_enriched", "ev_capabilities",
            "pricing_index", "ev_curves_index", "pricing_catalog", "ev_power_model"]
    missing = [k for k in need if k not in bundle or bundle[k] is None]
    if missing:
        raise RuntimeError(
            f"load_all_ready missing required bundle keys: {missing}. "
            f"Ensure data_loader returns pricing_catalog and ev_power_model."
        )

    # ---- env config (tune as needed) ----
    cfg = PPOEnvConfig(
        obs_top_k=5,            # keep small; SUMO heavy
        dt_minutes=5.0,         # decision interval
        max_steps=120,          # 10 hours max horizon -> adjust if you prefer shorter
        prefer="hybrid",        # hybrid cost+time objective
        value_of_time_per_min=0.05,  # £3/h
        charge_efficiency=0.92,
    )

    # ---- trip iterator ----
    trip_iter = make_trip_iterator(sim_csv, seed)

    # ---- build env ----
    env_fn = make_env(bundle, trip_iter, cfg, out_dir, seed)
    vec_env = DummyVecEnv([env_fn])

    # ---- model ----
    # Reasonable PPO defaults for MLP with single env
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=str(tb_log_dir),
        n_steps=1024,              # rollout size
        batch_size=1024,           # one update per rollout
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,
        max_grad_norm=0.5,
        device="auto",
        seed=seed,
    )

    # ---- callbacks ----
    kpi_cb = EpisodeKpiLogger(kpi_csv)

    # # Optional evaluation (uses same env factory; OK since SUMO is read-only for routing)
    # eval_env = DummyVecEnv([env_fn])
    # eval_cb = EvalCallback(
    #     eval_env=eval_env,
    #     best_model_save_path=str(out_dir / "best"),
    #     log_path=str(out_dir / "eval"),
    #     eval_freq=10_000,          # evaluate every 10k timesteps
    #     n_eval_episodes=10,
    #     deterministic=False,
    #     render=False,
    # )

    # ---- save config for reproducibility ----
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "ppo_env_config": asdict(cfg),
            "algo": "PPO",
            "hyperparams": {
                "n_steps": 1024,
                "batch_size": 1024,
                "learning_rate": 3e-4,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.1,
                "max_grad_norm": 0.5
            }
        }, f, indent=2)

    # ---- graceful shutdown on SIGINT ----
    stop_training = {"flag": False}

    def _sigint_handler(sig, frame):
        print("\n[train] Caught Ctrl+C — stopping after current update...")
        stop_training["flag"] = True
    signal.signal(signal.SIGINT, _sigint_handler)

    # ---- learn ----
    TOTAL_STEPS = 200_000
    rolled = 0
    while rolled < TOTAL_STEPS and not stop_training["flag"]:
        chunk = min(20_000, TOTAL_STEPS - rolled)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=[kpi_cb])
        rolled += chunk
        # checkpoint
        model.save(str(model_path))
        print(f"[train] checkpoint saved at {rolled}/{TOTAL_STEPS} steps → {model_path}")

    # ---- save final model ----
    model.save(str(out_dir / "model_final.zip"))
    print(f"[train] done. final model → {out_dir/'model_final.zip'}")
    # close envs to stop SUMO
    try:
        vec_env.envs[0].unwrapped.close()
        # eval_env.envs[0].unwrapped.close()
    except Exception:
        pass
    vec_env.close()
    # eval_env.close()


if __name__ == "__main__":
    main()
