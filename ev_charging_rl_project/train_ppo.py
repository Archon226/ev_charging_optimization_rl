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

import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

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

        for i, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue

            # Prefer values emitted by env in 'info' at termination
            total_minutes = info.get("episode_minutes", None)
            total_cost    = info.get("episode_cost_gbp", None)
            remaining_km  = info.get("remaining_km", None)
            soc_final     = info.get("soc_final", None)
            episode_steps = info.get("episode_steps", None)
            charge_events = info.get("charge_events", None)
            term_reason   = info.get("termination_reason", None)

            # Fallback to unwrapped env attributes if 'info' is missing (backward-compatible)
            def F(val, name, default):
                return val if val is not None else (GA(name, default))

            total_minutes = F(total_minutes, "total_minutes", 0.0) or 0.0
            total_cost    = F(total_cost,    "total_cost",    0.0) or 0.0
            remaining_km  = F(remaining_km,  "remaining_km",  -1.0)
            soc_final     = F(soc_final,     "soc",           -1.0)
            episode_steps = F(episode_steps, "step_count",    -1)
            charge_events = F(charge_events, "charge_events", 0)
            term_reason   = term_reason or ("success" if (remaining_km is not None and remaining_km <= 0.0)
                                            else "stranded" if (soc_final is not None and soc_final <= 0.0)
                                            else "time_limit" if (episode_steps is not None and episode_steps >= 0)
                                            else "unknown")

            # Flags
            success  = int((remaining_km is not None) and (remaining_km <= 0.0))
            stranded = int((soc_final is not None) and (soc_final <= 0.0))

            row = {
                "time_s":         int(time.time()),
                "timesteps":      int(self.num_timesteps),
                "episode_steps":  int(episode_steps if episode_steps is not None else -1),
                "total_minutes":  float(total_minutes),
                "total_cost_gbp": float(total_cost),
                "success":        success,
                "stranded":       stranded,
                "soc_final":      float(soc_final if soc_final is not None else -1.0),
                "remaining_km":   float(remaining_km if remaining_km is not None else -1.0),
                "charge_events":  int(charge_events if charge_events is not None else 0),
                "termination_reason": term_reason,
                # === Phase 3: new metrics ===
                "violations_repeat":   int(info.get("violations_repeat", 0)),
                "violations_overlimit":int(info.get("violations_overlimit", 0)),
                "violations_cooldown": int(info.get("violations_cooldown", 0)),
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
    sim_csv = data_dir / "sim_users_train_calibrated.csv"
    
    # =========================
    # EXPERIMENT A — (You)
    # =========================
    # Baseline HYBRID, coarse 10-min decisions, standard value-of-time
    # EXPERIMENT = "A"
    # seed = 42
    # RUN_TAG = "Hatim_hybrid_dt10_vot0p05"
    # TOTAL_STEPS = 100_000
    # cfg = PPOEnvConfig(
    #     obs_top_k=5,
    #     dt_minutes=10.0,     # coarser decisions => fewer, longer charges
    #     max_steps=60,        # 60 * 10min = ~10h horizon
    #     prefer="hybrid",     # env still overrides per-episode from TripPlan if set in _reset_state
    #     value_of_time_per_min=0.05,  # £3/h
    #     charge_efficiency=0.92,
    #     charge_session_overhead_min=3.0,  # you just added this in ppo_env.py
    #     traffic_mode="none",
    #     # (optional overrides if you want to tweak peaks) keep these commented
    #     # traffic_peak_factor_am=1.6,
    #     # traffic_peak_factor_pm=1.5,
    #     # traffic_offpeak_factor=1.0,
    # )


    # =========================
    # EXPERIMENT B — (Vishesh)
    # =========================
    # HYBRID but time is more valuable → pushes fewer stops, faster routes
    # Uncomment this block for teammate 1 and comment the others.
    # EXPERIMENT = "B"
    # seed = 101
    # RUN_TAG = "Vishesh_hybrid_dt10_vot0p10"
    # TOTAL_STEPS = 100_000
    # cfg = PPOEnvConfig(
    #     obs_top_k=5,
    #     dt_minutes=10.0,
    #     max_steps=60,
    #     prefer="hybrid",
    #     value_of_time_per_min=0.10,  # £6/h — increases time pressure
    #     charge_efficiency=0.92,
    #     charge_session_overhead_min=3.0,
    #     traffic_mode="none",
        # (optional overrides if you want to tweak peaks)
        # traffic_peak_factor_am=1.6,
        # traffic_peak_factor_pm=1.5,
        # traffic_offpeak_factor=1.0,
    # )



    # EXPERIMENT C — (Divya)
    # =========================
    # HYBRID with fewer candidates for speed/stability (obs_top_k=3)
    # Uncomment this block for teammate 2 and comment the others.
    EXPERIMENT = "C"
    seed = 203
    RUN_TAG = "Hatim_cost_sumo_traffic"
    TOTAL_STEPS = 200_000

    cfg = PPOEnvConfig(
        # --- observation & horizon ---
        obs_top_k=5,                 # small candidate set → faster, stabler learning
        dt_minutes=10.0,             # coarse decisions → fewer, longer charges
        max_steps=84,                # 60 * 10min = 10 hours horizon (enough for long trips)

        # --- objective & costs ---
        prefer="cost",
        respect_trip_objective= False,
        value_of_time_per_min=0.05,
        charge_efficiency=0.92,
        charge_session_overhead_min=3.0,

        # --- Phase 3 constraints (already in your env) ---
        # keep your current penalties & flags in PPOEnvConfig as-is

        # --- Phase 5 shaping (from earlier patch) ---
        enable_shaping=True,
        shaping_gamma=1.0,           # shaping neutral on charge steps
        enable_potential_time=True,
        potential_vref_kmh=25.0,
        idle_penalty_per_step=0.05,
        idle_progress_epsilon_km=0.15,
        micro_charge_penalty=0.5,
        micro_charge_min_kwh=1.0,
        micro_charge_min_minutes=6.0,

        # --- Traffic + SUMO ---
        traffic_mode="light",
        traffic_peak_factor_am=1.6,
        traffic_peak_factor_pm=1.5,
        traffic_offpeak_factor=1.0,

        use_sumo_drive=True,
        sumo_mode="route_time",
        sumo_net_path="london_inner.net.xml",  # set if not at project root
        sumo_gui=False,                       # enable for visual debugging only
        
        max_charges_per_trip=2,
        terminate_on_overlimit=True,   
    )

    
    run_name = time.strftime(f"{RUN_TAG}_ppo_ev_%Y%m%d_%H%M%S")
    out_dir = project_root / "runs" / run_name
    tb_log_dir = out_dir / "tb"
    kpi_csv = out_dir / "kpi_episodes.csv"
    model_path = out_dir / "model.zip"
    config_path = out_dir / "train_config.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- seeds ----
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

    # ---- SUMO sanity (optional) ----
    if cfg.use_sumo_drive:
        if not os.environ.get("SUMO_HOME"):
            print("[warn] SUMO_HOME is not set. traci/sumolib require SUMO to be installed and SUMO_HOME exported.")
        try:
            import traci, sumolib  # noqa: F401
        except Exception as e:
            print(f"[warn] SUMO Python libs not importable: {e}. Training will fail if use_sumo_drive=True.")


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

        # Larger rollouts reduce update noise with single env
        n_steps=4096,                 # collect 8k steps/update
        batch_size=2048,              # large minibatches for stable gradients
        n_epochs=10,                  # PPO default (good here)

        # Optimisation
        learning_rate=lambda f: 1e-4 * f,  # linear decay over training
        gamma=0.995,                 # long horizon (10-min steps ≈ 55h effective)
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=0.2,           # value clipping prevents VF runaway
        ent_coef=0.01,             # gentler exploration to cut dithering
        vf_coef=0.7,
        max_grad_norm=0.5,
        target_kl=0.02,             # guardrails against destabilising updates

        # Network
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU,
            ortho_init=True,
        ),

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
                "n_steps": 4096,
                "batch_size": 2048,
                "n_epochs": 10,
                "learning_rate": "linear_1e-4",
                "gamma": 0.997,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "clip_range_vf": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.7,
                "max_grad_norm": 0.5,
                "target_kl": 0.02,
                "policy_net_arch": [256, 256],
                "value_net_arch": [256, 256],
                "activation": "ReLU",
                "ortho_init": True
            }
        }, f, indent=2)

    # ---- graceful shutdown on SIGINT ----
    stop_training = {"flag": False}

    def _sigint_handler(sig, frame):
        print("\n[train] Caught Ctrl+C — stopping after current update...")
        stop_training["flag"] = True
    signal.signal(signal.SIGINT, _sigint_handler)

    # ---- learn ----
    rolled = 0
    lr_dropped = False            # NEW: track whether we already lowered LR
    while rolled < TOTAL_STEPS and not stop_training["flag"]:
        chunk = min(10_000, TOTAL_STEPS - rolled)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=[kpi_cb])
        rolled += chunk

        # checkpoint
        model.save(str(model_path))
        print(f"[train] checkpoint saved at {rolled}/{TOTAL_STEPS} steps → {model_path}")

        # === OPTIONAL: lower LR mid-run WITHOUT restarting env/model ===
        # Trigger this based on your own heuristic (e.g., tensorboard, CSV tail, or a manual flag).
        # Example: after 100k steps, clamp LR to 7.5e-5 one time.
        if (rolled >= 100_000) and (not lr_dropped):
            new_lr = 7.5e-5
            # 1) Override SB3's lr schedule to a constant
            model.lr_schedule = (lambda _progress_remaining: new_lr)
            # 2) Set optimizer param groups now (so next update uses new LR)
            for pg in model.policy.optimizer.param_groups:
                pg["lr"] = new_lr
            # 3) (Optional) relax/adjust target_kl on-the-fly if you also want gentler updates
            model.target_kl = 0.02
            lr_dropped = True
            print(f"[train] LR clamped mid-run to {new_lr}; target_kl={model.target_kl}")


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
