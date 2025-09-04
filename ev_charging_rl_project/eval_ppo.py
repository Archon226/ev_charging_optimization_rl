# evaluate_ppo.py
from __future__ import annotations

import os, json, time, argparse
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# --- project imports ---
# repo root is the cwd when you run this; adjust if needed
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig
from rl.episodes import TripPlan
from utils.data_loader import load_all_ready


# ----------------------------
# Episode iterator (eval CSV)
# ----------------------------
def _parse_eval_csv(sim_csv: Path, default_target_soc: float = 80.0) -> List[TripPlan]:
    """
    Minimal loader to avoid changing your training code:
    Expects columns compatible with rl/episodes.load_episodes.
    """
    from rl.episodes import load_episodes as _load
    return _load(sim_csv, default_target_soc=default_target_soc)


def make_trip_iterator(sim_csv: Path, seed: int) -> Iterator[TripPlan]:
    rng = np.random.default_rng(seed)
    while True:
        episodes = _parse_eval_csv(sim_csv)
        if not episodes:
            raise RuntimeError(f"No episodes found in {sim_csv}")
        idx = np.arange(len(episodes))
        rng.shuffle(idx)
        for i in idx:
            yield episodes[i]


# ----------------------------
# Env wrapper: inject TripPlan
# ----------------------------
class PPOTripEnv(PPOChargingEnv):
    """Wrapper that always injects a TripPlan on reset() for SB3 compatibility."""
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


# ----------------------------
# Load PPOEnvConfig from run
# ----------------------------
def load_cfg_from_train(model_path: Path, overrides: Dict[str, Any]) -> PPOEnvConfig:
    """
    Reads train_config.json next to the model (created in train_ppo.py),
    extracts "ppo_env_config", then applies overrides (e.g., sumo_mode/gui).
    Falls back to defaults if file/keys are missing.
    """
    run_dir = model_path.parent
    cfg_json = run_dir / "train_config.json"
    base = {}
    if cfg_json.exists():
        with cfg_json.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            base = obj.get("ppo_env_config", {})
    # Phase-5 knobs: ensure present with safe defaults (in case run predates Phase 5)
    phase5_safe_defaults = dict(
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
    merged = {**base, **phase5_safe_defaults, **overrides}
    return PPOEnvConfig(**merged)


# ----------------------------
# Evaluation loop
# ----------------------------
def evaluate(
    model_path: Path,
    data_dir: Path,
    sim_csv: Path,
    out_dir: Path,
    sumo_mode: str = "route_time",    # or "microsim"
    episodes: int = 50,
    seed: int = 202,
    sumo_gui: bool = False,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # SUMO sanity (warn early if missing)
    if sumo_mode in ("route_time", "microsim"):
        try:
            import traci, sumolib  # noqa: F401
        except Exception as e:
            print(f"[warn] SUMO Python libs not importable: {e}. If use_sumo_drive=True, eval will fail.")

    # --- data bundle (fast lookups) ---
    bundle = load_all_ready(data_dir, strict=True)

    # --- env config from training, override SUMO mode/gui ---
    cfg = load_cfg_from_train(
        model_path,
        overrides=dict(
            sumo_mode=sumo_mode,
            sumo_gui=bool(sumo_gui),
            # common defaults if a saved config had relative paths:
            sumo_net_path=str((Path(".") / "london_inner.net.xml").resolve()),
            use_sumo_drive=True,          # ensure we're exercising SUMO path times
        ),
    )

    # --- build vec env with trip injection ---
    trip_iter = make_trip_iterator(sim_csv, seed)
    env_fn = make_env(bundle, trip_iter, cfg, out_dir, seed)
    vec_env = DummyVecEnv([env_fn])

    # --- load PPO model ---
    print(f"[eval] loading model: {model_path}")
    model = PPO.load(str(model_path), env=vec_env, device="auto")

    # --- per-episode KPI CSV ---
    kpi_path = out_dir / "eval_kpi_episodes.csv"
    header = [
        "time_s","episode","episode_steps",
        "total_minutes","total_cost_gbp",
        "success","stranded",
        "soc_final","remaining_km","charge_events",
        # Phase 3 penalties (if your env logs them)
        "violations_repeat","violations_overlimit","violations_cooldown",
        # Phase 5 fields (optional, present if you merged the changes)
        "progress_km","shaping_time","penalty_idle","penalty_micro_charge",
        # Misc telemetry you already write
        "termination_reason","detour_minutes","action_mask_size",
    ]
    with kpi_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")

    # --- run episodes (deterministic policy) ---
    done_eps = 0
    t0 = time.time()
    obs = vec_env.reset()
    while done_eps < episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = vec_env.step(action)

        if done[0]:
            info = infos[0]
            ep_info = info.get("terminal_kpis", {}) or {}
            # defaults if fields are absent (back-compat)
            row = dict(
                time_s=round(time.time() - t0, 3),
                episode=done_eps,
                episode_steps=int(info.get("episode_steps", ep_info.get("steps", 0))),
                total_minutes=float(info.get("episode_minutes", ep_info.get("minutes", 0.0))),
                total_cost_gbp=float(info.get("episode_cost_gbp", ep_info.get("cost", 0.0))),
                success=int(ep_info.get("success", info.get("success", False))) or 0,
                stranded=int(ep_info.get("stranded", info.get("stranded", False))) or 0,
                soc_final=float(ep_info.get("soc_final", info.get("soc_final", 0.0))),
                remaining_km=float(ep_info.get("remaining_km", info.get("remaining_km", 0.0))),
                charge_events=int(ep_info.get("charge_events", info.get("charge_events", 0))),
                violations_repeat=int(info.get("violations_repeat", 0)),
                violations_overlimit=int(info.get("violations_overlimit", 0)),
                violations_cooldown=int(info.get("violations_cooldown", 0)),
                progress_km=float(info.get("progress_km", 0.0)),
                shaping_time=float(info.get("shaping_time", 0.0)),
                penalty_idle=float(info.get("penalty_idle", 0.0)),
                penalty_micro_charge=float(info.get("penalty_micro_charge", 0.0)),
                termination_reason=str(ep_info.get("termination_reason", info.get("termination_reason", ""))),
                detour_minutes=float(info.get("detour_minutes", 0.0)),
                action_mask_size=int(info.get("action_mask_size", 0)),
            )
            with kpi_path.open("a", encoding="utf-8") as f:
                f.write(",".join(str(row[k]) for k in header) + "\n")

            print(
                f"[eval] ep={row['episode']:>3} steps={row['episode_steps']:>4} "
                f"min={row['total_minutes']:.1f} £={row['total_cost_gbp']:.2f} "
                f"success={row['success']} stranded={row['stranded']} "
                f"charges={row['charge_events']}"
            )
            done_eps += 1
            obs = vec_env.reset()

    # --- aggregate summary ---
    summary = aggregate_summary(kpi_path)
    with (out_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # --- clean shutdown ---
    try:
        vec_env.envs[0].unwrapped.close()
    except Exception:
        pass
    vec_env.close()
    print(f"[eval] done.\n  KPIs → {kpi_path}\n  Summary → {out_dir/'eval_summary.json'}")


def aggregate_summary(kpi_csv: Path) -> Dict[str, Any]:
    import pandas as pd
    df = pd.read_csv(kpi_csv)
    n = len(df)
    if n == 0:
        return {}
    succ = df["success"].sum()
    stranded = df["stranded"].sum()
    out = dict(
        episodes=int(n),
        success_rate=float(succ / n),
        strand_rate=float(stranded / n),
        avg_minutes=float(df["total_minutes"].mean()),
        med_minutes=float(df["total_minutes"].median()),
        avg_cost_gbp=float(df["total_cost_gbp"].mean()),
        med_cost_gbp=float(df["total_cost_gbp"].median()),
        avg_steps=float(df["episode_steps"].mean()),
        avg_charges=float(df["charge_events"].mean()),
        p95_minutes=float(df["total_minutes"].quantile(0.95)),
        p95_cost_gbp=float(df["total_cost_gbp"].quantile(0.95)),
        violations_repeat_total=int(df["violations_repeat"].sum()),
        violations_overlimit_total=int(df["violations_overlimit"].sum()),
        violations_cooldown_total=int(df["violations_cooldown"].sum()),
        # Phase 5 diagnostics (if present)
        avg_progress_km=float(df.get("progress_km", 0.0).mean()) if "progress_km" in df else 0.0,
        avg_shaping_time=float(df.get("shaping_time", 0.0).mean()) if "shaping_time" in df else 0.0,
        micro_charge_hits=int((df.get("penalty_micro_charge", 0.0) > 0).sum()) if "penalty_micro_charge" in df else 0,
        idle_hits=int((df.get("penalty_idle", 0.0) > 0).sum()) if "penalty_idle" in df else 0,
    )
    return out


# ----------------------------
# CLI
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO model on the EV charging env.")
    p.add_argument("--model", required=True, type=Path,
                   help="Path to runs/<run>/model.zip (e.g., runs/ppo_ev_20250903/model.zip)")
    p.add_argument("--episodes", type=int, default=50, help="Number of eval episodes")
    p.add_argument("--sumo-mode", choices=["route_time","microsim"], default="route_time",
                   help="Which SUMO integration to use during eval")
    p.add_argument("--sumo-gui", action="store_true", help="Show SUMO GUI during eval")
    p.add_argument("--sim-csv", type=Path, default=Path("data") / "sim_users_eval.csv",
                   help="Path to evaluation episode CSV")
    p.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write eval logs (default: alongside model as eval_<mode>/)")
    p.add_argument("--seed", type=int, default=202, help="RNG seed")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = model_path.parent / f"eval_{args.sumo_mode}"
    evaluate(
        model_path=model_path,
        data_dir=args.data_dir.resolve(),
        sim_csv=args.sim_csv.resolve(),
        out_dir=out_dir.resolve(),
        sumo_mode=args.sumo_mode,
        episodes=args.episodes,
        seed=args.seed,
        sumo_gui=args.sumo_gui,
    )


if __name__ == "__main__":
    main()
