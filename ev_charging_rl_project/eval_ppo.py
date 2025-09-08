# eval_ppo.py
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

# --- your project imports (same as train_ppo.py)
from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes, TripPlan
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig

# -----------------------------
# Finite Trip iterator for evaluation
# -----------------------------
def load_eval_trips(eval_csv: Path) -> list[TripPlan]:
    trips = list(iter_episodes(eval_csv))  # same generator you use in training
    if not trips:
        raise RuntimeError(f"No episodes found in {eval_csv}")
    return trips

class EvalTripEnv(PPOChargingEnv):
    """
    Same env as training but pulls TripPlan from a finite list.
    """
    def __init__(self, cfg: PPOEnvConfig, bundle: dict, trips: list[TripPlan]):
        super().__init__(cfg, data_bundle=bundle)
        self._trips = trips
        self._idx = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self._idx >= len(self._trips):
            raise StopIteration("All evaluation trips consumed. Re-create env to run again.")
        trip = self._trips[self._idx]
        self._idx += 1
        # base class expects options={"trip": TripPlan}
        return super().reset(seed=seed, options={"trip": trip})

# -----------------------------
# Evaluation config (edit paths)
# -----------------------------
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
EVAL_CSV = DATA_DIR / "sim_users_eval_calibrated.csv"

# Point these to your final models
MODELS = {
    "cost":   PROJECT_ROOT / "runs" / "Hatim_cost_sumo_traffic_ppo_ev_20250907_024758"  / "model_final.zip",
    "time":   PROJECT_ROOT / "runs" / "Hatim_time_sumo_traffic_ppo_ev_20250906_195009"  / "model_final.zip",
    "hybrid": PROJECT_ROOT / "runs" / "Hatim_hybrid_sumo_traffic_ppo_ev_20250907_110917" / "model_final.zip",
}

OUT_DIR = PROJECT_ROOT / "eval_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Build shared bundle and trips
# -----------------------------
bundle = load_all_ready(DATA_DIR, strict=True)  # same as training
eval_trips = load_eval_trips(EVAL_CSV)         # finite list (one pass)  :contentReference[oaicite:3]{index=3}
N_EVAL = len(eval_trips)

# -----------------------------
# Evaluate all policies
# -----------------------------
all_rows = []

for policy_name, model_path in MODELS.items():
    run_dir = Path(model_path).parent

    # cfg.prefer is irrelevant at inference, but we set it for clarity/logs.
    cfg = PPOEnvConfig(
        obs_top_k=5,
        dt_minutes=10.0,
        max_steps=84,
        prefer=policy_name,            # "cost" | "time" | "hybrid"
        respect_trip_objective=False,  # IMPORTANT: evaluation uses the trained policy; we still log user's objective below
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

    # Create a FRESH env with a fresh copy of trips (so each policy sees the same set, in the same order)
    trips_copy = list(eval_trips)  # shallow copy OK; TripPlan is immutable enough for eval use
    env = EvalTripEnv(cfg, bundle, trips_copy)

    # Load model
    model = PPO.load(str(model_path), env=env)

    # Roll through the entire eval set once
    for i in range(N_EVAL):
        obs, info = env.reset()
        # record the user's declared objective from the active trip
        user_obj = getattr(env._trip, "objective", None)
        done = False
        terminated = False
        truncated = False
        last_info = {}
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)

        # episode-end KPIs are in last_info (set by env.step), consistent with training logger  :contentReference[oaicite:4]{index=4}
        row = {
            "policy": policy_name,
            "objective_user": (user_obj or "").lower(),
            "total_minutes": float(last_info.get("episode_minutes", np.nan)),
            "total_cost_gbp": float(last_info.get("episode_cost_gbp", np.nan)),
            "success": 1 if last_info.get("termination_reason") == "success" else 0,
            "termination_reason": last_info.get("termination_reason", "unknown"),
            "charge_events": int(last_info.get("charge_events", 0)),
            "soc_final": float(last_info.get("soc_final", np.nan)),
        }
        all_rows.append(row)

# Save episode-level results
df = pd.DataFrame(all_rows)
df.to_csv(OUT_DIR / "evaluation_results.csv", index=False)

# -----------------------------
# Summaries (global + per-user-objective)
# -----------------------------
def summarise(g):
    return pd.Series({
        "success_rate": g["success"].mean(),
        "median_minutes": g["total_minutes"].median(),
        "iqr_minutes": g["total_minutes"].quantile(0.75) - g["total_minutes"].quantile(0.25),
        "median_cost": g["total_cost_gbp"].median(),
        "iqr_cost": g["total_cost_gbp"].quantile(0.75) - g["total_cost_gbp"].quantile(0.25),
        "mean_charges": g["charge_events"].mean(),
        "median_soc_final": g["soc_final"].median(),
    })

summary_global = df.groupby("policy").apply(summarise).sort_index()
summary_global.to_csv(OUT_DIR / "summary_global.csv")

summary_matrix = df.groupby(["policy","objective_user"]).apply(summarise).reset_index()
summary_matrix.to_csv(OUT_DIR / "summary_matrix_policy_by_user_objective.csv", index=False)

# Termination breakdown
term = df.groupby(["policy","termination_reason"]).size().unstack(fill_value=0)
term_prop = term.div(term.sum(axis=1), axis=0)
term_prop.to_csv(OUT_DIR / "termination_breakdown.csv")

# -----------------------------
# Plots
# -----------------------------
# 1) Cost boxplot
ax = df.boxplot(column="total_cost_gbp", by="policy")
plt.ylabel("Total Cost (£)")
plt.title("Trip Cost by Policy"); plt.suptitle("")
plt.savefig(OUT_DIR / "plot_cost_box.png"); plt.close()

# 2) Time boxplot
ax = df.boxplot(column="total_minutes", by="policy")
plt.ylabel("Trip Time (min)")
plt.title("Trip Time by Policy"); plt.suptitle("")
plt.savefig(OUT_DIR / "plot_time_box.png"); plt.close()

# 3) Pareto scatter (episode cloud)
for pol, grp in df.groupby("policy"):
    plt.scatter(grp["total_cost_gbp"], grp["total_minutes"], alpha=0.25, label=pol, s=8)
plt.xlabel("Total Cost (£)")
plt.ylabel("Trip Time (min)")
plt.legend()
plt.title("Pareto: Episode Cost vs Time")
plt.savefig(OUT_DIR / "plot_pareto_episodes.png"); plt.close()

# 4) Pareto (run-level medians)
med = df.groupby("policy")[["total_cost_gbp","total_minutes"]].median()
plt.scatter(med["total_cost_gbp"], med["total_minutes"])
for pol, (x,y) in med.iterrows():
    plt.annotate(pol, (x,y), xytext=(3,3), textcoords="offset points")
plt.xlabel("Median Cost (£)"); plt.ylabel("Median Time (min)")
plt.title("Pareto (Run-level Medians)")
plt.savefig(OUT_DIR / "plot_pareto_medians.png"); plt.close()

# 5) Termination breakdown (stacked bar)
term_prop.plot(kind="bar", stacked=True)
plt.ylabel("Proportion")
plt.title("Termination Reasons by Policy")
plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_termination_stacked.png"); plt.close()

# 6) Policy × User Objective heatmaps (median cost/time; success)
pivot_cost = df.pivot_table(index="policy", columns="objective_user", values="total_cost_gbp", aggfunc="median")
pivot_time = df.pivot_table(index="policy", columns="objective_user", values="total_minutes", aggfunc="median")
pivot_succ = df.pivot_table(index="policy", columns="objective_user", values="success", aggfunc="mean")

for name, pv in [("median_cost", pivot_cost), ("median_minutes", pivot_time), ("success_rate", pivot_succ)]:
    plt.imshow(pv.values, aspect="auto")
    plt.xticks(range(pv.shape[1]), pv.columns)
    plt.yticks(range(pv.shape[0]), pv.index)
    plt.colorbar()
    plt.title(f"Policy × User Objective — {name}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"heatmap_{name}.png"); plt.close()

# -----------------------------
# Training curves (from run dirs)
# -----------------------------
def maybe_plot_training_curves(tag: str, run_dir: Path):
    mon = run_dir / "monitor.csv"
    kpi = run_dir / "kpi_episodes.csv"
    if mon.exists():
        try:
            m = pd.read_csv(mon, skiprows=1)
            m["r_roll"] = m["r"].rolling(200, min_periods=1).mean()
            plt.plot(m["t"], m["r_roll"], label=tag)
        except Exception:
            pass
    return kpi.exists(), kpi

plt.figure()
for pol, model_path in MODELS.items():
    run_dir = Path(model_path).parent
    _exists, _ = maybe_plot_training_curves(pol, run_dir)
plt.xlabel("Timesteps"); plt.ylabel("Rolling Reward")
plt.title("Training Reward (Monitor.csv)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "plot_training_reward.png"); plt.close()

plt.figure()
for pol, model_path in MODELS.items():
    run_dir = Path(model_path).parent
    kpi_path = run_dir / "kpi_episodes.csv"
    if kpi_path.exists():
        kpi = pd.read_csv(kpi_path)
        kpi["succ_roll"] = kpi["success"].rolling(500, min_periods=1).mean()
        plt.plot(kpi["timesteps"], kpi["succ_roll"], label=pol)
plt.xlabel("Timesteps"); plt.ylabel("Rolling Success Rate")
plt.title("Training Success (KPI Episodes)")
plt.legend(); plt.tight_layout()
plt.savefig(OUT_DIR / "plot_training_success.png"); plt.close()

print(f"[eval] Done. Outputs → {OUT_DIR}")
