from __future__ import annotations
import os, sys, glob, shutil
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Resolve project paths (run from project root) ----------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = PROJECT_ROOT / "runs"
OUT_DIR = PROJECT_ROOT / "eval_outputs2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Make local packages importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------- Project imports ----------
from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes, TripPlan
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig
from stable_baselines3 import PPO

# ---------- SUMO / TraCI helpers ----------
def _sumo_available() -> bool:
    net_ok = (PROJECT_ROOT / "london_inner.net.xml").exists()
    bin_ok = shutil.which("sumo") is not None or shutil.which("sumo-gui") is not None or os.environ.get("SUMO_HOME")
    return bool(net_ok and bin_ok)

def _force_close_traci():
    """
    Close any stale TraCI connection to avoid: TraCIException: Connection 'default' is already active.
    Safe to call even if TraCI isn't loaded or no connection exists.
    """
    try:
        import traci
        try:
            traci.close(False)  # don't raise if already closed
        except Exception:
            pass
        # also try to kill any named/default connection explicitly
        try:
            conn = traci.getConnection("default")
            try: conn.close(False)
            except Exception: pass
        except Exception:
            pass
    except Exception:
        pass

# ---------- Discovery / trips ----------
def _find_latest_model(policy_hint: str) -> Path:
    pats = [
        str(RUNS_DIR / f"*{policy_hint}*" / "model_final.zip"),
        str(RUNS_DIR / f"*{policy_hint}*" / "**" / "model_final.zip"),
    ]
    cands: list[str] = []
    for p in pats: cands.extend(glob.glob(p, recursive=True))
    if not cands:
        cands = glob.glob(str(RUNS_DIR / "**" / "model_final.zip"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"No model_final.zip found under runs/ for hint='{policy_hint}'.")
    cands = sorted(cands, key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return Path(cands[0]).resolve()

def _load_eval_trip(trip_index: int, eval_csv: Path | None = None) -> TripPlan:
    csv_path = eval_csv if eval_csv is not None else (DATA_DIR / "sim_users_eval.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")
    trips = list(iter_episodes(csv_path))
    if not (0 <= trip_index < len(trips)):
        raise IndexError(f"trip_index {trip_index} out of range [0, {len(trips)-1}] for {csv_path.name}")
    return trips[trip_index]

# ---------- Env wrapper to pin one trip ----------
class OneTripEnv(PPOChargingEnv):
    def __init__(self, cfg: PPOEnvConfig, bundle: Dict, trip: TripPlan):
        super().__init__(cfg, data_bundle=bundle)
        self._trip = trip
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return super().reset(seed=seed, options={"trip": self._trip})

# ---------- rollout + plotting ----------
def run_policy_on_trip(model_path: Path, prefer: str, cfg: PPOEnvConfig, bundle: Dict, trip: TripPlan):
    """
    Returns (history, ep):
      history: dict with minutes[], soc[], remaining_km[], step_type[]
      ep: dict with termination_reason, episode_minutes, episode_cost_gbp, charge_events
    """
    # close any stale TraCI before bringing up a new env/runner
    _force_close_traci()

    # SUMO fallback
    cfg.prefer = prefer
    if not _sumo_available():
        cfg.use_sumo_drive = False

    env = None
    model = None
    last_info = {}
    try:
        env = OneTripEnv(cfg, bundle, trip)
        model = PPO.load(str(model_path), env=env)

        obs, info = env.reset()
        done = False; truncated = False

        hist = {
            "minutes": [float(getattr(env, "total_minutes", 0.0))],
            "soc": [float(getattr(env, "soc", np.nan))],
            "remaining_km": [float(getattr(env, "remaining_km", np.nan))],
            "step_type": [],
        }
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, last_info = env.step(int(action))

            stype = last_info.get("step_type", "drive")
            hist["step_type"].append(stype)
            hist["minutes"].append(float(getattr(env, "total_minutes", np.nan)))
            hist["soc"].append(float(getattr(env, "soc", np.nan)))
            hist["remaining_km"].append(float(getattr(env, "remaining_km", np.nan)))

            if len(hist["minutes"]) > 5000:
                break

        ep = {
            "termination_reason": last_info.get("termination_reason", "unknown"),
            "episode_minutes": float(last_info.get("episode_minutes", hist["minutes"][-1])),
            "episode_cost_gbp": float(last_info.get("episode_cost_gbp", 0.0)),
            "charge_events": int(last_info.get("charge_events", hist["step_type"].count("charge"))),
        }
        return hist, ep

    finally:
        # mirror eval_ppo.py: close vec env and raw env to free TraCI before next policy
        try:
            if model is not None and hasattr(model, "env") and hasattr(model.env, "close"):
                model.env.close()
        except Exception:
            pass
        try:
            if env is not None and hasattr(env, "close"):
                env.close()
        except Exception:
            pass
        # double-safety: if anything leaked, force close again
        _force_close_traci()

def plot_soc_progression(out_path: Path, traces: Dict[str, Dict[str, Dict]]):
    plt.figure(figsize=(7.0, 4.0))
    for pol, pack in traces.items():
        hist = pack["history"]; ep = pack["ep"]
        t = np.array(hist["minutes"]); soc = np.array(hist["soc"])
        plt.plot(t, soc, label=f"{pol} (end: {ep['termination_reason']})")
        # mark charge steps
        for i, st in enumerate(hist["step_type"]):
            if st == "charge":
                plt.scatter(t[i+1], soc[i+1], marker="o")
    plt.xlabel("Elapsed time (min)")
    plt.ylabel("State of charge (fraction)")
    plt.title("SoC progression for a representative trip")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    # zero-based; change if you want a different trip
    trip_index = 0
    out_img = OUT_DIR / "figure_44_soc_progression.png"
    out_csv = OUT_DIR / "figure_44_soc_progression.csv"

    # 1) data bundle from ./data (no random paths)
    bundle = load_all_ready(DATA_DIR)

    # 2) pick the same trip for all policies
    trip = _load_eval_trip(trip_index, eval_csv=DATA_DIR / "sim_users_eval.csv")

    # 3) auto-discover the latest models in ./runs
    model_cost   = _find_latest_model("cost")
    model_hybrid = _find_latest_model("hybrid")
    model_time   = _find_latest_model("time")

    # 4) env config (SUMO fallback handled inside run_policy_on_trip)
    cfg = PPOEnvConfig(
        obs_top_k=5,
        max_steps=240,
        dt_minutes=5.0,
        start_soc_range=(0.10, 0.30),
        trip_distance_km_range=(12.0, 25.0),
        value_of_time_per_min=0.05,
        charge_efficiency=0.92,
        charge_session_overhead_min=3.0,
        traffic_mode="light",
        prefer="hybrid",
        respect_trip_objective=False,
        success_bonus=50.0,
        strand_penalty=200.0,
        invalid_action_penalty=2.0,
        rng_seed=42,
        use_sumo_drive=True,
        sumo_net_path=str(PROJECT_ROOT / "london_inner.net.xml"),
        sumo_gui=False,
        sumo_step_length_s=1.0,
        sumo_mode="route_time",
        sumo_vehicle_type="passenger",
        disallow_repeat_station=True,
        max_charges_per_trip=2,
        min_charge_gap_min=12.0,
        penalty_repeat=-5.0,
        penalty_overlimit=-20.0,
        penalty_cooldown=-5.0,
        terminate_on_overlimit=True,
        penalty_unreachable=-3.0,
        enable_shaping=True,
        shaping_gamma=1.0,
        enable_potential_time=True,
        potential_vref_kmh=25.0,
        idle_penalty_per_step=0.0,
        micro_charge_penalty=0.0,
        micro_charge_min_kwh=1.0,
        micro_charge_min_minutes=6.0,
    )

    traces: Dict[str, Dict[str, Dict]] = {}
    for pol, mpath in (("cost", model_cost), ("hybrid", model_hybrid), ("time", model_time)):
        hist, ep = run_policy_on_trip(mpath, pol, cfg, bundle, trip)
        traces[pol] = {"history": hist, "ep": ep}

    plot_soc_progression(out_img, traces)

    # save CSV for appendix
    long_rows = []
    for pol, pack in traces.items():
        hist = pack["history"]; ep = pack["ep"]
        for i in range(len(hist["minutes"])):
            long_rows.append({
                "policy": pol,
                "step": i,
                "t_min": hist["minutes"][i],
                "soc": hist["soc"][i],
                "remaining_km": hist["remaining_km"][i],
                "termination_reason": ep["termination_reason"],
                "episode_minutes": ep["episode_minutes"],
                "episode_cost_gbp": ep["episode_cost_gbp"],
                "charge_events": ep["charge_events"],
            })
    pd.DataFrame(long_rows).to_csv(out_csv, index=False)

    print(f"[figure] Wrote: {out_img}")
    print(f"[csv]    Wrote: {out_csv}")
    for pol, pack in traces.items():
        ep = pack["ep"]
        print(f" - {pol}: end={ep['termination_reason']}, "
              f"minutes={ep['episode_minutes']:.2f}, cost=£{ep['episode_cost_gbp']:.2f}, "
              f"charges={ep['charge_events']}")

if __name__ == "__main__":
    main()
