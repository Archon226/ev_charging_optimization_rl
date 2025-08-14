# rl/train.py — PPO training using simulated_users.csv (no trips.csv needed)
import os, random
from dataclasses import dataclass
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Package imports (run from project root: `python -m rl.train`)
from rl.ev_env import EVChargingEnv
from utils.data_loader import load_all_data
from utils.charging_curves import EVPowerModel
from sumo.sumo_interface import SumoInterface

# -------- trip wrapper built from simulated_users.csv --------
@dataclass
class Trip:
    origin: tuple[float, float]
    dest: tuple[float, float]
    ev_model: str
    start_soc: float  # fraction in [0,1]

def trips_from_simulated_users(bundle, data_dir="data"):
    import pandas as pd
    if bundle.users is None:
        users_csv = os.path.join(data_dir, "simulated_users.csv")
        if not os.path.exists(users_csv):
            raise FileNotFoundError("simulated_users.csv not found (run utils.generate_users)")
        df = pd.read_csv(users_csv)
    else:
        df = bundle.users

    trips = []
    for _, r in df.iterrows():
        trips.append(
            Trip(
                origin=(float(r.start_lat), float(r.start_lon)),
                dest=(float(r.dest_lat), float(r.dest_lon)),
                ev_model=str(r.ev_model),
                start_soc=float(r.init_soc_pct) / 100.0
            )
        )
    if not trips:
        raise ValueError("No trips built from simulated_users.csv")
    return trips

def main():
    # ---- load data & helpers ----
    bundle = load_all_data("data")
    stations  = bundle.stations
    connectors = bundle.connectors
    trips = trips_from_simulated_users(bundle, "data")
    trips = [random.choice(trips)]
    # EV power/efficiency model (for per-trip specs)
    epm = EVPowerModel.from_frames(bundle)

    # ---- SUMO (fast routing for training) ----
    sumo = SumoInterface(
        sumo_binary="sumo",               # use "sumo-gui" for visual later
        net_file="london_boroughs.net.xml",  # set to your SUMO .net.xml
        fast_mode=True
    )
    sumo.start(gui=False)

    # ---- Env wrapper that picks a random trip and derives EV spec per episode ----
    class TripEnv(EVChargingEnv):
        def reset(self, *, seed=None, options=None):
            trip = random.choice(trips)
            # derive ev_spec from EV model
            cap = epm.battery_kwh(trip.ev_model) or 60.0
            eff = epm.kwh_per_km(trip.ev_model) or 0.18
            # simple connector defaults; adjust if you map models→connectors
            ev_spec = {
                "battery_kwh": cap,
                "eff_kwh_per_km": eff,
                "allowed_connectors": ("CCS", "Type2", "CHAdeMO"),
                "max_dc_kw": 150,
                "max_ac_kw": 11
            }
            self.ev_spec = ev_spec  # update per episode
            # hand trip to base env
            return super().reset(seed=seed, options={
                "trip": type("T", (), {
                    "origin": trip.origin,
                    "dest": trip.dest,
                    "ev_model": trip.ev_model,
                    "start_soc": trip.start_soc
                })
            })

    env = TripEnv(
        sim=sumo,
        stations=stations,
        connectors=connectors,
        pricing=bundle.pricing_catalog,
        ev_spec={  # will be overridden each reset
            "battery_kwh": 60, "eff_kwh_per_km": 0.18,
            "allowed_connectors": ("CCS",), "max_dc_kw": 150, "max_ac_kw": 11
        },
        normalize_obs=True,
        normalize_rewards=True,
        reward_norm_factor=100.0,
        fast_mode=True
    )

    # ---- PPO ----
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99
    )

    eval_cb = EvalCallback(
        env,
        best_model_save_path="./ppo_models/",
        log_path="./ppo_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=100_000, callback=eval_cb)
    model.save("ppo_ev")

    sumo.close()

if __name__ == "__main__":
    main()
