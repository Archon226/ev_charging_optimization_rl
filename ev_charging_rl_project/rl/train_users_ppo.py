# rl/train_users_ppo.py
import os, pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simulator.sumo_adapter import SumoSim
from utils.pricing import load_pricing_catalog
from utils.data_loader import build_stations_merged_files
from rl.episodes import load_user_plans
from rl.ev_env_users import EVUsersEnv

DATA="data"; NET=r"london_boroughs.net.xml"
stations = pd.read_csv(os.path.join(DATA,"stations_in_net.csv")) if os.path.exists(os.path.join(DATA,"stations_in_net.csv")) \
           else build_stations_merged_files(DATA)
ev_meta = pd.read_csv(f"{DATA}/EV_Metadata.csv")
curves  = pd.read_csv(f"{DATA}/EV_Charging_Curve_Data.csv")
catalog = load_pricing_catalog(DATA)
plans   = load_user_plans(os.path.join(DATA, "simulated_users.csv"))
sim = SumoSim(NET)

def make_env_fn():
    return EVUsersEnv(sim, stations, catalog, ev_meta, curves, plans)

env = make_vec_env(make_env_fn, n_envs=8)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("models/ppo_users_pickbest")
print("done")
