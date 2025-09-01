import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from rl.episodes import iter_episodes
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig
from utils.data_loader import load_all_ready

DATA_DIR = Path("data")
bundle = load_all_ready(DATA_DIR, strict=True)

# pull one TripPlan from your simulated users
trip = next(iter_episodes(DATA_DIR / "simulated_users.csv"))

cfg = PPOEnvConfig(obs_top_k=6, dt_minutes=5.0, prefer="hybrid")
env = PPOChargingEnv(cfg, data_bundle=bundle)

obs, info = env.reset(seed=0, options={"trip": trip})
print("episode EV:", info["ev_id"], "start_soc:", obs[0], "remaining_km:", obs[1])

for _ in range(20):
    a = env.action_space.sample()
    obs, r, term, trunc, inf = env.step(a)
    if term or trunc:
        break

print("episode EV:", info["ev_id"], "start_soc:", obs[0], "remaining_km:", obs[1])
# peek at candidates encoded in obs
K = env.cfg.obs_top_k
base = 2
feat = 5
for i in range(min(K, 3)):
    off = base + i*feat
    dist_km, price, fast, rapid, ultra = obs[off:off+5]
    print(f"cand{i}: dist_km≈{dist_km:.2f}, price={price:.2f}, type="
          f"{'Fast' if fast>0.5 else ('Rapid' if rapid>0.5 else ('Ultra' if ultra>0.5 else 'N/A'))}")
