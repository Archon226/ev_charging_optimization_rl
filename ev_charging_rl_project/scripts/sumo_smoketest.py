# scripts/sumo_smoketest.py
import os, sys
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig

def main():
    DATA_DIR = Path("data")

    # Load bundle
    bundle = load_all_ready(DATA_DIR, strict=True)

    # Grab one TripPlan from simulated users
    trip = next(iter_episodes(DATA_DIR / "simulated_users.csv"))

    # Configure env (SUMO provider is default now in ppo_env.py)
    cfg = PPOEnvConfig(obs_top_k=5, dt_minutes=5.0, prefer="hybrid")
    env = PPOChargingEnv(cfg, data_bundle=bundle)

    # Reset with this trip
    obs, info = env.reset(options={"trip": trip})
    print("Episode EV:", info["ev_id"])
    print("Start SoC:", obs[0], "Remaining km:", obs[1])

    # Peek at candidates from SUMO
    K = env.cfg.obs_top_k
    base = 2
    feat = env._feat_per_cand
    for i in range(K):
        off = base + i*feat
        dist_km, price, fast, rapid, ultra = obs[off:off+5]
        charger_type = "Fast" if fast > 0.5 else ("Rapid" if rapid > 0.5 else ("Ultra" if ultra > 0.5 else "N/A"))
        print(f"Candidate {i}: dist_km≈{dist_km:.2f}, price={price:.2f}, type={charger_type}")

    # Step through a few actions to confirm env runs
    for _ in range(5):
        a = env.action_space.sample()
        obs, r, term, trunc, inf = env.step(a)
        print(f"Action={a}, Reward={r:.2f}, Terminated={term}, Truncated={trunc}")
        if term or trunc:
            break
    
    env.close()
if __name__ == "__main__":
    main()
