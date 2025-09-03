import numpy as np
from utils.data_loader import load_all_ready
from rl.episodes import iter_episodes
from rl.ppo_env import PPOChargingEnv, PPOEnvConfig

def main():
    cfg = PPOEnvConfig(dt_minutes=5.0, max_steps=40, prefer="hybrid", use_sumo_drive=False)
    bundle = load_all_ready()
    rng = np.random.default_rng(0)

    trips = list(iter_episodes("data/sim_users_eval_calibrated.csv"))[:3]
    for i, trip in enumerate(trips, 1):
        env = PPOChargingEnv(cfg, data_bundle=bundle, pricing_catalog=bundle.get("pricing_catalog"))
        obs, info = env.reset(options={"trip": trip})
        last_remaining = env.remaining_km
        charge_progress_violation = False
        neg_charge_rewards = True

        done = False
        total_return = 0.0
        while not done:
            # random action among (Drive + available candidates)
            a = rng.integers(low=0, high=1 + len(env.candidates))
            obs, rew, term, trunc, step_info = env.step(int(a))
            done = term or trunc
            total_return += rew

            if step_info.get("step_type") == "charge":
                # remaining_km must NOT change on charge
                if env.remaining_km != last_remaining:
                    charge_progress_violation = True
                # reward on charge must be strictly negative
                if rew >= 0:
                    neg_charge_rewards = False

            last_remaining = env.remaining_km

        print(f"[Ep {i}] steps={step_info.get('episode_steps')} "
              f"drive={step_info.get('drive_steps')} charge={step_info.get('charge_steps')} "
              f"arrival_via={step_info.get('arrival_via')} "
              f"mins={step_info.get('episode_minutes'):.1f} "
              f"cost=£{step_info.get('episode_cost_gbp'):.2f} "
              f"return={total_return:.2f} "
              f"charge_changed_distance={charge_progress_violation} "
              f"charge_rewards_all_negative={neg_charge_rewards}")

if __name__ == "__main__":
    main()
