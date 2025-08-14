# playback.py – Visualize trained PPO agent in SUMO-GUI
import random
from stable_baselines3 import PPO
from rl.ev_env import EVChargingEnv
from rl.episodes import load_episodes
from utils.data_loader import load_all_data
from utils.pricing import PricingCatalog
from utils.charging_curves import EVPowerModel
from sumo.sumo_interface import SumoInterface


def main():
    # --- Load data ---
    bundle = load_all_data()
    stations = bundle.stations
    connectors = bundle.connectors
    trips = load_episodes("data/trips.csv")

    pricing_catalog = PricingCatalog.from_frames(bundle)
    ev_power_model = EVPowerModel.from_frames(bundle)

    ev_spec = {
        "battery_kwh": 75,
        "eff_kwh_per_km": 0.18,
        "allowed_connectors": ("CCS",),
        "max_dc_kw": 150,
        "max_ac_kw": 11
    }

    # --- Start SUMO in GUI mode ---
    sumo = SumoInterface(
        sumo_binary="sumo-gui",
        net_file="sumo_network.net.xml",
        fast_mode=False  # Detailed routing for realism
    )
    sumo.start(gui=True)

    # --- Use a single random trip ---
    trip = random.choice(trips)

    env = EVChargingEnv(
        sim=sumo,
        stations=stations,
        connectors=connectors,
        pricing=pricing_catalog,
        ev_spec=ev_spec,
        normalize_obs=True,
        normalize_rewards=True,
        reward_norm_factor=100.0,
        fast_mode=False
    )

    obs, _ = env.reset(options={"trip": trip})

    # --- Load trained model ---
    model = PPO.load("ppo_ev")

    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        env.render()  # print SOC & position
        sumo.step()   # advance SUMO simulation

    sumo.close()


if __name__ == "__main__":
    main()
