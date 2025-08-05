import random
import pandas as pd
from datetime import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import load_ev_metadata

def generate_users(num_users=10, seed=42):
    random.seed(seed)

    ev_specs = load_ev_metadata()
    user_list = []

    # Sample inner London bounding box (roughly)
    lat_range = (51.48, 51.54)
    lon_range = (-0.15, -0.05)
    objectives = ["cost", "time", "hybrid"]

    for i in range(num_users):
        ev_row = ev_specs.sample(1).iloc[0]

        start_lat = round(random.uniform(*lat_range), 6)
        start_lon = round(random.uniform(*lon_range), 6)
        end_lat = round(random.uniform(*lat_range), 6)
        end_lon = round(random.uniform(*lon_range), 6)

        user = {
            "user_id": f"EV-{i+1:03}",
            "ev_model": f"{ev_row['brand_name']} {ev_row['model']} {ev_row['release_year']}",
            "start_lat": start_lat,
            "start_lon": start_lon,
            "end_lat": end_lat,
            "end_lon": end_lon,
            "start_soc_percent": random.choice([20, 30, 40, 50]),
            "objective": random.choice(objectives),
            "max_budget": round(random.uniform(3, 10), 2),
            "max_time_slack": random.choice([5, 10, 15, 20, 30]),
            "departure_time": time(random.randint(6, 20), random.choice([0, 15, 30, 45])).strftime("%H:%M")
        }

        user_list.append(user)

    df = pd.DataFrame(user_list)
    return df

if __name__ == "__main__":
    df = generate_users(20)
    df.to_csv("data/simulated_users.csv", index=False)
    print("Generated simulated_users.csv with", len(df), "agents.")
