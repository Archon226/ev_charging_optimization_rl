import pandas as pd
import random

# Load the charging station metadata dataset
file_path = "data/charging_station_metadata.csv"
stations_df = pd.read_csv(file_path)

# Step 1: Add 'region' column with all values set to 'London'
stations_df['region'] = 'London'

# Step 2: Add 'specific_location' column
stations_df['specific_location'] = None

# Step 3: For rows where deviceNetworks == 'POD Point', randomly assign 'Tesco' or 'Lidl' in 'specific_location'
mask_pod_point = stations_df['deviceNetworks'].str.strip().str.lower() == 'pod point'.lower()
stations_df.loc[mask_pod_point, 'specific_location'] = [
    random.choice(['Tesco', 'Lidl']) for _ in range(mask_pod_point.sum())
]

# Save the updated file
output_path = "data/charging_station_metadata.csv"
stations_df.to_csv(output_path, index=False)

