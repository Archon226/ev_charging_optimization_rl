import pandas as pd

# Load your stations data
stations = pd.read_csv("data\charging_station_metadata.csv")

# Padding in degrees (~0.01 degrees ≈ 1.1 km in lat, ~0.01 degrees ≈ 0.7 km in lon near London)
PAD_LAT = 0.01
PAD_LON = 0.015

# Get min/max lat/lon
min_lat = stations['latitude'].min() - PAD_LAT
max_lat = stations['latitude'].max() + PAD_LAT
min_lon = stations['longitude'].min() - PAD_LON
max_lon = stations['longitude'].max() + PAD_LON

# Print bbox in Overpass order (lon_min, lat_min, lon_max, lat_max)
print(f"Bounding box: {min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}")

# Generate curl command for Overpass API
print(
    f'curl -L "https://overpass-api.de/api/map?bbox={min_lon:.6f},{min_lat:.6f},{max_lon:.6f},{max_lat:.6f}" -o net/london_auto.osm'
)
