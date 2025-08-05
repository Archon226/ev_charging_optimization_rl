from utils.data_loader import load_ev_metadata, load_charging_curves, load_station_metadata, load_connectors, load_pricing_data
from utils.charging_curves import ChargingCurveLibrary
from env.agent import EVAgent
from env.charger import Charger

import random
from datetime import datetime

# === Load data ===
ev_specs = load_ev_metadata()
curve_data = load_charging_curves()
station_data = load_station_metadata()
connector_data = load_connectors()
pricing_data = load_pricing_data()

charging_library = ChargingCurveLibrary(curve_data)

# === Select EV and charger ===
ev_row = ev_specs.iloc[0]  # e.g., Nissan Leaf 2020
ev_model = ev_row['model']
brand = ev_row['brand_name']
year = ev_row['release_year']

agent = EVAgent(
    user_id="EV-001",
    ev_model=ev_model,
    battery_kWh=ev_row['battery_kWh'],
    efficiency_Wh_per_km=ev_row['avg_consumption_Wh_per_km'],
    ac_power_kW=ev_row['ac_max_power_kW'],
    dc_power_kW=ev_row['dc_max_power_kW'],
    start_soc_percent=30,
    origin=(51.512, -0.092),
    destination=(51.525, -0.071),
    objective='cost'
)

# Pick a random charger (we’ll assume it's compatible for now)
connector = connector_data.iloc[0]
station = station_data[station_data['chargeDeviceID'] == connector['chargeDeviceID']].iloc[0]
pricing_row = pricing_data[pricing_data['company_name'].str.lower().str.contains(station['deviceNetworks'].lower())].head(1)

charger = Charger(
    charger_id=connector['chargeDeviceID'],
    location=(station['latitude'], station['longitude']),
    operator=station['deviceNetworks'],
    connector_type=connector['connector_type'],
    rated_power_kW=connector['rated_power_kW'],
    charge_method=connector['charge_method'],
    base_price_per_kWh=pricing_row['price_per_kWh'].values[0] if not pricing_row.empty else 0.4,
    pricing_scheme=pricing_row.to_dict('records')[0] if not pricing_row.empty else {}
)

# === Charging Logic ===
current_soc = agent.soc_percent
target_soc = 80  # Charge up to 80%

# Charging curve
curve = charging_library.get_curve(brand, ev_model, year)
segment = charging_library.get_curve_slice(curve, current_soc, target_soc)

# Estimate time and energy
charging_time_min = charger.estimate_charging_time(current_soc, target_soc, segment, agent.battery_kWh)
energy_needed = ((target_soc - current_soc) / 100) * agent.battery_kWh
charging_cost = charger.estimate_cost(
    energy_needed,
    timestamp=datetime.now(),
    idle_minutes=5  # Or dynamically simulate this later
)


# Apply charge
agent.charge(energy_kWh=energy_needed, time_spent_min=charging_time_min, cost=charging_cost)

# === Output summary ===
print(f"\n=== EV Charging Summary ===")
print(f"Model: {brand} {ev_model} ({year})")
print(f"Charged from {current_soc}% to {agent.soc_percent:.1f}%")
print(f"Energy used: {energy_needed:.2f} kWh")
print(f"Charging time: {charging_time_min:.1f} min")
print(f"Charging cost: £{charging_cost:.2f}")
print(f"New total cost: £{agent.total_cost:.2f}")
print(f"New total time: {agent.total_time:.1f} min")
