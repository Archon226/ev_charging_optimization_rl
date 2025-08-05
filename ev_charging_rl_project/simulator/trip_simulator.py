import pandas as pd
from datetime import datetime
from geopy.distance import geodesic

from env.agent import EVAgent
from env.charger import Charger

from utils.charging_curves import ChargingCurveLibrary


class TripSimulator:
    def __init__(self, user_df, station_df, connector_df, pricing_df, ev_metadata_df, charging_library: ChargingCurveLibrary):
        self.user_df = user_df
        self.station_df = station_df
        self.connector_df = connector_df
        self.pricing_df = pricing_df
        self.ev_metadata_df = ev_metadata_df
        self.charging_library = charging_library
        self.agents = []
        self.chargers = []

        self.initialize_chargers()
        self.initialize_agents()

    def initialize_chargers(self):
        # Normalize company names for merging
        self.station_df["deviceNetworks_lower"] = self.station_df["deviceNetworks"].str.lower()
        self.pricing_df["company_name_lower"] = self.pricing_df["company_name"].str.lower()

        merged = self.connector_df.merge(
            self.station_df,
            on="chargeDeviceID",
            how="left"
        ).merge(
            self.pricing_df,
            left_on="deviceNetworks_lower",
            right_on="company_name_lower",
            how="left"
        )

        for _, row in merged.iterrows():
            pricing_scheme = row.to_dict()

            charger = Charger(
                charger_id=row['chargeDeviceID'],
                location=(row['latitude'], row['longitude']),
                operator=row['deviceNetworks'],
                connector_type=row['connector_type'],
                rated_power_kW=row['rated_power_kW'],
                charge_method=row['charge_method'],
                base_price_per_kWh=row.get('price_per_kWh', 0.4),
                pricing_scheme=pricing_scheme,
                edge_id=row.get('edge_id', None)
            )
            self.chargers.append(charger)

    def initialize_agents(self):
        for _, row in self.user_df.iterrows():
            brand, model, year = self.split_model(row['ev_model'])
            ev_row = self.match_ev_spec(brand, model, year)
            if ev_row is None:
                continue

            agent = EVAgent(
                user_id=row['user_id'],
                ev_model=row['ev_model'],
                battery_kWh=ev_row['battery_kWh'],
                efficiency_Wh_per_km=ev_row['avg_consumption_Wh_per_km'],
                ac_power_kW=ev_row['ac_max_power_kW'],
                dc_power_kW=ev_row['dc_max_power_kW'],
                start_soc_percent=row['start_soc_percent'],
                origin=(row['start_lat'], row['start_lon']),
                destination=(row['end_lat'], row['end_lon']),
                objective=row['objective'],
                max_budget=row.get('max_budget', None),
                max_time_slack=row.get('max_time_slack', None)
            )
            self.agents.append(agent)

    def split_model(self, ev_model_str):
        try:
            parts = ev_model_str.strip().split(" ")
            brand = parts[0]
            year = int(parts[-1])
            model = " ".join(parts[1:-1])
            return brand, model, year
        except:
            return None, None, None

    def match_ev_spec(self, brand, model, year):
        df = self.ev_metadata_df
        match = df[
            (df['brand_name'].str.lower() == brand.lower()) &
            (df['model'].str.lower() == model.lower()) &
            (df['release_year'] == year)
        ]
        return match.iloc[0] if not match.empty else None

    def assign_nearest_charger(self, agent):
        def distance(charger):
            return geodesic(agent.location, charger.location).km

        compatible = [c for c in self.chargers if c.is_available]
        if not compatible:
            return None
        return min(compatible, key=distance)

    def simulate_trip(self, agent):
        charger = self.assign_nearest_charger(agent)
        if charger is None:
            print(f"No charger available for agent {agent.user_id}")
            return

        curve = self.charging_library.get_curve(*self.split_model(agent.ev_model))
        segment = self.charging_library.get_curve_slice(curve, agent.soc_percent, 80)

        energy_needed = ((80 - agent.soc_percent) / 100) * agent.battery_kWh
        charge_time = charger.estimate_charging_time(agent.soc_percent, 80, segment, agent.battery_kWh)
        cost = charger.estimate_cost(energy_needed, timestamp=datetime.now(), idle_minutes=5)

        charger.occupy(agent.user_id)
        agent.charge(energy_kWh=energy_needed, time_spent_min=charge_time, cost=cost)
        charger.release()

    def run_all(self):
        for agent in self.agents:
            self.simulate_trip(agent)
            print(f"Agent {agent.user_id} | SOC: {agent.soc_percent:.1f}% | Cost: £{agent.total_cost:.2f} | Time: {agent.total_time:.1f} min")
