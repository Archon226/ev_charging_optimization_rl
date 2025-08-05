import pandas as pd
import math

class Charger:
    def __init__(self, charger_id, location, operator, connector_type, rated_power_kW,
                 charge_method, base_price_per_kWh=0.0, pricing_scheme=None, edge_id=None):
        self.charger_id = charger_id
        self.location = location  # (lat, lon) or SUMO node
        self.operator = operator
        self.connector_type = connector_type
        self.rated_power_kW = rated_power_kW
        self.charge_method = charge_method
        self.base_price_per_kWh = base_price_per_kWh
        self.pricing_scheme = pricing_scheme or {}

        self.edge_id = edge_id  # for SUMO routing
        self.is_available = True  # required by Person B

        # Simulation flags
        self.occupied = False
        self.current_user_id = None

    def is_available_sim(self):
        return not self.occupied

    def occupy(self, user_id):
        self.occupied = True
        self.current_user_id = user_id
        self.is_available = False

    def release(self):
        self.occupied = False
        self.current_user_id = None
        self.is_available = True

    def update_status(self, sumo_interface):
        """
        Update availability via SUMO logic (external state).
        """
        self.is_available = sumo_interface.check_availability(self.charger_id)

    def estimate_charging_time(self, start_soc, target_soc, ev_curve, ev_battery_kWh):
        usable_curve = ev_curve[(ev_curve['soc_percent'] >= start_soc) &
                                (ev_curve['soc_percent'] < target_soc)]

        if usable_curve.empty:
            return 0.0

        total_time_min = 0.0
        for _, row in usable_curve.iterrows():
            power = min(row['charging_power_kW'], self.rated_power_kW)
            if power > 0:
                energy_chunk = ev_battery_kWh / 100  # 1% SOC
                time_chunk = (energy_chunk / power) * 60
                total_time_min += time_chunk

        return total_time_min

    def estimate_cost(self, kWh_needed, timestamp=None, idle_minutes=0):
        scheme = self.pricing_scheme
        base_price_per_kWh = self.base_price_per_kWh

        if scheme.get("time_sensitive_flag") == 1 and timestamp:
            hour = timestamp.hour
            if 0 <= hour < 7 and "off_peak_price" in scheme:
                base_price_per_kWh = scheme.get("off_peak_price", base_price_per_kWh)
            elif "peak_price" in scheme:
                base_price_per_kWh = scheme.get("peak_price", base_price_per_kWh)

        energy_cost = kWh_needed * base_price_per_kWh

        def safe(value):
            return 0.0 if pd.isna(value) or value is None else float(value)

        subscription_fee = safe(scheme.get("subscription_fee"))
        connection_fee = safe(scheme.get("connection_fee"))
        pre_auth_fee = safe(scheme.get("pre_auth_avg_fee"))

        idle_fee = 0.0
        if scheme.get("idle_fee_flag") == 1 and idle_minutes > 0:
            idle_fee = idle_minutes * 0.5  # Placeholder until rate is available

        total_cost = energy_cost + subscription_fee + connection_fee + idle_fee + pre_auth_fee
        return total_cost
