class Charger:
    def __init__(self, charger_id, location, operator, connector_type, rated_power_kW,
                 charge_method, base_price_per_kWh=0.0, pricing_scheme=None):
        self.charger_id = charger_id
        self.location = location  # (lat, lon) or SUMO node
        self.operator = operator
        self.connector_type = connector_type
        self.rated_power_kW = rated_power_kW
        self.charge_method = charge_method
        self.base_price_per_kWh = base_price_per_kWh
        self.pricing_scheme = pricing_scheme or {}

        self.occupied = False  # Simple simulation flag
        self.current_user_id = None

    def is_available(self):
        return not self.occupied

    def occupy(self, user_id):
        self.occupied = True
        self.current_user_id = user_id

    def release(self):
        self.occupied = False
        self.current_user_id = None

    def estimate_charging_time(self, start_soc, target_soc, ev_curve, ev_battery_kWh):
        """
        Estimate charging time in minutes using the EV's charging curve and charger's power limit.
        """
        # Extract curve data for this EV and interpolate
        usable_curve = ev_curve[(ev_curve['soc_percent'] >= start_soc) &
                                (ev_curve['soc_percent'] < target_soc)]

        if usable_curve.empty:
            return 0.0

        total_energy = ((target_soc - start_soc) / 100) * ev_battery_kWh  # kWh needed
        total_time_min = 0.0

        for _, row in usable_curve.iterrows():
            soc = row['soc_percent']
            power = min(row['charging_power_kW'], self.rated_power_kW)
            if power > 0:
                # Estimate time for 1% of SOC gain
                energy_chunk = ev_battery_kWh / 100  # 1% of battery in kWh
                time_chunk = (energy_chunk / power) * 60  # in minutes
                total_time_min += time_chunk

        return total_time_min

    def estimate_cost(self, kWh_needed, timestamp=None, idle_minutes=0):
        """
        Estimate total cost including energy + optional fees.
        """
        scheme = self.pricing_scheme
        base_price_per_kWh = self.base_price_per_kWh

        # Time-sensitive price adjustment
        if scheme.get("time_sensitive_flag") == 1 and timestamp:
            hour = timestamp.hour
            if 0 <= hour < 7 and "off_peak_price" in scheme:
                base_price_per_kWh = scheme["off_peak_price"]
            elif "peak_price" in scheme:
                base_price_per_kWh = scheme["peak_price"]

        energy_cost = kWh_needed * base_price_per_kWh

        # Optional costs
        subscription_fee = 0.0
        if scheme.get("subscription_required", "").lower() == "yes":
            subscription_fee = scheme.get("subscription_fee", 0.0) or 0.0

        connection_fee = scheme.get("connection_fee", 0.0) or 0.0

        idle_fee = 0.0
        if scheme.get("idle_fee_flag") == 1 and idle_minutes > 0:
            # Assume a flat fee or 0.5 per idle min if no rate is given
            idle_fee = idle_minutes * 0.5

        pre_auth_fee = 0.0
        if scheme.get("pre_auth_fee_flag") == 1:
            pre_auth_fee = scheme.get("pre_auth_avg_fee", 0.0) or 0.0

        total_cost = energy_cost + subscription_fee + connection_fee + idle_fee + pre_auth_fee
        return total_cost

