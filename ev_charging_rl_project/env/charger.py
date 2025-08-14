class Charger:
    def __init__(self, station_id, power_kw, connector_type, pricing_catalog, fast_mode=False):
        self.station_id = station_id
        self.power_kw = power_kw
        self.connector_type = connector_type
        self.pricing_catalog = pricing_catalog
        self.fast_mode = fast_mode

    def estimate_session_cost(self, ev_spec, start_soc, end_soc):
        added_kwh = (end_soc - start_soc) * ev_spec["battery_kwh"]
        if self.fast_mode:
            # Skip detailed breakdown
            return self.pricing_catalog.simple_price(self.station_id, added_kwh)
        else:
            return self.pricing_catalog.compute_price(self.station_id, added_kwh)

    def charge_time_seconds(self, ev_spec, start_soc, end_soc):
        added_kwh = (end_soc - start_soc) * ev_spec["battery_kwh"]
        return (added_kwh / self.power_kw) * 3600
