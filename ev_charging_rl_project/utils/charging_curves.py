import pandas as pd

class ChargingCurveLibrary:
    def __init__(self, curve_df):
        """
        Expects the raw EV_Charging_Curve_Data.csv as a DataFrame.
        """
        self.curve_df = curve_df

    def get_curve(self, brand_name, model, release_year):
        """
        Returns the charging curve (SOC % → charging power in kW) for a specific EV.
        """
        curve = self.curve_df[
            (self.curve_df["brand_name"].str.lower() == brand_name.lower()) &
            (self.curve_df["model"].str.lower() == model.lower()) &
            (self.curve_df["release_year"] == release_year)
        ].sort_values("soc_percent")

        if curve.empty:
            raise ValueError(f"No charging curve found for {brand_name} {model} {release_year}")

        return curve.reset_index(drop=True)

    def get_curve_slice(self, curve, start_soc, target_soc):
        """
        Returns a slice of the curve between start_soc and target_soc (inclusive).
        """
        return curve[(curve["soc_percent"] >= start_soc) &
                     (curve["soc_percent"] <= target_soc)].copy().reset_index(drop=True)
