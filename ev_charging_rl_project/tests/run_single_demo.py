from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import PricingCatalog  # from load_all_data().pricing_catalog
from utils.data_loader import load_all_data
from utils.charging_curves import build_ev_specs, build_power_curve, integrate_charge_session
from env.charger import charger_from_row
from env.agent import EVAgent

def main():
    ds = load_all_data("data")

    # pick an EV model present in your EV_Metadata.csv
    model_col = "model" if "model" in ds.ev_metadata.columns else "Model"
    model = ds.ev_metadata.iloc[0][model_col]

    ev_specs = build_ev_specs(ds.ev_metadata)
    ev = ev_specs[model]

    # pick a charger row that has lat/lon + company_id + charger_type + rated_power_kw
    row = ds.stations_merged.dropna(subset=["latitude","longitude"]).iloc[0]

    # agent
    eff = ev.efficiency_wh_per_km or 170.0
    agent = EVAgent(
        agent_id="demo1",
        model=model,
        user_type="Payg",
        objective="hybrid",
        efficiency_Wh_per_km=eff,
        battery_kwh=ev.usable_battery_kwh,
        soc=20.0,
        alpha_cost=0.5
    )

    # integrate a single session 20% -> 80% at this station
    curve = build_power_curve(ds.charging_curves, model)
    integ = integrate_charge_session(
        ev_spec=ev,
        power_curve=curve,
        start_soc=agent.soc,
        target_soc=80.0,
        station_power_kw=float(row["rated_power_kw"]),
        is_dc=row["charger_type"] in ("Rapid","Ultra"),
    )

    # price it
    charger = charger_from_row(row, ds.pricing_catalog)
    breakdown = charger.estimate_session_cost(
        kwh=integ["delivered_kwh"],
        start_dt=datetime(2025,8,9,18,30),   # Sat 18:30 to exercise peak/weekend logic
        user_type=agent.user_type,
        session_minutes=integ["minutes"],
        include_subscription=False
    )

    # apply and print
    agent.apply_charge(integ["delivered_kwh"], integ["minutes"], breakdown)
    print("Delivered:", integ["delivered_kwh"], "kWh")
    print("Duration:", integ["minutes"], "min")
    print("Unit price:", breakdown["unit_price"], f"({breakdown['price_source']})")
    print("Total cost:", breakdown["total_cost"], "GBP")
    print("Final SOC:", agent.soc, "%")

if __name__ == "__main__":
    main()
