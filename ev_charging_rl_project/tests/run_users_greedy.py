# tests/run_users_greedy.py
import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd

# Make project importable when running from tests/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.data_loader import load_all_data
from utils.charging_curves import build_ev_specs, build_power_curve
from simulator.sumo_adapter import SumoSim
from simulator.candidates import find_charger_candidates, EVSpecLite
from rl.episodes import load_user_plans
from rl.candidate_eval import evaluate_candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--net", default=r"london_boroughs.net.xml")
    ap.add_argument("--num-users", type=int, default=20, help="evaluate first N users")
    ap.add_argument("--prefilter-km", type=float, default=3.0, help="corridor width before SUMO (km)")
    ap.add_argument("--max-evals", type=int, default=300, help="max stations to evaluate per user with SUMO")
    args = ap.parse_args()

    DATA = args.data_dir
    NET = args.net

    # ---- single source of truth: load everything via the loader ----
    ds = load_all_data(DATA)

    # Prefer stations filtered to the SUMO net if you created them earlier
    st_in_net = os.path.join(DATA, "stations_in_net.csv")
    if os.path.exists(st_in_net):
        stations_df = pd.read_csv(st_in_net)
    else:
        stations_df = ds.stations_merged

    ev_meta = ds.ev_metadata
    curves = ds.charging_curves
    catalog = ds.pricing_catalog
    ev_specs = build_ev_specs(ev_meta)

    # SUMO net
    sim = SumoSim(NET)

    # Plans from simulated_users.csv
    plans = load_user_plans(os.path.join(DATA, "simulated_users.csv"))
    if args.num_users and args.num_users < len(plans):
        plans = plans[:args.num_users]

    rows = []
    t0 = time.time()
    for idx, plan in enumerate(plans, 1):
        t_user = time.time()
        try:
            # EV spec + curve
            spec = ev_specs[plan.model]
            ev_curve = build_power_curve(curves, plan.model)

            # Efficiency and battery (kWh)
            eff_kwh_per_km = float(
                ev_meta.loc[ev_meta["model"] == plan.model].iloc[0]["avg_consumption_Wh_per_km"]
            ) / 1000.0
            batt_kwh = float(getattr(spec, "usable_battery_kWh", getattr(spec, "battery_kWh", 0.0)))

            # Candidate search EV lite (connector rules can be made stricter later)
            ev_lite = EVSpecLite(
                battery_kwh=batt_kwh,
                eff_kwh_per_km=eff_kwh_per_km,
                allowed_connectors=("CCS2", "Type2"),
                max_dc_kw=getattr(spec, "dc_max_power_kW", 120),
                max_ac_kw=getattr(spec, "ac_max_power_kW", 11),
            )

            # Find charger candidates (fast path: prefilter + cap)
            cands = find_charger_candidates(
                sim,
                (plan.start_lat, plan.start_lon),
                (plan.dest_lat, plan.dest_lon),
                stations_df,          # merged table (connector_type / rated_power_kw / station_connectors / company_id)
                None,                 # connectors_df not needed when passing merged table
                ev_lite,
                current_soc=plan.init_soc_pct / 100.0,
                top_k=plan.top_k_candidates,
                max_detour_km=plan.max_detour_km,
                require_connector_ok=False,     # flip True when you want strict plug matching
                prefilter_km=args.prefilter_km,
                max_evals=args.max_evals,
            )

            if not cands:
                rows.append({"agent_id": plan.agent_id, "objective": plan.objective, "why": "no_candidates"})
            else:
                # Score candidates using your pricing + charging curve + simple wait model
                evals = evaluate_candidates(
                    sim, stations_df, cands, plan, spec, ev_curve,
                    eff_kwh_per_km, batt_kwh, catalog, plan.depart_dt
                )
                best = evals[0]
                rows.append({
                    "agent_id": plan.agent_id, "objective": plan.objective,
                    "station_id": best.station_id, "ctype": best.charger_type, "kW": best.rated_power_kw,
                    "detour_min": best.detour_min, "wait_min": best.wait_min, "charge_min": best.charge_min,
                    "delivered_kwh": best.delivered_kwh, "unit_price": best.unit_price, "cost": best.total_cost,
                    "score": best.score
                })

            dt_user = time.time() - t_user
            print(f"[{idx}/{len(plans)}] {plan.agent_id} objective={plan.objective}  "
                  f"cands={len(cands) if cands else 0}  {dt_user:.2f}s")

        except Exception as e:
            dt_user = time.time() - t_user
            print(f"[{idx}/{len(plans)}] {plan.agent_id} ERROR after {dt_user:.2f}s: {e}")
            rows.append({"agent_id": plan.agent_id, "objective": plan.objective, "why": f"error: {e}"})

    out = pd.DataFrame(rows)
    print(out.head(10).to_string(index=False))
    os.makedirs(DATA, exist_ok=True)
    out_path = os.path.join(DATA, "greedy_user_runs.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}  |  total {len(rows)} users  |  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
