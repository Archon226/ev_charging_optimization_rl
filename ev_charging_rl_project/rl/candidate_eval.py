# rl/candidate_eval.py
from dataclasses import dataclass
from datetime import timedelta
import pandas as pd
from utils.charging_curves import build_power_curve, integrate_charge_session
from env.charger import charger_from_row

@dataclass
class CandEval:
    station_id: str
    charger_type: str
    rated_power_kw: float
    company_id: int
    detour_km: float
    detour_min: float
    wait_min: float
    delivered_kwh: float
    charge_min: float
    unit_price: float
    total_cost: float
    score: float        # smaller is better

def evaluate_candidates(sim,  # SumoSim or None
                        stations_df: pd.DataFrame,
                        candidates: list,            # from find_charger_candidates
                        plan,                         # TripPlan
                        ev_spec, ev_power_curve,
                        eff_kwh_per_km: float,
                        batt_kwh: float,
                        pricing_catalog,
                        now_dt):
    out = []
    for c in candidates:
        # 1) Get station row
        row = stations_df.loc[stations_df["station_id"] == c.station_id]
        if row.empty: 
            continue
        st = row.iloc[0].to_dict()
        rated = float(st["rated_power_kw"]); ctype = str(st["charger_type"]).title()
        company_id = int(st["company_id"])

        # 2) Detour leg time from SUMO candidate
        detour_min = c.detour_time_s/60.0
        wait_min = _estimate_wait_minutes(ctype, now_dt)

        # 3) Required energy to finish after visiting this station
        # For a quick but consistent estimate, target = min(plan.target, enough to finish + reserve)
        # We’ll compute charging session from CURRENT SOC (plan.init_soc_pct is acceptable for a one-shot decision)
        # If you have current SOC tracked stepwise in your loop, pass that instead.
        # Charge curve integration (your function signature → dict out)
        is_dc = (ctype == "Rapid")
        target_soc_pct = _target_soc_to_finish(sim, c, plan, eff_kwh_per_km, batt_kwh)
        int_res = integrate_charge_session(
            ev_spec=ev_spec,
            power_curve=ev_power_curve,
            start_soc=float(plan.init_soc_pct),               # replace with current SOC if inside loop
            target_soc=float(min(99.0, target_soc_pct)),
            station_power_kw=rated,
            is_dc=is_dc
        )
        delivered = float(int_res["delivered_kwh"])
        ch_min    = float(int_res["minutes"])

        # 4) Price the session (your pricing logic)
        chg = charger_from_row(st, pricing_catalog)
        bd  = chg.estimate_session_cost(
            start_dt=now_dt, kwh=delivered, session_minutes=ch_min,
            user_type=plan.user_type,
            include_subscription=plan.include_subscription,
            sessions_per_month=plan.sessions_per_month
        )
        # tolerate dict or dataclass
        unit_price = bd.get("unit_price") if isinstance(bd, dict) else getattr(bd, "unit_price", 0.0)
        total_cost = bd.get("total_cost") if isinstance(bd, dict) else getattr(bd, "total_cost", 0.0)

        # 5) Score (lower is better)
        if plan.objective == "time":
            score = detour_min + wait_min + ch_min
        elif plan.objective == "cost":
            score = total_cost
        else:  # hybrid 50/50; tune weights later
            score = 0.5 * (detour_min + wait_min + ch_min) + 0.5 * total_cost

        out.append(CandEval(
            station_id=c.station_id, charger_type=ctype, rated_power_kw=rated, company_id=company_id,
            detour_km=c.detour_km, detour_min=detour_min, wait_min=wait_min,
            delivered_kwh=delivered, charge_min=ch_min, unit_price=unit_price, total_cost=total_cost,
            score=score
        ))
    # sort by best score
    return sorted(out, key=lambda x: x.score)

def _target_soc_to_finish(sim, c, plan, eff_kwh_per_km, batt_kwh):
    # simple: enough to finish from station to destination + reserve, capped at plan.target_soc_pct
    if sim:
        rem_km = sim.route_between((c.poi_lat, c.poi_lon), (plan.dest_lat, plan.dest_lon)).dist_km
    else:
        rem_km = 5.0  # fallback; not used in SUMO runs
    energy_kwh = max(0.0, rem_km * eff_kwh_per_km)
    need_pct = (energy_kwh / max(1e-6, batt_kwh)) * 100.0 + plan.reserve_soc_pct
    return min(plan.target_soc_pct, max(plan.reserve_soc_pct, need_pct))

def _estimate_wait_minutes(charger_type, now_dt):
    tod = now_dt.hour + now_dt.minute/60.0
    rush = (17 <= tod <= 19)
    base = 2.0 if charger_type == "Rapid" else 1.0
    return float((8.0 if rush else 3.0) * base)
