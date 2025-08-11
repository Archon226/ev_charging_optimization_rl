# simulator/trip_simulator.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict

import math
import numpy as np
import pandas as pd

from utils.data_loader import DatasetBundle
from utils.charging_curves import build_ev_specs, build_power_curve, integrate_charge_session
from env.agent import EVAgent
from env.charger import charger_from_row, Charger


# --- geo ---
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Scalar haversine in km."""
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _haversine_km_vec(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized haversine (lat/lon arrays)."""
    R = 6371.0088
    lat1r = math.radians(lat1)
    lon1r = math.radians(lon1)
    latsr = np.radians(lats.astype(float))
    lonsr = np.radians(lons.astype(float))
    dphi = latsr - lat1r
    dl = lonsr - lon1r
    a = np.sin(dphi/2)**2 + np.cos(lat1r) * np.cos(latsr) * np.sin(dl/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def _is_dc(charger_type: str) -> bool:
    t = (charger_type or "").strip().title()
    return t in ("Rapid", "Ultra")



@dataclass
class TripPlan:
    start_lat: float
    start_lon: float
    dest_lat: float
    dest_lon: float
    depart_dt: datetime
    target_soc: float = 80.0
    reserve_soc: float = 10.0
    avg_speed_kph: float = 28.0          # Static for now; SUMO can replace later
    search_radius_km: float = 2.0        # initial charger search radius
    expand_radius_km: float = 3.0        # how much to expand if none found
    max_radius_km: float = 8.0           # cap expansion


class TripSimulator:
    def __init__(self, datasets: DatasetBundle):
        self.ds = datasets
        self.ev_specs = build_ev_specs(datasets.ev_metadata)
        self.curves_df = datasets.charging_curves

        # Only keep chargers with coordinates and a pricing company_id
        st = datasets.stations_merged.copy()
        st = st.dropna(subset=["latitude", "longitude", "company_id"]).reset_index(drop=True)

        # enforce numeric rated_power_kw
        st["rated_power_kw"] = pd.to_numeric(st["rated_power_kw"], errors="coerce").fillna(7.0)
        self.chargers_df = st

    # --------- helpers ---------
    def _nearest_chargers(self, lat: float, lon: float, radius_km: float) -> pd.DataFrame:
        df = self.chargers_df.copy()
        dists = _haversine_km_vec(lat, lon, df["latitude"].to_numpy(), df["longitude"].to_numpy())
        df["dist_km"] = dists
        return df[df["dist_km"] <= radius_km].sort_values("dist_km")

    # --------- main API ---------
    def run_trip(self, agent: EVAgent, plan: TripPlan) -> Dict:
        # setup
        now = plan.depart_dt
        cur_lat, cur_lon = plan.start_lat, plan.start_lon
        dest_lat, dest_lon = plan.dest_lat, plan.dest_lon

        # EV curve prep
        if agent.model not in self.ev_specs:
            raise KeyError(f"EV model '{agent.model}' not found in EV metadata")
        ev_spec = self.ev_specs[agent.model]
        power_curve = build_power_curve(self.curves_df, agent.model)

        # loop until destination reached
        finished = False
        safety_iter = 0
        while not finished and safety_iter < 100:
            safety_iter += 1
            remaining_km = _haversine_km(cur_lat, cur_lon, dest_lat, dest_lon)

            # if we can reach destination with reserve, drive and finish
            if agent.can_reach(remaining_km, reserve_soc=plan.reserve_soc):
                drive = agent.apply_drive(remaining_km, plan.avg_speed_kph)
                now += timedelta(minutes=drive["minutes"])
                cur_lat, cur_lon = dest_lat, dest_lon
                finished = True
                break

            # need to charge: find nearby charger (expanding radius if needed)
            radius = plan.search_radius_km
            candidates = pd.DataFrame()
            while candidates.empty and radius <= plan.max_radius_km:
                candidates = self._nearest_chargers(cur_lat, cur_lon, radius)
                if candidates.empty:
                    radius += plan.expand_radius_km

            if candidates.empty:
                # fail safe: no chargers nearby; abort
                return {
                    "status": "fail_no_charger",
                    "agent_id": agent.agent_id,
                    "reason": f"No charger within {plan.max_radius_km} km",
                    "partial": self._summary(agent, plan, now, cur_lat, cur_lon),
                }

            # pick candidate: nearest; if tie, prefer higher power
            cand = candidates.sort_values(["dist_km", "rated_power_kw"], ascending=[True, False]).iloc[0]

            # drive to charger
            dist_to_ch = float(cand["dist_km"])
            drive = agent.apply_drive(dist_to_ch, plan.avg_speed_kph)
            now += timedelta(minutes=drive["minutes"])
            cur_lat, cur_lon = float(cand["latitude"]), float(cand["longitude"])

            # decide target SOC: enough to reach destination + reserve, up to plan.target_soc
            km_needed = _haversine_km(cur_lat, cur_lon, dest_lat, dest_lon)
            kwh_needed = (km_needed * agent.efficiency_Wh_per_km) / 1000.0
            soc_needed = (kwh_needed / agent.battery_kwh) * 100.0 + plan.reserve_soc
            desired_soc = min(100.0, max(plan.target_soc, soc_needed))

            # integrate session at this charger
            station_power = float(cand.get("rated_power_kw", 50.0))
            is_dc_flag = _is_dc(str(cand.get("charger_type", "")))
            integ = integrate_charge_session(
                ev_spec=ev_spec,
                power_curve=power_curve,
                start_soc=agent.soc,
                target_soc=desired_soc,
                station_power_kw=station_power,
                is_dc=is_dc_flag,
            )

            # price the session
            charger: Charger = charger_from_row(cand, self.ds.pricing_catalog)
            breakdown = charger.estimate_session_cost(
                kwh=integ["delivered_kwh"],
                start_dt=now,
                user_type=agent.user_type,
                session_minutes=integ["minutes"],
                include_subscription=False,
            )

            # apply to agent
            agent.apply_charge(integ["delivered_kwh"], integ["minutes"], breakdown)
            now += timedelta(minutes=integ["minutes"])

        return {
            "status": "ok" if finished else "iter_limit",
            **self._summary(agent, plan, now, cur_lat, cur_lon),
        }

    def _summary(self, agent: EVAgent, plan: TripPlan, now: datetime, lat: float, lon: float) -> Dict:
        return {
            "agent_id": agent.agent_id,
            "model": agent.model,
            "objective": agent.objective,
            "user_type": agent.user_type,
            "depart_dt": plan.depart_dt.isoformat(),
            "arrive_dt": now.isoformat(),
            "distance_km": round(agent.distance_driven_km, 3),
            "total_minutes": round(agent.total_minutes, 2),
            "total_cost": round(agent.total_cost, 2),
            "final_soc": round(agent.soc, 2),
            "charging_sessions": agent.charging_sessions,
            "last_lat": lat,
            "last_lon": lon,
        }
