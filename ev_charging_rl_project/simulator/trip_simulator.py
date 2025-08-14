from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# ----- your existing modules -----
from utils.charging_curves import build_ev_specs, build_power_curve, integrate_charge_session
from utils.pricing import PricingCatalog
from env.charger import charger_from_row
# If you use an EVAgent class elsewhere, this file stays agent-agnostic.

# ----- SUMO (optional) -----
try:
    from simulator.sumo_adapter import SumoSim
    from simulator.candidates import find_charger_candidates, EVSpecLite
    HAS_SUMO = True
except Exception:
    HAS_SUMO = False


# =========================
# Dataclasses for I/O
# =========================

@dataclass
class TripPlan:
    agent_id: str
    model: str                 # EV model key present in your EV metadata
    user_type: str             # "Payg" | "Member" | "Fleet"
    objective: str             # "time" | "cost" | "hybrid"
    start_lat: float
    start_lon: float
    dest_lat: float
    dest_lon: float
    depart_dt: datetime
    init_soc_pct: float        # e.g. 35.0
    reserve_soc_pct: float = 10.0
    target_soc_pct: float = 80.0
    avg_speed_kph: float = 28.0          # used only in Haversine fallback
    include_subscription: bool = True
    sessions_per_month: int = 12
    max_detour_km: float = 5.0           # SUMO candidate search
    top_k_candidates: int = 5
    step_horizon_s: int = 180            # SUMO: advance horizon per "continue" step


@dataclass
class ChargeEvent:
    station_id: str
    company_id: int
    charger_type: str
    rated_power_kw: float
    lat: float
    lon: float
    detour_km: float
    wait_min: float
    delivered_kwh: float
    charge_min: float
    unit_price: float
    unit_source: str
    energy_cost: float
    total_cost: float
    start_dt: datetime


@dataclass
class TripResult:
    agent_id: str
    model: str
    user_type: str
    objective: str
    depart_dt: datetime
    arrive_dt: datetime
    total_drive_min: float
    total_charge_min: float
    total_wait_min: float
    total_cost: float
    distance_km: float
    arrived: bool
    charges: List[ChargeEvent]


# =========================
# Helpers
# =========================

def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    s = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(s))

def _haversine_km_vec(lat0: float, lon0: float, lat_arr, lon_arr):
    import numpy as np
    R = 6371.0
    lat0r = np.radians(lat0); lon0r = np.radians(lon0)
    lat2r = np.radians(lat_arr); lon2r = np.radians(lon_arr)
    dlat = lat2r - lat0r; dlon = lon2r - lon0r
    a = np.sin(dlat/2.0)**2 + np.cos(lat0r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    return 2.0 * R * np.arcsin(np.sqrt(a))

def _energy_needed_kwh(eff_kwh_per_km: float, km: float) -> float:
    return max(0.0, eff_kwh_per_km * max(0.0, km))


def _soc_to_kwh(soc_pct: float, batt_kwh: float) -> float:
    return max(0.0, min(100.0, soc_pct)) / 100.0 * batt_kwh


def _kwh_to_soc_pct(kwh: float, batt_kwh: float) -> float:
    return 0.0 if batt_kwh <= 0 else 100.0 * max(0.0, kwh) / batt_kwh

def _bd(bd_obj, key, default=0.0):
    """Read a field from PriceBreakdown or dict, with a default."""
    if isinstance(bd_obj, dict):
        return bd_obj.get(key, default)
    return getattr(bd_obj, key, default)

# =========================
# Main simulator
# =========================

class TripSimulator:
    """
    Drop-in trip simulator that:
      - Uses SUMO if sumo_net is provided (route + detour-based chargers)
      - Falls back to Haversine + fixed speed otherwise
      - Always uses your existing charging-curve + pricing logic
    """

    def __init__(
        self,
        ev_metadata: pd.DataFrame,
        charging_curves: pd.DataFrame,
        stations_merged: pd.DataFrame,
        pricing_catalog: PricingCatalog,
        sumo_net: Optional[str] = None,
    ):
        """
        Args:
          ev_metadata: canonical EV metadata (expects: 'model', 'battery_kWh', 'avg_consumption_Wh_per_km', max power cols)
          charging_curves: your EV_Charging_Curve_Data.csv as DataFrame
          stations_merged: must include columns:
              ['station_id','latitude','longitude','company_id','charger_type','rated_power_kw', ...]
          pricing_catalog: loaded via utils.pricing.load_pricing_catalog(data_dir)
          sumo_net: path to SUMO .net.xml; if None or SUMO not installed, Haversine fallback is used
        """
        self.ev_meta = ev_metadata.copy()
        self.curves_df = charging_curves.copy()
        self.stations = stations_merged.dropna(
            subset=["latitude", "longitude", "company_id"]
        ).copy()
        self.stations["rated_power_kw"] = pd.to_numeric(
            self.stations.get("rated_power_kw", 7.0), errors="coerce"
        ).fillna(7.0)

        self.catalog = pricing_catalog

        # Prebuild EV specs, power curves are built per model lazily
        self.ev_specs = build_ev_specs(self.ev_meta)

        self.use_sumo = bool(sumo_net and HAS_SUMO)
        self.sumo: Optional[SumoSim] = SumoSim(sumo_net) if self.use_sumo else None

    # ------------- public API -------------

    def run_single_trip(self, plan: TripPlan) -> TripResult:
        # Validate EV model
        if plan.model not in self.ev_specs:
            raise ValueError(f"EV model '{plan.model}' not found in metadata/specs")

        spec = self.ev_specs[plan.model]
        batt_kwh = float(getattr(spec, "usable_battery_kWh", None) or getattr(spec, "battery_kWh", None) or 0.0) or \
                   float(self.ev_meta.loc[self.ev_meta["model"] == plan.model].iloc[0].get("battery_kWh", 0.0))
        eff_kwh_per_km = float(self.ev_meta.loc[self.ev_meta["model"] == plan.model].iloc[0].get(
            "avg_consumption_Wh_per_km", 180.0)) / 1000.0

        # Build power curve once
        ev_power_curve = build_power_curve(self.curves_df, plan.model)

        # Time & state trackers
        now = plan.depart_dt
        soc_pct = float(plan.init_soc_pct)
        total_drive_min = 0.0
        total_charge_min = 0.0
        total_wait_min = 0.0
        total_cost = 0.0
        total_distance_km = 0.0
        charges: List[ChargeEvent] = []

        A = (plan.start_lat, plan.start_lon)
        B = (plan.dest_lat, plan.dest_lon)
        cur_lat, cur_lon = A

        # SUMO setup
        if self.use_sumo:
            route = self.sumo.route_between(A, B)
            edge_idx = 0
            remaining_km_est = route.dist_km
        else:
            remaining_km_est = _haversine_km(A, B)

        # Main loop (single-stop or multi-stop; we’ll allow multiple if needed)
        safety_guard = 0
        while True:
            safety_guard += 1
            if safety_guard > 50:
                # avoid infinite loops in degenerate data
                break

            # Distance still to go (estimate)
            if self.use_sumo:
                remaining_km_est = max(0.0, route.dist_km - total_distance_km)
            else:
                remaining_km_est = _haversine_km((cur_lat, cur_lon), B)

            # Check if we can arrive without charging
            energy_needed = _energy_needed_kwh(eff_kwh_per_km, remaining_km_est)
            avail_kwh = _soc_to_kwh(soc_pct - plan.reserve_soc_pct, batt_kwh)
            if avail_kwh >= energy_needed:
                # Drive final leg
                if self.use_sumo:
                    # advance until we consume the whole remaining route
                    # we’ll chunk by step horizon to keep accounting consistent
                    while edge_idx < len(route.per_edge):
                        adv = self.sumo.advance(route, edge_idx, plan.step_horizon_s)
                        if adv["time_s"] <= 0:
                            break
                        step_km = float(adv["dist_km"])
                        step_min = float(adv["time_s"] / 60.0)
                        edge_idx = int(adv["end_idx"])

                        # apply drive
                        total_drive_min += step_min
                        total_distance_km += step_km
                        # SOC drop
                        soc_pct -= _kwh_to_soc_pct(_energy_needed_kwh(eff_kwh_per_km, step_km), batt_kwh)
                        now += timedelta(minutes=step_min)
                    # snap to destination
                    cur_lat, cur_lon = B
                else:
                    # Haversine single shot
                    drive_min = (remaining_km_est / max(1e-6, plan.avg_speed_kph)) * 60.0
                    total_drive_min += drive_min
                    total_distance_km += remaining_km_est
                    soc_pct -= _kwh_to_soc_pct(energy_needed, batt_kwh)
                    now += timedelta(minutes=drive_min)
                    cur_lat, cur_lon = B

                # Done
                arrive_dt = now
                return TripResult(
                    agent_id=plan.agent_id,
                    model=plan.model,
                    user_type=plan.user_type,
                    objective=plan.objective,
                    depart_dt=plan.depart_dt,
                    arrive_dt=arrive_dt,
                    total_drive_min=round(total_drive_min, 2),
                    total_charge_min=round(total_charge_min, 2),
                    total_wait_min=round(total_wait_min, 2),
                    total_cost=round(total_cost, 2),
                    distance_km=round(total_distance_km, 3),
                    arrived=True,
                    charges=charges,
                )

            # Need to charge before arriving
            # ------------------------------------------------------------------
            # Candidate selection
            if self.use_sumo:
                # Build a light EV spec for reachability & power caps
                ev_lite = EVSpecLite(
                    battery_kwh=batt_kwh,
                    eff_kwh_per_km=eff_kwh_per_km,
                    # If you track connectors in stations_merged, wire them here.
                    # Otherwise allow common pair so we don't over-filter.
                    allowed_connectors=("CCS2", "Type2"),
                    max_dc_kw=getattr(spec, "dc_max_power_kW", 120.0),
                    max_ac_kw=getattr(spec, "ac_max_power_kW", 11.0),
                )
                candidates = find_charger_candidates(
                    sim=self.sumo,
                    current_latlon=(cur_lat, cur_lon),
                    dest_latlon=B,
                    stations_df=self.stations,
                    connectors_df=pd.DataFrame(),     # merged table already used; connector check disabled below
                    ev=ev_lite,
                    current_soc=soc_pct/100.0,
                    top_k=plan.top_k_candidates,
                    max_detour_km=plan.max_detour_km,
                    require_connector_ok=False,        # set True if you have station_connectors merged
                )
                if not candidates:
                    # Expand search a bit once
                    candidates = find_charger_candidates(
                        sim=self.sumo, current_latlon=(cur_lat, cur_lon), dest_latlon=B,
                        stations_df=self.stations, connectors_df=pd.DataFrame(), ev=ev_lite,
                        current_soc=soc_pct/100.0, top_k=plan.top_k_candidates,
                        max_detour_km=plan.max_detour_km * 2.0, require_connector_ok=False
                    )
                if not candidates:
                    # Failed to find any feasible stop
                    return TripResult(
                        agent_id=plan.agent_id, model=plan.model, user_type=plan.user_type,
                        objective=plan.objective, depart_dt=plan.depart_dt, arrive_dt=now,
                        total_drive_min=round(total_drive_min, 2),
                        total_charge_min=round(total_charge_min, 2),
                        total_wait_min=round(total_wait_min, 2),
                        total_cost=round(total_cost, 2),
                        distance_km=round(total_distance_km, 3),
                        arrived=False, charges=charges,
                    )
                # Greedy: pick min detour_time_s
                cand = sorted(candidates, key=lambda c: (c.detour_time_s, c.detour_km))[0]
                detour_this_stop_km = float(cand.detour_km)

                # Find station row
                st_row = self.stations.loc[self.stations["station_id"] == cand.station_id]
                if st_row.empty:
                    # shouldn't happen, skip and continue
                    # widen search next loop
                    plan.max_detour_km *= 1.5
                    continue
                station = st_row.iloc[0].to_dict()
                station_lat = float(station["latitude"])
                station_lon = float(station["longitude"])
                rated_power_kw = float(station["rated_power_kw"])
                charger_type = str(station["charger_type"]).strip().title()
                company_id = int(station["company_id"])
                station_id = str(station["station_id"])

                # Travel to station (use SUMO sub-route A->POI)
                leg_to_station = self.sumo.route_between((cur_lat, cur_lon), (station_lat, station_lon))
                # Consume along that leg
                # We'll step once with "exact" leg time to keep accounting clean
                step_km = float(leg_to_station.dist_km)
                step_min = float(leg_to_station.time_s / 60.0)
                total_drive_min += step_min
                total_distance_km += step_km
                soc_pct -= _kwh_to_soc_pct(_energy_needed_kwh(eff_kwh_per_km, step_km), batt_kwh)
                now += timedelta(minutes=step_min)
                cur_lat, cur_lon = (station_lat, station_lon)

                # Wait time model (simple placeholder; replace with your availability model when ready)
                wait_min = self._estimate_wait_minutes(charger_type, now)
                total_wait_min += wait_min
                now += timedelta(minutes=wait_min)

            else:
                # Haversine fallback: choose nearest station (greedy)
                lat_arr = self.stations["latitude"].to_numpy()
                lon_arr = self.stations["longitude"].to_numpy()
                dists = _haversine_km_vec(cur_lat, cur_lon, lat_arr, lon_arr)
                ix = int(dists.argmin())
                station = self.stations.iloc[ix].to_dict()

                station_lat = float(station["latitude"])
                station_lon = float(station["longitude"])
                rated_power_kw = float(station["rated_power_kw"])
                charger_type = str(station["charger_type"]).strip().title()
                company_id = int(station["company_id"])
                station_id = str(station["station_id"])

                # drive to station
                step_km = float(dists[ix])
                drive_min = (step_km / max(1e-6, plan.avg_speed_kph)) * 60.0
                total_drive_min += drive_min
                total_distance_km += step_km
                soc_pct -= _kwh_to_soc_pct(_energy_needed_kwh(eff_kwh_per_km, step_km), batt_kwh)
                now += timedelta(minutes=drive_min)
                cur_lat, cur_lon = (station_lat, station_lon)

                # record detour for this stop
                detour_this_stop_km = step_km

                # simple queue model (as before)
                wait_min = self._estimate_wait_minutes(charger_type, now)
                total_wait_min += wait_min
                now += timedelta(minutes=wait_min)


            # ------------------------------------------------------------------
            # Charge session at chosen station using charging curve + pricing
            # ------------------------------------------------------------------

            # Target SOC policy: enough to finish + buffer, capped by plan.target_soc_pct
            if self.use_sumo:
                remaining_km_for_finish = self.sumo.route_between((cur_lat, cur_lon), B).dist_km
            else:
                remaining_km_for_finish = _haversine_km((cur_lat, cur_lon), B)

            energy_to_finish_kwh = _energy_needed_kwh(eff_kwh_per_km, remaining_km_for_finish)
            target_soc_need_pct = soc_pct + _kwh_to_soc_pct(energy_to_finish_kwh, batt_kwh) + plan.reserve_soc_pct
            target_soc_pct = min(plan.target_soc_pct, max(soc_pct, target_soc_need_pct, plan.reserve_soc_pct))

            # Build/ensure the EV power curve using your helper (already done earlier; keep if you cache it)
            # ev_power_curve = build_power_curve(self.curves_df, plan.model)      # from utils.charging_curves

            # Integrate charging curve using YOUR function signature (dict out)
            # utils.charging_curves.integrate_charge_session(ev_spec, power_curve, start_soc, target_soc, station_power_kw, is_dc)
            is_dc = (charger_type == "Rapid")
            int_res = integrate_charge_session(
                ev_spec=spec,
                power_curve=ev_power_curve,
                start_soc=float(soc_pct),
                target_soc=float(min(99.0, target_soc_pct)),
                station_power_kw=float(rated_power_kw),
                is_dc=is_dc,
            )
            delivered_kwh = float(int_res["delivered_kwh"])
            charge_min    = float(int_res["minutes"])

            # PRICE BY SESSION START (so peak/off-peak is correct)
            session_start_dt = now
            charger = charger_from_row(station, self.catalog)  # uses your PricingCatalog
            bd = charger.estimate_session_cost(
                start_dt=session_start_dt,
                kwh=delivered_kwh,
                session_minutes=charge_min,
                user_type=plan.user_type,
                include_subscription=plan.include_subscription,
                sessions_per_month=plan.sessions_per_month,
            )

            # Advance time after pricing
            total_charge_min += charge_min
            now += timedelta(minutes=charge_min)
            total_cost += float(_bd(bd, "total_cost", 0.0))

            # detour for this stop should have been recorded earlier as detour_this_stop_km (SUMO/Haversine branches)
            charges.append(ChargeEvent(
                station_id=station_id,
                company_id=company_id,
                charger_type=charger_type,
                rated_power_kw=rated_power_kw,
                lat=cur_lat, lon=cur_lon,
                detour_km=float(detour_this_stop_km) if 'detour_this_stop_km' in locals() else 0.0,
                wait_min=float(wait_min),
                delivered_kwh=delivered_kwh,
                charge_min=charge_min,
                unit_price=float(_bd(bd, "unit_price", 0.0)),
                unit_source=str(_bd(bd, "unit_source", "")),
                energy_cost=float(_bd(bd, "energy_cost", 0.0)),
                total_cost=float(_bd(bd, "total_cost", 0.0)),
                start_dt=session_start_dt,
            ))


            # Update SOC after charge
            soc_pct = min(100.0, soc_pct + _kwh_to_soc_pct(delivered_kwh, batt_kwh))

            # Continue; loop will check if we can arrive now


        # Fallback (should not hit)
        return TripResult(
            agent_id=plan.agent_id, model=plan.model, user_type=plan.user_type,
            objective=plan.objective, depart_dt=plan.depart_dt, arrive_dt=now,
            total_drive_min=round(total_drive_min, 2),
            total_charge_min=round(total_charge_min, 2),
            total_wait_min=round(total_wait_min, 2),
            total_cost=round(total_cost, 2), distance_km=round(total_distance_km, 3),
            arrived=False, charges=charges
        )

    # ------------- simple availability model (stub) -------------

    def _estimate_wait_minutes(self, charger_type: str, now: datetime) -> float:
        """Toy queue model: busy at evening for Rapids. Replace with your data-driven model later."""
        tod = now.hour + now.minute/60.0
        rush = (17 <= tod <= 19)
        base = 2.0 if charger_type == "Rapid" else 1.0
        mean_wait = (8.0 if rush else 3.0) * base
        # exponential sample; keep deterministic here by returning mean (or random if desired)
        return float(mean_wait)
