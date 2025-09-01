# utils/session_planner.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple, Optional, Any

from utils.charging_curves import EVPowerModel, _norm_category
from utils.pricing import PricingCatalog


@dataclass(frozen=True)
class SessionResult:
    station_id: str
    company_id: int
    company_name: str
    category: str                 # Fast/Rapid/Ultra
    start_soc: float
    target_soc: float
    delivered_kwh_batt: float     # energy into battery
    session_minutes: float
    avg_power_kw: float
    total_price: float
    unit_price_source: str        # e.g. by_type, peak_window, base, etc.


def get_candidate_pairs(
    ev_model: str,
    station_ids: Iterable[str],
    station_capabilities: Dict[str, Any],   # station_id -> StationCaps (from data_loader.load_all_ready)
    ev_capabilities: Dict[str, Any],        # ev_id or model -> EVCaps
) -> List[Tuple[str, str]]:
    """
    Returns feasible (station_id, category) pairs by intersecting:
      - station categories_available
      - ev categories_supported
    We accept either exact key 'ev_model' or the simpler model key in ev_capabilities.
    """
    # Try exact key first, else model-only key
    ev_caps = ev_capabilities.get(ev_model) or ev_capabilities.get(str(ev_model))
    if not ev_caps or not ev_caps.get("categories_supported"):
        return []

    ev_cats = set(ev_caps["categories_supported"])
    out: List[Tuple[str, str]] = []
    for sid in station_ids:
        sc = station_capabilities.get(sid)
        if not sc:
            continue
        cats = set(sc.categories_available) if isinstance(sc.categories_available, (set, list, tuple)) else set(sc.get("categories_available", []))
        for c in (ev_cats & cats):
            out.append((sid, c))
    return out


def evaluate_session(
    *,
    ev_model: str,
    station_id: str,
    category: str,                                # Fast/Rapid/Ultra (case-insensitive ok)
    start_soc_pct: float,
    target_soc_pct: float,
    when: datetime,
    pricing: PricingCatalog,
    ev_power: EVPowerModel,
    station_capabilities: Dict[str, Any],         # station_id -> StationCaps
    user_type: str = "Default",
    fast_is_dc: Optional[bool] = None,            # resolves Fast ambiguity; None -> infer by power
    efficiency: float = 1.0,                      # battery-side efficiency (<=1.0). Price grid energy by dividing delivered_kwh if desired
    include_subscription: bool = False,
    sessions_per_month: int = 20,
    idle_minutes: float = 0.0,
    detailed_pricing_breakdown: bool = False,     # True -> calls estimate_session to get unit_source, etc.
) -> SessionResult:
    """
    One-shot: simulate charging at (station_id, category) and price it.

    Contract:
      - Uses the SAME `category` throughout (no AC/DC drift).
      - Station power cap is taken from station_capabilities[station_id].power_stats_by_category[category]
      - If efficiency<1.0 and you charge by grid energy, either:
          (a) call with detailed_pricing_breakdown=True and multiply kWh accordingly before pricing, or
          (b) leave as-is to price battery energy (your policy call).
    """
    cat = _norm_category(category)
    sc = station_capabilities.get(station_id)
    if sc is None:
        raise KeyError(f"Unknown station_id '{station_id}'")

    # Pull company data
    company_id = int(sc.company_id) if hasattr(sc, "company_id") else int(sc.get("company_id"))
    company_name = sc.company_name if hasattr(sc, "company_name") else sc.get("company_name")

    # Pull station power cap for that category
    p_by_cat = sc.power_stats_by_category if hasattr(sc, "power_stats_by_category") else sc.get("power_stats_by_category", {})
    if cat not in p_by_cat or p_by_cat[cat] is None:
        raise ValueError(f"Station '{station_id}' does not advertise power for category '{cat}'")
    station_cap_kw = float(p_by_cat[cat])

    # Simulate session (battery-side kWh & minutes)
    sim = ev_power.integrate_session_by_category(
        model=ev_model,
        start_soc_pct=start_soc_pct,
        target_soc_pct=target_soc_pct,
        category=cat,
        station_cap_kw_for_category=station_cap_kw,
        fast_is_dc=fast_is_dc,
        ev_capabilities=None,  # optional: pass in your EV caps dict if you want hard enforcement here
        soc_step=1.0,
        efficiency=efficiency,
    )
    delivered_kwh_batt = float(sim["delivered_kwh"])
    session_minutes = float(sim["minutes"])
    avg_power_kw = float(sim["avg_power_kw"])

    # Decide what energy to bill on:
    # - If you want to bill grid energy: bill_kwh = delivered_kwh_batt / efficiency (if efficiency<=1.0)
    # - If you bill battery energy: bill_kwh = delivered_kwh_batt
    bill_kwh = delivered_kwh_batt if efficiency >= 1.0 else (delivered_kwh_batt / max(efficiency, 1e-9))

    if detailed_pricing_breakdown:
        bd = pricing.estimate_session(
            company_id=company_id,
            charger_type=cat,
            user_type=user_type,
            start_dt=when,
            kwh=bill_kwh,
            session_minutes=session_minutes,
            include_subscription=include_subscription,
            sessions_per_month=sessions_per_month,
        )
        total_price = float(bd.total_cost)
        unit_src = bd.unit_source
    else:
        # Lightweight path for RL
        total_price = pricing.compute_price(
            company_id=company_id,
            charger_type=cat,
            user_type=user_type,
            start_dt=when,
            kwh=bill_kwh,
            session_minutes=session_minutes,
            idle_minutes=idle_minutes,
            include_subscription=include_subscription,
            sessions_per_month=sessions_per_month,
        )
        # Reuse the unit price source by resolving once (cheap)
        _, unit_src = pricing.resolve_unit_price(company_id, cat, user_type, when)

    return SessionResult(
        station_id=str(station_id),
        company_id=company_id,
        company_name=str(company_name),
        category=cat,
        start_soc=float(start_soc_pct),
        target_soc=float(target_soc_pct),
        delivered_kwh_batt=round(delivered_kwh_batt, 4),
        session_minutes=round(session_minutes, 2),
        avg_power_kw=round(avg_power_kw, 3),
        total_price=float(total_price),
        unit_price_source=str(unit_src),
    )


# -------- optional: batch helpers for efficiency in env --------

def rank_candidates_by_objective(
    *,
    ev_model: str,
    candidates: List[Tuple[str, str]],            # list of (station_id, category)
    start_soc_pct: float,
    target_soc_pct: float,
    when: datetime,
    pricing: PricingCatalog,
    ev_power: EVPowerModel,
    station_capabilities: Dict[str, Any],
    user_type: str = "Default",
    objective: str = "cost",                      # "cost" | "time" | "hybrid"
    time_weight: float = 0.5,                     # for hybrid: reward = w*time + (1-w)*cost
    cost_weight: float = 0.5,
    efficiency: float = 1.0,
    include_subscription: bool = False,
    sessions_per_month: int = 20,
) -> List[Tuple[Tuple[str, str], SessionResult, float]]:
    """
    Evaluates and ranks candidates by the chosen objective.
    Returns a list of (candidate, result, score) sorted ascending (lower is better).
    """
    results: List[Tuple[Tuple[str, str], SessionResult, float]] = []
    for sid, cat in candidates:
        res = evaluate_session(
            ev_model=ev_model,
            station_id=sid,
            category=cat,
            start_soc_pct=start_soc_pct,
            target_soc_pct=target_soc_pct,
            when=when,
            pricing=pricing,
            ev_power=ev_power,
            station_capabilities=station_capabilities,
            user_type=user_type,
            efficiency=efficiency,
            include_subscription=include_subscription,
            sessions_per_month=sessions_per_month,
            detailed_pricing_breakdown=False,
        )
        if objective == "time":
            score = res.session_minutes
        elif objective == "cost":
            score = res.total_price
        else:
            # hybrid: normalized simple blend (you can switch to a better normalization if desired)
            score = time_weight * res.session_minutes + cost_weight * res.total_price
        results.append(((sid, cat), res, float(score)))
    return sorted(results, key=lambda x: x[2])
