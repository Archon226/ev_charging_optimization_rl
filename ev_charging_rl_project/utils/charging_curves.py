# utils/charging_curves.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


# =============================
# EV spec container
# =============================
@dataclass(frozen=True)
class EVSpec:
    model: str
    usable_battery_kwh: float
    max_dc_kw: float
    max_ac_kw: float
    efficiency_wh_per_km: float  # Wh/km

    @property
    def capacity(self) -> float:
        return float(self.usable_battery_kwh or 0.0)

    @property
    def kwh_per_km(self) -> float:
        # convert Wh/km -> kWh/km
        return float(self.efficiency_wh_per_km or 180.0) / 1000.0


# =============================
# Category helpers
# =============================
_FAST = "Fast"
_RAPID = "Rapid"
_ULTRA = "Ultra"
_VALID_CATS = {_FAST, _RAPID, _ULTRA}

def _norm_category(cat: str) -> str:
    s = (cat or "").strip().title()
    if s in _VALID_CATS: return s
    # accept a few legacy synonyms
    if s in {"Ac", "Slow"}: return _FAST
    if s in {"Dc50", "Dc_50", "Dc-50kw"}: return _RAPID
    if s in {"High Power", "Hpc", "Dc100", "Dc150", "Dc350"}: return _ULTRA
    return s  # will be validated later

def _infer_is_dc_from_category(category: str, station_cap_kw: float, fast_is_dc: Optional[bool]) -> bool:
    """
    We need AC/DC to choose the correct EV limit (ac_max_power_kW vs dc_max_power_kW).
    Policy:
      - Rapid/Ultra -> DC
      - Fast -> use caller hint fast_is_dc if provided; else infer by power (>=50kW -> DC, otherwise AC).
    """
    cat = _norm_category(category)
    if cat in (_RAPID, _ULTRA):
        return True
    if fast_is_dc is not None:
        return bool(fast_is_dc)
    return float(station_cap_kw or 0.0) >= 50.0


# =============================
# EVPowerModel (memoized)
# =============================
class EVPowerModel:
    """
    Memoized access to EV specs and per-SOC charging power.

    Exposes:
      - battery_kwh(model) -> float
      - kwh_per_km(model) -> float
      - power_at_soc(model, soc, station_cap_kw, is_dc=True) -> float
      - integrate_session(model, start_soc_pct, target_soc_pct, station_cap_kw, is_dc=True) -> dict

    New category-aware APIs (aligns with pricing categories):
      - integrate_session_by_category(model, start_soc_pct, target_soc_pct,
          category, station_cap_kw, fast_is_dc=None, ev_capabilities=None, soc_step=1.0, efficiency=1.0)
    """

    def __init__(self, ev_metadata: pd.DataFrame, curves_df: pd.DataFrame):
        # SPEC columns expected in EV_Metadata.csv:
        #   model, battery_kWh, dc_max_power_kW, ac_max_power_kW, avg_consumption_Wh_per_km
        self.specs = _build_specs_index(ev_metadata)           # Dict[str, EVSpec]
        # CURVE columns expected in EV_Charging_Curve_Data.csv:
        #   model, soc_percent, charging_power_kW
        self.curves_df = curves_df.copy()
        self._curve_cache: Dict[str, np.ndarray] = {}          # model -> 101-length power array (kW)

    @classmethod
    def from_frames(cls, bundle) -> "EVPowerModel":
        # kept for compatibility with your previous code (bundle likely DataBundle)
        return cls(bundle.ev_metadata, bundle.charging_curves)

    # ---------- simple getters ----------
    def battery_kwh(self, model: str) -> float:
        return self.specs[_norm_model(model)].capacity

    def kwh_per_km(self, model: str) -> float:
        return self.specs[_norm_model(model)].kwh_per_km

    # ---------- power queries ----------
    def power_at_soc(self, model: str, soc: float, station_cap_kw: float, is_dc: bool = True) -> float:
        spec = self.specs[_norm_model(model)]
        curve_kw = _curve_power_at(self._get_curve(model), soc)
        ev_cap = spec.max_dc_kw if is_dc else spec.max_ac_kw
        # effective power is bounded by curve, station cap, and EV-side cap
        return float(max(0.0, min(curve_kw, station_cap_kw, ev_cap if np.isfinite(ev_cap) and ev_cap > 0 else 1e9)))

    # ---------- core integration (power/time) ----------
    def integrate_session(
        self,
        model: str,
        start_soc_pct: float,
        target_soc_pct: float,
        station_cap_kw: float,
        is_dc: bool = True,
        soc_step: float = 1.0,
        efficiency: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Vectorized integration over SOC buckets.
        Returns dict with delivered_kwh (into battery), minutes, avg_power_kw, trace, caps.
        If you pass efficiency<1.0, we scale the grid energy so delivered_kwh is what goes *into* the battery,
        while energy drawn from the grid would be delivered_kwh/efficiency (the pricing layer can decide which it uses).
        """
        spec = self.specs[_norm_model(model)]
        cap = float(spec.capacity)
        s0 = float(np.clip(start_soc_pct, 0, 100))
        s1 = float(np.clip(target_soc_pct, 0, 100))
        if s1 <= s0:
            return {
                "delivered_kwh": 0.0, "minutes": 0.0, "avg_power_kw": 0.0,
                "start_soc": s0, "target_soc": s1, "steps": [],
                "caps": {"station_cap_hits": 0, "ev_cap_hits": 0, "curve_zero_hits": 0},
            }

        # SOC grid
        soc_grid = np.arange(np.floor(s0), np.ceil(s1) + 1e-9, soc_step, dtype=float)
        soc_grid[0] = s0
        soc_grid[-1] = s1
        dsoc = np.diff(soc_grid)  # percentage points

        curve = self._get_curve(model)                           # 101-length array, kW at integer SOC
        curve_kw = _curve_values(curve, soc_grid[:-1])           # nearest-index sample
        ev_cap_kw = spec.max_dc_kw if is_dc else spec.max_ac_kw
        ev_cap_kw = ev_cap_kw if (np.isfinite(ev_cap_kw) and ev_cap_kw > 0) else 1e9
        eff_kw = np.minimum(np.minimum(curve_kw, station_cap_kw), ev_cap_kw)

        # Energy per SOC slice (into battery)
        slice_kwh_batt = cap * (dsoc / 100.0)                    # battery energy delta
        # If charging inefficiency exists, grid energy would be higher; caller can compute grid_kwh = delivered_kwh/efficiency if needed.
        delivered_kwh = float(np.sum(slice_kwh_batt) * float(efficiency))

        # Time per slice = battery energy / effective power (hours → minutes)
        # Use battery energy; equivalent if you consistently apply efficiency on the price side.
        hours = np.divide(slice_kwh_batt, np.maximum(eff_kw, 1e-9))
        minutes = hours * 60.0
        total_minutes = float(np.sum(minutes))
        avg_power_kw = delivered_kwh / (total_minutes / 60.0) if total_minutes > 0 else 0.0

        station_hits = int(np.sum(station_cap_kw + 1e-9 < np.minimum(curve_kw, ev_cap_kw)))
        ev_hits = int(np.sum(ev_cap_kw + 1e-9 < np.minimum(curve_kw, station_cap_kw)))
        curve_zeros = int(np.sum(curve_kw <= 1e-9))

        # sparse trace for debugging
        keep = max(1, int(round(5.0 / max(1e-6, soc_step))))
        steps = []
        for i in range(0, len(dsoc), keep):
            steps.append({
                "soc_from": round(soc_grid[i], 3),
                "soc_to": round(soc_grid[min(i + 1, len(soc_grid) - 1)], 3),
                "slice_kwh": round(slice_kwh_batt[i], 4),
                "p_curve_kw": round(float(curve_kw[i]), 3),
                "p_effective_kw": round(float(eff_kw[i]), 3),
                "minutes": round(float(minutes[i]), 3),
            })

        return {
            "delivered_kwh": round(delivered_kwh, 4),       # into battery
            "minutes": round(total_minutes, 2),
            "avg_power_kw": round(avg_power_kw, 3),
            "start_soc": s0,
            "target_soc": s1,
            "steps": steps,
            "caps": {"station_cap_hits": station_hits, "ev_cap_hits": ev_hits, "curve_zero_hits": curve_zeros},
        }

    # ---------- category-aware integration ----------
    def integrate_session_by_category(
        self,
        model: str,
        start_soc_pct: float,
        target_soc_pct: float,
        category: str,
        station_cap_kw_for_category: float,
        *,
        fast_is_dc: Optional[bool] = None,
        ev_capabilities: Optional[Dict[str, Any]] = None,
        soc_step: float = 1.0,
        efficiency: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Category-aligned simulation so pricing & time use the SAME notion of charger.

        Args:
          model: EV model (string, must match EV_Metadata.csv 'model')
          start_soc_pct, target_soc_pct: SOC %
          category: 'Fast' | 'Rapid' | 'Ultra'
          station_cap_kw_for_category: station power cap derived for that category (e.g., max kW of that category at the station)
          fast_is_dc: override for 'Fast' ambiguity; if None, infer by power (>=50 kW -> DC else AC)
          ev_capabilities: optional dict[ev_id or model] -> { 'categories_supported': set(...) }
          soc_step: SOC integration step (default 1%)
          efficiency: charger efficiency factor for energy into battery (<=1.0). If you bill grid energy, compute price on delivered_kwh/efficiency.

        Returns:
          dict with delivered_kwh (battery), minutes, avg_power_kw, and caps/trace.
        """
        cat = _norm_category(category)
        if cat not in _VALID_CATS:
            raise ValueError(f"Unknown charger category '{category}' (expected one of {sorted(_VALID_CATS)})")

        # Optional compatibility enforcement (if provided)
        if ev_capabilities is not None:
            # Try exact model key first, else loose 'model' key
            caps = ev_capabilities.get(model) or ev_capabilities.get(str(model))
            if caps and "categories_supported" in caps:
                if cat not in caps["categories_supported"]:
                    raise ValueError(f"EV '{model}' does not support category '{cat}' per ev_capabilities")

        is_dc = _infer_is_dc_from_category(cat, station_cap_kw_for_category, fast_is_dc)
        return self.integrate_session(
            model=model,
            start_soc_pct=start_soc_pct,
            target_soc_pct=target_soc_pct,
            station_cap_kw=float(station_cap_kw_for_category or 0.0),
            is_dc=is_dc,
            soc_step=soc_step,
            efficiency=efficiency,
        )

    # ---------- internals ----------
    def _get_curve(self, model: str) -> np.ndarray:
        key = _norm_model(model)
        if key not in self._curve_cache:
            self._curve_cache[key] = _build_curve_array(self.curves_df, key)
        return self._curve_cache[key]


# =============================
# Legacy API (kept for compat)
# =============================
def build_ev_specs(ev_metadata: pd.DataFrame) -> Dict[str, EVSpec]:
    return _build_specs_index(ev_metadata)

def build_power_curve(curves_df: pd.DataFrame, model: str) -> pd.DataFrame:
    arr = _build_curve_array(curves_df, _norm_model(model))
    return pd.DataFrame({"soc": np.arange(101, dtype=float), "power_kw": arr})

def integrate_charge_session(
    ev_spec: EVSpec,
    power_curve: pd.DataFrame,
    start_soc: float,
    target_soc: float,
    station_power_kw: float,
    is_dc: bool = True,
    soc_step: float = 1.0,
    efficiency: float = 1.0,
) -> Dict[str, Any]:
    # Adapt legacy into the new fast path
    epm = EVPowerModel(
        ev_metadata=pd.DataFrame([{
            "model": ev_spec.model,
            "battery_kWh": ev_spec.usable_battery_kwh,
            "dc_max_power_kW": ev_spec.max_dc_kw,
            "ac_max_power_kW": ev_spec.max_ac_kw,
            "avg_consumption_Wh_per_km": ev_spec.efficiency_wh_per_km
        }]),
        curves_df=pd.DataFrame({
            "model": [ev_spec.model]*len(power_curve),
            "soc_percent": power_curve["soc"].to_numpy(),
            "charging_power_kW": power_curve["power_kw"].to_numpy()
        })
    )
    return epm.integrate_session(
        ev_spec.model, start_soc, target_soc, station_power_kw, is_dc, soc_step, efficiency
    )


# =============================
# Helpers (dataset-aligned)
# =============================
def _norm_model(m: str) -> str:
    return str(m).strip()

def _build_specs_index(ev_metadata: pd.DataFrame) -> Dict[str, EVSpec]:
    # Expected columns in EV_Metadata.csv (as you specified):
    req = ["model", "battery_kWh", "dc_max_power_kW", "ac_max_power_kW", "avg_consumption_Wh_per_km"]
    miss = [c for c in req if c not in ev_metadata.columns]
    if miss:
        raise KeyError(f"EV_Metadata missing columns: {miss}")

    df = ev_metadata.copy()
    # Ensure numeric
    for c in ["battery_kWh", "dc_max_power_kW", "ac_max_power_kW", "avg_consumption_Wh_per_km"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    specs: Dict[str, EVSpec] = {}
    for _, r in df.iterrows():
        m = _norm_model(r["model"])
        raw_eff = float(r["avg_consumption_Wh_per_km"])

        # Unit sanity: most datasets are Wh/km already; small values may be kWh/100km.
        # If it's <60, treat as kWh/100km -> Wh/km * 10.
        if raw_eff < 60.0:
            eff_wh_per_km = raw_eff * 10.0
        else:
            eff_wh_per_km = raw_eff

        specs[m] = EVSpec(
            model=m,
            usable_battery_kwh=float(r["battery_kWh"]),
            max_dc_kw=float(r["dc_max_power_kW"]),
            max_ac_kw=float(r["ac_max_power_kW"]),
            efficiency_wh_per_km=float(eff_wh_per_km),
        )
    return specs

def _build_curve_array(curves_df: pd.DataFrame, model: str) -> np.ndarray:
    # Expected columns in EV_Charging_Curve_Data.csv:
    #   'model', 'soc_percent', 'charging_power_kW'
    req = ["model", "soc_percent", "charging_power_kW"]
    miss = [c for c in req if c not in curves_df.columns]
    if miss:
        raise KeyError(f"EV_Charging_Curve_Data missing columns: {miss}")

    df = curves_df.loc[curves_df["model"].astype(str) == str(model), ["soc_percent", "charging_power_kW"]].dropna()
    if df.empty:
        raise ValueError(f"No charging curve rows for model='{model}'")

    df["soc_percent"] = pd.to_numeric(df["soc_percent"], errors="raise").clip(0, 100)
    df["charging_power_kW"] = pd.to_numeric(df["charging_power_kW"], errors="raise").clip(lower=0)
    df = df.sort_values("soc_percent").groupby("soc_percent", as_index=False)["charging_power_kW"].mean()

    # Build 0..100 grid (kW at integer SOC)
    grid = np.arange(0, 101, 1, dtype=float)
    interp = np.interp(grid, df["soc_percent"].to_numpy(), df["charging_power_kW"].to_numpy())
    return np.maximum(interp, 0.0)

def _curve_power_at(curve: np.ndarray, soc: float) -> float:
    idx = int(np.clip(round(soc), 0, 100))
    return float(curve[idx])

def _curve_values(curve: np.ndarray, socs: np.ndarray) -> np.ndarray:
    idx = np.clip(np.round(socs).astype(int), 0, 100)
    return curve[idx]
