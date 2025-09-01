# utils/pricing.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Iterable

import math
import numpy as np
import pandas as pd


# ----------------------------
# Normalization & small utils
# ----------------------------
def _norm_user_type(x: str) -> str:
    return (x or "").strip().title() or "Default"

def _norm_charger_type(x: str) -> str:
    # Expect exactly one of: Fast / Rapid / Ultra (title-cased)
    s = (x or "").strip().lower()
    if s in ("fast", "ac", "slow"):   # allow legacy synonyms; normalize to Fast
        return "Fast"
    if s in ("rapid", "dc50", "dc_50", "dc_50kw", "dc-50kw"):
        return "Rapid"
    if s in ("ultra", "high power", "hpc", "dc100", "dc150", "dc350"):
        return "Ultra"
    # Fall back to title-case of the input
    return (x or "").strip().title() or "Fast"

def _to_bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "t")
    if isinstance(x, (float, int, np.floating, np.integer)):
        return bool(x) and not pd.isna(x)
    return bool(x)

def _nz(x, default=0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (float, int, np.floating, np.integer)):
            return 0.0 if pd.isna(x) else float(x)
        s = str(x).strip().lower()
        if s in ("", "nan", "none", "null"):
            return float(default)
        v = float(x)
        return 0.0 if math.isnan(v) else v
    except Exception:
        return float(default)


# ----------------------------
# Data classes
# ----------------------------
@dataclass(frozen=True)
class FeePolicy:
    connection_fee: float = 0.0
    infrastructure_fee: float = 0.0
    minimum_charge_amount: float = 0.0
    idle_fee_flag: int = 0
    idle_fee_per_min: float = 0.0
    idle_fee_start_min: float = 0.0
    max_overstay_fee: float = 0.0
    overstay_fee_start_min: float = 0.0
    overstay_fee_per_min: float = 0.0
    pre_auth_fee_flag: int = 0
    pre_auth_avg_fee: float = 0.0
    subscription_required: bool = False
    subscription_fee_per_month: float = 0.0

@dataclass
class PriceBreakdown:
    unit_price: float
    unit_source: str
    energy_cost: float
    connection_fee: float
    infrastructure_fee: float
    overstay_fee: float
    idle_fee: float
    min_charge_applied: float
    subscription_applied: float
    preauth_hold: float
    total_cost: float
    notes: Dict[str, Any]


# ============================================================
# PricingCatalog
# ============================================================
class PricingCatalog:
    """
    Fast tariff engine keyed by (company_id, charger_type[, user_type]).
    Column names expected from datasets:
      - pricing_core.csv:           'company_id', 'company_name', optional flags:
            ['time_sensitive_flag','day_sensitive_flag','subscription_required',
             'subscription_fee_per_month','connection_fee', ...]
      - pricing_by_charger_type.csv: 'company_id', 'charger_type', optional:
            ['user_type','exists_flag','price_per_kwh', ...]
        (charger_type normalizes to: Fast/Rapid/Ultra)
      - pricing_conditions.csv:     'company_id', windows/prices:
            ['peak_hours','off_peak_hours','peak_price','off_peak_price','weekend_price',
             'infrastructure_fee','minimum_charge_amount',
             'idle_fee_flag','idle_fee_per_min','idle_fee_start_min',
             'max_overstay_fee','overstay_fee_start_min','overstay_fee_per_min',
             'pre_auth_fee_flag','pre_auth_avg_fee', ...]

    Strictness:
      - strict=True (default): if (company_id, charger_type) is missing or 'exists_flag' is False,
        we RAISE instead of silently falling back. This protects training signals.
      - strict=False: allow base price fallback.
    """
    def __init__(self,
                 core: pd.DataFrame,
                 by_type: pd.DataFrame,
                 cond: pd.DataFrame,
                 strict: bool = True):
        self.strict = bool(strict)

        # Defensive copies
        self.core = core.copy()
        self.by_type = by_type.copy()
        self.cond = cond.copy()

        # Enforce company_id numeric
        for f in ("company_id",):
            for df in (self.core, self.by_type, self.cond):
                if f in df.columns:
                    df[f] = pd.to_numeric(df[f], errors="raise").astype(int)

        # Normalize charger_type and user_type if present
        if "charger_type" in self.by_type.columns:
            self.by_type["charger_type"] = self.by_type["charger_type"].map(_norm_charger_type)
        if "user_type" in self.by_type.columns:
            self.by_type["user_type"] = self.by_type["user_type"].map(_norm_user_type)
        else:
            self.by_type["user_type"] = "Default"

        # Build indices
        self.core_idx = self.core.set_index("company_id")
        # Some datasets may not include 'user_type' (per-company single price per charger_type)
        if {"company_id", "charger_type", "user_type"}.issubset(self.by_type.columns):
            self.by_type_idx = self.by_type.set_index(["company_id", "charger_type", "user_type"])
        else:
            self.by_type_idx = None  # fallback to scanning rows per (company_id, charger_type)

        self.cond_idx = self.cond.set_index("company_id") if "company_id" in self.cond.columns else pd.DataFrame().set_index(pd.Index([]))

        # Pre-parse time windows once per company
        self._windows: Dict[int, Dict[str, Tuple[Tuple[int, int], ...]]] = {}
        if not self.cond_idx.empty:
            for cid, row in self.cond_idx.iterrows():
                self._windows[int(cid)] = {
                    "peak": self._parse_windows(row.get("peak_hours")),
                    "off":  self._parse_windows(row.get("off_peak_hours")),
                }

    # ----------------------------
    # Alt constructor from in-memory index (from data_loader.load_all_ready)
    # ----------------------------
    @classmethod
    def from_index(cls, pricing_index: Dict[str, Any], strict: bool = True) -> "PricingCatalog":
        """
        Build a PricingCatalog from pre-indexed dicts:
          pricing_index = {
            "core": { company_id -> dict(...) },
            "by_type": { (company_id, category) -> dict(... price_per_kwh, exists_flag, user_type, ...) },
            "conditions": { company_id -> dict(...) }
          }
        """
        # Core
        core_rows = []
        for cid, row in pricing_index.get("core", {}).items():
            r = dict(row)
            r["company_id"] = int(cid)
            core_rows.append(r)
        core_df = pd.DataFrame(core_rows) if core_rows else pd.DataFrame(columns=["company_id"])

        # By type
        bt_rows = []
        for (cid, cat), row in pricing_index.get("by_type", {}).items():
            r = dict(row)
            r["company_id"] = int(cid)
            # Accept both 'charger_category' and 'charger_type'
            cat_val = r.pop("charger_category", None) or r.get("charger_type", cat)
            r["charger_type"] = _norm_charger_type(cat_val)
            # Ensure user_type column exists
            r["user_type"] = _norm_user_type(r.get("user_type", "Default"))
            # Normalize presence flag
            if "exists_flag" not in r:
                r["exists_flag"] = True
            bt_rows.append(r)
        by_type_df = pd.DataFrame(bt_rows) if bt_rows else pd.DataFrame(columns=["company_id", "charger_type", "user_type"])

        # Conditions
        cond_rows = []
        for cid, row in pricing_index.get("conditions", {}).items():
            r = dict(row)
            r["company_id"] = int(cid)
            cond_rows.append(r)
        cond_df = pd.DataFrame(cond_rows) if cond_rows else pd.DataFrame(columns=["company_id"])

        return cls(core_df, by_type_df, cond_df, strict=strict)

    # ----------------------------
    # Time windows helpers
    # ----------------------------
    @staticmethod
    def _parse_windows(txt: Optional[str]):
        if not isinstance(txt, str) or "-" not in txt:
            return tuple()
        def to_min(s: str) -> int:
            h, m = s.split(":")
            return int(h) * 60 + int(m)
        wins = []
        for chunk in txt.split(";"):
            chunk = chunk.strip()
            if "-" in chunk:
                a, b = [x.strip() for x in chunk.split("-")]
                wins.append((to_min(a), to_min(b)))
        return tuple(wins)

    @staticmethod
    def _in_window(mins: int, win: Tuple[int, int]) -> bool:
        s, e = win
        return s <= mins < e if s <= e else (mins >= s or mins < e)

    def _in_any_window(self, mins: int, wins: Tuple[Tuple[int, int], ...]) -> bool:
        return any(self._in_window(mins, w) for w in (wins or ()))

    def _cond_row(self, company_id: int) -> Optional[pd.Series]:
        if company_id in self.cond_idx.index:
            row = self.cond_idx.loc[company_id]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        return None

    # ----------------------------
    # Unit price resolution (strict by default)
    # ----------------------------
    def resolve_unit_price(self, company_id: int, charger_type: str, user_type: str, dt: datetime) -> Tuple[float, str]:
        """
        Returns (unit_price_gbp_per_kwh, source_tag)
        Resolution order (strict=True):
          1) by_type exact (company_id, charger_type, user_type) and exists_flag==True
          2) by_type any-user if unique and exists_flag==True
          3) time windows (if time_sensitive_flag) using conditions
          4) weekend (if day_sensitive_flag) using conditions
          5) base price from core
        If no resolvable price and strict=True -> raises ValueError.
        """
        user_type = _norm_user_type(user_type)
        charger_type = _norm_charger_type(charger_type)

        # Pull core/cond rows
        if company_id not in self.core_idx.index:
            raise ValueError(f"Unknown company_id={company_id}")
        core = self.core_idx.loc[company_id]
        cond = self._cond_row(company_id)

        # 1) exact by_type match
        if self.by_type_idx is not None:
            key = (company_id, charger_type, user_type)
            if key in self.by_type_idx.index:
                row = self.by_type_idx.loc[key]
                exists = row["exists_flag"] if "exists_flag" in row else True
                price = row.get("price_per_kwh")
                if _to_bool(exists) and pd.notna(price):
                    return float(price), "by_type"
                # exists_flag False → do not allow price from this row
                if self.strict:
                    raise ValueError(f"Tariff exists_flag=False for {key}")

            # 2) any-user (unique) for this company_id & charger_type
            bt = self.by_type[(self.by_type["company_id"] == company_id) &
                              (self.by_type["charger_type"] == charger_type)]
            bt = bt[pd.notna(bt.get("price_per_kwh"))]
            # filter to rows that have exists_flag True (or missing -> treat as True)
            if "exists_flag" in bt.columns:
                bt = bt[bt["exists_flag"].map(_to_bool)]
            uniq = bt["price_per_kwh"].dropna().unique()
            if len(uniq) == 1:
                return float(uniq[0]), "by_type_any_user"

            if self.strict:
                # If we have any rows for that (company, type) but none valid, raise
                if not bt.empty or ((company_id, charger_type, user_type) in self.by_type_idx.index):
                    raise ValueError(f"No valid by_type price for company_id={company_id}, charger_type={charger_type}")

        else:
            # No multi-index: scan rows
            bt = self.by_type[(self.by_type["company_id"] == company_id) &
                              (self.by_type["charger_type"] == charger_type)]
            if not bt.empty:
                # Prefer matching user_type
                pref = bt[bt["user_type"].map(_norm_user_type) == user_type] if "user_type" in bt.columns else bt
                cand = pref if not pref.empty else bt
                # respect exists_flag
                if "exists_flag" in cand.columns:
                    cand = cand[cand["exists_flag"].map(_to_bool)]
                cand = cand[pd.notna(cand.get("price_per_kwh"))]
                if not cand.empty:
                    return float(cand["price_per_kwh"].iloc[0]), "by_type"
                if self.strict:
                    raise ValueError(f"No valid by_type price for company_id={company_id}, charger_type={charger_type}")

        # 3) time windows (only if flagged)
        if _to_bool(core.get("time_sensitive_flag", 0)) and cond is not None:
            minutes = dt.hour * 60 + dt.minute
            wins = self._windows.get(int(company_id), {})
            peak_wins = wins.get("peak", tuple())
            off_wins = wins.get("off", tuple())
            if self._in_any_window(minutes, peak_wins) and pd.notna(cond.get("peak_price")):
                return float(cond["peak_price"]), "peak_window"
            if self._in_any_window(minutes, off_wins) and pd.notna(cond.get("off_peak_price")):
                return float(cond["off_peak_price"]), "offpeak_window"

        # 4) weekend (only if flagged)
        if _to_bool(core.get("day_sensitive_flag", 0)) and cond is not None:
            if dt.weekday() >= 5 and pd.notna(cond.get("weekend_price")):
                return float(cond["weekend_price"]), "weekend"

        # 5) base price
        base = core.get("base_price_per_kwh")
        if pd.notna(base):
            # If strict and we KNOW this company advertises charger_type in by_type but none valid, we already raised above.
            return float(base), "base"

        # No price found
        if self.strict:
            raise ValueError(f"No resolvable unit price for company_id={company_id}, charger_type={charger_type}")
        # Non-strict fallback
        return float(0.0), "fallback_zero"

    # Simple helpers
    def price_per_kwh_gbp(self, company_id: int, charger_type: str, user_type: str, dt: datetime) -> float:
        price, _ = self.resolve_unit_price(company_id, charger_type, user_type, dt)
        return float(price)

    def price_p_per_kwh(self, company_id: int, charger_type: str, user_type: str, dt: datetime) -> float:
        return self.price_per_kwh_gbp(company_id, charger_type, user_type, dt) * 100.0

    # ----------------------------
    # Fees aggregation
    # ----------------------------
    def fees_for_company(self, company_id: int) -> FeePolicy:
        if company_id not in self.core_idx.index:
            # Defensive default; most training workflows should validate beforehand
            return FeePolicy()
        core = self.core_idx.loc[company_id]
        cond = self._cond_row(company_id)
        get = (cond.get if cond is not None else (lambda *_args, **_kw: None))
        return FeePolicy(
            connection_fee=_nz(core.get("connection_fee", 0)),
            infrastructure_fee=_nz(get("infrastructure_fee", 0)),
            minimum_charge_amount=_nz(get("minimum_charge_amount", 0)),
            idle_fee_flag=int(_nz(get("idle_fee_flag", 0))),
            idle_fee_per_min=_nz(get("idle_fee_per_min", 0)),
            idle_fee_start_min=_nz(get("idle_fee_start_min", 0)),
            max_overstay_fee=_nz(get("max_overstay_fee", 0)),
            overstay_fee_start_min=_nz(get("overstay_fee_start_min", 0)),
            overstay_fee_per_min=_nz(get("overstay_fee_per_min", 0)),
            pre_auth_fee_flag=int(_nz(get("pre_auth_fee_flag", 0))),
            pre_auth_avg_fee=_nz(get("pre_auth_avg_fee", 0)),
            subscription_required=_to_bool(core.get("subscription_required", False)),
            subscription_fee_per_month=_nz(core.get("subscription_fee_per_month", 0)),
        )

    # ----------------------------
    # Full session estimator (unchanged API; strict on unit price)
    # ----------------------------
    def estimate_session(self,
                         company_id: int,
                         charger_type: str,
                         user_type: str,
                         start_dt: datetime,
                         kwh: float,
                         session_minutes: float,
                         include_subscription: bool = False,
                         sessions_per_month: int = 20) -> PriceBreakdown:
        unit_price, src = self.resolve_unit_price(company_id, charger_type, user_type, start_dt)
        fees = self.fees_for_company(company_id)

        energy_cost = round(_nz(unit_price) * _nz(kwh), 2)
        conn = _nz(fees.connection_fee)
        infra = _nz(fees.infrastructure_fee)

        mins = _nz(session_minutes)
        idle_fee = 0.0
        if _nz(fees.idle_fee_flag) and _nz(fees.idle_fee_per_min) > 0 and mins > _nz(fees.idle_fee_start_min):
            idle_minutes = max(0.0, mins - _nz(fees.idle_fee_start_min))
            idle_fee = round(idle_minutes * _nz(fees.idle_fee_per_min), 2)

        overstay_fee = 0.0
        if _nz(fees.overstay_fee_per_min) > 0 and mins > _nz(fees.overstay_fee_start_min):
            overstay_minutes = max(0.0, mins - _nz(fees.overstay_fee_start_min))
            overstay_fee = round(overstay_minutes * _nz(fees.overstay_fee_per_min), 2)
            if _nz(fees.max_overstay_fee) > 0:
                overstay_fee = float(min(overstay_fee, _nz(fees.max_overstay_fee)))

        sub_applied = 0.0
        if include_subscription and fees.subscription_required and _nz(fees.subscription_fee_per_month) > 0 and sessions_per_month > 0:
            sub_applied = round(_nz(fees.subscription_fee_per_month) / float(sessions_per_month), 2)

        subtotal = energy_cost + conn + infra + idle_fee + overstay_fee + sub_applied
        min_charge = _nz(fees.minimum_charge_amount)
        min_applied = float(max(0.0, min_charge - subtotal))
        total = round(max(subtotal, min_charge), 2)

        return PriceBreakdown(
            unit_price=float(unit_price), unit_source=src,
            energy_cost=round(energy_cost, 2),
            connection_fee=round(conn, 2), infrastructure_fee=round(infra, 2),
            overstay_fee=round(overstay_fee, 2), idle_fee=round(idle_fee, 2),
            min_charge_applied=round(min_applied, 2), subscription_applied=round(sub_applied, 2),
            preauth_hold=round(_nz(fees.pre_auth_avg_fee) if _nz(fees.pre_auth_fee_flag) else 0.0, 2),
            total_cost=total,
            notes={
                "charger_type": _norm_charger_type(charger_type),
                "user_type": _norm_user_type(user_type),
                "kwh": round(_nz(kwh), 3),
                "minutes": round(mins, 2)
            },
        )

    # ----------------------------
    # Lightweight price helper for RL loop
    # ----------------------------
    def compute_price(self,
                      company_id: int,
                      charger_type: str,
                      user_type: str,
                      start_dt: datetime,
                      kwh: float,
                      session_minutes: float,
                      idle_minutes: float = 0.0,
                      include_subscription: bool = False,
                      sessions_per_month: int = 20) -> float:
        """
        Returns just the total £ for performance-sensitive calls in training.
        Same logic as estimate_session (unit price strictness included).
        """
        unit_price, _ = self.resolve_unit_price(company_id, charger_type, user_type, start_dt)
        fees = self.fees_for_company(company_id)

        energy_cost = _nz(unit_price) * _nz(kwh)
        conn = _nz(fees.connection_fee)
        infra = _nz(fees.infrastructure_fee)

        idle_fee = 0.0
        if _nz(fees.idle_fee_flag) and _nz(fees.idle_fee_per_min) > 0 and _nz(idle_minutes) > 0:
            idle_fee = _nz(idle_minutes) * _nz(fees.idle_fee_per_min)

        overstay_fee = 0.0
        if _nz(fees.overstay_fee_per_min) > 0 and _nz(session_minutes) > _nz(fees.overstay_fee_start_min):
            overstay_minutes = max(0.0, _nz(session_minutes) - _nz(fees.overstay_fee_start_min))
            overstay_fee = overstay_minutes * _nz(fees.overstay_fee_per_min)
            if _nz(fees.max_overstay_fee) > 0:
                overstay_fee = float(min(overstay_fee, _nz(fees.max_overstay_fee)))

        sub_applied = 0.0
        if include_subscription and fees.subscription_required and _nz(fees.subscription_fee_per_month) > 0 and sessions_per_month > 0:
            sub_applied = _nz(fees.subscription_fee_per_month) / float(sessions_per_month)

        subtotal = energy_cost + conn + infra + idle_fee + overstay_fee + sub_applied
        total = float(max(subtotal, _nz(self._cond_row(company_id).get("minimum_charge_amount", 0) if self._cond_row(company_id) is not None else 0)))
        return round(total, 2)


# ============================================================
# Small loader helper (kept for compatibility with your codebase)
# ============================================================
def load_pricing_catalog(data_dir) -> PricingCatalog:
    from pathlib import Path
    data_dir = Path(data_dir)

    core = pd.read_csv(data_dir / "pricing_core.csv")
    by_type = pd.read_csv(data_dir / "pricing_by_charger_type.csv")
    cond = pd.read_csv(data_dir / "pricing_conditions.csv")

    # Ensure canonical column names coming from datasets, without altering them:
    # - We expect 'company_id' in all three files.
    # - We expect 'charger_type' and 'price_per_kwh' in pricing_by_charger_type.csv.
    # - If datasets used 'charger_category' instead, map it here.
    if "charger_type" not in by_type.columns and "charger_category" in by_type.columns:
        by_type = by_type.rename(columns={"charger_category": "charger_type"})

    # Default user_type column if missing
    if "user_type" not in by_type.columns:
        by_type["user_type"] = "Default"

    return PricingCatalog(core, by_type, cond, strict=True)
