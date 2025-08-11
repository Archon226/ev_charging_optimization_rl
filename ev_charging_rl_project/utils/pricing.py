# utils/pricing.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import math
import numpy as np
import pandas as pd


# -----------------------
# Normalisers / helpers
# -----------------------
def _norm_user_type(x: str) -> str:
    return (x or "").strip().title()

def _norm_charger_type(x: str) -> str:
    return (x or "").strip().title()


def _to_bool(x) -> bool:
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y")
    return bool(x)

def _nz(x, default=0.0) -> float:
    """Numeric zero if None/NaN/empty."""
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


# -----------------------
# Dataclasses
# -----------------------
@dataclass(frozen=True)
class FeePolicy:
    connection_fee: float = 0.0
    infrastructure_fee: float = 0.0
    minimum_charge_amount: float = 0.0
    idle_fee_flag: int = 0
    idle_fee_per_min: float = 0.0         # OPTIONAL COLUMN (if present)
    idle_fee_start_min: float = 0.0       # OPTIONAL COLUMN (if present)
    max_overstay_fee: float = 0.0
    overstay_fee_start_min: float = 0.0
    overstay_fee_per_min: float = 0.0     # OPTIONAL COLUMN (if present)
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


# -----------------------
# Pricing catalog
# -----------------------
class PricingCatalog:
    """
    Data-backed tariff engine that combines:
      - pricing_core.csv
      - pricing_by_charger_type.csv
      - pricing_conditions.csv

    Unit price cascade (deterministic):
      1) by_type(company_id, charger_type, user_type) if exists_flag and price
      2) by_type(company_id, charger_type, ANY user_type) if unique price
      3) time windows (peak/off-peak) if enabled
      4) weekend price if enabled
      5) base_price_per_kwh from core

    Fees:
      - connection_fee (core)
      - infrastructure_fee (conditions)
      - minimum_charge_amount (conditions) applied to final total
      - idle/overstay (per-minute if columns exist; capped by max_overstay_fee)
      - pre_auth_avg_fee reported as HOLD if flag set (not added to total)
      - optional subscription amortisation per-session if requested
    """
    def __init__(self, core: pd.DataFrame, by_type: pd.DataFrame, cond: pd.DataFrame):
        # Raw copies
        self.core = core.copy()
        self.by_type = by_type.copy()
        self.cond = cond.copy()

        # Types
        for f in ("company_id",):
            for df in (self.core, self.by_type, self.cond):
                if f in df.columns:
                    df[f] = pd.to_numeric(df[f], errors="raise").astype(int)

        if "charger_type" in self.by_type.columns:
            self.by_type["charger_type"] = self.by_type["charger_type"].map(_norm_charger_type)
        if "user_type" in self.by_type.columns:
            self.by_type["user_type"] = self.by_type["user_type"].map(_norm_user_type)

        # Indexes
        self.core_idx = self.core.set_index("company_id")
        self.by_type_idx = None
        if {"company_id", "charger_type", "user_type"}.issubset(self.by_type.columns):
            self.by_type_idx = self.by_type.set_index(["company_id", "charger_type", "user_type"])
        self.cond_idx = self.cond.set_index("company_id")

    # -------- core accessors --------
    def _cond_row(self, company_id: int) -> Optional[pd.Series]:
        if company_id in self.cond_idx.index:
            row = self.cond_idx.loc[company_id]
            return row.iloc[0] if isinstance(row, pd.DataFrame) else row
        return None

    @staticmethod
    def _parse_windows(txt: Optional[str]):
        if not isinstance(txt, str) or "-" not in txt:
            return None
        def to_min(s: str) -> int:
            h, m = s.split(":"); return int(h) * 60 + int(m)
        wins = []
        for chunk in txt.split(";"):
            chunk = chunk.strip()
            if "-" in chunk:
                a, b = [x.strip() for x in chunk.split("-")]
                wins.append((to_min(a), to_min(b)))
        return tuple(wins) if wins else None

    @staticmethod
    def _in_window(mins: int, win) -> bool:
        s, e = win
        return s <= mins < e if s <= e else (mins >= s or mins < e)

    def _in_any_window(self, mins: int, wins) -> bool:
        return any(self._in_window(mins, w) for w in (wins or ()))

    # -------- unit price resolution --------
    def resolve_unit_price(self, company_id: int, charger_type: str, user_type: str, dt: datetime) -> Tuple[float, str]:
        user_type = _norm_user_type(user_type)
        charger_type = _norm_charger_type(charger_type)

        core = self.core_idx.loc[company_id]
        cond = self._cond_row(company_id)

        # 1) exact by_type
        if self.by_type_idx is not None:
            key = (company_id, charger_type, user_type)
            if key in self.by_type_idx.index:
                row = self.by_type_idx.loc[key]
                exists = row["exists_flag"] if "exists_flag" in row else True
                price = row.get("price_per_kwh")
                if _to_bool(exists) and pd.notna(price):
                    return float(price), "by_type"

            # 2) by_type ignoring user_type (unique price among user types)
            bt = self.by_type[(self.by_type["company_id"] == company_id) &
                              (self.by_type["charger_type"] == charger_type)]
            bt = bt[pd.notna(bt["price_per_kwh"])]
            uniq_prices = bt["price_per_kwh"].dropna().unique()
            if len(uniq_prices) == 1:
                return float(uniq_prices[0]), "by_type_any_user"

        # 3) time windows (peak/off-peak) if enabled
        if _to_bool(core.get("time_sensitive_flag", 0)) and cond is not None:
            minutes = dt.hour * 60 + dt.minute
            peak_wins = self._parse_windows(cond.get("peak_hours"))
            off_wins  = self._parse_windows(cond.get("off_peak_hours"))
            if self._in_any_window(minutes, peak_wins) and pd.notna(cond.get("peak_price")):
                return float(cond["peak_price"]), "peak_window"
            if self._in_any_window(minutes, off_wins) and pd.notna(cond.get("off_peak_price")):
                return float(cond["off_peak_price"]), "offpeak_window"

        # 4) weekend price if enabled
        if _to_bool(core.get("day_sensitive_flag", 0)) and cond is not None:
            if dt.weekday() >= 5 and pd.notna(cond.get("weekend_price")):
                return float(cond["weekend_price"]), "weekend"

        # 5) base
        base = core.get("base_price_per_kwh")
        if pd.notna(base):
            return float(base), "base"

        raise ValueError(f"No resolvable unit price for company_id={company_id}")

    # -------- fee policy --------
    def fees_for_company(self, company_id: int) -> FeePolicy:
        core = self.core_idx.loc[company_id]
        cond = self._cond_row(company_id)
        get = (cond.get if cond is not None else (lambda *_args, **_kw: None))

        return FeePolicy(
            connection_fee=_nz(core.get("connection_fee", 0)),
            infrastructure_fee=_nz(get("infrastructure_fee", 0)),
            minimum_charge_amount=_nz(get("minimum_charge_amount", 0)),
            idle_fee_flag=int(_nz(get("idle_fee_flag", 0))),
            idle_fee_per_min=_nz(get("idle_fee_per_min", 0)),                 # if column exists
            idle_fee_start_min=_nz(get("idle_fee_start_min", 0)),             # if column exists
            max_overstay_fee=_nz(get("max_overstay_fee", 0)),
            overstay_fee_start_min=_nz(get("overstay_fee_start_min", 0)),
            overstay_fee_per_min=_nz(get("overstay_fee_per_min", 0)),         # if column exists
            pre_auth_fee_flag=int(_nz(get("pre_auth_fee_flag", 0))),
            pre_auth_avg_fee=_nz(get("pre_auth_avg_fee", 0)),
            subscription_required=_to_bool(core.get("subscription_required", False)),
            subscription_fee_per_month=_nz(core.get("subscription_fee_per_month", 0)),
        )

    # -------- full session estimator --------
    def estimate_session(
        self,
        company_id: int,
        charger_type: str,
        user_type: str,
        start_dt: datetime,
        kwh: float,
        session_minutes: float,
        include_subscription: bool = False,
        sessions_per_month: int = 20,    # amortisation divisor if include_subscription=True
    ) -> PriceBreakdown:

        # Resolve unit price
        unit_price, src = self.resolve_unit_price(company_id, charger_type, user_type, start_dt)

        # Fees / policy
        fees = self.fees_for_company(company_id)

        # Energy
        energy_cost = round(_nz(unit_price) * _nz(kwh), 2)

        # Fixed fees
        conn = _nz(fees.connection_fee)
        infra = _nz(fees.infrastructure_fee)

        # Idle & overstay (per-minute if rates exist; both capped by max_overstay_fee where relevant)
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

        # Subscription amortisation (optional)
        sub_applied = 0.0
        if include_subscription and fees.subscription_required and _nz(fees.subscription_fee_per_month) > 0 and sessions_per_month > 0:
            sub_applied = round(_nz(fees.subscription_fee_per_month) / float(sessions_per_month), 2)

        # Pre-auth HOLD (not added to total)
        preauth_hold = _nz(fees.pre_auth_avg_fee) if _nz(fees.pre_auth_fee_flag) else 0.0

        # Subtotal
        subtotal = energy_cost + conn + infra + idle_fee + overstay_fee + sub_applied

        # Minimum charge
        min_charge = _nz(fees.minimum_charge_amount)
        min_applied = float(max(0.0, min_charge - subtotal))
        total = round(max(subtotal, min_charge), 2)

        return PriceBreakdown(
            unit_price=float(unit_price),
            unit_source=src,
            energy_cost=round(energy_cost, 2),
            connection_fee=round(conn, 2),
            infrastructure_fee=round(infra, 2),
            overstay_fee=round(overstay_fee, 2),
            idle_fee=round(idle_fee, 2),
            min_charge_applied=round(min_applied, 2),
            subscription_applied=round(sub_applied, 2),
            preauth_hold=round(preauth_hold, 2),
            total_cost=total,
            notes={
                "charger_type": charger_type,
                "user_type": user_type,
                "kwh": round(_nz(kwh), 3),
                "minutes": round(mins, 2),
            },
        )


# Convenience loader
def load_pricing_catalog(data_dir) -> PricingCatalog:
    from pathlib import Path
    data_dir = Path(data_dir)
    core = pd.read_csv(data_dir / "pricing_core.csv")
    by_type = pd.read_csv(data_dir / "pricing_by_charger_type.csv")
    cond = pd.read_csv(data_dir / "pricing_conditions.csv")
    return PricingCatalog(core, by_type, cond)
