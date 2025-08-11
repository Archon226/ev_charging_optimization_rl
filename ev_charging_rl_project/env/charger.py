# env/charger.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import math
import sys
import os

# allow "utils" imports when running files directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pricing import PricingCatalog  # new ground-up pricing engine


@dataclass
class ChargerInfo:
    """Static metadata for a physical charger (connector group)."""
    station_id: str
    company_id: int
    operator_name: str
    charger_type: str            # e.g., "Fast" | "Rapid" | "Ultra"
    rated_power_kw: float
    connector_type: Optional[str] = None
    edge_id: Optional[str] = None


class Charger:
    """
    Session-cost interface for a charger.

    Notes:
    - This class does NOT compute delivered energy or duration. Feed it (kWh, minutes)
      from charging curves.
    - Availability hooks are here for SUMO/TraCI coordination.
    """

    def __init__(self, info: ChargerInfo, pricing_catalog: PricingCatalog):
        self.info = info
        self.pricing = pricing_catalog
        self._available = True

    # -------- availability (for SUMO integration) --------
    def is_available(self) -> bool:
        return self._available

    def update_status(self, available: bool) -> None:
        self._available = available

    # -------- pricing --------
    def estimate_session_cost(
        self,
        kwh: float,
        start_dt: datetime,
        user_type: str,
        session_minutes: float,
        include_subscription: bool = False,
        sessions_per_month: int = 20,
    ) -> Dict:
        """
        Compute a transparent cost breakdown for one charging session.

        Parameters
        ----------
        kwh : float
            Energy delivered during the session (from charging curves).
        start_dt : datetime
            Session start time (affects peak/off-peak/weekend pricing).
        user_type : str
            "Subscriber" | "Payg" | "Contactless"
        session_minutes : float
            Duration of the session (used for idle/overstay fee logic).
        include_subscription : bool
            If True and operator requires subscription, adds a naive amortised per‑session fee.
        sessions_per_month : int
            Divisor for subscription amortisation when include_subscription=True.

        Returns
        -------
        dict with keys compatible with existing simulator logs:
          station_id, operator, company_id, charger_type, connector_type, rated_power_kw,
          user_type, kwh, unit_price, unit_source, price_source, energy_cost,
          connection_fee, infrastructure_fee, idle_fee, overstay_fee,
          subscription_cost, min_charge_topup, preauth_hold, total_cost,
          start_dt, session_minutes
        """
        # Guard: company_id must be valid
        if self.info.company_id is None or (isinstance(self.info.company_id, float) and math.isnan(self.info.company_id)):
            # Return a no-price breakdown to avoid crashes; caller can decide how to handle
            return {
                "station_id": self.info.station_id,
                "operator": self.info.operator_name,
                "company_id": None,
                "charger_type": self.info.charger_type,
                "connector_type": self.info.connector_type,
                "rated_power_kw": self.info.rated_power_kw,
                "user_type": user_type,
                "kwh": float(kwh),
                "unit_price": 0.0,
                "unit_source": "unpriced_no_company_id",
                "price_source": "unpriced_no_company_id",
                "energy_cost": 0.0,
                "connection_fee": 0.0,
                "infrastructure_fee": 0.0,
                "idle_fee": 0.0,
                "overstay_fee": 0.0,
                "subscription_cost": 0.0,
                "min_charge_topup": 0.0,
                "preauth_hold": 0.0,
                "total_cost": float("nan"),
                "start_dt": start_dt.isoformat(),
                "session_minutes": float(session_minutes) if session_minutes is not None else None,
                "edge_id": self.info.edge_id

            }

        bk = self.pricing.estimate_session(
            company_id=int(self.info.company_id),
            charger_type=self.info.charger_type,
            user_type=user_type,
            start_dt=start_dt,
            kwh=float(kwh),
            session_minutes=float(session_minutes),
            include_subscription=include_subscription,
            sessions_per_month=sessions_per_month,
        )

        return {
            "station_id": self.info.station_id,
            "operator": self.info.operator_name,
            "company_id": int(self.info.company_id),
            "charger_type": self.info.charger_type,
            "connector_type": self.info.connector_type,
            "rated_power_kw": float(self.info.rated_power_kw),
            "user_type": user_type,
            "kwh": float(bk.notes.get("kwh", kwh)),
            "unit_price": float(bk.unit_price),
            "unit_source": bk.unit_source,          # new name
            "price_source": bk.unit_source,         # backward-compat mirror
            "energy_cost": float(bk.energy_cost),
            "connection_fee": float(bk.connection_fee),
            "infrastructure_fee": float(bk.infrastructure_fee),
            "idle_fee": float(bk.idle_fee),
            "overstay_fee": float(bk.overstay_fee),
            "subscription_cost": float(bk.subscription_applied),
            "min_charge_topup": float(bk.min_charge_applied),
            "preauth_hold": float(bk.preauth_hold),  # not charged; info only
            "total_cost": float(bk.total_cost),
            "start_dt": start_dt.isoformat(),
            "session_minutes": float(bk.notes.get("minutes", session_minutes)),
            "edge_id": self.info.edge_id

        }


# -------- convenience factory --------
def charger_from_row(row, pricing_catalog: PricingCatalog) -> Charger:
    """
    Build a Charger from a row of datasets.stations_merged
    (see utils.data_loader.load_all_data().stations_merged).
    """
    # NaN-safe company_id conversion
    cid = row.get("company_id")
    company_id = None
    try:
        if cid is not None:
            fcid = float(cid)
            if not math.isnan(fcid):
                company_id = int(fcid)
    except Exception:
        company_id = int(cid)

    info = ChargerInfo(
        station_id=str(row.get("station_id")),
        company_id=company_id if company_id is not None else -1,
        operator_name=(row.get("operator_name") or row.get("company_name") or "")[:255],
        charger_type=str(row.get("charger_type", "Rapid")),
        rated_power_kw=float(row.get("rated_power_kw", 50.0)),
        connector_type=row.get("connector_type"),
        edge_id=row.get("edge_id"),
    )
    return Charger(info, pricing_catalog)
