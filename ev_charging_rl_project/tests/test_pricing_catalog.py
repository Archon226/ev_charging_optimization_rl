# tests/test_pricing_catalog.py
import pandas as pd
from datetime import datetime, date
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import PricingCatalog

def _mk_catalog():
    core = pd.DataFrame([
        {"company_id": 1, "base_price_per_kwh": 0.55, "time_sensitive_flag": 1, "day_sensitive_flag": 1,
         "connection_fee": 0.30, "subscription_required": "No", "subscription_fee_per_month": 0},
        {"company_id": 2, "base_price_per_kwh": 0.45, "time_sensitive_flag": 0, "day_sensitive_flag": 0,
         "connection_fee": 0.00, "subscription_required": "Yes", "subscription_fee_per_month": 12.99},
    ])
    by_type = pd.DataFrame([
        # by-type override for PAYG Rapid
        {"company_id": 1, "charger_type": "Rapid", "user_type": "Payg", "price_per_kwh": 0.60, "exists_flag": 1},
        # exists but NaN price should be ignored
        {"company_id": 1, "charger_type": "Ultra", "user_type": "Subscriber", "price_per_kwh": None, "exists_flag": 1},
    ])
    cond = pd.DataFrame([
        {"company_id": 1, "peak_hours": "17:00-20:00", "off_peak_hours": "22:00-06:00",
         "peak_price": 0.70, "off_peak_price": 0.40,
         "weekend_price": 0.50,
         "infrastructure_fee": 0.05,
         "idle_fee_flag": 1, "max_overstay_fee": 10.0, "overstay_fee_start_min": 30.0,
         "pre_auth_fee_flag": 0, "pre_auth_avg_fee": 0},
        {"company_id": 2, "peak_hours": None, "off_peak_hours": None, "peak_price": None, "off_peak_price": None,
         "weekend_price": None,
         "infrastructure_fee": 0.00}
    ])
    return PricingCatalog(core, by_type, cond)

def test_resolution_by_type_overrides_windows():
    cat = _mk_catalog()
    # Rapid + PAYG should use by_type regardless of peak window
    price, src = cat.get_unit_price(1, "Rapid", "Payg", datetime(2025, 8, 9, 18, 0))
    assert round(price, 2) == 0.60
    assert src == "by_type"

def test_resolution_peak_window():
    cat = _mk_catalog()
    # Not Rapid → by_type doesn't match; should fall to peak window at 18:00
    price, src = cat.get_unit_price(1, "Fast", "Payg", datetime(2025, 8, 9, 18, 0))
    assert round(price, 2) == 0.70
    assert src == "peak_window"

def test_resolution_offpeak_midnight_wrap():
    cat = _mk_catalog()
    # Off-peak wraps: 22:00-06:00 → 02:00 is off-peak
    price, src = cat.get_unit_price(1, "Fast", "Payg", datetime(2025, 8, 10, 2, 0))
    assert round(price, 2) == 0.40
    assert src == "offpeak_window"

def test_resolution_weekend_only_when_no_time_window_match():
    cat = _mk_catalog()
    # Saturday at 12:00 is not in peak/offpeak; weekend price applies
    price, src = cat.get_unit_price(1, "Fast", "Payg", datetime(2025, 8, 9, 12, 0))  # Sat
    assert round(price, 2) == 0.50
    assert src == "weekend"

def test_base_price_fallback():
    cat = _mk_catalog()
    price, src = cat.get_unit_price(2, "Rapid", "Payg", datetime(2025, 8, 9, 12, 0))
    assert round(price, 2) == 0.45
    assert src == "base"

def test_fees_returned_safely():
    cat = _mk_catalog()
    fees = cat.fees_for_company(1)
    assert fees.connection_fee == 0.30
    assert fees.infrastructure_fee == 0.05
    assert fees.idle_fee_flag == 1
    assert fees.max_overstay_fee == 10.0
    assert fees.overstay_fee_start_min == 30.0
