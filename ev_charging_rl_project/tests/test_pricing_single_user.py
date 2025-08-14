"""
Smoke test: run the pricing engine for ONE simulated user to verify the full
pricing flow (resolve unit price -> apply all fees -> compute total).

Usage options:
  # 1) Run as a script:
  #    python tests/test_pricing_single_user.py --data-dir data --agent-id U001 \
  #           --charger-type Rapid --kwh 20 --minutes 45

  # 2) Run with pytest:
  #    pytest -q tests/test_pricing_single_user.py

Assumptions:
- Pricing code lives in utils/pricing.py with:
    - load_pricing_catalog(data_dir) -> PricingCatalog
    - PricingCatalog.estimate_session(...)
- CSVs exist in <data-dir>/:
    pricing_core.csv
    pricing_by_charger_type.csv
    pricing_conditions.csv
    simulated_users.csv
- We take `user_type` and `depart_dt` from the chosen simulated user.
- By default we pick a company that can price either via by-type or via
  peak/off-peak or base price. You can override company_id via CLI.
"""

from __future__ import annotations
import argparse
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd

# --- Import your actual pricing engine ---
try:
    # preferred import if your repo structure matches previous discussion
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.pricing import load_pricing_catalog
except ModuleNotFoundError:
    # fallback if it's packaged differently
    try:
        from pricing import load_pricing_catalog  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Could not import load_pricing_catalog from utils.pricing or pricing.\n"
            "Please adjust the import to match your repository structure."
        ) from e


def _pick_company_id(data_dir: Path) -> int:
    """
    Pick a company_id that is likely to succeed in pricing:
    - Prefer a company with time windows AND a non-null peak/offpeak price
    - Else any company with a non-zero base_price_per_kwh
    """
    core = pd.read_csv(data_dir / "pricing_core.csv")
    cond = pd.read_csv(data_dir / "pricing_conditions.csv")

    core["company_id"] = core["company_id"].astype(int)
    cond["company_id"] = cond["company_id"].astype(int)

    # left join so we can see flags + window prices
    df = core.merge(cond, on="company_id", how="left")

    # 1) time-sensitive with usable window prices
    ts = df[(df.get("time_sensitive_flag", 0) == 1) & (
        (df.get("peak_price").fillna(0) > 0) | (df.get("off_peak_price").fillna(0) > 0)
    )]
    if len(ts):
        return int(ts.iloc[0]["company_id"])

    # 2) fallback to any positive base price
    base_ok = df[df.get("base_price_per_kwh", 0).fillna(0) > 0]
    if len(base_ok):
        return int(base_ok.iloc[0]["company_id"])

    # 3) last resort: just take the first company_id (will likely raise in pricing)
    return int(df.iloc[0]["company_id"])


def _auto_defaults_from_user(user_row: pd.Series):
    """
    Derive reasonable defaults from a simulated user row.
    We only *require* user_type and depart_dt here; kWh/min can be CLI'd.
    """
    # depart_dt is ISO string in the dataset; fall back to 'now' if missing
    dt_str = str(user_row.get("depart_dt", "")).strip()
    try:
        start_dt = datetime.fromisoformat(dt_str)
    except Exception:
        start_dt = datetime.now()

    # user_type should be like "Payg", "Member", etc.
    user_type = str(user_row.get("user_type", "Payg")).strip().title() or "Payg"
    return start_dt, user_type


def pretty(obj):
    if is_dataclass(obj):
        return asdict(obj)
    # support namedtuple / dict-like returns as well
    try:
        return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
    except Exception:
        return obj


def run_single_pricing(
    data_dir: Path,
    agent_id: str | None,
    charger_type: str,
    kwh: float,
    minutes: float,
    include_subscription: bool,
    sessions_per_month: int,
    company_override: int | None,
):
    # Load catalog (this loads the three pricing CSVs under the hood)
    catalog = load_pricing_catalog(str(data_dir))

    # Pick a user
    users = pd.read_csv(data_dir / "simulated_users.csv")
    users["agent_id"] = users["agent_id"].astype(str)

    if agent_id:
        row = users.loc[users["agent_id"] == agent_id]
        if row.empty:
            raise SystemExit(f"agent_id {agent_id} not found in simulated_users.csv")
        user = row.iloc[0]
    else:
        user = users.iloc[0]

    start_dt, user_type = _auto_defaults_from_user(user)

    # Company selection logic (can be overridden)
    company_id = company_override or _pick_company_id(data_dir)

    # Call the pricing engine
    breakdown = catalog.estimate_session(
        company_id=company_id,
        charger_type=str(charger_type).title(),  # normalizes like the engine does
        user_type=str(user_type).title(),
        start_dt=start_dt,
        kwh=float(kwh),
        session_minutes=float(minutes),
        include_subscription=include_subscription,
        sessions_per_month=int(sessions_per_month),
    )

    # Print a friendly, auditable summary
    print("\n=== Pricing smoke test ===")
    print(f"Agent:        {user.get('agent_id')}  ({user_type})")
    print(f"Company ID:   {company_id}")
    print(f"Charger type: {charger_type}")
    print(f"Start time:   {start_dt.isoformat(timespec='seconds')}")
    print(f"Energy:       {kwh:.2f} kWh")
    print(f"Duration:     {minutes:.1f} min")
    print(f"Subscription: {'ON' if include_subscription else 'OFF'} "
          f"(sessions_per_month={sessions_per_month})")

    # The PriceBreakdown dataclass from your engine typically has these fields;
    # but we handle surprises by pretty-printing whatever it returns.
    bd = pretty(breakdown)
    # Try to extract familiar fields if present
    unit_price = bd.get("unit_price", None) if isinstance(bd, dict) else None
    unit_source = bd.get("unit_source", None) if isinstance(bd, dict) else None
    total_cost = bd.get("total_cost", None) if isinstance(bd, dict) else None

    print("\n--- Breakdown ---")
    if unit_price is not None:
        print(f"unit_price:   {unit_price}")
    if unit_source is not None:
        print(f"unit_source:  {unit_source}")

    # Print all top-level line items if the object is dict-like
    if isinstance(bd, dict):
        for k in [
            "energy_cost",
            "connection_fee",
            "infrastructure_fee",
            "idle_fee",
            "overstay_fee",
            "min_charge_applied",
            "subscription_applied",
            "preauth_hold",
            "total_cost",
        ]:
            if k in bd:
                print(f"{k:18s}{bd[k]}")

        # Also print notes if present
        notes = bd.get("notes", None)
        if isinstance(notes, dict) and notes:
            print("\nnotes:")
            for nk, nv in notes.items():
                print(f"  {nk}: {nv}")
    else:
        # fallback dump for unknown return types
        print(bd)

    # A couple of sanity assertions when run under pytest
    assert unit_price is None or unit_price >= 0
    assert total_cost is None or total_cost >= 0

    return breakdown


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data"),
                   help="Directory containing pricing_*.csv and simulated_users.csv")
    p.add_argument("--agent-id", type=str, default=None,
                   help="Agent ID from simulated_users.csv, e.g., U001")
    p.add_argument("--company-id", type=int, default=None,
                   help="Override company_id (otherwise auto-picked)")
    p.add_argument("--charger-type", type=str, default="Rapid",
                   help="Charger type, e.g. Rapid | Fast | Slow")
    p.add_argument("--kwh", type=float, default=20.0,
                   help="Energy delivered (kWh)")
    p.add_argument("--minutes", type=float, default=45.0,
                   help="Total session duration (minutes)")
    p.add_argument("--include-subscription", action="store_true",
                   help="Amortize monthly membership across sessions")
    p.add_argument("--sessions-per-month", type=int, default=12,
                   help="Used only if --include-subscription is set")
    args = p.parse_args()

    run_single_pricing(
        data_dir=args.data_dir,
        agent_id=args.agent_id,
        charger_type=args.charger_type,
        kwh=args.kwh,
        minutes=args.minutes,
        include_subscription=args.include_subscription,
        sessions_per_month=args.sessions_per_month,
        company_override=args.company_id,
    )


# pytest entrypoint
def test_pricing_single_user():
    """pytest will call this with default params against ./data by default."""
    data_dir = Path("data")
    if not (data_dir / "simulated_users.csv").exists():
        # When run in CI without data present, skip gracefully.
        import pytest
        pytest.skip("data/ not present in this environment")
    run_single_pricing(
        data_dir=data_dir,
        agent_id=None,
        charger_type="Rapid",
        kwh=20.0,
        minutes=45.0,
        include_subscription=True,
        sessions_per_month=12,
        company_override=None,
    )


if __name__ == "__main__":
    _cli()
