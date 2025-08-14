import pandas as pd
import logging
from typing import Optional

log = logging.getLogger(__name__)

# --------------------
# Company ID helpers
# --------------------

def _is_nationwide(x: str) -> bool:
    """True if region value represents 'nationwide'."""
    return x in (
        "nationwide", "uk-wide", "uk wide", "ukwide",
        "all uk", "all-uk", "alluk", ""
    )

def _normalize_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Strip, lowercase, and fill NA for given string columns."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip().str.lower()
    return df

# --------------------
# Main loader
# --------------------

def load_stations_with_connectors(
    stations_path: str,
    connectors_path: str,
    pricing_companies_path: str
) -> pd.DataFrame:
    """
    Load station metadata and connectors, attach company_id from pricing companies,
    and return clean merged DataFrame with fixed schema.
    """
    # Read core files
    stations = pd.read_csv(stations_path)
    connectors = pd.read_csv(connectors_path)
    companies = pd.read_csv(pricing_companies_path)

    # Normalize all relevant string columns for matching
    str_cols = ["company_name", "region", "specific_location", "city", "postcode"]
    stations = _normalize_strings(stations, str_cols)
    companies = _normalize_strings(companies, str_cols)

    # Normalized aliases for internal match columns
    rename_map = {
        "latitude": "lat",
        "longitude": "lon",
        "specific_location": "loc_n",
        "region": "region_n",
        "city": "city_n",
        "postcode": "pc_n",
        "company_name": "company_name_n"
    }
    stations.rename(columns=rename_map, inplace=True, errors="ignore")
    companies.rename(columns=rename_map, inplace=True, errors="ignore")

    # Ensure numeric company_id in pricing companies
    if "company_id" in companies.columns:
        companies["company_id"] = pd.to_numeric(
            companies["company_id"], errors="coerce"
        ).astype("Int64")

    stations["company_id"] = pd.NA

    # --------------------
    # Resolve company_id for each station
    # --------------------
    def pick_company_id(row) -> Optional[int]:
        cands = companies[companies["company_name_n"] == row["company_name_n"]]
        if cands.empty:
            return None

        # Region preference
        if row["region_n"]:
            same_region = cands[cands["region_n"] == row["region_n"]]
            if not same_region.empty:
                cands = same_region
            else:
                nw = cands[cands["region_n"].map(_is_nationwide)]
                if not nw.empty:
                    cands = nw
        else:
            nw = cands[cands["region_n"].map(_is_nationwide)]
            if not nw.empty:
                cands = nw

        # Specific location
        if len(cands) > 1 and row["loc_n"]:
            m = cands[cands["loc_n"] == row["loc_n"]]
            if not m.empty:
                cands = m

        # Postcode
        if len(cands) > 1 and row["pc_n"]:
            m = cands[cands["pc_n"] == row["pc_n"]]
            if not m.empty:
                cands = m

        # City
        if len(cands) > 1 and row["city_n"]:
            m = cands[cands["city_n"] == row["city_n"]]
            if not m.empty:
                cands = m

        # Prefer generic (empty loc and postcode)
        if len(cands) > 1:
            generic = cands[(cands["loc_n"] == "") & (cands["pc_n"] == "")]
            if not generic.empty:
                cands = generic

        # Deterministic pick
        cands = cands.sort_values("company_id")
        return int(cands["company_id"].iloc[0]) if not cands.empty else None

    stations["company_id"] = stations.apply(pick_company_id, axis=1)

    # Warn if any unresolved
    if stations["company_id"].isna().any():
        cols = ["company_name", "region", "specific_location", "city", "postcode"]
        bad = stations.loc[stations["company_id"].isna(), cols].drop_duplicates()
        log.warning(
            "Unresolved company_id for some stations (up to 10):\n%s",
            bad.head(10).to_string(index=False)
        )

    # Operator alias for backward compatibility
    stations["operator_name"] = stations.get("company_name", stations["company_name_n"])

    # Merge with connectors
    merged = connectors.merge(stations, on="station_id", how="left")
    if "edge_id" not in merged.columns:
        merged["edge_id"] = None

    # Ensure output schema is consistent
    cols = [
        "station_id", "company_id", "operator_name", "company_name",
        "charger_type", "connector_type", "rated_power_kw",
        "lat", "lon", "edge_id"
    ]
    for c in cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    return merged[cols].copy()

# --- add below existing imports/helpers in utils/data_loader.py ---
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from typing import Optional
from utils.pricing import load_pricing_catalog

@dataclass
class DataBundle:
    stations: pd.DataFrame
    connectors: pd.DataFrame
    ev_metadata: pd.DataFrame
    charging_curves: pd.DataFrame
    users: Optional[pd.DataFrame]
    pricing_catalog: any

def load_all_data(data_dir: str | Path = "data") -> DataBundle:
    """
    Minimal unified loader used by rl/train.py.
    Reads core CSVs and builds a PricingCatalog.
    """
    data_dir = Path(data_dir)

    # Core tables
    stations = pd.read_csv(data_dir / "charging_station_metadata.csv")
    if "latitude" in stations.columns and "lat" not in stations.columns:
        stations = stations.rename(columns={"latitude": "lat", "longitude": "lon"})
    if "chargeDeviceID" in stations.columns and "station_id" not in stations.columns:
        stations = stations.rename(columns={"chargeDeviceID": "station_id"})

    connectors = pd.read_csv(data_dir / "charging_station_connectors.csv")
    if "chargeDeviceID" in connectors.columns and "station_id" not in connectors.columns:
        connectors = connectors.rename(columns={"chargeDeviceID": "station_id"})
    if "rated_power_kW" in connectors.columns and "rated_power_kw" not in connectors.columns:
        connectors = connectors.rename(columns={"rated_power_kW": "rated_power_kw"})

    ev_metadata = pd.read_csv(data_dir / "EV_Metadata.csv")
    charging_curves = pd.read_csv(data_dir / "EV_Charging_Curve_Data.csv")
    users = pd.read_csv(data_dir / "simulated_users.csv") if (data_dir / "simulated_users.csv").exists() else None

    pricing_catalog = load_pricing_catalog(data_dir)

    return DataBundle(
        stations=stations,
        connectors=connectors,
        ev_metadata=ev_metadata,
        charging_curves=charging_curves,
        users=users,
        pricing_catalog=pricing_catalog,
    )
