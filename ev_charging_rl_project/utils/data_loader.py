# utils/data_loader.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any

import pandas as pd

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ------------------------------
# Data container used by rl.train
# ------------------------------
@dataclass
class DataBundle:
    stations: pd.DataFrame
    connectors: pd.DataFrame
    ev_metadata: pd.DataFrame
    charging_curves: pd.DataFrame
    users: Optional[pd.DataFrame]
    pricing_catalog: Any  # constructed in load_pricing_catalog


# ------------------------------
# Helpers
# ------------------------------
def _normalize_strings(s: pd.Series) -> pd.Series:
    """Lowercase + strip; tolerant to missing column."""
    if s is None:
        return pd.Series(dtype="object")
    return s.astype(str).str.strip().str.lower()


def _is_nationwide(region_n: str) -> bool:
    """Heuristic for nationwide / UK-wide region markers."""
    if not isinstance(region_n, str):
        return False
    r = region_n.strip().lower()
    return r in {"uk", "united kingdom", "nationwide", "all", "gb", "uk-wide", "u.k."}


def _ensure_cols(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
    """Ensure the given columns exist; create empty strings if missing."""
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def _ensure_norm_cols(df: pd.DataFrame, pairs: Tuple[Tuple[str, str], ...]) -> pd.DataFrame:
    """
    Make sure normalized columns (like company_name_n) exist.
    If the normalized column is missing, derive from its raw counterpart (or empty string).
    """
    for raw, norm in pairs:
        if norm not in df.columns:
            base = df.get(raw, "")
            df[norm] = _normalize_strings(base)
    return df


def _derive_charger_type(connector_name: str) -> str:
    """Map connector label to AC/DC bucket."""
    if not isinstance(connector_name, str):
        return "AC"
    n = connector_name.strip()
    if any(k in n for k in ["CCS", "CHADEMO", " DC"]):
        return "DC"
    if "dc" in n.lower():
        return "DC"
    return "AC"


def _primary_from_device_networks(series: pd.Series) -> pd.Series:
    """
    deviceNetworks can contain multiple providers separated by ';' or ','.
    Take the first non-empty token as primary.
    """
    s = series.astype(str).fillna("")
    return s.str.split(r"[;,]").str[0].str.strip()


# ------------------------------
# Pricing catalog loader
# ------------------------------
def load_pricing_catalog(data_dir: Path):
    """
    Construct your PricingCatalog from CSVs.
    We import here to avoid circular imports.
    """
    from utils.pricing import PricingCatalog  # your pricing module
    data_dir = Path(data_dir)

    core = pd.read_csv(data_dir / "pricing_core.csv")
    by_type = pd.read_csv(data_dir / "pricing_by_charger_type.csv")
    conditions = pd.read_csv(data_dir / "pricing_conditions.csv")

    return PricingCatalog(core, by_type, conditions)


# ------------------------------
# Station <-> Company mapping + merge with connectors
# ------------------------------
def load_stations_with_connectors(
    stations_csv: str | Path,
    connectors_csv: str | Path,
    pricing_core_csv: str | Path,
) -> pd.DataFrame:
    """
    Return a merged table of stations ⨝ connectors with reliable columns:

      ['station_id','company_id','company_name',
       'charger_type','connector_type','rated_power_kw','lat','lon','edge_id']

    Robust to missing columns and earlier renames. Does NOT drop rows silently.
    Uses deviceNetworks -> company_name mapping and resolves company_id via pricing_core.
    """
    stations = pd.read_csv(stations_csv)
    # Drop unpriceable providers early
    if "deviceNetworks" in stations.columns:
        stations["deviceNetworks"] = stations["deviceNetworks"].astype(str)
        stations = stations[stations["deviceNetworks"].str.strip() != ""].copy()
        stations = stations[~stations["deviceNetworks"].str.lower().isin(["nan", "none", "null"])].copy()

    connectors = pd.read_csv(connectors_csv)
    companies = pd.read_csv(pricing_core_csv).copy()

    # --- Canonicalize stations ---
    if "chargeDeviceID" in stations.columns and "station_id" not in stations.columns:
        stations = stations.rename(columns={"chargeDeviceID": "station_id"})
    if "latitude" in stations.columns and "lat" not in stations.columns:
        stations = stations.rename(columns={"latitude": "lat"})
    if "longitude" in stations.columns and "lon" not in stations.columns:
        stations = stations.rename(columns={"longitude": "lon"})

    # Provider name: map deviceNetworks -> company_name
    if "deviceNetworks" in stations.columns:
        primary = _primary_from_device_networks(stations["deviceNetworks"])
        if "company_name" not in stations.columns:
            stations["company_name"] = primary
        else:
            mask = stations["company_name"].astype(str).str.strip().eq("") | stations["company_name"].isna()
            stations.loc[mask, "company_name"] = primary

    # Ensure matcher fields exist (raw)
    stations = _ensure_cols(
        stations,
        ("company_name", "region", "specific_location", "city", "postcode"),
    )

    # --- Canonicalize connectors ---
    if "chargeDeviceID" in connectors.columns and "station_id" not in connectors.columns:
        connectors = connectors.rename(columns={"chargeDeviceID": "station_id"})
    if "rated_power_kW" in connectors.columns and "rated_power_kw" not in connectors.columns:
        connectors = connectors.rename(columns={"rated_power_kW": "rated_power_kw"})
    if "max_power_kW" in connectors.columns and "rated_power_kw" not in connectors.columns:
        connectors = connectors.rename(columns={"max_power_kW": "rated_power_kw"})
    if "charger_type" not in connectors.columns and "connector_type" in connectors.columns:
        connectors["charger_type"] = connectors["connector_type"].map(_derive_charger_type)

    # --- Canonicalize companies (pricing_core) ---
    companies = _ensure_cols(
        companies,
        ("company_name", "region", "specific_location", "city", "postcode"),
    )
    keep_cols = ["company_id", "company_name", "region", "specific_location", "city", "postcode"]
    have_cols = [c for c in keep_cols if c in companies.columns]
    companies = companies[have_cols].copy()

    # --- Create normalized versions on both sides (safe, idempotent) ---
    norm_pairs = (
        ("company_name", "company_name_n"),
        ("region", "region_n"),
        ("specific_location", "loc_n"),
        ("city", "city_n"),
        ("postcode", "pc_n"),
    )
    stations = _ensure_norm_cols(stations, norm_pairs)
    companies = _ensure_norm_cols(companies, norm_pairs)

    # --- Match company_id for each station row ---
    # (Keep your meticulous logic intact)
    def pick_company_id(row) -> Optional[int]:
        cname = row.get("company_name_n", "")
        if not cname:
            return None

        cands = companies.loc[companies["company_name_n"] == cname]
        if cands.empty:
            return None

        # Try region
        reg = row.get("region_n", "")
        if reg:
            same_region = cands.loc[cands["region_n"] == reg]
            if not same_region.empty:
                cands = same_region
            else:
                nw = cands.loc[cands["region_n"].map(_is_nationwide)]
                if not nw.empty:
                    cands = nw
        else:
            nw = cands.loc[cands["region_n"].map(_is_nationwide)]
            if not nw.empty:
                cands = nw

        # Try specific location
        loc = row.get("loc_n", "")
        if len(cands) > 1 and loc:
            m = cands.loc[cands["loc_n"] == loc]
            if not m.empty:
                cands = m

        # Try postcode
        pc = row.get("pc_n", "")
        if len(cands) > 1 and pc:
            m = cands.loc[cands["pc_n"] == pc]
            if not m.empty:
                cands = m

        # Try city
        city = row.get("city_n", "")
        if len(cands) > 1 and city:
            m = cands.loc[cands["city_n"] == city]
            if not m.empty:
                cands = m

        # Prefer generic entries
        if len(cands) > 1:
            generic = cands.loc[(cands["loc_n"] == "") & (cands["pc_n"] == "")]
            if not generic.empty:
                cands = generic

        cands = cands.sort_values("company_id", kind="stable")
        return int(cands["company_id"].iloc[0]) if not cands.empty else None

    stations["company_id"] = stations.apply(pick_company_id, axis=1)

    # Warn (but do NOT crash) if unresolved
    if stations["company_id"].isna().any():
        col_pairs = [
            ("company_name", "company_name_n"),
            ("region", "region_n"),
            ("specific_location", "loc_n"),
            ("city", "city_n"),
            ("postcode", "pc_n"),
        ]
        display_cols = [raw if raw in stations.columns else norm for (raw, norm) in col_pairs]
        bad = (
            stations.loc[stations["company_id"].isna(), display_cols]
            .fillna("")
            .drop_duplicates()
        )
        log.warning(
            "Unresolved company_id for some stations (showing up to 10):\n%s",
            bad.head(10).to_string(index=False)
        )

    # --- Final merge: station ⨝ connectors (inner to keep only known connectors) ---
    merged = connectors.merge(
        stations,
        on="station_id",
        how="inner",
        suffixes=("", "_st")
    )

    # Ensure required columns exist
    for col in ["lat", "lon"]:
        if col not in merged.columns:
            merged[col] = None
    if "edge_id" not in merged.columns:
        merged["edge_id"] = None

    if "rated_power_kw" not in merged.columns and "rated_power_kW" in merged.columns:
        merged["rated_power_kw"] = merged["rated_power_kW"]

    cols_final = [
        "station_id", "company_id", "company_name",
        "charger_type", "connector_type", "rated_power_kw",
        "lat", "lon", "edge_id"
    ]
    cols_present = [c for c in cols_final if c in merged.columns]
    merged = merged[cols_present].copy()

    return merged


# ------------------------------
# Main loader used by rl.train
# ------------------------------
def load_all_data(data_dir: str | Path = "data") -> DataBundle:
    data_dir = Path(data_dir)

    # Core tables
    stations = pd.read_csv(data_dir / "charging_station_metadata.csv")
    # Drop unpriceable providers early
    if "deviceNetworks" in stations.columns:
        stations["deviceNetworks"] = stations["deviceNetworks"].astype(str)
        stations = stations[stations["deviceNetworks"].str.strip() != ""].copy()
        stations = stations[~stations["deviceNetworks"].str.lower().isin(["nan", "none", "null"])].copy()

    # Canonicalize stations early (id + coords)
    if "chargeDeviceID" in stations.columns and "station_id" not in stations.columns:
        stations = stations.rename(columns={"chargeDeviceID": "station_id"})
    if "latitude" in stations.columns and "lat" not in stations.columns:
        stations = stations.rename(columns={"latitude": "lat"})
    if "longitude" in stations.columns and "lon" not in stations.columns:
        stations = stations.rename(columns={"longitude": "lon"})

    # Map deviceNetworks -> company_name if present
    if "deviceNetworks" in stations.columns:
        primary = _primary_from_device_networks(stations["deviceNetworks"])
        if "company_name" not in stations.columns:
            stations["company_name"] = primary
        else:
            mask = stations["company_name"].astype(str).str.strip().eq("") | stations["company_name"].isna()
            stations.loc[mask, "company_name"] = primary

    # Connectors
    connectors = pd.read_csv(data_dir / "charging_station_connectors.csv")
    if "chargeDeviceID" in connectors.columns and "station_id" not in connectors.columns:
        connectors = connectors.rename(columns={"chargeDeviceID": "station_id"})
    if "rated_power_kW" in connectors.columns and "rated_power_kw" not in connectors.columns:
        connectors = connectors.rename(columns={"rated_power_kW": "rated_power_kw"})
    if "max_power_kW" in connectors.columns and "rated_power_kw" not in connectors.columns:
        connectors = connectors.rename(columns={"max_power_kW": "rated_power_kw"})
    if "charger_type" not in connectors.columns and "connector_type" in connectors.columns:
        connectors["charger_type"] = connectors["connector_type"].map(_derive_charger_type)

    # EV data
    ev_metadata = pd.read_csv(data_dir / "EV_Metadata.csv")
    # Canonicalize EV metadata naming
    rename_map_ev = {
        "battery_kWh": "battery_kwh",
        "avg_consumption_Wh_per_km": "kwh_per_km",  # later divide by 1000 if needed
    }
    for old, new in rename_map_ev.items():
        if old in ev_metadata.columns and new not in ev_metadata.columns:
            ev_metadata[new] = ev_metadata[old]
    # If avg consumption is Wh/km, convert to kWh/km
    if "kwh_per_km" in ev_metadata.columns and ev_metadata["kwh_per_km"].max() > 5:
        ev_metadata["kwh_per_km"] = ev_metadata["kwh_per_km"] / 1000.0

    charging_curves = pd.read_csv(data_dir / "EV_Charging_Curve_Data.csv")

    # Users (optional)
    users_path = data_dir / "simulated_users.csv"
    users = pd.read_csv(users_path) if users_path.exists() else None

    # Pricing catalog
    pricing_catalog = load_pricing_catalog(data_dir)

    # Enrich stations with company_id using pricing_core + connectors
    merged = load_stations_with_connectors(
        data_dir / "charging_station_metadata.csv",
        data_dir / "charging_station_connectors.csv",
        data_dir / "pricing_core.csv",
    )
    # Pull clean station subset with company_id; keep lat/lon/company_name
    base_cols = ["station_id", "company_id", "company_name", "lat", "lon"]
    have = [c for c in base_cols if c in merged.columns]
    stations_w_company = merged[have].drop_duplicates()

    # Left-join back to original stations to preserve extra station columns if present
    join_keys = [c for c in ["station_id", "lat", "lon"]
                 if c in stations.columns and c in stations_w_company.columns]
    if join_keys:
        swc = stations_w_company[join_keys + ["company_id"] + (
            ["company_name"] if "company_name" not in stations.columns else []
        )]
        stations = stations.merge(swc, on=join_keys, how="left", suffixes=("", "_swc"))
        if "company_name_swc" in stations.columns:
            stations["company_name"] = stations["company_name"].where(
                stations["company_name"].notna() & (stations["company_name"].astype(str).str.strip() != ""),
                stations["company_name_swc"]
            )
            stations.drop(columns=["company_name_swc"], inplace=True)
        if "company_id" not in stations.columns:
            stations["company_id"] = None
    else:
        if "company_id" not in stations.columns:
            stations["company_id"] = None

    # --- Final schema guards (fail-fast, but informative) ---
    def _must_have(df: pd.DataFrame, cols: Tuple[str, ...], name: str):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} missing columns: {missing}")

    _must_have(stations, ("station_id", "lat", "lon", "company_id"), "stations")
    _must_have(connectors, ("station_id", "connector_type", "charger_type", "rated_power_kw"), "connectors")
    _must_have(ev_metadata, ("model", "battery_kwh", "kwh_per_km"), "EV_Metadata")
    _must_have(charging_curves, ("model",), "EV_Charging_Curve_Data")

    return DataBundle(
        stations=stations,
        connectors=connectors,
        ev_metadata=ev_metadata,
        charging_curves=charging_curves,
        users=users,
        pricing_catalog=pricing_catalog,
    )
