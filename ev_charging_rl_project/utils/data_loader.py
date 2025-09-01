# utils/data_loader.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple
import os, sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from utils.charging_curves import EVPowerModel

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# ============================================================
# Existing container kept for backward compatibility
# ============================================================
@dataclass
class DataBundle:
    stations: pd.DataFrame
    connectors: pd.DataFrame
    ev_metadata: pd.DataFrame
    charging_curves: pd.DataFrame
    users: Optional[pd.DataFrame]
    pricing_catalog: Any  # constructed in load_pricing_catalog


# ============================================================
# New ready-to-train bundle (pre-indexed, fast lookups)
# ============================================================
@dataclass
class StationCaps:
    station_id: str
    company_id: str
    company_name: str
    categories_available: Set[str]
    power_stats_by_category: Dict[str, float]  # e.g. {"Fast": 22.0, "Rapid": 50.0, "Ultra": 150.0}


@dataclass
class EVCaps:
    ev_id: str  # "brand|model|year"
    ac_max_power_kW: Optional[float]
    dc_max_power_kW: Optional[float]
    ac_phases: Optional[int]
    categories_supported: Set[str]


TrainBundle = Dict[str, Any]  # typed at the return site for clarity


# ============================================================
# Config defaults
# ============================================================
DEFAULT_THRESHOLDS = {
    "AC_FAST_MAX": 22,     # AC treated as Fast by default
    "DC_FAST_MAX": 49,     # <50kW
    "DC_RAPID_MAX": 99,    # 50-99kW Rapid, >=100kW Ultra
}
DEFAULT_STATUS_ALLOWLIST = {"In service", "Operational", "Available", "Working", "In Service"}


# ============================================================
# Helpers
# ============================================================
def _normalize_strings(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="object")
    return s.astype(str).str.strip().str.lower()


def _is_nationwide(region_n: str) -> bool:
    if not isinstance(region_n, str):
        return False
    r = region_n.strip().lower()
    return r in {"uk", "united kingdom", "nationwide", "all", "gb", "uk-wide", "u.k."}


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df


def _ensure_norm_cols(df: pd.DataFrame, pairs: Iterable[Tuple[str, str]]) -> pd.DataFrame:
    for raw, norm in pairs:
        if norm not in df.columns:
            base = df.get(raw, "")
            df[norm] = _normalize_strings(base)
    return df


def _primary_from_device_networks(series: pd.Series) -> pd.Series:
    """
    deviceNetworks can contain multiple providers separated by ';' or ','.
    Take the first non-empty token as primary (keeps your intent).
    """
    s = series.astype(str).fillna("")
    return s.str.split(r"[;,]").str[0].str.strip()


def _derive_charge_method_from_connector_name(name: str) -> str:
    """Fallback: map connector label to AC/DC if 'charge_method' not given."""
    if not isinstance(name, str):
        return "AC"
    n = name.strip()
    if any(k in n for k in ["CCS", "CHADEMO", " DC"]):
        return "DC"
    if "dc" in n.lower():
        return "DC"
    return "AC"


def map_category(charge_method: str, rated_power_kw: float, thresholds: Dict[str, float]) -> str:
    """
    Single source of truth for Fast/Rapid/Ultra mapping.
    Default policy (aligned with our design):
      - AC: treat as Fast (<= AC_FAST_MAX). (No 'Rapid AC' by default)
      - DC: < DC_FAST_MAX -> Fast; <= DC_RAPID_MAX -> Rapid; else Ultra
    """
    cm = (charge_method or "AC").strip().upper()
    p = float(rated_power_kw or 0.0)

    if cm == "AC":
        return "Fast"
    if p <= thresholds["DC_FAST_MAX"]:
        return "Fast"
    if p <= thresholds["DC_RAPID_MAX"]:
        return "Rapid"
    return "Ultra"


def _normalize_pricing_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize pricing_by_charger_type.csv (strict schema):
      - company_id (int)
      - charger_type ∈ {Fast, Rapid, Ultra}
      - user_type ∈ {Payg, Member, Default}
      - price_per_kwh (float, £/kWh)
      - exists_flag (bool)
    """
    df = df.copy()

    # 0) Clean header names (strip + remove NBSP / zero-width)
    def _clean_header(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = s.replace("\u00A0", " ").replace("\u200B", "")  # NBSP, zero-width
        return s.strip()

    df.columns = [_clean_header(c) for c in df.columns]

    # Strict schema: require official names only
    required = ["company_id", "charger_type", "user_type", "price_per_kwh", "exists_flag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"pricing_by_charger_type.csv missing required columns: {missing}. "
            f"Got {list(df.columns)}"
        )

    # company_id
    df["company_id"] = pd.to_numeric(df["company_id"], errors="raise").astype(int)

    # charger_type values (normalize values only, not header)
    def _norm_ct(x: str) -> str:
        s = (str(x) or "").strip().title()
        return s if s in {"Fast", "Rapid", "Ultra"} else "Fast"
    df["charger_type"] = df["charger_type"].map(_norm_ct)

    # user_type normalization
    map_user = {
        "Subscriber": "Member",
        "Contactless": "Payg",
        "Payg": "Payg",
        "Member": "Member"
    }
    df["user_type"] = df["user_type"].astype(str).map(map_user).fillna("Default")

    # price_per_kwh
    df["price_per_kwh"] = pd.to_numeric(df["price_per_kwh"], errors="coerce")

    # exists_flag
    df["exists_flag"] = df["exists_flag"].astype(bool)

    # Final strict schema
    keep = ["company_id", "charger_type", "user_type", "price_per_kwh", "exists_flag"]
    df = df[keep].reset_index(drop=True)

    return df


# ============================================================
# Pricing catalog loader (existing dependency)
# ============================================================
def load_pricing_catalog(data_dir: Path):
    """
    Construct your PricingCatalog from CSVs.
    We import here to avoid circular imports.
    """
    from utils.pricing import PricingCatalog  # your pricing module
    data_dir = Path(data_dir)

    core = pd.read_csv(data_dir / "pricing_core.csv")
    by_type = pd.read_csv(data_dir / "pricing_by_charger_type.csv")
    by_type = _normalize_pricing_by_type(by_type)
    conditions = pd.read_csv(data_dir / "pricing_conditions.csv")

    return PricingCatalog(core, by_type, conditions)


# ============================================================
# Company mapping using deviceNetworks (keeps your concrete logic)
# ============================================================
def _match_company_ids_on_stations(stations: pd.DataFrame, pricing_core_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map stations.deviceNetworks/company_name -> company_id using pricing_core.
    Keeps your resolution strategy but makes it explicit and testable.
    """
    stations = stations.copy()
    companies = pricing_core_df.copy()

    stations = _ensure_cols(stations, ("company_name", "region", "specific_location", "city", "postcode"))
    companies = _ensure_cols(companies, ("company_name", "region", "specific_location", "city", "postcode"))

    norm_pairs = (
        ("company_name", "company_name_n"),
        ("region", "region_n"),
        ("specific_location", "loc_n"),
        ("city", "city_n"),
        ("postcode", "pc_n"),
    )
    stations = _ensure_norm_cols(stations, norm_pairs)
    companies = _ensure_norm_cols(companies, norm_pairs)

    keep_cols = ["company_id", "company_name", "company_name_n", "region_n", "loc_n", "city_n", "pc_n"]
    companies = companies[[c for c in keep_cols if c in companies.columns]].copy()

    def pick_company_id(row) -> Optional[str]:
        cname = row.get("company_name_n", "")
        if not cname:
            return None

        cands = companies.loc[companies["company_name_n"] == cname]
        if cands.empty:
            return None

        # Try region match (or nationwide)
        reg = row.get("region_n", "")
        if reg:
            m = cands.loc[cands["region_n"] == reg]
            if not m.empty:
                cands = m
            else:
                m = cands.loc[cands["region_n"].map(_is_nationwide)]
                if not m.empty:
                    cands = m
        else:
            m = cands.loc[cands["region_n"].map(_is_nationwide)]
            if not m.empty:
                cands = m

        # Specific location
        if len(cands) > 1:
            loc = row.get("loc_n", "")
            if loc:
                m = cands.loc[cands["loc_n"] == loc]
                if not m.empty:
                    cands = m

        # Postcode
        if len(cands) > 1:
            pc = row.get("pc_n", "")
            if pc:
                m = cands.loc[cands["pc_n"] == pc]
                if not m.empty:
                    cands = m

        # City
        if len(cands) > 1:
            city = row.get("city_n", "")
            if city:
                m = cands.loc[cands["city_n"] == city]
                if not m.empty:
                    cands = m

        # Prefer generic entries if still ambiguous
        if len(cands) > 1 and {"loc_n", "pc_n"}.issubset(cands.columns):
            generic = cands.loc[(cands["loc_n"] == "") & (cands["pc_n"] == "")]
            if not generic.empty:
                cands = generic

        cands = cands.sort_values("company_id", kind="stable")
        return str(cands["company_id"].iloc[0]) if not cands.empty else None

    stations["company_id"] = stations.apply(pick_company_id, axis=1)
    return stations


# ============================================================
# Stations ⨝ connectors with company_id and categories
# ============================================================
def load_stations_with_connectors(
    stations_csv: str | Path,
    connectors_csv: str | Path,
    pricing_core_csv: str | Path,
    thresholds: Dict[str, float] = DEFAULT_THRESHOLDS,
    status_allowlist: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Return per-connector rows with reliable columns:
      ['station_id','company_id','company_name','charge_method','connector_type',
       'rated_power_kw','charger_type','lat','lon','edge_id','status']
    """
    status_allowlist = status_allowlist or DEFAULT_STATUS_ALLOWLIST

    # Load
    stations = pd.read_csv(stations_csv)
    if "deviceNetworks" in stations.columns:
        stations["deviceNetworks"] = stations["deviceNetworks"].astype(str)
        stations = stations[stations["deviceNetworks"].str.strip() != ""].copy()
        stations = stations[~stations["deviceNetworks"].str.lower().isin(["nan", "none", "null"])].copy()

    connectors = pd.read_csv(connectors_csv)
    companies = pd.read_csv(pricing_core_csv).copy()

    # Normalize company location columns for matching
    ren = {}
    if "City" in companies.columns and "city" not in companies.columns:
        ren["City"] = "city"
    if "Postcode" in companies.columns and "postcode" not in companies.columns:
        ren["Postcode"] = "postcode"
    if ren:
        companies = companies.rename(columns=ren)

    # Canonicalize stations
    if "chargeDeviceID" in stations.columns and "station_id" not in stations.columns:
        stations = stations.rename(columns={"chargeDeviceID": "station_id"})
    if "latitude" in stations.columns and "lat" not in stations.columns:
        stations = stations.rename(columns={"latitude": "lat"})
    if "longitude" in stations.columns and "lon" not in stations.columns:
        stations = stations.rename(columns={"longitude": "lon"})

    # deviceNetworks → primary company name
    if "deviceNetworks" in stations.columns:
        primary = _primary_from_device_networks(stations["deviceNetworks"])
        if "company_name" not in stations.columns:
            stations["company_name"] = primary
        else:
            mask = stations["company_name"].astype(str).str.strip().eq("") | stations["company_name"].isna()
            stations.loc[mask, "company_name"] = primary

    # Map to company_id (keep your "pick company" logic formalized)
    stations = _match_company_ids_on_stations(stations, companies)

    # Canonicalize connectors
    if "chargeDeviceID" in connectors.columns and "station_id" not in connectors.columns:
        connectors = connectors.rename(columns={"chargeDeviceID": "station_id"})
    if "rated_power_kW" in connectors.columns and "rated_power_kw" not in connectors.columns:
        connectors = connectors.rename(columns={"rated_power_kW": "rated_power_kw"})
    if "max_power_kW" in connectors.columns and "rated_power_kw" not in connectors.columns:
        connectors = connectors.rename(columns={"max_power_kW": "rated_power_kw"})

    # Ensure charge_method
    if "charge_method" not in connectors.columns:
        if "connector_type" in connectors.columns:
            connectors["charge_method"] = connectors["connector_type"].map(_derive_charge_method_from_connector_name)
        else:
            connectors["charge_method"] = "AC"

    # Filter by status (keep only active/available) if column exists
    if "status" in connectors.columns:
        connectors = connectors[connectors["status"].isin(status_allowlist)].copy()

    # Merge station info into connectors
    merged = connectors.merge(
        stations[["station_id", "company_id", "company_name", "lat", "lon"]].copy(),
        on="station_id",
        how="inner",
    )

    # Derive charger_type (Fast/Rapid/Ultra)
    merged["charger_type"] = merged.apply(
        lambda r: map_category(r.get("charge_method"), r.get("rated_power_kw"), thresholds), axis=1
    )

    # Edge id optional
    if "edge_id" not in merged.columns:
        merged["edge_id"] = None

    # Final order (only keep those present)
    cols = [
        "station_id", "company_id", "company_name",
        "charge_method", "connector_type", "rated_power_kw",
        "charger_type", "lat", "lon", "edge_id", "status"
    ]
    merged = merged[[c for c in cols if c in merged.columns]].copy()
    return merged


# ============================================================
# Aggregations: Station capabilities & EV capabilities
# ============================================================
def build_station_capabilities(connectors_enriched: pd.DataFrame) -> Dict[str, StationCaps]:
    caps: Dict[str, StationCaps] = {}
    if connectors_enriched.empty:
        return caps

    req_cols = {"station_id", "company_id", "company_name", "charger_type"}
    if not req_cols.issubset(connectors_enriched.columns):
        raise KeyError(f"connectors_enriched missing required columns: {sorted(req_cols)}")

    def _max_kw(df: pd.DataFrame) -> float:
        if "rated_power_kw" in df.columns and len(df["rated_power_kw"].dropna()) > 0:
            return float(df["rated_power_kw"].max())
        return 0.0

    for (sid, cid, cname), g in connectors_enriched.groupby(["station_id", "company_id", "company_name"], dropna=False):
        by_type = g.groupby("charger_type")
        types = set(by_type.groups.keys())
        pstats = {ct: _max_kw(by_type.get_group(ct)) for ct in types}
        caps[str(sid)] = StationCaps(
            station_id=str(sid),
            company_id=str(cid),
            company_name=str(cname),
            categories_available=types,
            power_stats_by_category=pstats,
        )
    return caps


def build_ev_capabilities(ev_meta_df: pd.DataFrame) -> Dict[str, EVCaps]:
    """
    Map EV AC/DC limits to supported categories (Fast/Rapid/Ultra).
    Default policy:
      - AC present -> include Fast
      - DC: <50 Fast, 50-99 Rapid, >=100 Ultra
    """
    req = {"brand_name", "model", "release_year", "ac_max_power_kW", "dc_max_power_kW", "ac_phases"}
    if not req.issubset(set(ev_meta_df.columns)):
        missing = sorted(list(req - set(ev_meta_df.columns)))
        raise KeyError(f"EV_Metadata missing columns: {missing}")

    ev_caps: Dict[str, EVCaps] = {}
    for _, r in ev_meta_df.iterrows():
        brand = str(r["brand_name"])
        model = str(r["model"])
        year = str(r["release_year"])
        ev_id = f"{brand}|{model}|{year}"

        ac_max = float(r["ac_max_power_kW"]) if pd.notna(r["ac_max_power_kW"]) else None
        dc_max = float(r["dc_max_power_kW"]) if pd.notna(r["dc_max_power_kW"]) else None
        phases = int(r["ac_phases"]) if pd.notna(r["ac_phases"]) else None

        cats: Set[str] = set()
        if ac_max and ac_max > 0:
            cats.add("Fast")  # default: AC treated as Fast
        if dc_max and dc_max > 0:
            if dc_max <= DEFAULT_THRESHOLDS["DC_FAST_MAX"]:
                cats.add("Fast")
            elif dc_max <= DEFAULT_THRESHOLDS["DC_RAPID_MAX"]:
                cats.add("Rapid")
            else:
                cats.add("Ultra")

        ev_caps[ev_id] = EVCaps(
            ev_id=ev_id,
            ac_max_power_kW=ac_max,
            dc_max_power_kW=dc_max,
            ac_phases=phases,
            categories_supported=cats,
        )
    return ev_caps


def build_ev_curves_index(ev_curve_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Index raw charging curves per EV for fast access.
    Expected columns minimally include: ['brand_name','model','release_year','soc','kW'] (your file may differ;
    we keep it flexible and return per-EV slices).
    """
    req_base = {"brand_name", "model", "release_year"}
    if not req_base.issubset(set(ev_curve_df.columns)):
        # Fall back to 'model' only if that's all we have (keep backward compatible)
        if "model" not in ev_curve_df.columns:
            raise KeyError("EV_Charging_Curve_Data requires ['brand_name','model','release_year'] or at least 'model'.")

    index: Dict[str, pd.DataFrame] = {}
    if req_base.issubset(set(ev_curve_df.columns)):
        for (brand, model, year), g in ev_curve_df.groupby(["brand_name", "model", "release_year"]):
            ev_id = f"{brand}|{model}|{year}"
            index[ev_id] = g.sort_values("soc") if "soc" in g.columns else g.copy()
    else:
        for model, g in ev_curve_df.groupby("model"):
            ev_id = str(model)
            index[ev_id] = g.sort_values("soc") if "soc" in g.columns else g.copy()
    return index


# ============================================================
# Pricing pre-index (no price calculation here)
# ============================================================
def build_pricing_index(pricing_core_df, pricing_by_type_df, pricing_conditions_df):
    """
    Build a pricing lookup for all (company_id, charger_type).
    """

    # Preflight: ensure normalized schema (self-heal if a raw frame slipped through)
    required_cols = ("company_id", "charger_type", "user_type", "price_per_kwh", "exists_flag")
    if any(c not in pricing_by_type_df.columns for c in required_cols):
        pricing_by_type_df = _normalize_pricing_by_type(pricing_by_type_df)
        still_missing = [c for c in required_cols if c not in pricing_by_type_df.columns]
        if still_missing:
            raise KeyError(
                f"build_pricing_index expected {required_cols}, "
                f"got {list(pricing_by_type_df.columns)}; missing {still_missing}"
            )

    by_type_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, row in pricing_by_type_df.iterrows():
        if bool(row.get("exists_flag", True)):
            cid = str(row["company_id"])
            ct = str(row["charger_type"])
            price = float(row["price_per_kwh"]) if pd.notna(row["price_per_kwh"]) else None
            if price is not None:
                by_type_map[(cid, ct)] = {
                    "price_per_kwh": price,
                    "exists": True,
                    "user_type": row.get("user_type", "Default"),
                }

    # core defaults (company-level)
    if "company_id" not in pricing_core_df.columns:
        raise KeyError("pricing_core.csv must include company_id")
    if "default_price" not in pricing_core_df.columns:
        if "base_price" in pricing_core_df.columns:
            pricing_core_df = pricing_core_df.rename(columns={"base_price": "default_price"})
        else:
            pricing_core_df = pricing_core_df.copy()
            pricing_core_df["default_price"] = 0.30
    core_defaults = {str(r["company_id"]): float(r.get("default_price", 0.30))
                     for _, r in pricing_core_df.iterrows()}

    # fill any missing (company, charger_type) with company default
    all_companies = set(pricing_core_df["company_id"].astype(str))
    all_types = {"Fast", "Rapid", "Ultra"}
    for cid in all_companies:
        for ct in all_types:
            if (cid, ct) not in by_type_map:
                by_type_map[(cid, ct)] = {
                    "price_per_kwh": core_defaults.get(cid, 0.30),
                    "exists": False,
                    "user_type": "Fallback",
                }

    # conditions
    conditions_map: Dict[Tuple[str, str], list] = {}
    if pricing_conditions_df is not None and not pricing_conditions_df.empty:
        pc = pricing_conditions_df.copy()
        # try to find charger_type column if provided under different label
        if "charger_type" not in pc.columns:
            for alt in ("tier", "speed_tier", "tariff_type", "speed_class"):
                if alt in pc.columns:
                    pc["charger_type"] = pc[alt]
                    break
        if "charger_type" in pc.columns:
            for _, row in pc.iterrows():
                cid = str(row["company_id"])
                ct = str(row["charger_type"])
                conditions_map.setdefault((cid, ct), []).append({
                    "condition": row.get("condition"),
                    "multiplier": float(row.get("multiplier", 1.0)),
                })

    return {"by_type": by_type_map, "conditions": conditions_map, "core_defaults": core_defaults}


# ============================================================
# Validators
# ============================================================
def validate_all(
    stations_df: "pd.DataFrame",
    connectors_df: "pd.DataFrame",
    station_capabilities: dict,
    pricing_index: dict,
    ev_capabilities: dict,
    *,
    strict: bool = True,
) -> dict:
    """
    Consistency checks across loaded artifacts.

    Critical only if:
      - stations_without_connectors > 0 (data gap for routing)
      - stations_with_connectors_missing_company > 0 (pricing lookup would break)

    Stations that have no connectors are NOT also counted as "missing company".
    """
    issues = {
        "stations_total": 0,
        "stations_with_connectors": 0,
        "stations_without_connectors": 0,
        "stations_with_connectors_missing_company": 0,
        "pricing_missing_pairs": [],
        "evs_without_categories": [],
    }

    # --- presence ---
    if stations_df is None or connectors_df is None:
        raise ValueError("validate_all: stations_df and connectors_df are required.")
    for col in ("station_id",):
        if col not in stations_df.columns:
            raise KeyError(f"stations_df missing '{col}'")
        if col not in connectors_df.columns:
            raise KeyError(f"connectors_df missing '{col}'")

    all_stations = set(stations_df["station_id"].astype(str))
    with_connectors = set(connectors_df["station_id"].astype(str))
    without_connectors = all_stations - with_connectors

    issues["stations_total"] = len(all_stations)
    issues["stations_with_connectors"] = len(with_connectors)
    issues["stations_without_connectors"] = len(without_connectors)

    # --- only check company for stations that DO have connectors ---
    missing_company = 0
    for sid in with_connectors:
        sc = station_capabilities.get(sid)
        if not sc:
            missing_company += 1
            continue
        cid = getattr(sc, "company_id", None) if not isinstance(sc, dict) else sc.get("company_id")
        if cid in (None, "", "nan"):
            missing_company += 1
    issues["stations_with_connectors_missing_company"] = missing_company

    # --- pricing coverage ---
    by_type = pricing_index.get("by_type", {})
    core_defaults = pricing_index.get("core_defaults", {})
    companies_seen = set()
    for sid in with_connectors:
        sc = station_capabilities.get(sid)
        if sc:
            cid = getattr(sc, "company_id", None) if not isinstance(sc, dict) else sc.get("company_id")
            if cid is not None and cid != "":
                companies_seen.add(str(cid))

    cats = {"Fast", "Rapid", "Ultra"}
    missing_pairs = []
    for cid in sorted(companies_seen):
        for ct in cats:
            if (cid, ct) not in by_type:
                missing_pairs.append((cid, ct))
            else:
                entry = by_type[(cid, ct)]
                if (entry.get("price_per_kwh") is None) and (str(cid) not in core_defaults):
                    missing_pairs.append((cid, ct))
    issues["pricing_missing_pairs"] = missing_pairs
    if missing_pairs:
        preview = ", ".join(map(str, missing_pairs[:10]))
        print(f"[WARN] Missing pricing pairs ({len(missing_pairs)}). "
              f"Filled by company defaults / fallback. Examples: {preview}")

    # --- criticality ---
    critical = []
    if issues["stations_without_connectors"] > 0:
        critical.append(f"stations_without_connectors={issues['stations_without_connectors']}")
    if issues["stations_with_connectors_missing_company"] > 0:
        critical.append(f"stations_with_connectors_missing_company={issues['stations_with_connectors_missing_company']}")

    if strict and critical:
        raise ValueError(f"Validation failed (critical): {', '.join(critical)}")

    return issues



# ============================================================
# Public API 1 (kept): original loader
# ============================================================
def load_all_ready(
    data_dir: str | Path = "data",
    strict: bool = True,
    thresholds: Dict[str, float] = DEFAULT_THRESHOLDS,
    status_allowlist: Optional[Set[str]] = None,
) -> TrainBundle:
    """
    One-shot loader that returns everything the env/agent needs:
      {
        "station_capabilities": dict[station_id -> StationCaps],
        "station_connectors_enriched": pd.DataFrame,
        "ev_capabilities": dict[ev_id -> EVCaps],
        "pricing_index": {"core":..., "by_type":..., "conditions":...},
        "ev_curves_index": dict[ev_id -> curve_df]
      }
    """
    data_dir = Path(data_dir)

    # 1) Enriched connectors with company_id + categories
    connectors_enriched = load_stations_with_connectors(
        data_dir / "charging_station_metadata.csv",
        data_dir / "charging_station_connectors.csv",
        data_dir / "pricing_core.csv",
        thresholds=thresholds,
        status_allowlist=None,  # keep all statuses; don't drop stations silently
    )

    # --- Path B: drop stations that have no connectors (and KEEP this filtered frame)
    has_conn = set(connectors_enriched["station_id"].astype(str))
    stations_df = pd.read_csv(data_dir / "charging_station_metadata.csv")
    if "chargeDeviceID" in stations_df.columns and "station_id" not in stations_df.columns:
        stations_df = stations_df.rename(columns={"chargeDeviceID": "station_id"})
    stations_df["station_id"] = stations_df["station_id"].astype(str)
    stations_df = stations_df[stations_df["station_id"].isin(has_conn)].copy()

    # 2) Station caps
    station_caps = build_station_capabilities(connectors_enriched)

    # 3) EV capabilities + curves
    ev_meta = pd.read_csv(data_dir / "EV_Metadata.csv")
    ev_caps = build_ev_capabilities(ev_meta)

    ev_curves_df = pd.read_csv(data_dir / "EV_Charging_Curve_Data.csv")
    ev_curves_index = build_ev_curves_index(ev_curves_df)

    # 4) Pricing pre-index
    pricing_core_df = pd.read_csv(data_dir / "pricing_core.csv")
    pricing_by_type_df = pd.read_csv(data_dir / "pricing_by_charger_type.csv")
    pricing_by_type_df = _normalize_pricing_by_type(pricing_by_type_df)
    pricing_conditions_df = pd.read_csv(data_dir / "pricing_conditions.csv")
    pricing_index = build_pricing_index(pricing_core_df, pricing_by_type_df, pricing_conditions_df)
    
    # 4b) Full pricing catalog (strict) for per-session billing
    pricing_catalog = load_pricing_catalog(data_dir)

    # 3b) Build EVPowerModel from raw frames (for curve-based power at SoC)
    # EVPowerModel expects:
    #   - EV metadata columns: model, battery_kWh, dc_max_power_kW, ac_max_power_kW, avg_consumption_Wh_per_km
    #   - Curves columns:       model, soc_percent, charging_power_kW
    # If your EV_Charging_Curve_Data.csv uses different names (e.g. 'soc','kW'),
    # keep using your existing ev_curves_index for other features; for EVPowerModel we need a canonical frame.
    # Try to map if needed:
    curves_df_for_model = ev_curves_df.copy()
    if "soc_percent" not in curves_df_for_model.columns and "soc" in curves_df_for_model.columns:
        curves_df_for_model = curves_df_for_model.rename(columns={"soc": "soc_percent"})
    if "charging_power_kW" not in curves_df_for_model.columns:
        for c in ("kW", "power_kW", "charging_kW"):
            if c in curves_df_for_model.columns:
                curves_df_for_model = curves_df_for_model.rename(columns={c: "charging_power_kW"})
                break
    ev_power_model = EVPowerModel(ev_meta, curves_df_for_model)

    # 5) Validate coverage (use the FILTERED stations_df — do NOT re-read)
    validate_all(
        stations_df=stations_df,
        connectors_df=connectors_enriched,
        station_capabilities=station_caps,
        pricing_index=pricing_index,
        ev_capabilities=ev_caps,
        strict=strict,
    )

    return {
        "station_capabilities": station_caps,
        "station_connectors_enriched": connectors_enriched,
        "ev_capabilities": ev_caps,
        "pricing_index": pricing_index,
        "pricing_catalog": pricing_catalog,
        "ev_curves_index": ev_curves_index,
        "ev_power_model": ev_power_model,
    }
