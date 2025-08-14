# utils/data_loader.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from utils.pricing import PricingCatalog, load_pricing_catalog  # moved out
from typing import Tuple, Iterable
import os
log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


# ---------- bundle ----------
@dataclass
class DatasetBundle:
    ev_metadata: pd.DataFrame
    charging_curves: pd.DataFrame
    stations: pd.DataFrame
    connectors: pd.DataFrame
    stations_merged: pd.DataFrame
    users: Optional[pd.DataFrame]
    pricing_catalog: PricingCatalog


# ---------- IO ----------
def _read_csv(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        log.warning("Optional dataset missing: %s", path)
        return None
    return pd.read_csv(path)


# ---------- merge with cascade ----------
def _merge_station_connector(
    stations: pd.DataFrame,
    connectors: pd.DataFrame,
    pricing_core: pd.DataFrame,
) -> pd.DataFrame:
    """
    Resolve company_id for each station using this cascade:

      1) filter by company_name (deviceNetworks)
      2) if duplicates, prefer same region over 'nationwide'
      3) if still multiple, match specific_location
      4) if still multiple, match postcode (then city as tiebreaker if present)
      5) if none matched by specific_location/postcode, prefer a generic row
         (empty specific_location & empty postcode)
      6) if still multiple, choose first deterministically (and WARN)

    Expected columns:
      stations:   chargeDeviceID, deviceNetworks, latitude, longitude,
                  [region], [specific_location], [city], [postcode]
      connectors: chargeDeviceID, connector_type, rated_power_kW, charge_method
      pricing_core: company_id, company_name, [region], [specific_location], [city], [postcode]
    """
    # ---- required ----
    st_req = ["chargeDeviceID", "deviceNetworks", "latitude", "longitude"]
    cn_req = ["chargeDeviceID", "connector_type", "rated_power_kW", "charge_method"]
    pc_req = ["company_id", "company_name"]
    for req, df, name in (
        (st_req, stations, "stations"),
        (cn_req, connectors, "connectors"),
        (pc_req, pricing_core, "pricing_core"),
    ):
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise KeyError(f"{name} missing: {miss}")

    # ---- select/rename ----
    st = stations[st_req + [c for c in ["region", "specific_location", "city", "postcode"] if c in stations.columns]].rename(
        columns={"chargeDeviceID": "station_id", "deviceNetworks": "company_name"}
    ).copy()
    # drop rows with missing operator (no way to map)
    st = st[st["company_name"].notna()].copy()
    for c in ["region", "specific_location", "city", "postcode"]:
        if c not in st.columns:
            st[c] = ""

    cn = connectors[cn_req].rename(columns={
        "chargeDeviceID": "station_id",
        "rated_power_kW": "rated_power_kw",
        "charge_method": "charger_type",
    }).copy()

    # numeric safety
    cn["rated_power_kw"] = pd.to_numeric(cn["rated_power_kw"], errors="raise")
    st["latitude"] = pd.to_numeric(st["latitude"], errors="raise")
    st["longitude"] = pd.to_numeric(st["longitude"], errors="raise")

    pc = pricing_core[pc_req + [c for c in ["region", "specific_location", "city", "postcode"] if c in pricing_core.columns]].copy()
    for c in ["region", "specific_location", "city", "postcode"]:
        if c not in pc.columns:
            pc[c] = ""

    # ---- normalisation (keep exact headers; only trim/case-fold) ----
    def _norm(s: pd.Series) -> pd.Series:
        return (
            s.astype(str)
             .str.strip()
             .str.replace(r"\s+", " ", regex=True)
             .str.replace("\u00A0", " ")
             .str.replace("’", "'")
             .str.lower()
        )

    st["company_name_n"] = _norm(st["company_name"])
    pc["company_name_n"] = _norm(pc["company_name"])
    st["region_n"] = _norm(st["region"])
    pc["region_n"] = _norm(pc["region"])
    st["loc_n"] = _norm(st["specific_location"])
    pc["loc_n"] = _norm(pc["specific_location"])
    st["pc_n"] = _norm(st["postcode"])
    pc["pc_n"] = _norm(pc["postcode"])
    if "city" in st.columns:
        st["city_n"] = _norm(st["city"])
    else:
        st["city_n"] = ""
    if "city" in pc.columns:
        pc["city_n"] = _norm(pc["city"])
    else:
        pc["city_n"] = ""

    # treat “nationwide” synonyms
    def _is_nationwide(x: str) -> bool:
        return x in ("nationwide", "uk-wide", "uk wide", "ukwide", "all uk", "all-uk", "alluk", "")

    # ---- resolve for each station row ----
    st["company_id"] = pd.NA

    def pick_company_id(row) -> Optional[int]:
        # 1) by name
        cands = pc[pc["company_name_n"] == row["company_name_n"]]
        if cands.empty:
            return None

        # 2) region preference
        # If station has region, prefer exact match; otherwise prefer 'nationwide'
        if row["region_n"]:
            same_region = cands[cands["region_n"] == row["region_n"]]
            if not same_region.empty:
                cands = same_region
            else:
                # if no same-region, try nationwide fallback
                nw = cands[cands["region_n"].map(_is_nationwide)]
                if not nw.empty:
                    cands = nw
                # else keep existing cands (multiple other regions → try loc/postcode)
        else:
            # station has no region → prefer nationwide if present
            nw = cands[cands["region_n"].map(_is_nationwide)]
            if not nw.empty:
                cands = nw

        # 3) specific_location exact
        if len(cands) > 1 and bool(row["loc_n"]):
            m = cands[cands["loc_n"] == row["loc_n"]]
            if not m.empty:
                cands = m

        # 4) postcode exact
        if len(cands) > 1 and bool(row["pc_n"]):
            m = cands[cands["pc_n"] == row["pc_n"]]
            if not m.empty:
                cands = m

        # (optional) city as additional tiebreaker
        if len(cands) > 1 and bool(row["city_n"]):
            m = cands[cands["city_n"] == row["city_n"]]
            if not m.empty:
                cands = m

        # 5) prefer generic (empty loc & empty postcode) if still multiple
        if len(cands) > 1:
            generic = cands[(cands["loc_n"] == "") & (cands["pc_n"] == "")]
            if not generic.empty:
                cands = generic

        # 6) deterministic pick (first by company_id)
        cands = cands.sort_values("company_id")
        return int(cands["company_id"].iloc[0]) if not cands.empty else None

    st["company_id"] = st.apply(pick_company_id, axis=1)

    # warn unresolved (but do not crash)
    if st["company_id"].isna().any():
        cols = ["company_name", "region", "specific_location", "city", "postcode"]
        bad = st.loc[st["company_id"].isna(), cols].drop_duplicates()
        log.warning("Unresolved company_id for some stations (up to 10):\n%s", bad.head(10).to_string(index=False))

    # operator alias
    st["operator_name"] = st["company_name"]

    # ---- attach connectors
    out = cn.merge(st, on="station_id", how="left")
    if "edge_id" not in out.columns:
        out["edge_id"] = None

    cols = [
        "station_id", "company_id", "operator_name", "company_name",
        "charger_type", "connector_type", "rated_power_kw",
        "latitude", "longitude", "edge_id"
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols].copy()



# ---------- public loader ----------
def load_all_data(data_dir: str | Path) -> DatasetBundle:
    data_dir = Path(data_dir)

    ev = _read_csv(data_dir / "EV_Metadata.csv")
    curves = _read_csv(data_dir / "EV_Charging_Curve_Data.csv")
    stations = _read_csv(data_dir / "charging_station_metadata.csv")
    connectors = _read_csv(data_dir / "charging_station_connectors.csv")
    users = _read_csv(data_dir / "simulated_users.csv", required=False)

    pricing = load_pricing_catalog(data_dir)  # from utils.pricing

    stn_merged = _merge_station_connector(stations, connectors, pricing.core)

    return DatasetBundle(
        ev_metadata=ev,
        charging_curves=curves,
        stations=stations,
        connectors=connectors,
        stations_merged=stn_merged,
        users=users,
        pricing_catalog=pricing,
    )


def _dl_norm_name(x: str) -> str:
    return str(x).strip().lower()

def _dl_norm_connector(label: str) -> str:
    s = str(label).strip().lower()
    if "ccs" in s or "combined" in s:
        return "CCS2"
    if "chademo" in s:
        return "CHAdeMO"
    if "mennekes" in s or "type 2" in s or "iec62196" in s:
        return "Type2"
    if "type 1" in s or "j1772" in s:
        return "Type1"
    if "3-pin" in s or "3 pin" in s or "bs 1363" in s:
        return "UK 3-pin"
    return str(label).strip()

def _dl_is_dc(charge_method: str) -> bool:
    s = str(charge_method).strip().lower()
    return ("dc" in s) or ("rapid" in s) or ("ultra" in s)

def _dl_charger_type(is_dc: bool, power_kw: float) -> str:
    # simple buckets for candidate selection; does NOT affect pricing
    if is_dc or power_kw >= 50:
        return "Rapid"
    if power_kw >= 22:
        return "Fast"
    return "Slow"

def build_stations_merged_from_frames(
    stations_df: pd.DataFrame,
    connectors_df: pd.DataFrame,
    pricing_core_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pure function: merge raw station metadata + connectors + operator→company map
    into a canonical stations_merged table used by SUMO/candidate search.

    Does NOT modify or rely on pricing logic. Output columns:
      ['station_id','latitude','longitude','company_id','charger_type',
       'rated_power_kw','station_connectors','operator_name']
    """
    # Key is chargeDeviceID in your datasets
    st = stations_df.rename(columns={
        "chargeDeviceID": "station_id",
        "deviceNetworks": "operator_name",
    }).copy()

    conns = connectors_df.rename(columns={
        "chargeDeviceID": "station_id",
        "rated_power_kW": "rated_power_kw",
    }).copy()

    # normalize connector labels & methods for aggregation
    if "connector_type" in conns.columns:
        conns["connector_type_norm"] = conns["connector_type"].map(_dl_norm_connector)
    else:
        conns["connector_type_norm"] = ""

    if "charge_method" in conns.columns:
        conns["is_dc"] = conns["charge_method"].map(_dl_is_dc)
    else:
        conns["is_dc"] = False

    agg = (
        conns.groupby("station_id")
        .agg(
            rated_power_kw=("rated_power_kw", "max"),
            any_dc=("is_dc", "max"),
            station_connectors=("connector_type_norm",
                                lambda s: tuple(sorted({x for x in s if pd.notna(x) and str(x).strip()}))),
        )
        .reset_index()
    )

    merged = st.merge(agg, on="station_id", how="left")

    # defaults if missing
    merged["rated_power_kw"] = pd.to_numeric(merged["rated_power_kw"], errors="coerce").fillna(7.0)
    merged["any_dc"] = merged["any_dc"].fillna(False)
    merged["station_connectors"] = merged["station_connectors"].apply(
        lambda x: x if isinstance(x, tuple) else tuple()
    )

    # derive charger_type for candidate filtering (not used by pricing)
    merged["charger_type"] = [
        _dl_charger_type(bool(dc), float(p))
        for dc, p in zip(merged["any_dc"], merged["rated_power_kw"])
    ]

    # operator → company_id via pricing_core company_name
    core = pricing_core_df.rename(columns={"company_name": "company_name_raw"}).copy()
    core["company_name_norm"] = core["company_name_raw"].map(_dl_norm_name)
    merged["operator_name_norm"] = merged["operator_name"].fillna("").map(_dl_norm_name)

    merged = merged.merge(
        core[["company_id", "company_name_norm"]].drop_duplicates(),
        left_on="operator_name_norm",
        right_on="company_name_norm",
        how="left",
    )

    merged["company_id"] = merged["company_id"].fillna(-1).astype(int)

    out = merged[[
        "station_id", "latitude", "longitude", "company_id",
        "charger_type", "rated_power_kw", "station_connectors", "operator_name"
    ]].copy()

    out["latitude"]  = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude","longitude"])
    return out

def build_stations_merged_files(
    data_dir: str,
    stations_csv: str = "charging_station_metadata.csv",
    connectors_csv: str = "charging_station_connectors.csv",
    pricing_core_csv: str = "pricing_core.csv",
    save_csv: str | None = "data/stations_merged.csv",
) -> pd.DataFrame:
    """
    File-based convenience wrapper around build_stations_merged_from_frames.
    Safe ADD-ONLY: does not alter existing code paths.
    """
    stations = pd.read_csv(os.path.join(data_dir, stations_csv))
    connectors = pd.read_csv(os.path.join(data_dir, connectors_csv))
    core = pd.read_csv(os.path.join(data_dir, pricing_core_csv))

    merged = build_stations_merged_from_frames(stations, connectors, core)

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        merged.to_csv(save_csv, index=False)
    return merged