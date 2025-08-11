# utils/data_loader.py
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from utils.pricing import PricingCatalog, load_pricing_catalog  # moved out

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
