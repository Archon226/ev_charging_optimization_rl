# tests/diagnose_stations_merge.py
import os, sys, pandas as pd
from pprint import pprint

DATA = "data"  # change if your csvs live elsewhere

def _pick(df, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing column for {label}. Looked for {candidates}\nHave: {df.columns.tolist()}")

def load_csv(name):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        print(f"!! Missing: {path}")
        return None
    try:
        df = pd.read_csv(path)
        print(f"Loaded {name}: shape={df.shape}")
        return df
    except Exception as e:
        print(f"!! Failed to read {name}: {e}")
        return None

def main():
    stations = load_csv("charging_station_metadata.csv")
    connectors = load_csv("charging_station_connectors.csv")

    if stations is None:
        print("Cannot continue without stations metadata.")
        sys.exit(1)

    print("\n--- Stations columns ---")
    print(stations.columns.tolist())
    print("Sample station row:")
    pprint(stations.iloc[0].to_dict(), width=120)

    # Does the merged table already have connectors?
    existing_connector_cols = [c for c in stations.columns if "connector" in c.lower()]
    if existing_connector_cols:
        print("\n✅ Stations already has connector-related columns:", existing_connector_cols)
    else:
        print("\nℹ️  Stations does not seem to have connector columns yet.")

    if connectors is None:
        print("\n(No connectors CSV found, skipping join preview.)")
        sys.exit(0)

    print("\n--- Connectors columns ---")
    print(connectors.columns.tolist())
    print("Sample connector row:")
    pprint(connectors.iloc[0].to_dict(), width=120)

    # Try to detect join keys
    station_id_col = _pick(stations, ["station_id","cp_id","charger_id","evse_id","EVSE_ID"], "station_id (stations)")
    conn_station_id_col = _pick(connectors, ["station_id","cp_id","charger_id","evse_id","EVSE_ID"], "station_id (connectors)")
    conn_type_col = _pick(connectors, ["connector_type","ConnectorType","connector","plug_type","PlugType"], "connector_type")

    print(f"\nDetected keys:\n  stations.{station_id_col}  <->  connectors.{conn_station_id_col}\n  connector type column: {conn_type_col}")

    # Build a preview of the aggregated connectors per station
    agg = (
        connectors[[conn_station_id_col, conn_type_col]]
        .dropna()
        .groupby(conn_station_id_col)[conn_type_col]
        .apply(lambda s: tuple(sorted(set(str(x).strip() for x in s if str(x).strip()))))
        .rename("station_connectors")
        .reset_index()
    )
    merged = stations.merge(agg, left_on=station_id_col, right_on=conn_station_id_col, how="left")
    merged["station_connectors"] = merged["station_connectors"].apply(lambda x: x if isinstance(x, tuple) else tuple())

    print(f"\nPreview merged shape: {merged.shape}")
    # Show a few with non-empty connectors
    non_empty = merged[merged["station_connectors"].apply(len) > 0].head(5)
    if len(non_empty):
        print("\nExample stations with connectors attached:")
        cols_to_show = [station_id_col, "station_connectors"]
        print(non_empty[cols_to_show].to_string(index=False))
    else:
        print("\nNo non-empty connector tuples found in preview (maybe connectors CSV has different keys/types).")

    # Save a temporary preview (optional)
    out = os.path.join(DATA, "stations_with_connectors_preview.csv")
    merged.to_csv(out, index=False)
    print(f"\nSaved preview to: {out}")

if __name__ == "__main__":
    main()
