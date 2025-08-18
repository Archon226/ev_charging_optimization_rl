#!/usr/bin/env python3
"""
Robust inner-London SUMO net builder:
- Reads data/charging_station_metadata.csv
- Clips outliers via lat/lon quantiles (2%..98% by default) OR uses a radius bbox around median
- Downloads OSM from multiple Overpass mirrors with retries (or uses a local .osm/.pbf)
- Runs netconvert -> london_inner.net.xml
- Patches rl/train.py (NET_FILE)

CLI (optional):
  --radius-km 20        # use circle bbox around median (km); overrides quantile clip if provided
  --qlo 0.02 --qhi 0.98 # quantile clipping (if no radius provided)
  --osm-file path.osm   # skip download, use local .osm/.pbf
"""

import argparse, csv, math, os, re, shutil, subprocess, sys, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CSV = PROJECT_ROOT / "data" / "charging_station_metadata.csv"
NET_DIR = PROJECT_ROOT 
OUT_OSM = NET_DIR / "london_inner.osm"
OUT_NET = PROJECT_ROOT / "london_inner.net.xml"
TRAIN_PY = PROJECT_ROOT / "rl" / "train.py"

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/map",
    "https://overpass.kumi.systems/api/map",
    "https://overpass.openstreetmap.ru/api/map",
    "https://lz4.overpass-api.de/api/map",
]

def die(msg): print(f"❌ {msg}", file=sys.stderr); sys.exit(1)
def info(msg): print(msg, flush=True)

def detect_lat_lon_headers(headers):
    hl = [h.lower() for h in headers]
    lat = None; lon = None
    for i,h in enumerate(hl):
        if h in ("lat","latitude"): lat = headers[i]
        if h in ("lon","lng","longitude"): lon = headers[i]
    if lat and lon: return lat, lon
    for i,h in enumerate(hl):
        if ("lat" in h) and not lat: lat = headers[i]
        if (("lon" in h) or ("lng" in h)) and not lon: lon = headers[i]
    return lat, lon

def read_coords(path: Path):
    if not path.exists(): die(f"Not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f); rows = list(reader)
    if not rows: die(f"Empty CSV: {path}")
    lat_col, lon_col = detect_lat_lon_headers(reader.fieldnames or [])
    if not lat_col or not lon_col:
        die(f"Could not detect lat/lon columns in {path}. Headers: {reader.fieldnames}")
    lat, lon = [], []
    for r in rows:
        try:
            la = float(r[lat_col]); lo = float(r[lon_col])
        except (ValueError, KeyError, TypeError):
            continue
        if -90 <= la <= 90 and -180 <= lo <= 180:
            lat.append(la); lon.append(lo)
    if not lat: die("No valid lat/lon parsed.")
    return lat, lon

def quantiles(arr, q):
    # arr must be non-empty
    s = sorted(arr); n = len(s)
    if n == 1: return s[0]
    p = max(0.0, min(1.0, q))
    idx = p*(n-1); lo = int(math.floor(idx)); hi = int(math.ceil(idx))
    if lo == hi: return s[lo]
    w = idx - lo
    return s[lo]*(1-w) + s[hi]*w

def deg_per_km_lat(): return 1.0/111.0
def deg_per_km_lon_at_lat(lat_deg): return math.cos(math.radians(lat_deg))/111.0

def bbox_from_radius(lat, lon, radius_km):
    dlat = radius_km * deg_per_km_lat()
    dlon = radius_km * deg_per_km_lon_at_lat(lat)
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)

def bbox_from_quantiles(lats, lons, qlo, qhi, pad_km=1.0):
    lo_lat = quantiles(lats, qlo); hi_lat = quantiles(lats, qhi)
    lo_lon = quantiles(lons, qlo); hi_lon = quantiles(lons, qhi)
    lat_mid = (lo_lat + hi_lat)/2.0
    pad_lat = pad_km*deg_per_km_lat()
    pad_lon = pad_km*deg_per_km_lon_at_lat(lat_mid)
    return (lo_lon - pad_lon, lo_lat - pad_lat, hi_lon + pad_lon, hi_lat + pad_lat)

def check_netconvert_on_path():
    exe = "netconvert.exe" if os.name == "nt" else "netconvert"
    path = shutil.which(exe)
    if not path:
        die("Could not find 'netconvert' on PATH. Add SUMO\\bin to PATH and reopen your shell.")
    info(f"[INFO] netconvert: {path}")
    return path

def download_osm(bbox, out_path: Path, max_retries=6, min_bytes=50_000):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    minlon, minlat, maxlon, maxlat = bbox
    qs = f"bbox={minlon:.7f},{minlat:.7f},{maxlon:.7f},{maxlat:.7f}"
    ua = {"User-Agent": "ev-net-builder/1.1"}
    delay = 2.0
    for attempt in range(max_retries):
        url = f"{OVERPASS_SERVERS[attempt % len(OVERPASS_SERVERS)]}?{qs}"
        info(f"[INFO] Download attempt {attempt+1}/{max_retries}: {url}")
        try:
            req = Request(url, headers=ua)
            with urlopen(req, timeout=120) as resp, out_path.open("wb") as f:
                shutil.copyfileobj(resp, f)
        except (HTTPError, URLError) as e:
            info(f"[WARN] Overpass error: {e}. Retrying after {delay:.1f}s...")
            time.sleep(delay); delay *= 1.8
            continue
        size = out_path.stat().st_size
        if size < min_bytes:
            info(f"[WARN] OSM looks too small ({size} bytes). Retrying after {delay:.1f}s...")
            time.sleep(delay); delay *= 1.8
            continue
        info(f"[OK] Downloaded OSM: {size/1_000_000:.2f} MB")
        return
    die("Overpass download failed after retries (rate-limited or bbox too big). Try smaller radius/quantiles.")

def run_netconvert(netconvert_path, osm_path: Path, out_net: Path):
    info(f"[INFO] Running netconvert → {out_net}")
    cmd = [
        netconvert_path,
        "--osm-files", str(osm_path),
        "--keep-edges.by-vclass", "passenger",
        "--geometry.remove",
        "--remove-edges.isolated",
        "--no-internal-links",
        "--tls.discard-loaded",
        "--output-file", str(out_net),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if proc.stdout:
            last = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            if last: info("[netconvert] " + last[-1])
    except subprocess.CalledProcessError as e:
        print(e.stdout or "", file=sys.stderr)
        print(e.stderr or "", file=sys.stderr)
        die("netconvert failed.")
    if not out_net.exists() or out_net.stat().st_size < 10_000:
        die("netconvert produced no/too-small output.")

def patch_train_py(train_py: Path, new_name: str):
    if not train_py.exists(): die(f"Cannot find {train_py}")
    text = train_py.read_text(encoding="utf-8")
    pattern = r'(?m)^\s*NET_FILE\s*=\s*".*?"\s*$'
    replacement = f'NET_FILE = "{new_name}"'
    if re.search(pattern, text):
        new_text = re.sub(pattern, replacement, text)
    else:
        # Insert just after DATA_DIR assignment if NET_FILE not present
        new_text = re.sub(r'(?m)^(DATA_DIR\s*=\s*.*)$',
                          r'\1\nNET_FILE = "' + new_name + '"',
                          text, count=1)
    if new_text != text:
        backup = train_py.with_suffix(train_py.suffix + ".bak")
        backup.write_text(text, encoding="utf-8")
        train_py.write_text(new_text, encoding="utf-8")
        info(f"[OK] Patched rl/train.py (backup: {backup.name})")
    else:
        info("[WARN] Could not patch rl/train.py automatically. Please set NET_FILE manually.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius-km", type=float, default=None,
                    help="Use circular bbox around median lat/lon (km). If set, overrides quantiles.")
    ap.add_argument("--qlo", type=float, default=0.02, help="Lower quantile for clipping")
    ap.add_argument("--qhi", type=float, default=0.98, help="Upper quantile for clipping")
    ap.add_argument("--osm-file", type=str, default=None, help="Use local .osm/.pbf instead of downloading")
    args = ap.parse_args()

    info(f"[BOOT] Project root: {PROJECT_ROOT}")
    if not DATA_CSV.exists(): die(f"Expected {DATA_CSV}")

    lats, lons = read_coords(DATA_CSV)
    info(f"[INFO] Stations parsed: {len(lats)}")

    if args.radius_km:
        lat_med = quantiles(lats, 0.5); lon_med = quantiles(lons, 0.5)
        bbox = bbox_from_radius(lat_med, lon_med, args.radius_km)
        info(f"[INFO] Using radius bbox ({args.radius_km} km) around median: "
             f"{bbox[0]:.7f},{bbox[1]:.7f},{bbox[2]:.7f},{bbox[3]:.7f}")
    else:
        bbox = bbox_from_quantiles(lats, lons, args.qlo, args.qhi, pad_km=1.0)
        info(f"[INFO] Using quantile bbox (qlo={args.qlo}, qhi={args.qhi}) + 1km pad: "
             f"{bbox[0]:.7f},{bbox[1]:.7f},{bbox[2]:.7f},{bbox[3]:.7f}")

    check_netconvert_on_path()

    if args.osm_file:
        osm_path = Path(args.osm_file)
        if not osm_path.exists(): die(f"--osm-file not found: {osm_path}")
        info(f"[INFO] Using local OSM file: {osm_path}")
    else:
        osm_path = OUT_OSM
        download_osm(bbox, osm_path)

    run_netconvert(shutil.which("netconvert.exe" if os.name == "nt" else "netconvert"), osm_path, OUT_NET)
    patch_train_py(TRAIN_PY, OUT_NET.name)
    info(f"[DONE] New net ready: {OUT_NET}")
    info("[NEXT] Probe with SUMO:\n"
         "       set DRY_NO_SUMO=\n"
         "       set DRY_RUN_STEPS=100\n"
         "       python -m rl.train")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr); sys.exit(1)
