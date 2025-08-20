# tests/quick_validation.py
import os, random, pandas as pd
import sys, pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
# tests/validation.py
import os, random, math, importlib, pandas as pd

def hav_km(a_lat, a_lon, b_lat, b_lon):
    R = 6371.0088
    dlat = math.radians(b_lat - a_lat)
    dlon = math.radians(b_lon - a_lon)
    x = (math.sin(dlat/2)**2
         + math.cos(math.radians(a_lat))*math.cos(math.radians(b_lat))*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(x))

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_episodes_csv():
    ep_mod = importlib.import_module("rl.episodes")
    trips = ep_mod.load_episodes(os.path.join(PROJECT, "data", "simulated_users.csv"))
    return trips

def load_data_bundle():
    dl = importlib.import_module("utils.data_loader")
    return dl.load_all_data()  # stations, connectors, ev_metadata, pricing_* frames

def build_pricing(bundle):
    pr = importlib.import_module("utils.pricing")
    return pr.PricingCatalog(bundle.pricing_core, bundle.pricing_by_type, bundle.pricing_conditions)

def build_sim():
    from simulator.sumo_adapter import SumoSim
    net_path = os.path.join(PROJECT, "london_inner.net.xml")
    return SumoSim(net_path, prefer_time=True, fallback_speed_kph=30.0)

def resolve_env_ctor():
    try:
        mod = importlib.import_module("env.ev_env")
        return mod.EVChargingEnv, "env.ev_env"
    except Exception:
        mod = importlib.import_module("rl.ev_env")
        return mod.EVChargingEnv, "rl.ev_env"

def maybe_logging_wrapper(env):
    try:
        lw_mod = importlib.import_module("env.logging_wrapper")
        return lw_mod.LoggingWrapper(env, log_dir=os.path.join(PROJECT, "runs", "quick_check"), run_id="quick")
    except Exception:
        return env

def make_env(sim, bundle, trips):
    EnvCtor, where = resolve_env_ctor()
    print(f"[info] Using {where}.EVChargingEnv")

    pricing_catalog = None
    try:
        pricing_catalog = build_pricing(bundle)
    except Exception:
        pass

    # Try a few ctor signatures, ALWAYS passing episodes=trips
    # 1) (sim, stations, connectors, pricing_catalog, ev_metadata, episodes, top_k_default=?)
    try:
        return EnvCtor(
            sim=sim,
            stations=bundle.stations,
            connectors=bundle.connectors,
            pricing_catalog=pricing_catalog,
            ev_metadata=bundle.ev_metadata,
            episodes=trips,
            top_k_default=6
        ), "rich(pc)"
    except Exception as e1:
        print(f"[warn] rich(pc) ctor failed: {e1}")

    # 2) Some repos name it 'pricing'
    try:
        return EnvCtor(
            sim=sim,
            stations=bundle.stations,
            connectors=bundle.connectors,
            pricing=pricing_catalog,
            ev_metadata=bundle.ev_metadata,
            episodes=trips,
            top_k_default=6
        ), "rich(p)"
    except Exception as e2:
        print(f"[warn] rich(p) ctor failed: {e2}")

    # 3) Simple positional fallback: (sim, stations, connectors, pricing_or_none, ev_metadata_or_ev_spec, episodes)
    try:
        return EnvCtor(sim, bundle.stations, bundle.connectors, pricing_catalog, bundle.ev_metadata, trips), "simple-pos(ev_meta)"
    except Exception as e3:
        print(f"[warn] simple-pos(ev_meta) ctor failed: {e3}")

    # 4) Last resort: simple with a generic EV spec
    ev_spec = {
        "battery_kwh": 75.0,
        "eff_kwh_per_km": 0.18,
        "allowed_connectors": ("CCS", "Type2"),
        "max_dc_kw": 150.0,
        "max_ac_kw": 11.0
    }
    return EnvCtor(sim, bundle.stations, bundle.connectors, pricing_catalog, ev_spec, trips), "simple-pos(ev_spec)"

def route_sanity(sim, trips, n=10):
    print("\n[1] Route sanity check (route_km vs hav_km):")
    sample = random.sample(trips, min(n, len(trips)))
    for t in sample:
        rk, _ = sim.route_between_cached(t.origin[0], t.origin[1], t.dest[0], t.dest[1])
        hk = hav_km(t.origin[0], t.origin[1], t.dest[0], t.dest[1])
        print(f"  route_km={rk:.3f}  hav_km={hk:.3f}")

def run_episodes(env, trips, use_wrapper=True, n=20):
    if use_wrapper:
        env = maybe_logging_wrapper(env)
    drive_ct = charge_ct = 0
    zero_time_eps = 0

    import numpy as np
    for ep in range(n):
        trip = random.choice(trips)
        # Always pass a specific trip via options to satisfy envs that require it
        obs, info = env.reset(options={"trip": trip})
        done = False
        steps = 0
        while not done and steps < 10:
            a_n = getattr(env.action_space, "n", None)
            action = int(np.random.randint(0, a_n)) if a_n else 0
            obs, reward, done, truncated, info = env.step(action)
            steps += 1

    if use_wrapper:
        out_dir = os.path.join(PROJECT, "runs", "quick_check")
        ep_csv = os.path.join(out_dir, "episodes.csv")
        ev_csv = os.path.join(out_dir, "events.csv")
        if os.path.exists(ep_csv) and os.path.exists(ev_csv):
            eps = pd.read_csv(ep_csv)
            evs = pd.read_csv(ev_csv)
            zero_time_eps = int((eps.get("total_time_s", 0) == 0).sum())
            drive_ct = int((evs.get("event_type") == "drive").sum())
            charge_ct = int((evs.get("event_type") == "charge").sum())
    print("\n[2] Episode stats")
    print(f"  episodes with total_time_s == 0 : {zero_time_eps}")
    print(f"  total drive events              : {drive_ct}")
    print(f"  total charge events             : {charge_ct}")

if __name__ == "__main__":
    trips = load_episodes_csv()
    bundle = load_data_bundle()
    sim = build_sim()
    route_sanity(sim, trips, n=10)
    env, mode = make_env(sim, bundle, trips)
    run_episodes(env, trips, use_wrapper=True)
