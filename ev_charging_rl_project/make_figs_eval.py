# make_figs_eval.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def p95(x: pd.Series) -> float:
    return float(np.nanpercentile(x.to_numpy(dtype=float), 95))

def safe_col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    return df[name] if name in df.columns else pd.Series([default]*len(df))

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = {
        "episodes": len(df),
        "success_rate": float(df["success"].mean()) if "success" in df else np.nan,
        "strand_rate": float(df["stranded"].mean()) if "stranded" in df else np.nan,
        "avg_minutes": float(df["total_minutes"].mean()),
        "p95_minutes": p95(df["total_minutes"]),
        "avg_cost_gbp": float(df["total_cost_gbp"].mean()),
        "p95_cost_gbp": p95(df["total_cost_gbp"]),
        "avg_charges": float(df["charge_events"].mean()) if "charge_events" in df else np.nan,
        "violations_repeat_total": int(df["violations_repeat"].sum()) if "violations_repeat" in df else 0,
        "violations_overlimit_total": int(df["violations_overlimit"].sum()) if "violations_overlimit" in df else 0,
        "violations_cooldown_total": int(df["violations_cooldown"].sum()) if "violations_cooldown" in df else 0,
    }
    # Optional diagnostics from Phase 5
    if "progress_km" in df: out["avg_progress_km"] = float(df["progress_km"].mean())
    if "shaping_time" in df: out["avg_shaping_time"] = float(df["shaping_time"].mean())
    if "penalty_micro_charge" in df: out["micro_charge_hits"] = int((df["penalty_micro_charge"] > 0).sum())
    if "penalty_idle" in df: out["idle_hits"] = int((df["penalty_idle"] > 0).sum())
    return pd.DataFrame([out])

def minutes_vs_cost_plot(df: pd.DataFrame, out_png: Path, title: str):
    mean_min = df["total_minutes"].mean()
    mean_cost = df["total_cost_gbp"].mean()
    p95_min = p95(df["total_minutes"])
    p95_cost = p95(df["total_cost_gbp"])

    fig = plt.figure(figsize=(6,5), dpi=140)
    plt.errorbar([mean_min], [mean_cost],
                 xerr=[[0],[p95_min-mean_min]], yerr=[[0],[p95_cost-mean_cost]],
                 fmt='o')
    plt.xlabel("Minutes (mean, whisker to p95)")
    plt.ylabel("£ Cost (mean, whisker to p95)")
    plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def hist_charge_events(df: pd.DataFrame, out_png: Path, title: str):
    if "charge_events" not in df: return
    fig = plt.figure(figsize=(6,4), dpi=140)
    plt.hist(df["charge_events"], bins=range(int(df["charge_events"].max())+2), align='left', rwidth=0.85)
    plt.xlabel("Charge events per trip")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def box_detour(df: pd.DataFrame, out_png: Path, title: str):
    if "detour_minutes" not in df: return
    fig = plt.figure(figsize=(5,4), dpi=140)
    plt.boxplot(df["detour_minutes"].to_numpy(dtype=float), vert=True, labels=["detour_min"])
    plt.ylabel("Detour minutes")
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

def maybe_low_soc_slice(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = [c for c in df.columns if c.lower() in ("start_soc","start_soc_pct","soc_start","soc0")]
    if not cols: return None
    c = cols[0]
    s = df[c]
    # Assume [0,100] if values look like percents; else [0,1]
    if s.dropna().max() > 1.5:
        return df[s <= 15.0]
    else:
        return df[s <= 0.15]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpi", required=True, type=Path, help="Path to eval_kpi_episodes.csv")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for figures and summaries")
    ap.add_argument("--title", default="Evaluation (route_time)", help="Title for plots")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.kpi)

    # Overall summary (R1–R3)
    overall = summarise(df)
    overall.to_csv(out_dir / "summary_overall.csv", index=False)

    # Plots
    minutes_vs_cost_plot(df, out_dir / "minutes_vs_cost.png", args.title)
    hist_charge_events(df, out_dir / "hist_charge_events.png", "Charge events per trip")
    box_detour(df, out_dir / "box_detour_minutes.png", "Detour minutes distribution")

    # Optional: low-SoC slice (R1 robustness)
    low = maybe_low_soc_slice(df)
    if low is not None and len(low) > 0:
        low_summary = summarise(low)
        low_summary.to_csv(out_dir / "summary_low_soc.csv", index=False)
        minutes_vs_cost_plot(low, out_dir / "minutes_vs_cost_low_soc.png", args.title + " — Low SoC")

    # Print quick console summary
    print("[make_figs_eval] wrote:")
    print("  -", out_dir / "summary_overall.csv")
    if low is not None and len(low) > 0:
        print("  -", out_dir / "summary_low_soc.csv")
    print("  -", out_dir / "minutes_vs_cost.png")
    print("  -", out_dir / "hist_charge_events.png")
    if "detour_minutes" in df.columns:
        print("  -", out_dir / "box_detour_minutes.png")

if __name__ == "__main__":
    main()
