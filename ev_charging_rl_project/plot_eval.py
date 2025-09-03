# plot_eval.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def moving_avg(series, k):
    if k <= 1:
        return series
    return series.rolling(window=k, min_periods=1).mean()

def plot_series(ax, x, y, y_label, title, roll_k=None):
    ax.plot(x, y, linewidth=1, alpha=0.5, label="per-episode")
    if roll_k and roll_k > 1:
        ax.plot(x, moving_avg(y, roll_k), linewidth=2, label=f"rolling mean (k={roll_k})")
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")

def main():
    p = argparse.ArgumentParser(description="Plot PPO EV RL KPI CSVs")
    p.add_argument("--csv", required=True, type=Path, help="Path to kpi_episodes.csv or eval_kpi_episodes.csv")
    p.add_argument("--outdir", type=Path, default=None, help="Where to save PNGs (default: alongside CSV)")
    p.add_argument("--rolling", type=int, default=20, help="Rolling window size for smoothing")
    p.add_argument("--title", type=str, default="", help="Optional title prefix for plots")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    # robust column names
    # training CSV: timesteps, episode_steps, total_minutes, total_cost_gbp, success, stranded, charge_events
    # eval CSV:     episode,  episode_steps, total_minutes, total_cost_gbp, success, stranded, charge_events
    if "episode" in df.columns:
        ep = df["episode"].astype(int)
    else:
        # synthesize episode index from row order
        ep = pd.Series(range(1, len(df) + 1))

    minutes = df.get("total_minutes", pd.Series([float("nan")] * len(df)))
    cost    = df.get("total_cost_gbp", pd.Series([float("nan")] * len(df)))
    success = df.get("success", pd.Series([0] * len(df))).astype(int)
    charges = df.get("charge_events", pd.Series([0] * len(df))).astype(int)

    outdir = args.outdir or args.csv.parent
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Total minutes
    fig1, ax1 = plt.subplots()
    plot_series(ax1, ep, minutes, "Total minutes", f"{args.title}Episode travel time", roll_k=args.rolling)
    fig1.savefig(outdir / "plot_minutes.png", dpi=160, bbox_inches="tight")

    # 2) Total cost
    fig2, ax2 = plt.subplots()
    plot_series(ax2, ep, cost, "Total cost (£)", f"{args.title}Episode charging+energy cost", roll_k=args.rolling)
    fig2.savefig(outdir / "plot_cost.png", dpi=160, bbox_inches="tight")

    # 3) Success rate (rolling)
    fig3, ax3 = plt.subplots()
    sr = moving_avg(success, args.rolling)
    ax3.plot(ep, sr, linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel(f"Rolling success rate (k={args.rolling})")
    ax3.set_ylim(0, 1)
    ax3.set_title(f"{args.title}Success rate (rolling)")
    fig3.savefig(outdir / "plot_success_rate.png", dpi=160, bbox_inches="tight")

    # 4) Charge events distribution
    fig4, ax4 = plt.subplots()
    ax4.hist(charges, bins=range(int(charges.min()), int(charges.max()) + 2))
    ax4.set_xlabel("Charge events per episode")
    ax4.set_ylabel("Count")
    ax4.set_title(f"{args.title}Charge events distribution")
    fig4.savefig(outdir / "plot_charge_events_hist.png", dpi=160, bbox_inches="tight")

    # 5) Cost vs Minutes scatter (colored by success, simple)
    fig5, ax5 = plt.subplots()
    ax5.scatter(minutes, cost, s=12, alpha=0.6)
    ax5.set_xlabel("Total minutes")
    ax5.set_ylabel("Total cost (£)")
    ax5.set_title(f"{args.title}Cost vs Minutes (each point = episode)")
    fig5.savefig(outdir / "plot_cost_vs_minutes.png", dpi=160, bbox_inches="tight")

    print(f"[plot] saved plots to: {outdir}")

if __name__ == "__main__":
    main()
