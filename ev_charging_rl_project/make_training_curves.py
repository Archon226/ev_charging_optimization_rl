# make_training_curves.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rolling(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=max(5, w//5), center=False).mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpi", required=True, type=Path, help="Path to training kpi_episodes.csv")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument("--window", type=int, default=50, help="Rolling window (episodes)")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.kpi)
    # Expect columns: episode, minutes, cost, success (bool/int). Your file may name them slightly differently.
    # Try to align common variants:
    col_min = "minutes" if "minutes" in df.columns else "total_minutes"
    col_cost = "cost" if "cost" in df.columns else ("total_cost_gbp" if "total_cost_gbp" in df.columns else None)
    col_succ = "success" if "success" in df.columns else None

    if col_cost is None:
        raise RuntimeError("Could not find cost column; expected 'cost' or 'total_cost_gbp' in kpi_episodes.csv")

    if "episode" not in df.columns:
        df["episode"] = np.arange(len(df))

    df = df.sort_values("episode").reset_index(drop=True)

    # Rolling means
    w = args.window
    r_min = rolling(df[col_min].astype(float), w) if col_min in df else None
    r_cost = rolling(df[col_cost].astype(float), w)
    r_succ = rolling(df[col_succ].astype(float), w) if col_succ is not None else None

    # Plot minutes
    if r_min is not None:
        fig = plt.figure(figsize=(7,3.5), dpi=140)
        plt.plot(df["episode"], r_min)
        plt.xlabel("Episode")
        plt.ylabel("Minutes (rolling mean)")
        plt.title(f"Training curve — Minutes (window={w})")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / f"train_minutes_w{w}.png")
        plt.close(fig)

    # Plot cost
    fig = plt.figure(figsize=(7,3.5), dpi=140)
    plt.plot(df["episode"], r_cost)
    plt.xlabel("Episode")
    plt.ylabel("£ Cost (rolling mean)")
    plt.title(f"Training curve — Cost (window={w})")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / f"train_cost_w{w}.png")
    plt.close(fig)

    # Plot success
    if r_succ is not None:
        fig = plt.figure(figsize=(7,3.5), dpi=140)
        plt.plot(df["episode"], r_succ)
        plt.xlabel("Episode")
        plt.ylabel("Success rate (rolling mean)")
        plt.title(f"Training curve — Success (window={w})")
        plt.ylim(0,1)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / f"train_success_w{w}.png")
        plt.close(fig)

    print("[make_training_curves] wrote figures to:", out_dir)

if __name__ == "__main__":
    main()
