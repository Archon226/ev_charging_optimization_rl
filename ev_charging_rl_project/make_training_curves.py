# make_training_curves.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rolling(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(window=w, min_periods=max(5, w//5), center=False).mean()

def pick_col(df: pd.DataFrame, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kpi", required=True, type=Path, help="Path to training kpi_episodes.csv")
    ap.add_argument("--out", required=True, type=Path, help="Output directory")
    ap.add_argument("--window", type=int, default=50, help="Rolling window (episodes)")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.kpi)

    # Your file uses these exact names; keep robust fallbacks just in case
    col_min   = pick_col(df, "total_minutes", "minutes", "episode_minutes")
    col_cost  = pick_col(df, "total_cost_gbp", "cost", "episode_cost_gbp")
    col_succ  = pick_col(df, "success", "episode_success", "succ")
    col_chg   = pick_col(df, "charge_events", "charges")

    if col_cost is None or col_min is None:
        raise RuntimeError("Could not find minutes/cost columns. "
                           "Expected total_minutes and total_cost_gbp in kpi_episodes.csv")

    # Ensure an episode index
    if "episode" not in df.columns:
        df["episode"] = np.arange(len(df))

    df = df.sort_values("episode").reset_index(drop=True)

    # Rolling means
    w = args.window
    r_min  = rolling(df[col_min].astype(float),  w)
    r_cost = rolling(df[col_cost].astype(float), w)
    r_succ = rolling(df[col_succ].astype(float), w) if col_succ else None
    r_chg  = rolling(df[col_chg].astype(float),  w) if col_chg  else None

    # Plot minutes
    fig = plt.figure(figsize=(7,3.5), dpi=140)
    plt.plot(df["episode"], r_min)
    plt.xlabel("Episode")
    plt.ylabel("Minutes (rolling mean)")
    plt.title(f"Training curve — Minutes (window={w})")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / f"train_minutes_w{w}.png")
    plt.close(fig)

    # Plot cost (£)
    fig = plt.figure(figsize=(7,3.5), dpi=140)
    plt.plot(df["episode"], r_cost)
    plt.xlabel("Episode")
    plt.ylabel("£ Cost (rolling mean)")
    plt.title(f"Training curve — Cost (window={w})")
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / f"train_cost_w{w}.png")
    plt.close(fig)

    # Plot success rate
    if r_succ is not None:
        fig = plt.figure(figsize=(7,3.5), dpi=140)
        plt.plot(df["episode"], r_succ)
        plt.xlabel("Episode")
        plt.ylabel("Success rate (rolling mean)")
        plt.title(f"Training curve — Success (window={w})")
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / f"train_success_w{w}.png")
        plt.close(fig)

    # Plot charge events (optional)
    if r_chg is not None:
        fig = plt.figure(figsize=(7,3.5), dpi=140)
        plt.plot(df["episode"], r_chg)
        plt.xlabel("Episode")
        plt.ylabel("Charge events (rolling mean)")
        plt.title(f"Training curve — Charges (window={w})")
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(out_dir / f"train_charges_w{w}.png")
        plt.close(fig)

    print("[make_training_curves] wrote figures to:", out_dir)

if __name__ == "__main__":
    main()
