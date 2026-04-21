import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import os

DA_THRESHOLD = 0.50

MODEL_COLORS = {
    "baseline": {"bg": "#E6F1FB", "fg": "#185FA5"},
    "regime":   {"bg": "#FAEEDA", "fg": "#854F0B"},
}
DA_HIGH   = "#0F6E56"
DA_LOW    = "#993C1D"
HEADER_BG = "#F1EFE8"
ROW_BG    = "#FFFFFF"
BORDER    = "#000000"

def load(path):
    df = pd.read_csv(path)
    return df[["asset", "model_type", "test_mse", "test_mae", "test_da"]].copy()

def fmt_cell(col, val):
    if "da" in col:
        return f"{val*100:.1f}%"
    return f"{val:.4f}"

def render(df, out_path):
    col_labels = ["Asset", "Model", "MSE", "MAE", "Dir. acc."]
    data_cols  = ["asset", "model_type", "test_mse", "test_mae", "test_da"]
    col_widths = [1.2, 1.1, 0.9, 0.9, 1.0]
    n_rows  = len(df)
    total_w = sum(col_widths)
    row_h   = 0.42
    header_h = 0.46
    fig_h   = header_h + n_rows * row_h + 0.28

    fig, ax = plt.subplots(figsize=(total_w, fig_h))
    ax.set_xlim(0, total_w); ax.set_ylim(0, fig_h); ax.axis("off")

    xs  = [0] + list(np.cumsum(col_widths))
    mid = [(xs[i] + xs[i+1]) / 2 for i in range(len(col_labels))]

    def row_y(r):
        return fig_h - header_h - (r + 1) * row_h

    # ── header row ────────────────────────────────────────────────────────────
    ch_y = fig_h - header_h
    # full header background
    ax.add_patch(plt.Rectangle((xs[0], ch_y), total_w, header_h,
                               fc=HEADER_BG, ec=BORDER, lw=0.8))
    for c, label in enumerate(col_labels):
        # vertical dividers
        if c > 0:
            ax.plot([xs[c], xs[c]], [ch_y, ch_y + header_h], color=BORDER, lw=0.6)
        tx = xs[c] + 0.12 if c < 2 else mid[c]
        ax.text(tx, ch_y + header_h / 2, label,
                ha="left" if c < 2 else "center", va="center",
                fontsize=8, color="#1a1a18", fontweight="bold")

    # ── data rows ─────────────────────────────────────────────────────────────
    prev_asset = None
    for r, (_, row) in enumerate(df.iterrows()):
        y = row_y(r)
        ax.add_patch(plt.Rectangle((xs[0], y), total_w, row_h,
                                   fc=ROW_BG, ec=BORDER, lw=0.6))
        for c in range(1, len(col_labels)):
            ax.plot([xs[c], xs[c]], [y, y + row_h], color=BORDER, lw=0.4)

        for c, col in enumerate(data_cols):
            val = row[col]
            tx = xs[c] + 0.12 if c < 2 else mid[c]
            ty = y + row_h / 2

            if col == "asset":
                ax.text(tx, ty, val if val != prev_asset else "",
                        ha="left", va="center", fontsize=8,
                        color="#1a1a18", fontweight="bold")
            elif col == "model_type":
                mc = MODEL_COLORS.get(val, {"bg": "#F1EFE8", "fg": "#444441"})
                bw, bh = col_widths[c] - 0.30, row_h - 0.16
                ax.add_patch(mpatches.FancyBboxPatch(
                    (xs[c] + 0.08, ty - bh / 2), bw, bh,
                    boxstyle="round,pad=0.04", fc=mc["bg"], ec="none"))
                ax.text(xs[c] + 0.08 + bw / 2, ty, val,
                        ha="center", va="center", fontsize=7.5,
                        color=mc["fg"], fontweight="bold")
            else:
                text  = fmt_cell(col, float(val))
                color = (DA_HIGH if float(val) >= DA_THRESHOLD else DA_LOW) if "da" in col else "#1a1a18"
                fw    = "bold" if "da" in col else "normal"
                ax.text(tx, ty, text, ha="center", va="center",
                        fontsize=8, color=color, fontweight=fw)

        prev_asset = row["asset"]

    # ── legend ────────────────────────────────────────────────────────────────
    legend_y = row_y(n_rows - 1) - 0.20
    lx = xs[0] + 0.12
    for color, label in [(DA_HIGH, f"Dir. acc. ≥ {int(DA_THRESHOLD*100)}%"),
                         (DA_LOW,  f"Dir. acc. < {int(DA_THRESHOLD*100)}%")]:
        ax.add_patch(plt.Rectangle((lx, legend_y + 0.04), 0.13, 0.13, fc=color, ec="none"))
        ax.text(lx + 0.19, legend_y + 0.10, label,
                ha="left", va="center", fontsize=7, color="#5F5E5A")
        lx += 1.6

    plt.tight_layout(pad=0.1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved -> {out_path}")

def main():
    print("Rendering LSTM forecast performance summary...")
    in_path  = "outputs/lstm_model_output/summary_metrics.csv"
    out_path = "outputs/figures/lstm_forecast_performance_summary.png"
    render(load(in_path), out_path)

if __name__ == "__main__":
    main()