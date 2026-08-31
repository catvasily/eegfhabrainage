"""Plot calibration summary metrics from a classification summary CSV.

Creates a 5x4 grid of subplots with rows for physicians and columns for
(`Physician` suffix, `ignore_confidence`) combinations. Each subplot shows
grouped bars for AUC and 1-BS across calibration methods.

Usage:
    python plot_calibration_summary.py [--csv CSV_PATH] [--out PNG_PATH]

Arguments:
    --csv
        Input classification summary CSV file.
        Default: classification_results_summary.csv
    --out
        Output PNG file path.
        Default: calibration_summary.png

Example:
    python plot_calibration_summary.py --csv classification_results_summary_Abnormal_260602.csv \
        --out calibration_summary.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROW_PHYSICIANS = ["Athina", "Eleni", "Maria", "Sophia", "Zoe"]
YLIM = [0.1,1.0]

# (column title, Physician suffix, ignore_confidence value)
COLUMN_SPECS = [
    ("Conf Model / Conf Input", "Conf", False),
    ("Conf Model / All Input", "Conf", True),
    ("All Model / Conf Input", "All", False),
    ("All Model / All Input", "All", True),
]

CALIBRATION_ORDER = ["none", "platt", "betacal", "isotonic"]
CALIBRATION_LABELS = ["None", "Platt", "Beta", "Iso"]


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed boolean/text column values to bool series."""
    if series.dtype == bool:
        return series

    text = series.astype(str).str.strip().str.lower()
    return text.isin({"1", "true", "t", "yes", "y"})


def _prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load CSV and normalize columns needed for plotting."""
    df = pd.read_csv(csv_path)

    required = {"Physician", "ignore_confidence", "Calibration", "AUC"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    if "1-BS" in df.columns:
        one_minus_bs_col = "1-BS"
        df[one_minus_bs_col] = pd.to_numeric(df[one_minus_bs_col], errors="coerce")
    elif "BS" in df.columns:
        one_minus_bs_col = "1-BS"
        bs_numeric = pd.to_numeric(df["BS"], errors="coerce")
        df[one_minus_bs_col] = 1.0 - bs_numeric
    else:
        raise ValueError('CSV must contain either "1-BS" or "BS" column.')

    df["AUC"] = pd.to_numeric(df["AUC"], errors="coerce")
    df["ignore_confidence"] = _to_bool_series(df["ignore_confidence"])
    df["Physician"] = df["Physician"].astype(str)
    df["Calibration"] = df["Calibration"].astype(str).str.strip().str.lower()

    return df


def _subset_for_subplot(
    df: pd.DataFrame,
    physician_prefix: str,
    physician_suffix: str,
    ignore_confidence: bool,
) -> pd.DataFrame:
    """Select records for one subplot by prefix/suffix and ignore_confidence."""
    mask = (
        df["Physician"].str.startswith(physician_prefix)
        & df["Physician"].str.endswith(physician_suffix)
        & (df["ignore_confidence"] == ignore_confidence)
    )
    return df.loc[mask].copy()


def _metric_arrays(df_sub: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return AUC and 1-BS arrays ordered by calibration method."""
    if df_sub.empty:
        return np.full(len(CALIBRATION_ORDER), np.nan), np.full(len(CALIBRATION_ORDER), np.nan)

    grouped = (
        df_sub.groupby("Calibration", as_index=False)[["AUC", "1-BS"]]
        .mean(numeric_only=True)
        .set_index("Calibration")
    )

    auc_vals = np.array([grouped["AUC"].get(c, np.nan) for c in CALIBRATION_ORDER], dtype=float)
    one_minus_bs_vals = np.array([grouped["1-BS"].get(c, np.nan) for c in CALIBRATION_ORDER], dtype=float)
    return auc_vals, one_minus_bs_vals


def plot_calibration_summary(csv_path: Path, out_path: Path) -> Path:
    """Create and save the 5x4 calibration summary figure."""
    df = _prepare_dataframe(csv_path)

    fig, axes = plt.subplots(5, 4, figsize=(18, 20), sharex=True, sharey=True)

    x = np.arange(len(CALIBRATION_LABELS), dtype=float)
    bar_width = 0.36

    for row_idx, physician_prefix in enumerate(ROW_PHYSICIANS):
        for col_idx, (col_title, suffix, ignore_conf) in enumerate(COLUMN_SPECS):
            ax = axes[row_idx, col_idx]
            subset = _subset_for_subplot(
                df,
                physician_prefix=physician_prefix,
                physician_suffix=suffix,
                ignore_confidence=ignore_conf,
            )
            auc_vals, one_minus_bs_vals = _metric_arrays(subset)

            ax.bar(x - bar_width / 2, auc_vals, width=bar_width, label="AUC", color="#4C78A8")
            ax.bar(x + bar_width / 2, one_minus_bs_vals, width=bar_width, label="1-BS", color="#F58518")

            ax.set_xticks(x)
            ax.set_xticklabels(CALIBRATION_LABELS, rotation=0)
            ax.tick_params(axis="x", labelbottom=True)
            ax.set_ylim(*YLIM)
            ax.grid(axis="y", linestyle="--", alpha=0.35)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=11)

            if col_idx == 0:
                ax.set_ylabel(f"{physician_prefix}\nScore", fontsize=10)

            if subset.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="0.4",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, loc="upper left", frameon=False)
    fig.suptitle("Calibration Summary by Physician and Confidence Setting", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.975])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Plot calibration summary 5x4 subplot grid.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("classification_results_summary.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("calibration_summary.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    out_path = plot_calibration_summary(args.csv, args.out)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
