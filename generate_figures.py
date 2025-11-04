import os
import math
import warnings
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="talk")

DATA_PATH = "msha_accidents.csv"
OUT_DIR = "figures"
DPI = 250


def ensure_outdir(path: str = OUT_DIR):
    os.makedirs(path, exist_ok=True)


def savefig(name: str):
    plt.tight_layout()
    outfile = os.path.join(OUT_DIR, name)
    plt.savefig(outfile, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved {outfile}")


def read_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file: {path}")

    df = pd.read_csv(path, low_memory=False)

    # Parse dates if present
    for col in ["ACCIDENT_DT", "ACCIDENT_DATE", "ACCIDENT_DATE_TIME", "ACCIDENT_TIME_DT"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Normalize a standard accident date column
    if "ACCIDENT_DT" in df.columns:
        df["ACCIDENT_DATE"] = pd.to_datetime(df["ACCIDENT_DT"], errors="coerce")
    elif "ACCIDENT_DATE" in df.columns:
        df["ACCIDENT_DATE"] = pd.to_datetime(df["ACCIDENT_DATE"], errors="coerce")

    # Attempt to derive HourOfDay
    hour = None
    if "ACCIDENT_TIME" in df.columns:
        s = df["ACCIDENT_TIME"].astype(str).str.zfill(4).str[:2]
        hour = pd.to_numeric(s, errors="coerce")
    elif "ACCIDENT_HOUR" in df.columns:
        hour = pd.to_numeric(df["ACCIDENT_HOUR"], errors="coerce")

    if hour is not None:
        df["HourOfDay"] = hour.clip(0, 23)

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    if "ACCIDENT_DATE" in df.columns:
        df["Month"] = df["ACCIDENT_DATE"].dt.month
        df["DayOfWeek"] = df["ACCIDENT_DATE"].dt.dayofweek
    else:
        df["Month"] = np.nan
        df["DayOfWeek"] = np.nan

    if "HourOfDay" not in df.columns:
        df["HourOfDay"] = np.nan

    # Cyclical encoding
    def cyclical(x, period):
        return np.sin(2 * np.pi * x / period), np.cos(2 * np.pi * x / period)

    for col, period, sin_name, cos_name in [
        ("Month", 12, "Month_Sin", "Month_Cos"),
        ("DayOfWeek", 7, "Day_Sin", "Day_Cos"),
        ("HourOfDay", 24, "Hour_Sin", "Hour_Cos"),
    ]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            sinv, cosv = cyclical(s, period)
            df[sin_name] = sinv
            df[cos_name] = cosv

    return df


# ---------- Plot helpers ----------

def bar_with_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f"{int(height):,}",
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=10, rotation=0)


def plot_target_distribution(df: pd.DataFrame):
    target_col = None
    for cand in ["DEGREE_INJURY", "INJURY_SEVERITY", "SEVERITY"]:
        if cand in df.columns:
            target_col = cand
            break
    if target_col is None:
        print("Target column not found; skipping target distribution plot.")
        return

    plt.figure(figsize=(10, 6))
    order = df[target_col].value_counts().index
    ax = sns.countplot(data=df, x=target_col, order=order, palette="Blues_d")
    ax.set_title("Injury Severity Distribution")
    ax.set_xlabel("Severity Class")
    ax.set_ylabel("Count")
    plt.xticks(rotation=20)
    bar_with_labels(ax)
    savefig("01_target_class_distribution.png")


def plot_days_lost_distribution(df: pd.DataFrame):
    if "DAYS_LOST" not in df.columns:
        print("DAYS_LOST not found; skipping days lost distribution plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.histplot(df["DAYS_LOST"].dropna().clip(upper=df["DAYS_LOST"].quantile(0.99)), bins=50, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Days Lost (clipped 99th pct)")

    sns.histplot(np.log1p(df["DAYS_LOST"].dropna()), bins=50, ax=axes[1], color="#55A868")
    axes[1].set_title("log1p(Days Lost)")

    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    savefig("02_days_lost_distribution.png")


def plot_experience_histograms(df: pd.DataFrame):
    cols = [c for c in ["TOT_EXPER", "MINE_EXPER", "JOB_EXPER"] if c in df.columns]
    if not cols:
        print("Experience columns not found; skipping experience histograms.")
        return

    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna().clip(upper=df[col].quantile(0.99)), bins=40, ax=ax, color="#C44E52")
        ax.set_title(f"{col} (clipped 99th pct)")
        ax.set_xlabel("")
        ax.set_ylabel("Count")

    savefig("03_experience_histograms.png")


def plot_categorical_tops(df: pd.DataFrame, topn: int = 10):
    cats = [c for c in ["ACCIDENT_TYPE", "CLASSIFICATION", "SUBUNIT"] if c in df.columns]
    if not cats:
        print("Categorical columns not found; skipping categorical bar charts.")
        return

    n = len(cats)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, cats):
        vc = df[col].value_counts().head(topn)
        sns.barplot(x=vc.values, y=vc.index, ax=ax, palette="viridis")
        ax.set_title(f"Top {topn} {col}")
        ax.set_xlabel("Count")
        ax.set_ylabel("")
        for i, v in enumerate(vc.values):
            ax.text(v, i, f" {int(v):,}", va='center')

    savefig("04_categorical_top10.png")


def plot_coal_metal(df: pd.DataFrame):
    if "COAL_METAL_IND" not in df.columns:
        print("COAL_METAL_IND not found; skipping coal/metal split chart.")
        return

    plt.figure(figsize=(8, 6))
    order = df["COAL_METAL_IND"].value_counts().index
    ax = sns.countplot(data=df, x="COAL_METAL_IND", order=order, palette="Set2")
    ax.set_title("Incidents by COAL_METAL_IND")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Count")
    bar_with_labels(ax)
    savefig("05_coal_metal_split.png")


def plot_missingness(df: pd.DataFrame):
    miss_pct = df.isna().mean().sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]
    if miss_pct.empty:
        print("No missing values; skipping missingness plot.")
        return

    plt.figure(figsize=(8, max(4, len(miss_pct) * 0.25)))
    ax = sns.barplot(x=miss_pct.values * 100, y=miss_pct.index, palette="magma")
    ax.set_title("Missing Values (%)")
    ax.set_xlabel("Percent Missing")
    ax.set_ylabel("")
    for i, v in enumerate(miss_pct.values * 100):
        ax.text(v, i, f" {v:.1f}%", va='center')
    savefig("06_missingness.png")


def plot_correlations(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Keep a manageable set
    keep = [c for c in num_cols if c in [
        "DAYS_LOST", "TOT_EXPER", "MINE_EXPER", "JOB_EXPER", "NO_INJURIES", "DAYS_RESTRICT",
        "Month_Sin", "Month_Cos", "Day_Sin", "Day_Cos", "Hour_Sin", "Hour_Cos"
    ]]
    if len(keep) < 3:
        print("Insufficient numeric features for correlation heatmap; skipping.")
        return

    corr = df[keep].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Heatmap (Selected Features)")
    savefig("07_corr_heatmap.png")


def plot_temporal_trends(df: pd.DataFrame):
    if "ACCIDENT_DATE" not in df.columns:
        print("ACCIDENT_DATE not found; skipping temporal trends plot.")
        return

    # Monthly counts
    monthly = df.set_index("ACCIDENT_DATE").resample("M").size()

    # Day of week & hour
    dow = df["DayOfWeek"].value_counts().sort_index()
    hour = df["HourOfDay"].value_counts().sort_index()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    axes[0].plot(monthly.index, monthly.values, color="#4C72B0")
    axes[0].set_title("Incidents Over Time (Monthly)")
    axes[0].set_ylabel("Count")

    axes[1].bar(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], dow.values, color="#55A868")
    axes[1].set_title("Incidents by Day of Week")

    axes[2].bar(hour.index, hour.values, color="#C44E52")
    axes[2].set_title("Incidents by Hour of Day")
    axes[2].set_xlabel("Hour")

    savefig("08_temporal_trends.png")


def plot_cyclical_demo():
    # Visualize cyclical encoding for months
    months = np.arange(1, 13)
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(months, month_sin, label="sin", color="#4C72B0")
    axes[0].plot(months, month_cos, label="cos", color="#55A868")
    axes[0].set_title("Cyclical Encoding (Wave View)")
    axes[0].set_xticks(months)
    axes[0].legend()

    axes[1].scatter(month_cos, month_sin, c=months, cmap="viridis", s=120)
    for i, m in enumerate(months):
        axes[1].annotate(str(m), (month_cos[i], month_sin[i]))
    axes[1].set_title("Cyclical Encoding (Unit Circle)")
    axes[1].set_xlabel("cos")
    axes[1].set_ylabel("sin")
    axes[1].axhline(0, color='gray', lw=0.5)
    axes[1].axvline(0, color='gray', lw=0.5)
    axes[1].set_aspect('equal', 'box')

    savefig("09_cyclical_encoding_demo.png")


if __name__ == "__main__":
    ensure_outdir()
    df = read_data(DATA_PATH)
    df = add_temporal_features(df)

    plot_target_distribution(df)
    plot_days_lost_distribution(df)
    plot_experience_histograms(df)
    plot_categorical_tops(df)
    plot_coal_metal(df)
    plot_missingness(df)
    plot_correlations(df)
    plot_temporal_trends(df)
    plot_cyclical_demo()

    print("All figures generated in ./figures")
