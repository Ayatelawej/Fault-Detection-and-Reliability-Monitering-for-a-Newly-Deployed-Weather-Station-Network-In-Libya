from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.rules.config import RESULT_FIGURES_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_FIGURES_PATH = PROJECT_ROOT / RESULT_FIGURES_DIR
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#6C6C6C",
}
TIER_ORDER = ["benign", "single_channel", "inconclusive", "supported"]
LOCALIZED_STATIONS = ["IJANZO2", "IJABAL16", "IJANZO3", "INUQAT8"]


def _standardize_time(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ["time_utc", "hour_utc", "start_hour", "end_hour", "start", "end"]:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], utc=True)
    if "station_id" in result.columns:
        result["station_id"] = result["station_id"].astype(str)
    return result


def _reason_value(text: object, key: str) -> str:
    match = re.search(rf"{re.escape(key)}=([^|]+)", str(text))
    return "" if match is None else match.group(1)


def _reason_float(text: object, key: str) -> float:
    value = _reason_value(text, key)
    try:
        return float(value)
    except ValueError:
        return np.nan


def confirmed_offset_rows(reviewed: pd.DataFrame, channel: str | None = None) -> pd.DataFrame:
    frame = _standardize_time(reviewed)
    selected = frame.loc[frame["label"].eq("calibration_offset")].copy()
    selected["channel"] = selected["reasons"].map(lambda value: _reason_value(value, "channel"))
    selected["level_p50"] = selected["reasons"].map(lambda value: _reason_float(value, "level_p50"))
    if channel is not None:
        selected = selected.loc[selected["channel"].eq(channel)]
    return selected.reset_index(drop=True)


def pressure_offset_summary(
    external_features: pd.DataFrame,
    reviewed: pd.DataFrame,
) -> pd.DataFrame:
    frame = _standardize_time(external_features)
    stats = (
        frame.groupby("station_id")["r_pressure"]
        .agg(
            p25=lambda values: values.quantile(0.25),
            median="median",
            p75=lambda values: values.quantile(0.75),
            n="count",
        )
        .reset_index()
    )
    confirmed = confirmed_offset_rows(reviewed, "pressure")
    confirmed_levels = (
        confirmed.groupby("station_id")["level_p50"]
        .agg(
            confirmed_level_p50="median",
            confirmed_level_min="min",
            confirmed_level_max="max",
            confirmed_events="count",
        )
        .reset_index()
    )
    result = stats.merge(confirmed_levels, on="station_id", how="left")
    result["confirmed_offset"] = result["confirmed_events"].fillna(0).gt(0)
    return result.sort_values("median").reset_index(drop=True)


def solar_monthly_fleet_ratio(external_features: pd.DataFrame) -> pd.DataFrame:
    frame = _standardize_time(external_features)
    frame = frame.loc[frame["clear_sky_ratio"].notna()].copy()
    frame["month"] = frame["time_utc"].dt.strftime("%Y-%m")
    return (
        frame.groupby("month")["clear_sky_ratio"]
        .agg(fleet_median="median", p25=lambda values: values.quantile(0.25), p75=lambda values: values.quantile(0.75), n="count")
        .reset_index()
    )


def solar_station_clear_day_ratio(
    external_features: pd.DataFrame,
    top_fraction: float = 0.10,
) -> pd.DataFrame:
    frame = _standardize_time(external_features)
    frame = frame.loc[frame["clear_sky_ratio"].notna()].copy()
    frame["date"] = frame["time_utc"].dt.floor("D")
    daily = (
        frame.groupby(["station_id", "date"])["clear_sky_ratio"]
        .agg(daily_ratio="median", hours="count")
        .reset_index()
    )
    rows = []
    for station_id, group in daily.groupby("station_id", sort=True):
        clean = group.dropna(subset=["daily_ratio"])
        if clean.empty:
            rows.append({"station_id": station_id, "clear_day_ratio": np.nan, "n_days": 0})
            continue
        threshold = clean["daily_ratio"].quantile(1.0 - top_fraction)
        top = clean.loc[clean["daily_ratio"].ge(threshold)]
        rows.append(
            {
                "station_id": station_id,
                "clear_day_ratio": float(top["daily_ratio"].median()),
                "n_days": int(len(top)),
            }
        )
    return pd.DataFrame(rows).sort_values("clear_day_ratio").reset_index(drop=True)


def systemic_tier_counts(evidence: pd.DataFrame) -> pd.DataFrame:
    counts = evidence["support_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)
    return counts.rename_axis("support_tier").reset_index(name="episodes")


def spatial_external_pairs(
    external_features: pd.DataFrame,
    spatial_features: pd.DataFrame,
    reviewed: pd.DataFrame,
) -> pd.DataFrame:
    external = _standardize_time(external_features)
    spatial = _standardize_time(spatial_features)
    ext = (
        external.groupby("station_id")["r_pressure"]
        .median()
        .rename("external_median")
        .reset_index()
    )
    sp = (
        spatial.groupby("station_id")
        .agg(
            spatial_median=("spatial_offset_level_pressure", "median"),
            spatial_isolated=("spatial_isolated", "max"),
        )
        .reset_index()
    )
    confirmed = set(confirmed_offset_rows(reviewed, "pressure")["station_id"])
    pairs = ext.merge(sp, on="station_id", how="left")
    pairs["confirmed_offset"] = pairs["station_id"].isin(confirmed)
    pairs["spatial_isolated"] = pairs["spatial_isolated"].fillna(False).astype(bool)
    return pairs.sort_values("external_median").reset_index(drop=True)


def _overlap(left: pd.Series, right: pd.Series) -> bool:
    return (
        str(left["station_id"]) == str(right["station_id"])
        and pd.to_datetime(left["start_hour"], utc=True) <= pd.to_datetime(right["end_hour"], utc=True)
        and pd.to_datetime(right["start_hour"], utc=True) <= pd.to_datetime(left["end_hour"], utc=True)
    )


def localized_spatial_episodes(
    spatial_queue: pd.DataFrame,
    reviewed: pd.DataFrame,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    spatial = _standardize_time(spatial_queue)
    confirmed = confirmed_offset_rows(reviewed, "pressure")
    external = _standardize_time(external_features)
    rows = []
    for _, row in spatial.iterrows():
        overlaps = confirmed.loc[confirmed.apply(lambda candidate: _overlap(row, candidate), axis=1)]
        if not overlaps.empty:
            continue
        station_id = str(row["station_id"])
        start = pd.to_datetime(row["start_hour"], utc=True)
        end = pd.to_datetime(row["end_hour"], utc=True)
        window = external.loc[
            external["station_id"].eq(station_id)
            & external["time_utc"].between(start, end, inclusive="both")
        ]
        rows.append(
            {
                "station_id": station_id,
                "start_hour": start,
                "end_hour": end,
                "spatial_level_p50": _reason_float(row["reasons"], "level_p50"),
                "external_r_pressure_median": float(window["r_pressure"].median()),
            }
        )
    result = pd.DataFrame(rows)
    order = {station: position for position, station in enumerate(LOCALIZED_STATIONS)}
    result["order"] = result["station_id"].map(order)
    return result.sort_values("order").drop(columns="order").reset_index(drop=True)


def localized_pressure_series(
    merged: pd.DataFrame,
    external_features: pd.DataFrame,
    spatial_residuals: pd.DataFrame,
    episode: pd.Series,
) -> pd.DataFrame:
    merged_frame = _standardize_time(merged.rename(columns={"hour_utc": "time_utc"}))
    external = _standardize_time(external_features)
    spatial = _standardize_time(spatial_residuals)
    station_id = str(episode["station_id"])
    start = pd.to_datetime(episode["start_hour"], utc=True)
    end = pd.to_datetime(episode["end_hour"], utc=True)
    station = merged_frame.loc[
        merged_frame["station_id"].eq(station_id),
        ["station_id", "time_utc", "pressure_max_hpa"],
    ]
    frame = station.merge(
        external.loc[:, ["station_id", "time_utc", "r_pressure"]],
        on=["station_id", "time_utc"],
        how="left",
    )
    frame = frame.merge(
        spatial.loc[:, ["station_id", "time_utc", "r_spatial_pressure"]],
        on=["station_id", "time_utc"],
        how="left",
    )
    frame = frame.loc[frame["time_utc"].between(start, end, inclusive="both")].copy()
    frame["station_pressure"] = frame["pressure_max_hpa"]
    frame["healthy_neighbor_pressure"] = frame["station_pressure"] - frame["r_spatial_pressure"]
    frame["era5_pressure"] = frame["station_pressure"] - frame["r_pressure"]
    frame["date"] = frame["time_utc"].dt.floor("D")
    return (
        frame.groupby("date")[["station_pressure", "healthy_neighbor_pressure", "era5_pressure"]]
        .mean()
        .reset_index()
    )


def figure_path(name: str, output_dir: Path = RESULT_FIGURES_PATH) -> Path:
    return Path(output_dir) / name


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)


def plot_pressure_offsets(summary: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = summary.sort_values("median").reset_index(drop=True)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.5, 8.8))
    ax.axvspan(-6, -4, color=PALETTE["orange"], alpha=0.14, label="fleet baseline shift")
    for confirmed, group in frame.groupby("confirmed_offset", sort=False):
        idx = group.index.to_numpy()
        xerr = np.vstack(
            [
                group["median"].to_numpy() - group["p25"].to_numpy(),
                group["p75"].to_numpy() - group["median"].to_numpy(),
            ]
        )
        ax.errorbar(
            group["median"],
            idx,
            xerr=xerr,
            fmt="o",
            markersize=5.5,
            markerfacecolor=PALETTE["red"] if confirmed else "white",
            markeredgecolor=PALETTE["red"] if confirmed else PALETTE["gray"],
            ecolor=PALETTE["red"] if confirmed else "#A0A0A0",
            elinewidth=1.2,
            capsize=2,
            linestyle="none",
            label="confirmed offset" if confirmed else "not confirmed",
        )
    ijabal = frame.loc[frame["station_id"].eq("IJABAL13")]
    if not ijabal.empty and pd.notna(ijabal["confirmed_level_p50"].iloc[0]):
        pos = int(ijabal.index[0])
        value = float(ijabal["confirmed_level_p50"].iloc[0])
        ax.annotate(
            f"IJABAL13 episode p50 {value:.1f} hPa",
            xy=(-19.8, pos),
            xytext=(-18.8, pos + 1.6),
            arrowprops={"arrowstyle": "->", "color": PALETTE["red"], "lw": 1.0},
            color=PALETTE["red"],
            fontsize=8.5,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["station_id"])
    ax.set_xlim(-20, 4)
    ax.set_xlabel("Median pressure residual, station minus ERA5 (hPa)")
    ax.set_ylabel("Station")
    ax.set_title("External Pressure Residuals by Station")
    _style_axes(ax)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_solar_underread(
    monthly: pd.DataFrame,
    station_ratios: pd.DataFrame,
    path: Path,
    reference_ratio: float = 0.853,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    station_frame = station_ratios.sort_values("clear_day_ratio").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    ax.plot(monthly["month"], monthly["fleet_median"], marker="o", color=PALETTE["blue"], label="fleet median")
    ax.fill_between(
        monthly["month"],
        monthly["p25"],
        monthly["p75"],
        color=PALETTE["sky"],
        alpha=0.22,
        label="IQR",
    )
    ax.axhline(reference_ratio, color=PALETTE["gray"], linestyle="--", linewidth=1.0, label="0.853 healthy source ratio")
    ax.set_xlabel("Month")
    ax.set_ylabel("Clear-sky ratio")
    ax.set_title("Fleet Clear-Sky Ratio by Month")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    ax.grid(True, color="#D9D9D9", linewidth=0.7)
    ax = axes[1]
    y = np.arange(len(station_frame))
    colors = [
        PALETTE["red"] if station == "IJABAL13" else PALETTE["green"] if station == "IMISRA12" else PALETTE["blue"]
        for station in station_frame["station_id"]
    ]
    ax.scatter(station_frame["clear_day_ratio"], y, color=colors, s=30)
    ax.axvline(reference_ratio, color=PALETTE["gray"], linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(station_frame["station_id"])
    ax.set_xlabel("Median ratio on clearest days")
    ax.set_title("Station Clear-Day Ratios")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_systemic_adjudication(counts: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = counts.copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(frame["support_tier"], frame["episodes"], color=[PALETTE["gray"], PALETTE["orange"], PALETTE["sky"], PALETTE["green"]])
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 1.5, f"{int(height)}", ha="center", va="bottom")
    ax.set_ylim(0, max(frame["episodes"].max() * 1.18, 5))
    ax.set_ylabel("Episodes")
    ax.set_xlabel("ERA5 support tier")
    ax.set_title("External Adjudication of Systemic Co-Flags")
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_spatial_vs_external(pairs: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pairs.sort_values("external_median").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.0), gridspec_kw={"width_ratios": [1.1, 1.0]})
    ax = axes[0]
    y = np.arange(len(frame))
    x_min = -20.0
    x_max = 5.0
    ax.scatter(frame["external_median"], y - 0.13, color=PALETTE["blue"], s=24, label="external")
    spatial_valid = frame["spatial_median"].notna()
    spatial_x = frame.loc[spatial_valid, "spatial_median"].clip(lower=x_min + 0.5, upper=x_max - 0.5)
    ax.scatter(spatial_x, y[spatial_valid] + 0.13, color=PALETTE["green"], s=24, label="spatial")
    clipped = frame.loc[spatial_valid & frame["spatial_median"].lt(x_min + 0.5)]
    for index, row in clipped.iterrows():
        ax.annotate(
            f"{row['station_id']} spatial {row['spatial_median']:.1f}",
            xy=(x_min + 0.5, index + 0.13),
            xytext=(x_min + 2.0, index + 1.2),
            arrowprops={"arrowstyle": "->", "color": PALETTE["green"], "lw": 1.0},
            color=PALETTE["green"],
            fontsize=8.0,
        )
    ax.axvline(0, color="#333333", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["station_id"])
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Median pressure residual (hPa)")
    ax.set_title("External vs Healthy-Neighbour Reference")
    _style_axes(ax)
    ax.legend(frameon=False)
    ax = axes[1]
    regular = frame.loc[~frame["spatial_isolated"]]
    isolated = frame.loc[frame["spatial_isolated"]]
    regular_y = regular["spatial_median"].clip(lower=x_min + 0.5, upper=x_max - 0.5)
    ax.scatter(
        regular["external_median"],
        regular_y,
        facecolors=np.where(regular["confirmed_offset"], PALETTE["red"], "white"),
        edgecolors=np.where(regular["confirmed_offset"], PALETTE["red"], PALETTE["gray"]),
        s=np.where(regular["confirmed_offset"], 46, 34),
        label="not isolated",
    )
    if not isolated.empty:
        ax.scatter(isolated["external_median"], np.zeros(len(isolated)), marker="x", color=PALETTE["purple"], s=54, label="isolated, pinned y=0")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.axvline(0, color="#333333", linewidth=1.0)
    clipped_regular = regular.loc[regular["spatial_median"].lt(x_min + 0.5)]
    for _, row in clipped_regular.iterrows():
        ax.annotate(
            f"{row['station_id']} {row['spatial_median']:.1f}",
            xy=(row["external_median"], x_min + 0.5),
            xytext=(row["external_median"] + 1.2, x_min + 2.2),
            arrowprops={"arrowstyle": "->", "color": PALETTE["red"], "lw": 1.0},
            color=PALETTE["red"],
            fontsize=8.0,
        )
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("External median r_pressure (hPa)")
    ax.set_ylabel("Spatial median offset level (hPa)")
    ax.set_title("Station Residual Pairing")
    ax.grid(True, color="#D9D9D9", linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def plot_localized_spatial(
    episodes: pd.DataFrame,
    merged: pd.DataFrame,
    external_features: pd.DataFrame,
    spatial_residuals: pd.DataFrame,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex=False)
    axes_flat = axes.ravel()
    for ax, (_, episode) in zip(axes_flat, episodes.iterrows()):
        series = localized_pressure_series(merged, external_features, spatial_residuals, episode)
        ax.plot(series["date"], series["station_pressure"], color=PALETTE["blue"], label="station", linewidth=1.8)
        ax.plot(series["date"], series["healthy_neighbor_pressure"], color=PALETTE["green"], label="healthy neighbours", linewidth=1.5)
        ax.plot(series["date"], series["era5_pressure"], color=PALETTE["orange"], label="ERA5", linewidth=1.5)
        ax.axvspan(pd.to_datetime(episode["start_hour"]), pd.to_datetime(episode["end_hour"]), color=PALETTE["red"], alpha=0.09)
        ax.set_title(str(episode["station_id"]))
        ax.set_ylabel("Pressure (hPa)")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, color="#D9D9D9", linewidth=0.7)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Localized Spatial Pressure Anomalies")
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return path


def build_all_figures(
    external_features: pd.DataFrame,
    external_residuals: pd.DataFrame,
    spatial_features: pd.DataFrame,
    spatial_residuals: pd.DataFrame,
    evidence: pd.DataFrame,
    reviewed: pd.DataFrame,
    spatial_queue: pd.DataFrame,
    merged: pd.DataFrame,
    output_dir: Path = RESULT_FIGURES_PATH,
) -> dict[str, object]:
    pressure = pressure_offset_summary(external_features, reviewed)
    monthly = solar_monthly_fleet_ratio(external_features)
    solar_station = solar_station_clear_day_ratio(external_features)
    systemic = systemic_tier_counts(evidence)
    pairs = spatial_external_pairs(external_features, spatial_features, reviewed)
    localized = localized_spatial_episodes(spatial_queue, reviewed, external_features)
    paths = {
        "pressure_offsets": plot_pressure_offsets(pressure, figure_path("fig_pressure_offsets.png", output_dir)),
        "solar_underread": plot_solar_underread(monthly, solar_station, figure_path("fig_solar_underread.png", output_dir)),
        "systemic_adjudication": plot_systemic_adjudication(systemic, figure_path("fig_systemic_adjudication.png", output_dir)),
        "spatial_vs_external": plot_spatial_vs_external(pairs, figure_path("fig_spatial_vs_external.png", output_dir)),
        "localized_spatial": plot_localized_spatial(localized, merged, external_features, spatial_residuals, figure_path("fig_localized_spatial.png", output_dir)),
    }
    return {
        "paths": paths,
        "pressure": pressure,
        "solar_monthly": monthly,
        "solar_station": solar_station,
        "systemic": systemic,
        "pairs": pairs,
        "localized": localized,
    }
