from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rules.stuck_confirmation import (
    CONSTANCY_THRESHOLD,
    FIVE_MIN_DIR,
    LABELED_EPISODES_PATH,
    MIN_5MIN_OBS,
    RANGING_SPREAD_THRESHOLD,
    STUCK_EVENTS_PATH,
    categorize_stuck_5min,
    confirm_stuck_episodes,
    confirm_stuck_episodes_dominant,
    expand_stuck_channels,
)


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["status"].value_counts().to_dict()
    return {
        "confirmed_stuck": int(counts.get("confirmed_stuck", 0)),
        "not_constant": int(counts.get("not_constant", 0)),
        "insufficient_5min_data": int(counts.get("insufficient_5min_data", 0)),
        "unmapped_channel": int(counts.get("unmapped_channel", 0)),
    }


def _print_headline(name: str, frame: pd.DataFrame, episode_count: int) -> None:
    counts = _status_counts(frame)
    checked = int(len(frame))
    confirmed = counts["confirmed_stuck"]
    fraction = float(confirmed / checked) if checked else 0.0
    print(f"{name}:")
    print(f"  episodes={episode_count}")
    print(f"  checks={checked}")
    print(f"  confirmed={confirmed}")
    print(f"  confirmed_fraction={fraction:.3f}")
    for status, count in counts.items():
        print(f"  {status}={count}")


def _modal_bins(frame: pd.DataFrame) -> pd.Series:
    not_constant = frame.loc[frame["status"].eq("not_constant")]
    bins = {
        "0.95-0.99": not_constant["modal_fraction"].ge(0.95)
        & not_constant["modal_fraction"].lt(0.99),
        "0.90-0.95": not_constant["modal_fraction"].ge(0.90)
        & not_constant["modal_fraction"].lt(0.95),
        "0.80-0.90": not_constant["modal_fraction"].ge(0.80)
        & not_constant["modal_fraction"].lt(0.90),
        "<0.80": not_constant["modal_fraction"].lt(0.80),
    }
    return pd.Series({name: int(mask.sum()) for name, mask in bins.items()})


def _distinct_bins(frame: pd.DataFrame) -> pd.Series:
    not_constant = frame.loc[frame["status"].eq("not_constant")]
    bins = {
        "2-3": not_constant["n_distinct_values"].between(2, 3),
        "4-10": not_constant["n_distinct_values"].between(4, 10),
        ">10": not_constant["n_distinct_values"].gt(10),
    }
    return pd.Series({name: int(mask.sum()) for name, mask in bins.items()})


def _threshold_sweep(
    frame: pd.DataFrame,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        confirmed = (
            frame["n_5min_present"].ge(MIN_5MIN_OBS)
            & frame["modal_fraction"].ge(threshold)
        )
        insufficient = frame["n_5min_present"].lt(MIN_5MIN_OBS)
        unmapped = frame["status"].eq("unmapped_channel")
        not_constant = ~(confirmed | insufficient | unmapped)
        rows.append(
            {
                "threshold": threshold,
                "confirmed": int(confirmed.sum()),
                "not_constant": int(not_constant.sum()),
                "insufficient_5min_data": int(insufficient.sum()),
                "unmapped_channel": int(unmapped.sum()),
                "fraction_confirmed": int(confirmed.sum()) / len(frame)
                if len(frame)
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def _print_examples(frame: pd.DataFrame, n: int = 5) -> None:
    examples = frame.loc[frame["status"].eq("not_constant")].sort_values(
        ["modal_fraction", "n_distinct_values"],
        ascending=[False, True],
    )
    columns = [
        "station_id",
        "channel",
        "n_5min_present",
        "modal_value",
        "modal_fraction",
        "n_distinct_values",
        "longest_constant_run_hours",
        "min_reading",
        "max_reading",
    ]
    print("not_constant_examples=")
    if examples.empty:
        print("none")
    else:
        print(examples.loc[:, columns].head(n).to_string(index=False))


def _low_modal_profiles(frame: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    low_modal = frame.loc[frame["modal_fraction"].lt(0.80)].copy()
    if low_modal.empty:
        return low_modal

    low_modal["spread"] = low_modal["max_reading"] - low_modal["min_reading"]
    return low_modal.sort_values(
        ["spread", "n_distinct_values", "duration_hours"],
        ascending=[False, False, False],
    ).head(n)


def _print_low_modal_profiles(frame: pd.DataFrame) -> None:
    low_modal = frame.loc[frame["modal_fraction"].lt(0.80)].copy()
    if not low_modal.empty:
        low_modal["spread"] = low_modal["max_reading"] - low_modal["min_reading"]

    print("low_modal_profiles=")
    if low_modal.empty:
        print("none")
    else:
        columns = [
            "station_id",
            "channel",
            "duration_hours",
            "n_5min_present",
            "min_reading",
            "max_reading",
            "spread",
            "n_distinct_values",
            "modal_value",
            "modal_fraction",
            "longest_constant_run_hours",
        ]
        print(_low_modal_profiles(frame).loc[:, columns].to_string(index=False))

    spread_gt = int(low_modal["spread"].gt(RANGING_SPREAD_THRESHOLD).sum())
    spread_le = int(low_modal["spread"].le(RANGING_SPREAD_THRESHOLD).sum())
    print("low_modal_spread_summary=")
    print(f"  total={len(low_modal)}")
    print(f"  spread_gt_5={spread_gt}")
    print(f"  spread_le_5={spread_le}")
    if not low_modal.empty:
        print(low_modal["spread"].describe().to_string())


def _print_itripo33(frame: pd.DataFrame) -> None:
    rows = frame.loc[
        frame["station_id"].eq("ITRIPO33")
        & frame["channel"].eq("windspeed_avg_kmh")
    ]
    columns = [
        "station_id",
        "channel",
        "n_5min_present",
        "modal_value",
        "modal_fraction",
        "n_distinct_values",
        "longest_constant_run_hours",
        "status",
    ]
    print("ITRIPO33_windspeed=")
    if rows.empty:
        print("WARNING: ITRIPO33 windspeed stuck episode not found")
        return

    print(rows.loc[:, columns].to_string(index=False))
    if not rows["status"].eq("confirmed_stuck").all():
        print("WARNING: ITRIPO33 windspeed did not fully confirm as stuck")


def _key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["start_hour"] = pd.to_datetime(result["start_hour"], utc=True).astype(str)
    result["end_hour"] = pd.to_datetime(result["end_hour"], utc=True).astype(str)
    return result


def _relabel_labeled_file(labels: pd.DataFrame, dominant: pd.DataFrame) -> pd.DataFrame:
    updated = labels.copy()
    before = updated["label"].value_counts().sort_index()
    dominant_status = dominant.copy()
    dominant_status["stuck_5min_status"] = dominant_status.apply(
        categorize_stuck_5min,
        axis=1,
    )
    dominant_status["stuck_confirmed_5min"] = dominant_status[
        "stuck_5min_status"
    ].eq("confirmed_flatline")
    dominant_status["spread"] = (
        dominant_status["max_reading"] - dominant_status["min_reading"]
    )
    dominant_status = _key_frame(dominant_status)
    updated = _key_frame(updated)
    updated["stuck_5min_status"] = ""
    updated["stuck_confirmed_5min"] = False

    status_columns = [
        "station_id",
        "start_hour",
        "end_hour",
        "stuck_5min_status",
        "stuck_confirmed_5min",
        "spread",
    ]
    merged = updated.merge(
        dominant_status.loc[:, status_columns],
        on=["station_id", "start_hour", "end_hour"],
        how="left",
        suffixes=("", "_new"),
    )
    has_status = merged["stuck_5min_status_new"].notna()
    merged.loc[has_status, "stuck_5min_status"] = merged.loc[
        has_status,
        "stuck_5min_status_new",
    ]
    merged.loc[has_status, "stuck_confirmed_5min"] = merged.loc[
        has_status,
        "stuck_confirmed_5min_new",
    ].astype(bool)

    relabel_mask = (
        merged["label"].eq("stuck_flatline")
        & merged["stuck_5min_status"].eq("ranging_not_stuck")
    )
    if "evidence" not in merged.columns:
        merged["evidence"] = ""

    spreads = merged.loc[relabel_mask, "spread"].round(2).astype(str)
    merged.loc[relabel_mask, "label"] = "outlier_benign"
    merged.loc[relabel_mask, "evidence"] = (
        "hourly stuck flag not supported at 5-minute resolution; "
        "sensor ranging (min-max spread "
        + spreads
        + "), reclassified as benign low-variance"
    )
    merged = merged.drop(
        columns=[
            "stuck_5min_status_new",
            "stuck_confirmed_5min_new",
            "spread",
        ]
    )

    after = merged["label"].value_counts().sort_index()
    remain = merged.loc[merged["label"].eq("stuck_flatline")]
    moved = int(relabel_mask.sum())

    print("label_distribution_before=")
    print(before.to_string())
    print()
    print("label_distribution_after=")
    print(after.to_string())
    print()
    print(f"stuck_reclassified_to_outlier_benign={moved}")
    print(f"stuck_flatline_remaining={len(remain)}")
    print("remaining_stuck_5min_status=")
    print(remain["stuck_5min_status"].value_counts().to_string())
    print()
    print("ITRIPO33_after_relabel=")
    print(
        merged.loc[
            merged["station_id"].eq("ITRIPO33"),
            [
                "station_id",
                "start_hour",
                "end_hour",
                "label",
                "stuck_5min_status",
                "stuck_confirmed_5min",
                "evidence",
            ],
        ].to_string(index=False)
    )
    return merged


def main() -> None:
    labels = pd.read_csv(LABELED_EPISODES_PATH)
    if "stuck_5min_status" in labels.columns:
        status_present = labels["stuck_5min_status"].fillna("").ne("")
    else:
        status_present = pd.Series(False, index=labels.index)

    episodes = labels.loc[labels["label"].eq("stuck_flatline") | status_present].copy()
    events = pd.read_parquet(STUCK_EVENTS_PATH)
    per_channel_rows = expand_stuck_channels(episodes)
    per_channel = confirm_stuck_episodes(per_channel_rows, FIVE_MIN_DIR)
    dominant = confirm_stuck_episodes_dominant(episodes, events, FIVE_MIN_DIR)

    print(f"five_min_dir={FIVE_MIN_DIR}")
    print(f"default_threshold={CONSTANCY_THRESHOLD}")
    print(f"min_obs={MIN_5MIN_OBS}")
    print()
    _print_headline("per_channel", per_channel, int(len(episodes)))
    print()
    _print_headline("dominant_channel", dominant, int(len(episodes)))
    print()
    print("dominant_not_constant_modal_fraction_bins=")
    print(_modal_bins(dominant).to_string())
    print()
    print("dominant_not_constant_n_distinct_bins=")
    print(_distinct_bins(dominant).to_string())
    print()
    print("dominant_threshold_sweep=")
    print(_threshold_sweep(dominant, [0.99, 0.97, 0.95]).to_string(index=False))
    print()
    _print_examples(dominant)
    print()
    _print_itripo33(dominant)
    print()
    _print_low_modal_profiles(dominant)
    print()
    dominant = dominant.copy()
    dominant["stuck_5min_status"] = dominant.apply(categorize_stuck_5min, axis=1)
    print("dominant_stuck_5min_status=")
    print(dominant["stuck_5min_status"].value_counts().to_string())
    print()
    updated = _relabel_labeled_file(labels, dominant)
    updated.to_csv(LABELED_EPISODES_PATH, index=False)
    print()
    print(f"updated={LABELED_EPISODES_PATH}")


if __name__ == "__main__":
    main()
