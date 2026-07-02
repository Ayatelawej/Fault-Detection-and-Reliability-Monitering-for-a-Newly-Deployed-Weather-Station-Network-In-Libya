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
    STUCK_EVENTS_PATH,
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


def main() -> None:
    labels = pd.read_csv(LABELED_EPISODES_PATH)
    episodes = labels.loc[labels["label"].eq("stuck_flatline")].copy()
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


if __name__ == "__main__":
    main()
