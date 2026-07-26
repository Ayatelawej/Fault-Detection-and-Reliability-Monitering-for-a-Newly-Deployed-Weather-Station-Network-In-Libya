from __future__ import annotations

import pandas as pd

from src.rules.labelling import classify_episode


BASE_TIME = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")


def _episode(hours: int) -> dict[str, object]:
    return {
        "station_id": "STA",
        "start_hour": BASE_TIME,
        "end_hour": BASE_TIME + pd.Timedelta(hours=hours - 1),
        "duration_hours": hours,
    }


def _raw(hours: int, **values: object) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "station_id": ["STA"] * hours,
            "hour_utc": pd.date_range(BASE_TIME, periods=hours, freq="h", tz="UTC"),
        },
    )
    for channel, value in values.items():
        frame[channel] = value
    return frame


def _scores(hours: int, channel: str, stuck: bool = True) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["STA"] * hours,
            "hour_utc": pd.date_range(BASE_TIME, periods=hours, freq="h", tz="UTC"),
            "channel": [channel] * hours,
            "flag_stuck": [stuck] * hours,
            "rolling_variance": [0.0 if stuck else 1.0] * hours,
        },
    )


def test_hard_value_becomes_spike_impossible() -> None:
    result = classify_episode(
        _episode(1),
        _raw(1, temp_avg_c=[61.0]),
        pd.DataFrame(),
    )

    assert result["mechanisms"] == "spike_impossible"
    assert result["components"] == "thermo_hygrometer"


def test_suspect_solar_is_benign_without_full_gate_evidence() -> None:
    result = classify_episode(
        _episode(1),
        _raw(1, solar_radiation_high_wm2=[1200.0]),
        pd.DataFrame(),
    )

    assert result["binary_fault"] == 0
    assert result["mechanisms"] == ""


def test_full_gate_evidence_becomes_statistical_anomaly() -> None:
    evidence = pd.DataFrame(
        {
            "station_id": ["STA"],
            "hour_utc": [BASE_TIME],
            "channel": ["temp_avg_c"],
            "full_gate_passed": [True],
        },
    )

    result = classify_episode(
        _episode(1),
        _raw(1, temp_avg_c=[25.0]),
        pd.DataFrame(),
        evidence,
    )

    assert result["mechanisms"] == "statistical_anomaly"
    assert result["components"] == "thermo_hygrometer"


def test_path_b_evidence_becomes_statistical_anomaly() -> None:
    evidence = pd.DataFrame(
        {
            "station_id": ["STA"],
            "hour_utc": [BASE_TIME],
            "channel": ["temp_avg_c"],
            "evidence_path": ["B"],
            "full_gate_passed": [True],
        },
    )

    result = classify_episode(
        _episode(1),
        _raw(1, temp_avg_c=[25.0]),
        pd.DataFrame(),
        evidence,
    )

    assert result["mechanisms"] == "statistical_anomaly"


def test_failed_full_gate_evidence_stays_benign() -> None:
    evidence = pd.DataFrame(
        {
            "station_id": ["STA"],
            "hour_utc": [BASE_TIME],
            "channel": ["temp_avg_c"],
            "full_gate_passed": [False],
        },
    )

    result = classify_episode(
        _episode(1),
        _raw(1, temp_avg_c=[25.0]),
        pd.DataFrame(),
        evidence,
    )

    assert result["binary_fault"] == 0


def test_constant_pressure_becomes_stuck_flatline() -> None:
    result = classify_episode(
        _episode(24),
        _raw(24, pressure_max_hpa=[1000.0] * 24),
        pd.DataFrame(),
    )

    assert result["mechanisms"] == "stuck_flatline"
    assert result["components"] == "barometer"


def test_hard_spike_and_flatline_are_both_retained() -> None:
    result = classify_episode(
        _episode(24),
        _raw(
            24,
            temp_avg_c=[61.0] + [20.0] * 23,
            pressure_max_hpa=[1000.0] * 24,
        ),
        pd.DataFrame(),
    )

    assert result["mechanisms"] == "spike_impossible|stuck_flatline"
    assert result["components"] == "barometer|thermo_hygrometer"


def test_flat_direction_during_calm_is_not_a_fault() -> None:
    raw = _raw(
        24,
        winddir_avg_deg=[90.0] * 24,
        windspeed_avg_kmh=[0.0, 0.5] * 12,
    )
    scores = pd.concat(
        [_scores(24, "winddir_sin"), _scores(24, "winddir_cos")],
        ignore_index=True,
    )

    result = classify_episode(_episode(24), raw, scores)

    assert result["binary_fault"] == 0
    assert result["mechanisms"] == ""


def test_flat_direction_with_wind_is_a_vane_fault() -> None:
    raw = _raw(
        24,
        winddir_avg_deg=[90.0] * 24,
        windspeed_avg_kmh=[5.0, 6.0] * 12,
    )
    scores = pd.concat(
        [_scores(24, "winddir_sin"), _scores(24, "winddir_cos")],
        ignore_index=True,
    )

    result = classify_episode(_episode(24), raw, scores)

    assert result["mechanisms"] == "stuck_flatline"
    assert result["components"] == "wind_vane"


def test_joint_flat_speed_and_direction_is_anemometer_fault() -> None:
    raw = _raw(
        24,
        winddir_avg_deg=[90.0] * 24,
        windspeed_avg_kmh=[5.0] * 24,
    )
    scores = pd.concat(
        [
            _scores(24, "winddir_sin"),
            _scores(24, "winddir_cos"),
            _scores(24, "windspeed_avg_kmh"),
        ],
        ignore_index=True,
    )

    result = classify_episode(_episode(24), raw, scores)

    assert result["mechanisms"] == "stuck_flatline"
    assert result["components"] == "anemometer"


def test_physically_normal_short_episode_is_benign() -> None:
    result = classify_episode(
        _episode(1),
        _raw(1, temp_avg_c=[20.0]),
        pd.DataFrame(),
    )

    assert result["binary_fault"] == 0
    assert result["mechanisms"] == ""


def test_zero_rain_is_not_stuck_when_detector_exempts_it() -> None:
    scores = _scores(24, "precip_rate_mmh", stuck=False)
    scores["rolling_variance"] = 0.0

    result = classify_episode(
        _episode(24),
        _raw(24, precip_rate_mmh=[0.0] * 24),
        scores,
    )

    assert result["binary_fault"] == 0
    assert result["mechanisms"] == ""
