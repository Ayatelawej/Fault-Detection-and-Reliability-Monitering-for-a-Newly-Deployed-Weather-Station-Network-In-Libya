from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.rules.config import COVERAGE_FLOOR_HOURS, ROBUST_ZSCORE_FLAG_PERCENTILE


SCORE_CHANNELS = ["temp_avg_c", "winddir_avg_deg", "precip_total_mm"]
SCORE_OUTPUT_COLUMNS = [
    "station_id",
    "hour_utc",
    "channel",
    "baseline_source",
    "zscore",
    "rolling_variance",
    "iforest_score",
    "flag_zscore",
    "flag_stuck",
    "flag_iforest",
    "flag",
    "reason",
]
SCORE_SPIKE_OFFSETS = [240, 780, 1_260]
SCORE_STUCK_START = 640
SCORE_STUCK_LENGTH = 12


def _assert_detection_frame(
    result: pd.DataFrame,
    expected_index: pd.Index,
) -> tuple[pd.Series, pd.Series]:
    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(expected_index)
    assert {"score", "flag"} <= set(result.columns)
    return result["score"], result["flag"]


def _assert_stuck_frame(
    result: pd.DataFrame,
    expected_index: pd.Index,
) -> pd.Series:
    assert isinstance(result, pd.DataFrame)
    assert result.index.equals(expected_index)
    assert {"rolling_variance", "flag"} <= set(result.columns)
    return result["flag"]


def _assert_score_series(
    result: pd.Series,
    expected_index: pd.Index,
) -> pd.Series:
    assert isinstance(result, pd.Series)
    assert result.index.equals(expected_index)
    assert result.notna().all()
    return result


def _baseline_source(result: dict[str, object]) -> str:
    assert isinstance(result, dict)
    assert "source" in result
    return str(result["source"])


def _baseline_value(result: dict[str, object]) -> float:
    assert isinstance(result, dict)
    assert "baseline_value" in result
    return float(result["baseline_value"])


def _baseline_input_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=1_501, freq="h", tz="UTC")
    peer_timestamps = pd.date_range(
        "2024-02-01",
        periods=1_501,
        freq="h",
        tz="UTC",
    )
    sparse_timestamps = pd.date_range(
        "2024-03-01",
        periods=1_499,
        freq="h",
        tz="UTC",
    )
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "station_id": "OWN",
                    "timestamp_utc": timestamps,
                    "airtemp_avg_c": 12.0,
                },
            ),
            pd.DataFrame(
                {
                    "station_id": "PEER_A",
                    "timestamp_utc": timestamps,
                    "airtemp_avg_c": 20.0,
                },
            ),
            pd.DataFrame(
                {
                    "station_id": "PEER_B",
                    "timestamp_utc": peer_timestamps,
                    "airtemp_avg_c": 20.0,
                },
            ),
            pd.DataFrame(
                {
                    "station_id": "SPARSE",
                    "timestamp_utc": sparse_timestamps,
                    "airtemp_avg_c": 99.0,
                },
            ),
        ],
        ignore_index=True,
    )


def _station_score_frame(
    station_id: str,
    periods: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = np.arange(periods, dtype=float)
    hours = pd.date_range("2024-04-01", periods=periods, freq="h", tz="UTC")
    precip = np.zeros(periods, dtype=float)
    wet_positions = np.arange(seed % 37, periods, 113)
    precip[wet_positions] = rng.uniform(0.05, 0.55, size=len(wet_positions))
    return pd.DataFrame(
        {
            "station_id": station_id,
            "hour_utc": hours,
            "temp_avg_c": 20.0
            + 0.6 * np.sin(positions / 24.0)
            + rng.normal(0.0, 0.12, periods),
            "winddir_avg_deg": (positions * 9.0 + seed * 13.0) % 360.0,
            "precip_total_mm": precip,
        },
    )


def _score_input_frame() -> pd.DataFrame:
    normal = _station_score_frame("STA_NORMAL", 1_600, 11)
    spike = _station_score_frame("STA_SPIKE", 1_600, 22)
    stuck = _station_score_frame("STA_STUCK", 1_600, 33)
    sparse = _station_score_frame("STA_SPARSE", 1_400, 44)

    spike.loc[SCORE_SPIKE_OFFSETS, "temp_avg_c"] = (
        spike.loc[SCORE_SPIKE_OFFSETS, "temp_avg_c"] + 100.0
    )
    stuck.loc[
        SCORE_STUCK_START : SCORE_STUCK_START + SCORE_STUCK_LENGTH - 1,
        "temp_avg_c",
    ] = 20.0

    return pd.concat(
        [normal, spike, stuck, sparse],
        ignore_index=True,
    )


def _score_output() -> pd.DataFrame:
    from src.rules.score import compute_anomaly_scores

    return compute_anomaly_scores(
        _score_input_frame(),
        SCORE_CHANNELS,
        flag_percentile=ROBUST_ZSCORE_FLAG_PERCENTILE,
        random_state=42,
    )


class TestRobustZScoreContract:
    def test_robust_zscore_flags_single_injected_spike(self) -> None:
        from src.rules.detectors.robust_zscore import score_robust_zscore

        values = np.tile([9.0, 10.0, 11.0], 8).astype(float)
        series = pd.Series(values, name="airtemp_avg_c")
        series.loc[12] = 25.0

        result = score_robust_zscore(series, threshold=3.5)
        scores, flags = _assert_detection_frame(result, series.index)

        assert scores.loc[12] > 3.5
        assert bool(flags.loc[12])
        assert scores.drop(index=12).lt(3.5).all()
        assert not flags.drop(index=12).fillna(False).astype(bool).any()

    def test_robust_zscore_leaves_clean_gaussian_like_series_unflagged(
        self,
    ) -> None:
        from src.rules.detectors.robust_zscore import score_robust_zscore

        series = pd.Series(
            [-2.0, -1.4, -0.8, -0.2, 0.0, 0.2, 0.8, 1.4, 2.0],
            name="airtemp_avg_c",
        )

        result = score_robust_zscore(series, threshold=3.5)
        scores, flags = _assert_detection_frame(result, series.index)

        assert np.isfinite(scores).all()
        assert not flags.fillna(False).astype(bool).any()

    def test_robust_zscore_preserves_nan_scores_without_flags(self) -> None:
        from src.rules.detectors.robust_zscore import score_robust_zscore

        series = pd.Series(
            [9.0, 10.0, np.nan, 11.0, 10.0, np.nan, 9.0],
            name="airtemp_avg_c",
        )
        nan_rows = series[series.isna()].index

        result = score_robust_zscore(series, threshold=3.5)
        scores, flags = _assert_detection_frame(result, series.index)

        assert scores.loc[nan_rows].isna().all()
        assert flags.loc[nan_rows].eq(False).all()
        assert scores.drop(index=nan_rows).notna().all()

    def test_robust_zscore_handles_zero_mad_without_infinite_scores(
        self,
    ) -> None:
        from src.rules.detectors.robust_zscore import score_robust_zscore

        series = pd.Series([12.0] * 8, name="airtemp_avg_c")

        result = score_robust_zscore(series, threshold=3.5)
        scores, flags = _assert_detection_frame(result, series.index)

        assert not np.isinf(scores.dropna()).any()
        assert not flags.fillna(False).astype(bool).any()


class TestRollingVarianceStuckDetectorContract:
    def test_rolling_variance_flags_six_identical_values(self) -> None:
        from src.rules.detectors.rolling_variance import detect_stuck_values

        series = pd.Series(
            [1.0, 2.0, 3.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 4.0],
            name="humidity_avg_pct",
        )

        result = detect_stuck_values(series, window=6)
        flags = _assert_stuck_frame(result, series.index)

        assert flags.iloc[3:9].eq(True).all()
        assert not flags.iloc[[0, 1, 2, 9]].fillna(False).astype(bool).any()

    def test_rolling_variance_leaves_varying_series_unflagged(self) -> None:
        from src.rules.detectors.rolling_variance import detect_stuck_values

        series = pd.Series(
            [0.1, 0.8, 1.4, 0.6, 1.8, 0.3, 1.1, 2.0, 1.5, 0.7],
            name="humidity_avg_pct",
        )

        result = detect_stuck_values(series, window=6)
        flags = _assert_stuck_frame(result, series.index)

        assert not flags.fillna(False).astype(bool).any()

    def test_rolling_variance_flags_imurqu7_nonzero_rain_fault(self) -> None:
        from src.rules.detectors.rolling_variance import detect_stuck_values

        series = pd.Series(
            [0.0, 1.2, *([4697.98] * 12), 0.4, 0.0],
            name="precip_total_mm",
        )

        result = detect_stuck_values(series, window=6)
        flags = _assert_stuck_frame(result, series.index)

        assert flags.iloc[2:14].eq(True).all()
        assert not flags.iloc[[0, 1, 14, 15]].fillna(False).astype(bool).any()

    def test_rolling_variance_flags_itripo33_zero_wind_fault(self) -> None:
        from src.rules.detectors.rolling_variance import detect_stuck_values

        series = pd.Series(
            [3.4, 2.8, *([0.0] * 10), 4.1],
            name="windspeed_avg_kph",
        )

        result = detect_stuck_values(series, window=6)
        flags = _assert_stuck_frame(result, series.index)

        assert flags.iloc[2:12].eq(True).all()
        assert not flags.iloc[[0, 1, 12]].fillna(False).astype(bool).any()

    def test_rolling_variance_does_not_flag_run_shorter_than_window(
        self,
    ) -> None:
        from src.rules.detectors.rolling_variance import detect_stuck_values

        series = pd.Series(
            [1.0, 2.0, *([5.0] * 5), 3.0, 4.0],
            name="humidity_avg_pct",
        )

        result = detect_stuck_values(series, window=6)
        flags = _assert_stuck_frame(result, series.index)

        assert not flags.fillna(False).astype(bool).any()


class TestIsolationForestContract:
    def test_isolation_forest_ranks_far_points_as_most_anomalous(self) -> None:
        from src.rules.detectors.isolation_forest import score_isolation_forest

        normal_left = pd.Series(np.linspace(-1.0, 1.0, 20))
        normal_right = pd.Series(np.linspace(99.0, 101.0, 20))
        far_points = pd.Series([-45.0, 55.0, 145.0])
        series = pd.concat(
            [normal_left, normal_right, far_points],
            ignore_index=True,
        )
        far_index = pd.Index([40, 41, 42])
        normal_index = series.index.difference(far_index)

        result = score_isolation_forest(
            series,
            contamination=0.1,
            random_state=42,
        )
        scores = _assert_score_series(result, series.index)

        assert set(scores.nlargest(3).index) == set(far_index)
        assert scores.loc[far_index].min() > scores.loc[normal_index].max()

    def test_isolation_forest_scores_are_reproducible_with_random_state(
        self,
    ) -> None:
        from src.rules.detectors.isolation_forest import score_isolation_forest

        series = pd.Series(
            np.r_[np.linspace(-1.0, 1.0, 12), np.linspace(8.0, 10.0, 12), 40.0],
            name="airtemp_avg_c",
        )

        first = score_isolation_forest(
            series,
            contamination=0.08,
            random_state=123,
        )
        second = score_isolation_forest(
            series,
            contamination=0.08,
            random_state=123,
        )

        pd.testing.assert_series_equal(first, second)


class TestBaselineContract:
    def test_station_with_enough_present_hours_uses_own_baseline(self) -> None:
        from src.rules.baselines import select_baseline

        result = select_baseline(
            _baseline_input_frame(),
            station_id="OWN",
            channel="airtemp_avg_c",
            min_present_hours=1_500,
        )

        assert _baseline_source(result) == "station"
        assert _baseline_value(result) == pytest.approx(12.0)

    def test_station_below_present_hour_threshold_uses_network_baseline(
        self,
    ) -> None:
        from src.rules.baselines import select_baseline

        result = select_baseline(
            _baseline_input_frame(),
            station_id="SPARSE",
            channel="airtemp_avg_c",
            min_present_hours=1_500,
        )

        assert _baseline_source(result) == "network_pooled"
        assert _baseline_value(result) == pytest.approx(20.0)


class TestChannelHandlersContract:
    def test_wind_direction_zero_and_360_encode_to_same_pair(self) -> None:
        from src.rules.channel_handlers import encode_wind_direction

        directions = pd.Series([0.0, 360.0], name="winddir_avg_deg")

        encoded = encode_wind_direction(directions)

        assert isinstance(encoded, pd.DataFrame)
        assert {"sin", "cos"} <= set(encoded.columns)
        np.testing.assert_allclose(
            encoded.loc[0, ["sin", "cos"]].to_numpy(dtype=float),
            encoded.loc[1, ["sin", "cos"]].to_numpy(dtype=float),
            atol=1e-12,
        )

    def test_wind_direction_nan_encodes_to_nan_pair(self) -> None:
        from src.rules.channel_handlers import encode_wind_direction

        directions = pd.Series([np.nan], name="winddir_avg_deg")

        encoded = encode_wind_direction(directions)

        assert isinstance(encoded, pd.DataFrame)
        assert pd.isna(encoded.loc[0, "sin"])
        assert pd.isna(encoded.loc[0, "cos"])

    def test_precip_log_transform_keeps_zero_finite(self) -> None:
        from src.rules.channel_handlers import log_transform_precip

        precip = pd.Series([0.0, 1.0, 10.0], name="precip_total_mm")

        transformed = log_transform_precip(precip)

        assert isinstance(transformed, pd.Series)
        assert np.isfinite(transformed.iloc[0])
        assert transformed.iloc[0] == pytest.approx(0.0)
        assert transformed.iloc[2] > transformed.iloc[1] > transformed.iloc[0]


class TestScoreOrchestratorIntegration:
    def test_compute_anomaly_scores_returns_exact_columns(self) -> None:
        result = _score_output()

        assert list(result.columns) == SCORE_OUTPUT_COLUMNS

    def test_compute_anomaly_scores_returns_expected_long_scored_channels(
        self,
    ) -> None:
        frame = _score_input_frame()
        result = _score_output()
        scored_channels = set(result["channel"])

        assert len(result) == len(frame) * 4
        assert scored_channels == {
            "temp_avg_c",
            "winddir_sin",
            "winddir_cos",
            "precip_total_mm",
        }
        assert "winddir_avg_deg" not in scored_channels
        assert not result[["station_id", "hour_utc", "channel"]].duplicated().any()

    def test_compute_anomaly_scores_flags_injected_temperature_spikes(
        self,
    ) -> None:
        frame = _score_input_frame()
        result = _score_output()
        spike_hours = frame.loc[
            frame["station_id"].eq("STA_SPIKE"),
        ].iloc[SCORE_SPIKE_OFFSETS]["hour_utc"]
        spike_rows = result.loc[
            result["station_id"].eq("STA_SPIKE")
            & result["channel"].eq("temp_avg_c")
            & result["hour_utc"].isin(spike_hours)
        ]

        assert len(spike_rows) == len(SCORE_SPIKE_OFFSETS)
        assert spike_rows["flag"].eq(True).all()
        assert spike_rows["reason"].str.contains("mad_high", regex=False).all()

    def test_compute_anomaly_scores_flags_stuck_temperature_run(self) -> None:
        frame = _score_input_frame()
        result = _score_output()
        stuck_hours = frame.loc[
            frame["station_id"].eq("STA_STUCK"),
        ].iloc[
            SCORE_STUCK_START : SCORE_STUCK_START + SCORE_STUCK_LENGTH
        ]["hour_utc"]
        stuck_rows = result.loc[
            result["station_id"].eq("STA_STUCK")
            & result["channel"].eq("temp_avg_c")
            & result["hour_utc"].isin(stuck_hours)
        ]

        assert len(stuck_rows) == SCORE_STUCK_LENGTH
        assert stuck_rows["flag"].eq(True).all()
        assert stuck_rows["reason"].str.contains(
            "stuck_variance_zero",
            regex=False,
        ).all()

    def test_compute_anomaly_scores_leaves_normal_temperature_mostly_unflagged(
        self,
    ) -> None:
        result = _score_output()
        normal_temp = result.loc[
            result["station_id"].eq("STA_NORMAL")
            & result["channel"].eq("temp_avg_c")
        ]

        assert normal_temp["flag"].mean() < 0.02

    def test_compute_anomaly_scores_records_baseline_source(
        self,
    ) -> None:
        frame = _score_input_frame()
        result = _score_output()
        station_counts = frame.groupby("station_id").size()
        sparse_rows = result.loc[result["station_id"].eq("STA_SPARSE")]
        above_floor_rows = result.loc[result["station_id"].ne("STA_SPARSE")]

        assert station_counts.loc["STA_SPARSE"] < COVERAGE_FLOOR_HOURS
        assert sparse_rows["baseline_source"].eq("network_pooled").all()
        assert above_floor_rows["baseline_source"].eq("station").any()
