from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
