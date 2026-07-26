from __future__ import annotations

import numpy as np
import pandas as pd

from src.rules.config import (
    ROLLING_VARIANCE_FLAG_THRESHOLD,
    ROLLING_VARIANCE_WINDOW_HOURS,
)


def detect_stuck_values(
    series: pd.Series,
    window: int = ROLLING_VARIANCE_WINDOW_HOURS,
    threshold: float = ROLLING_VARIANCE_FLAG_THRESHOLD,
    ignore_zero: bool = False,
) -> pd.DataFrame:
    values = series.astype(float)
    rolling_variance = values.rolling(
        window=window,
        min_periods=window,
    ).var()
    rolling_variance = rolling_variance.rename("rolling_variance")

    flags = np.zeros(len(series), dtype=bool)
    stuck_windows = rolling_variance.lt(threshold).fillna(False)

    if ignore_zero:
        rolling_magnitude = values.abs().rolling(
            window=window,
            min_periods=window,
        ).mean()
        stuck_windows = stuck_windows & rolling_magnitude.gt(1e-9).fillna(False)

    for end_position, is_stuck in enumerate(stuck_windows.to_numpy()):
        if is_stuck:
            start_position = end_position - window + 1
            flags[start_position : end_position + 1] = True

    flag_series = pd.Series(flags, index=series.index, name="flag")
    return pd.DataFrame(
        {
            "rolling_variance": rolling_variance,
            "flag": flag_series,
        },
        index=series.index,
    )
