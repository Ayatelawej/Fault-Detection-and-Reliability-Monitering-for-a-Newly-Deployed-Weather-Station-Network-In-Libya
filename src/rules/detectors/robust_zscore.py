from __future__ import annotations

import numpy as np
import pandas as pd


def score_robust_zscore(
    series: pd.Series,
    threshold: float = 3.5,
    baseline: dict[str, object] | None = None,
) -> pd.DataFrame:
    values = series.astype(float)

    if baseline is None:
        present_values = values.dropna()

        if present_values.empty:
            scores = pd.Series(np.nan, index=series.index, name="score")
            flags = pd.Series(False, index=series.index, name="flag")
            return pd.DataFrame({"score": scores, "flag": flags}, index=series.index)

        center = float(np.nanmedian(present_values.to_numpy(dtype=float)))
        absolute_deviation = np.abs(present_values.to_numpy(dtype=float) - center)
        spread = float(np.nanmedian(absolute_deviation))
    else:
        center = float(baseline["baseline_value"])
        spread = float(baseline["baseline_spread"])

    if spread == 0.0 or np.isnan(spread):
        scores = pd.Series(np.nan, index=series.index, name="score")
        flags = pd.Series(False, index=series.index, name="flag")
        return pd.DataFrame({"score": scores, "flag": flags}, index=series.index)

    scores = (values - center) / spread
    scores = scores.rename("score")
    flags = scores.abs().gt(threshold).fillna(False).rename("flag")
    return pd.DataFrame({"score": scores, "flag": flags}, index=series.index)
