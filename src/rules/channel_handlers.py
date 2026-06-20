from __future__ import annotations

import numpy as np
import pandas as pd


def encode_wind_direction(series: pd.Series) -> pd.DataFrame:
    radians = np.deg2rad(series.astype(float))
    return pd.DataFrame(
        {
            "sin": np.sin(radians),
            "cos": np.cos(radians),
        },
        index=series.index,
    )


def log_transform_precip(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.log1p(series.astype(float)),
        index=series.index,
        name=series.name,
    )
