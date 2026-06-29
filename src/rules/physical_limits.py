from __future__ import annotations

import pandas as pd

from src.rules.config import PHYSICAL_LIMIT_RULES


REASON_PHYSICAL_LIMIT = "physical_limit_breach"


def physical_limit_flags(series: pd.Series, channel: str) -> pd.Series:
    rule = PHYSICAL_LIMIT_RULES.get(channel)
    if rule is None:
        return pd.Series(False, index=series.index)

    values = pd.to_numeric(series, errors="coerce")
    flags = pd.Series(False, index=series.index)

    if "min" in rule:
        flags = flags | values.lt(float(rule["min"]))
    if "max" in rule:
        flags = flags | values.gt(float(rule["max"]))
    if "max_abs" in rule:
        flags = flags | values.abs().gt(float(rule["max_abs"]))

    return flags.fillna(False).astype(bool)


def physical_limit_kind(channel: str) -> str:
    rule = PHYSICAL_LIMIT_RULES.get(channel, {})
    return str(rule.get("kind", "physical"))
