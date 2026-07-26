from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def require_files(
    workflow: str,
    required: Mapping[str, Path],
    recovery: str | None = None,
) -> None:
    missing = [(name, Path(path)) for name, path in required.items() if not Path(path).is_file()]
    if not missing:
        return
    lines = [f"{workflow} cannot start because required inputs are missing:"]
    lines.extend(f"- {name}: {path}" for name, path in missing)
    if recovery:
        lines.append(recovery)
    raise FileNotFoundError("\n".join(lines))
