"""Optional macro-bias adapter. Defaults to neutral; no dependency on Begonya files.

In autonomous mode, macro_bias is strictly 0 (neutral, no external bias).
In Begonya macro-anchored mode, reads macro_bias (-1, 0, +1) safely with fail-closed behavior.
Never modifies or writes to external project directories.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_macro_bias(
    path: str | Path | None = None,
    *,
    autonomous: bool = True,
) -> int:
    """Return -1, 0, or +1 macro bias.

    -1: Bearish bias (vetoes LONG trades)
     0: Neutral (permits both LONG and SHORT trades)
    +1: Bullish bias (vetoes SHORT trades)

    Missing, corrupt, or out-of-range input fails closed to 0 (neutral).
    """
    if autonomous or path is None:
        return 0

    target = Path(path)
    if not target.exists() or not target.is_file():
        logger.warning("Macro file '%s' does not exist; failing closed to neutral (0)", target)
        return 0

    try:
        content = target.read_text(encoding="utf-8")
        data = json.loads(content)
        if not isinstance(data, dict):
            logger.warning("Macro file '%s' is not a JSON object; failing closed to 0", target)
            return 0
        raw = data.get("macro_bias", 0)
        bias = int(raw)
        if bias in (-1, 0, 1):
            return bias
        logger.warning("Macro bias value '%s' out of valid range {-1, 0, 1}; failing closed to 0", raw)
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse macro file '%s' (%s); failing closed to 0", target, exc)
        return 0
