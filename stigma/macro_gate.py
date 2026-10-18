"""Optional macro-bias adapter. Defaults to neutral; no dependency on Begonya files."""
from __future__ import annotations
import json
from pathlib import Path

def load_macro_bias(path: str | Path | None = None, *, autonomous: bool = True) -> int:
    """Return -1/0/+1. Missing or invalid macro input fails closed to neutral."""
    if autonomous or path is None: return 0
    try:
        data=json.loads(Path(path).read_text(encoding="utf-8"))
        bias=int(data.get("macro_bias",0))
        return bias if bias in (-1,0,1) else 0
    except (OSError,ValueError,TypeError,json.JSONDecodeError):
        return 0
