from __future__ import annotations

import sys
from pathlib import Path
from pkgutil import extend_path


_THIS_DIR = Path(__file__).resolve().parent
_SRC_PARENT = _THIS_DIR.parent / "src"

if _SRC_PARENT.is_dir():
    src_parent_str = str(_SRC_PARENT)
    if src_parent_str not in sys.path:
        sys.path.insert(0, src_parent_str)

__path__ = extend_path(__path__, __name__)

