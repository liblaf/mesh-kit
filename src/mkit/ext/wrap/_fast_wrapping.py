import tempfile
from pathlib import Path
from typing import Any


def fast_wrapping(source: Any, target: Any) -> None:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir: Path = Path(tmpdir_str)
