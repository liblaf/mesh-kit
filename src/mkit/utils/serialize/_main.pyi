from typing import Any

from mkit.typing import StrPath

def load(fpath: StrPath, ext: str | None = None, **kwargs) -> Any: ...
def save(data: Any, fpath: StrPath, ext: str | None = None, **kwargs) -> None: ...
