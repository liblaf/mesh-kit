import json
from pathlib import Path
from typing import Any

from mkit.typing import StrPath


def load_json(fpath: StrPath, **kwargs) -> Any:
    fpath: Path = Path(fpath)
    with fpath.open() as fp:
        return json.load(fp, **kwargs)


def save_json(data: Any, fpath: StrPath, **kwargs) -> None:
    fpath: Path = Path(fpath)
    with fpath.open("w") as fp:
        json.dump(data, fp, **kwargs)
