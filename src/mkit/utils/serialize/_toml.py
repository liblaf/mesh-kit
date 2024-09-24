from pathlib import Path
from typing import Any

import tomlkit

from mkit.typing import StrPath


def load_toml(fpath: StrPath) -> tomlkit.TOMLDocument:
    fpath: Path = Path(fpath)
    with fpath.open() as fp:
        return tomlkit.load(fp)


def save_toml(data: Any, fpath: StrPath, *, sort_keys: bool = False) -> None:
    fpath: Path = Path(fpath)
    with fpath.open("w") as fp:
        tomlkit.dump(data, fp, sort_keys=sort_keys)
