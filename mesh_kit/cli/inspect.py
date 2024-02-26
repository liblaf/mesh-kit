import pathlib
import sys
import tempfile
from collections.abc import Mapping
from typing import Annotated, Any, Optional

import numpy as np
import rich
import typer

from mesh_kit import cli as _cli
from mesh_kit.typing import check_type as _check_type


def _inspect(data, key: str | None = None, *, raw: bool = False) -> None:
    if key:
        data = _check_type(data, Mapping[str, Any])  # type: ignore[type-abstract]
        _inspect(data.get(key), raw=raw)
    elif raw:
        match data:
            case np.ndarray():
                with tempfile.TemporaryFile(suffix=".txt") as fp:
                    np.savetxt(
                        fp,
                        data,
                        fmt="%d" if np.issubdtype(data.dtype, np.integer) else "%.18e",
                    )
                    fp.flush()
                    fp.seek(0)
                    sys.stdout.write(fp.read().decode())
                    return
            case _:
                raise NotImplementedError
    else:
        rich.inspect(data)


def main(
    path: Annotated[pathlib.Path, typer.Argument(exists=True)],
    *,
    key: Annotated[Optional[str], typer.Option()] = None,
    raw: Annotated[bool, typer.Option()] = False,
) -> None:
    match path.suffix:
        case ".npy" | ".npz":
            data = np.load(path)
            _inspect(data, key=key, raw=raw)
        case _:
            raise NotImplementedError


if __name__ == "__main__":
    _cli.run(main)
