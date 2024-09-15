from pathlib import Path
from typing import Any, Literal, TypeVar

import pydantic

from mkit.typing._types import StrPath

_C = TypeVar("_C", bound=pydantic.BaseModel)


def load_pydantic(
    cls: type[_C],
    fpath: StrPath = "params.json",
    ext: Literal["json", "toml", "yaml"] | None = None,
) -> _C:
    fpath: Path = Path(fpath)
    if ext is None:
        ext = fpath.suffix[1:]
    obj: Any
    match ext:
        case "json":
            obj = load_json(fpath)
        case "toml":
            obj = load_toml(fpath)
        case "yaml":
            obj = load_yaml(fpath)
        case _:
            msg: str = f"Unsupported extension: {ext}"
            raise ValueError(msg)
    return cls.model_validate(obj)


def save_pydantic(
    obj: pydantic.BaseModel,
    fpath: StrPath = "params.json",
    ext: Literal["json", "toml", "yaml"] | None = None,
) -> None:
    fpath: Path = Path(fpath)
    if ext is None:
        ext = fpath.suffix[1:]
    o: dict[str, Any] = obj.model_dump()
    match ext:
        case "json":
            save_json(o, fpath)
        case "toml":
            save_toml(o, fpath)
        case "yaml":
            save_yaml(o, fpath)
        case _:
            msg: str = f"Unsupported extension: {ext}"
            raise ValueError(msg)


def load_json(fpath: StrPath) -> Any:
    import json

    fpath: Path = Path(fpath)
    with fpath.open("r") as fp:
        return json.load(fp)


def load_toml(fpath: StrPath) -> Any:
    import toml

    fpath: Path = Path(fpath)
    with fpath.open("r") as fp:
        return toml.load(fp)


def load_yaml(fpath: StrPath) -> Any:
    import yaml

    fpath: Path = Path(fpath)
    with fpath.open("r") as fp:
        return yaml.safe_load(fp)


def save_json(obj: Any, fpath: StrPath) -> None:
    import json

    fpath: Path = Path(fpath)
    with fpath.open("w") as fp:
        json.dump(obj, fp)


def save_toml(obj: Any, fpath: StrPath) -> None:
    import toml

    fpath: Path = Path(fpath)
    with fpath.open("w") as fp:
        toml.dump(obj, fp)


def save_yaml(obj: Any, fpath: StrPath) -> None:
    import yaml

    fpath: Path = Path(fpath)
    with fpath.open("w") as fp:
        yaml.safe_dump(obj, fp)
