import json
import pathlib
from typing import Any

import pydantic


@pydantic.validate_call
def load_json(fpath: pydantic.FilePath, **kwargs) -> Any:
    with fpath.open() as fp:
        return json.load(fp, **kwargs)


@pydantic.validate_call
def save_json(data: Any, fpath: pathlib.Path, **kwargs) -> None:
    with fpath.open("w") as fp:
        json.dump(data, fp, **kwargs)
