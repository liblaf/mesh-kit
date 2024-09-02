from pathlib import Path
from typing import ClassVar

import confz
from loguru import logger


class BaseConfig(confz.BaseConfig):
    log_level: int | str = "INFO"
    log_file: Path | None = None
    CONFIG_SOURCES: ClassVar = [confz.FileSource("params.yaml"), confz.CLArgSource()]

    def __post_init__(self) -> None:
        logger.info("{}", self)
