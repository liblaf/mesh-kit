import csv
import functools
import io

from loguru import logger
from matplotlib import font_manager


@functools.cache
def find_font(font_family: str) -> str | None:
    fonts: list[str] = next(csv.reader(io.StringIO(font_family)))
    fonts = [s.strip().strip("'\"") for s in fonts]
    for font in fonts:
        try:
            return font_manager.findfont(font, fallback_to_default=False)
        except ValueError:  # noqa: PERF203
            logger.exception("")
    return None


monospace = functools.partial(
    find_font,
    font_family="'FiraCode Nerd Font Mono', Consolas, 'Courier New', monospace",
)
