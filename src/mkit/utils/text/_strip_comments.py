import re
from collections.abc import Generator


def strip_comments(text: str, comments: str = "#") -> Generator[str, None, None]:
    text = re.sub(re.escape(comments) + r".*$", "", text, flags=re.MULTILINE)
    for line in text.strip().splitlines():
        s: str = line.strip()
        if s:
            yield s
