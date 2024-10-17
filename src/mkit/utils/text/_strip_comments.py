from collections.abc import Iterable


def strip_comments(text: str, comments: str = "#") -> Iterable[str]:
    for line in text.strip().splitlines():
        stripped: str = line.split(comments, 1)[0].strip()
        if stripped:
            yield stripped
