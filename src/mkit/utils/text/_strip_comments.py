from collections.abc import Iterable


def strip_comments(text: str) -> Iterable[str]:
    for line in text.splitlines():
        s: str
        s, _, _ = line.partition("#")
        s = s.strip()
        if s:
            yield s
