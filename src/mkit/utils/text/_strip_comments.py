from collections.abc import Generator


def strip_comments(text: str) -> Generator[str, None, None]:
    for line in text.strip().splitlines():
        s: str
        s, _, _ = line.partition("#")
        s = s.strip()
        if s:
            yield s
