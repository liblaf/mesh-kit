def splitlines(text: str) -> list[str]:
    lines: list[str] = text.splitlines()
    lines = [s for line in lines if (s := line.partition("#")[0].strip())]
    return lines
