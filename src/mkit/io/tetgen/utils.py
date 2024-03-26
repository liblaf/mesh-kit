def comment_strip(text: str) -> list[str]:
    lines: list[str] = text.splitlines()
    lines = [line.partition("#")[0] for line in lines]
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    return lines
