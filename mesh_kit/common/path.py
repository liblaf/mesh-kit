from pathlib import Path


def landmarks_filepath(filepath: Path) -> Path:
    return filepath.with_stem(filepath.stem + "-landmarks").with_suffix(".txt")
