import pathlib

__all__ = ["landmarks"]


def landmarks(mesh_path: pathlib.Path) -> pathlib.Path:
    return mesh_path.with_stem(mesh_path.stem + "-landmarks").with_suffix(".txt")
