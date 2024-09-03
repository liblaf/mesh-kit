from pathlib import Path

import pyvista as pv

import mkit


class Config(mkit.cli.BaseConfig):
    ct: Path
    mesh: Path
    threshold: float = -200


def main(cfg: Config) -> None:
    ct: pv.ImageData = pv.read(cfg.ct, force_ext=".dcm", progress_bar=True)
    ct = ct.gaussian_smooth(progress_bar=True)
    contour: pv.PolyData = ct.contour(
        [cfg.threshold],  # pyright: ignore [reportArgumentType]
        scalars="DICOMImage",
        progress_bar=True,
    )
    contour.connectivity("largest", inplace=True, progress_bar=True)
    contour.save(cfg.mesh)


if __name__ == "__main__":
    mkit.cli.run(main)
