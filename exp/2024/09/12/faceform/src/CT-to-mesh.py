from pathlib import Path

import pyvista as pv

import mkit


class Config(mkit.cli.BaseConfig):
    ct: Path
    threshold: float = -200.0
    output: Path


@mkit.cli.auto_run()
def main(cfg: Config) -> None:
    ct: pv.ImageData = pv.read(cfg.ct, force_ext=".dcm")
    ct = ct.gaussian_smooth(progress_bar=True)
    contours: pv.PolyData = ct.contour([cfg.threshold], progress_bar=True)  # pyright: ignore [reportArgumentType]
    surface: pv.PolyData = contours.connectivity("largest", progress_bar=True)
    surface.save(cfg.output)
