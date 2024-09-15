from pathlib import Path

import mkit
import pyvista as pv


class Config(mkit.cli.BaseConfig):
    camera: Path | None = None
    fig: Path
    input: Path


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = pv.read(cfg.input)
    pl: pv.Plotter = pv.Plotter(off_screen=True)
    pl.add_axes()  # pyright: ignore [reportCallIssue]
    pl.add_mesh(mesh, color="white")
    if cfg.camera is not None:
        mkit.plot.load_camera(pl, cfg.camera)
    pl.save_graphic(cfg.fig)


if __name__ == "__main__":
    mkit.cli.run(main)
