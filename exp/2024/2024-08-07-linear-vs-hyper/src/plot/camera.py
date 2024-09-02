import pathlib

import mkit
import mkit.cli
import mkit.plot
import pyvista as pv
from icecream import ic


class Config(mkit.cli.BaseConfig):
    camera: pathlib.Path


def main(cfg: Config) -> None:
    pl: pv.Plotter = pv.Plotter(off_screen=True)
    camera: pv.Camera = pl.camera
    for filepath in pathlib.Path("data").rglob("*.vtu"):
        ic(filepath)
        mesh: pv.UnstructuredGrid = pv.read(filepath)
        if "solution" in mesh.point_data:
            mesh.warp_by_vector("solution", inplace=True, progress_bar=True)
        pl.add_mesh(mesh)
    camera.tight(0.1)

    mkit.plot.save_camera(cfg.camera, pl)


if __name__ == "__main__":
    mkit.cli.run(main)
