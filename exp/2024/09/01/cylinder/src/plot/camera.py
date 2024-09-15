from pathlib import Path

import mkit
import pyvista as pv


class Config(mkit.cli.BaseConfig):
    camera: Path
    input: Path


def main(cfg: Config) -> None:
    pl: pv.Plotter = pv.Plotter()
    for f in cfg.input.rglob("*.vtu"):
        mesh: pv.UnstructuredGrid = pv.read(f)
        if "solution" in mesh.point_data:
            mesh.warp_by_vector("solution", inplace=True, progress_bar=True)
        pl.add_mesh(mesh)
    renderer: pv.Renderer = pl.renderer
    renderer.view_isometric()
    pl.zoom_camera(1.2)
    mkit.plot.save_camera(cfg.camera, pl)


if __name__ == "__main__":
    mkit.cli.run(main)
