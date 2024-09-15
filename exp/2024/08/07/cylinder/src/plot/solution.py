import pathlib

import numpy as np
import pyvista as pv

import mkit.cli
import mkit.plot


class Config(mkit.cli.BaseConfig):
    camera: pathlib.Path | None = None
    fig: pathlib.Path
    solution: pathlib.Path


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = pv.read(cfg.solution)
    execution_time: float = np.asarray(mesh.field_data["execution_time"]).item()
    volume_rest: float = mesh.volume
    mesh.warp_by_vector("solution", inplace=True, progress_bar=True)
    relative_volume_change: float = mesh.volume / volume_rest
    pl: pv.Plotter = pv.Plotter(off_screen=True)
    pl.add_axes()  # pyright: ignore [reportCallIssue]
    pl.add_mesh(mesh, scalars="energy_density")
    if cfg.camera is not None:
        mkit.plot.load_camera(pl, cfg.camera)
    else:
        camera: pv.Camera = pl.camera
        camera.tight(0.1)
    pl.add_text(
        f"Relative Volume Change: {relative_volume_change:.2f}",
        position=(0, pl.window_size[1] - 1 * 18 * 2),  # pyright: ignore [reportArgumentType]
    )
    pl.add_text(
        f"Execution Time: {execution_time:.1f} s",
        position=(0, pl.window_size[1] - 2 * 18 * 2),  # pyright: ignore [reportArgumentType]
    )
    pl.save_graphic(cfg.fig)


if __name__ == "__main__":
    mkit.cli.run(main)
