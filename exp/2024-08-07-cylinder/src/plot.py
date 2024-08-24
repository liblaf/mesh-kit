import pyvista as pv
from icecream import ic
from mkit.physics.preset.elastic import MODELS


def main() -> None:
    pl: pv.Plotter = pv.Plotter(off_screen=True)
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")
    pl.add_mesh(mesh)
    for name in MODELS:
        mesh = pv.read(f"data/{name}.vtu")
        mesh.warp_by_vector("solution", inplace=True, progress_bar=True)
        pl.add_mesh(mesh, scalars="energy_density")
    camera: pv.Camera = pl.camera
    camera.tight(0.1)
    pl.clear_actors()
    pl.add_axes()  # pyright: ignore [reportCallIssue]
    mesh = pv.read("data/input.vtu")
    volume_rest: float = mesh.volume
    pl.add_mesh(mesh, color="white", name="mesh")
    pl.save_graphic("plot/input.svg")
    for name, cfg in MODELS.items():
        ic(cfg.name, cfg.params)
        mesh = pv.read(f"data/{name}.vtu")
        mesh.warp_by_vector("solution", inplace=True, progress_bar=True)
        ic(mesh.volume / volume_rest)
        pl.add_mesh(mesh, scalars="energy_density", name="mesh")
        pl.save_graphic(f"plot/{name}.svg")


if __name__ == "__main__":
    main()
