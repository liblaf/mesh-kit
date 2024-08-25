import mkit
import mkit.creation
import mkit.creation.tetmesh
import numpy as np
import pyvista as pv


def main() -> None:
    mesh: pv.UnstructuredGrid = mkit.creation.tetmesh.tetrahedron()
    pl: pv.Plotter = pv.Plotter(off_screen=True)
    camera: pv.Camera = pl.camera
    pl.add_axes()  # pyright: ignore [reportCallIssue]
    pl.add_mesh(mesh)
    pl.add_arrows(mesh.points[0], np.asarray([1.0, 0, 0]), name="arrow")
    camera.tight(0.1)
    pl.screenshot("plot/tet/rest.png")
    pl.add_arrows(mesh.points[2], np.asarray([0, 0, 1.0]), name="arrow")
    camera.tight(0.1)
    pl.update()
    pl.screenshot("plot/tet/shear.png")
    pl.remove_actor("arrow")  # pyright: ignore [reportArgumentType]
    mesh.points[0][0] -= 0.5
    camera.tight(0.1)
    pl.update()
    pl.screenshot("plot/tet/squash.png")
    mesh.points[0][0] += 1.5
    camera.tight(0.1)
    pl.update()
    pl.screenshot("plot/tet/stretch.png")


if __name__ == "__main__":
    main()
