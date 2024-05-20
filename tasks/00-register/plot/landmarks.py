import pathlib

import numpy as np
import pyvista as pv
from numpy import typing as npt
from pyvista.plotting.plotter import Plotter
from pyvista.plotting.themes import DocumentProTheme

DATA_DIR = pathlib.Path("/home/liblaf/Documents/data/target/120056/pre")
TEMPLATE_DIR = pathlib.Path("/home/liblaf/Documents/data/template")

pl = Plotter(window_size=(600, 800), theme=DocumentProTheme())
pl.add_axes()

mesh: pv.PolyData = pv.read(TEMPLATE_DIR / "99-face.vtu")
landmarks: npt.NDArray[np.floating] = np.loadtxt(DATA_DIR / "face-template.1.xyz")
pl.add_mesh(mesh, color="#FFCC99")
pl.view_vector([0.0, -1.0, 0.0], viewup=[0.0, 0.0, 1.0])
pl.zoom_camera(1.9)
pl.add_point_labels(
    landmarks,
    range(len(landmarks)),
    shape_color="green",
    render_points_as_spheres=True,
    always_visible=True,
)
pl.save_graphic("face-landmarks.pdf")
pl.clear_actors()

mesh: pv.PolyData = pv.read(TEMPLATE_DIR / "99-mandible.vtu")
landmarks: npt.NDArray[np.floating] = np.loadtxt(DATA_DIR / "mandible-template.1.xyz")
pl.add_mesh(mesh, color="#E3DAC9")
pl.add_point_labels(
    landmarks,
    range(len(landmarks)),
    shape_color="green",
    render_points_as_spheres=True,
    always_visible=True,
)
mesh: pv.PolyData = pv.read(TEMPLATE_DIR / "99-maxilla.vtu")
# landmarks: npt.NDArray[np.floating] = np.loadtxt(DATA_DIR / "maxilla-template.1.xyz")
pl.add_mesh(mesh, color="#E3DAC9")
pl.save_graphic("skull-landmarks.pdf")
