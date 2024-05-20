import pathlib

import pyvista as pv
from pyvista.plotting.plotter import Plotter
from pyvista.plotting.themes import DocumentProTheme

DATA_DIR = pathlib.Path("/home/liblaf/Documents/data/target/120056/pre")
TEMPLATE_DIR = pathlib.Path("/home/liblaf/Documents/data/template")

pl = Plotter(window_size=(600, 800), theme=DocumentProTheme())
pl.add_axes()

target: pv.PolyData = pv.read(DATA_DIR / "02-face.ply")
template: pv.PolyData = pv.read(DATA_DIR / "03-face.vtu")
pl.add_mesh(target, label="Target Face", opacity=0.3)
pl.add_mesh(template, label="Template Face", opacity=0.3)
pl.view_vector([1.0, 0.0, 0.0], viewup=[0.0, 0.0, -1.0])
pl.zoom_camera(1.6)
pl.add_legend(size=(0.5, 0.2), loc="upper center")
pl.save_graphic("align-face.pdf")
pl.clear_actors()

pl.set_color_cycler("default")
target: pv.PolyData = pv.read(DATA_DIR / "02-skull.ply")
pl.add_mesh(target, label="Target Skull", opacity=0.3)
template: pv.PolyData = pv.read(DATA_DIR / "03-mandible.vtu")
pl.add_mesh(template, label="Template Mandible", opacity=0.3)
template: pv.PolyData = pv.read(DATA_DIR / "03-maxilla.vtu")
pl.add_mesh(template, label="Template Maxilla", opacity=0.3)
pl.add_legend(size=(0.5, 0.2), loc="upper center")
pl.save_graphic("align-skull.pdf")
