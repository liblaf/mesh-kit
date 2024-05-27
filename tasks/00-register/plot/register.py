import pathlib

import pyvista as pv
from pyvista.plotting.plotter import Plotter
from pyvista.plotting.themes import DocumentProTheme
from vtkmodules.vtkRenderingAnnotation import vtkLegendBoxActor
from vtkmodules.vtkRenderingCore import vtkTextProperty

DATA_DIR = pathlib.Path("/home/liblaf/Documents/data/target/120056/pre")
TEMPLATE_DIR = pathlib.Path("/home/liblaf/Documents/data/template")

pl = Plotter(window_size=(600, 800), theme=DocumentProTheme())
pl.add_axes()

target: pv.PolyData = pv.read(DATA_DIR / "02-face.ply")
template: pv.PolyData = pv.read(DATA_DIR / "99-face.vtu")
pl.add_mesh(target, label="目标面部", opacity=0.3)
pl.add_mesh(template, label="配准结果", opacity=0.3)
pl.view_vector([1.0, 0.0, 0.0], viewup=[0.0, 0.0, -1.0])
pl.zoom_camera(1.6)
legend: vtkLegendBoxActor = pl.add_legend(size=(0.5, 0.2), loc="upper center")
text: vtkTextProperty = legend.GetEntryTextProperty()
text.SetFontFamily(4)
text.SetFontFile("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc")
pl.save_graphic("register-face.svg")
pl.clear_actors()

pl.set_color_cycler("default")
target: pv.PolyData = pv.read(DATA_DIR / "02-skull.ply")
pl.add_mesh(target, label="目标颌骨", opacity=0.3)
template: pv.PolyData = pv.read(DATA_DIR / "99-mandible.vtu")
pl.add_mesh(template, label="配准结果下颌骨", opacity=0.3)
template: pv.PolyData = pv.read(DATA_DIR / "99-maxilla.vtu")
pl.add_mesh(template, label="配准结果上颌骨", opacity=0.3)
legend: vtkLegendBoxActor = pl.add_legend(size=(0.5, 0.2), loc="upper center")
text: vtkTextProperty = legend.GetEntryTextProperty()
text.SetFontFamily(4)
text.SetFontFile("/usr/share/fonts/noto-cjk/NotoSansCJK-Light.ttc")
pl.save_graphic("register-skull.svg")
