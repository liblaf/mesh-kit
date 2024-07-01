import numpy as np
import pyvista as pv
from icecream import ic
from numpy import typing as npt
from pyvista.plotting.plotter import Plotter
from pyvista.plotting.themes import DocumentProTheme

reader = pv.DICOMReader("/home/liblaf/Documents/data/CT/120056/pre/")
ct: pv.ImageData = reader.read()
# ct = ct.gaussian_smooth()
contour: pv.PolyData = ct.contour([-200])
components: pv.PolyData = contour.connectivity("all")
region_ids: npt.NDArray[np.integer] = np.unique(components["RegionId"])
largest: pv.PolyData = contour.connectivity("largest")
small: pv.PolyData = contour.connectivity("specified", region_ids[1:])
small = small.clip(origin=largest.center)
largest = largest.clip(origin=largest.center)
ic(contour)

pl = Plotter(window_size=(800, 800), theme=DocumentProTheme())
pl.add_axes()
pl.add_mesh(small, color="red", opacity=0.5)
pl.add_mesh(largest, color="white")
# pl.add_mesh(largest, color="FFF0F5")
pl.view_vector([1, 0, 0], [0, 0, -1])
pl.zoom_camera(1.3)
pl.save_graphic("contour-face-gaussian-off.pdf")
pl.show()
