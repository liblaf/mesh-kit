import pyvista as pv
from icecream import ic
from pyvista.plotting.plotter import Plotter
from pyvista.plotting.themes import DocumentProTheme

reader = pv.DICOMReader("/home/liblaf/Documents/data/CT/120056/pre/")
ct: pv.ImageData = reader.read()
ic(ct)

pl = Plotter(off_screen=True, window_size=(800, 800), theme=DocumentProTheme())
pl.add_axes()
pl.add_volume(ct, opacity="sigmoid_5", cmap="bone")
pl.view_vector([0, 1, 0], [0, 0, -1])
pl.zoom_camera(1.8)
pl.save_graphic("ct-front.pdf")


pl.view_vector([1, 0, 0], [0, 0, -1])
pl.zoom_camera(1.8)
pl.save_graphic("ct-lateral.pdf")
