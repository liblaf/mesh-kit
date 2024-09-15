import pyvista as pv

mesh: pv.UnstructuredGrid = pv.read("slide.vtu")
mesh_warp: pv.UnstructuredGrid = mesh.warp_by_vector("solution", progress_bar=True)
mesh_clip: pv.UnstructuredGrid = mesh_warp.clip(
    "-x", (0, 0, 0), progress_bar=True, crinkle=True
)
pl = pv.Plotter(off_screen=True)
pl.add_axes()  # pyright: ignore [reportCallIssue]
pl.add_mesh(mesh_clip, scalars="energy_density", name="mesh")
camera: pv.Camera = pl.camera
camera.tight(0.2, view="zy")
camera.to_paraview_pvcc("camera.pvcc")
pl.camera = pv.Camera.from_paraview_pvcc("camera.pvcc")
pl.screenshot("slide.png")

mesh_clip = mesh.clip("-x", (0, 0, 0), progress_bar=True, crinkle=True)
pl.add_mesh(mesh_clip, scalars="slide_mask", name="mesh")
pl.screenshot("slide-mask.png")

mesh = pv.read("ref.vtu")
mesh_warp = mesh.warp_by_vector("solution", progress_bar=True)
mesh_clip = mesh_warp.clip("-x", (0, 0, 0), progress_bar=True, crinkle=True)
pl.add_mesh(mesh_clip, scalars="energy_density", name="mesh")
pl.screenshot("ref.png")
