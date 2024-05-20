import pathlib

import meshio
import mkit
import mkit.array
import mkit.array.mask
import mkit.io
import pyvista as pv
import trimesh
from pyvista.plotting.camera import Camera
from pyvista.plotting.plotter import Plotter
from pyvista.plotting.themes import DocumentProTheme

SKIN_COLOR = "#FFCC99"
BONE_COLOR = "#E3DAC9"
DATA_DIR = pathlib.Path("/home/liblaf/Documents/data/target/113286")
IMG_DIR = pathlib.Path("img/113286")
IMG_DIR.mkdir(parents=True, exist_ok=True)
VIEW = "front"

pl = Plotter(
    off_screen=True,
    line_smoothing=True,
    polygon_smoothing=True,
    theme=DocumentProTheme(),
)
pl.add_axes()
camera: Camera = pl.camera

pl.window_size = (800, 600)
valid_io: meshio.Mesh = mkit.io.load_meshio(DATA_DIR / "eval.vtu")
valid_tr: trimesh.Trimesh = mkit.io.as_trimesh(valid_io)
valid_tr.update_faces(
    mkit.array.mask.vertex_to_face(valid_tr.faces, valid_io.point_data["validation"])
)
valid_pv: pv.PolyData = pv.wrap(valid_tr)
pl.add_mesh(valid_pv, name="mesh", reset_camera=True)
pl.view_vector([0.0, 1.0, 0.0], viewup=[0.0, 0.0, -1.0])
pl.zoom_camera(1.9)
pl.clear_actors()
camera.clipping_range = (1e2, 1e3)


blinder: pv.PolyData = pv.Box((70.0, 170.0, 0.0, 300.0, 100.0, 120.0))
pre_face: pv.PolyData = pv.read(DATA_DIR / "pre/02-face.ply")
pl.add_mesh(pre_face, color=SKIN_COLOR, smooth_shading=True, name="mesh")
pl.add_mesh(blinder, color="black", name="blinder")
pl.save_graphic(IMG_DIR / f"pre-face-{VIEW}.pdf")
pl.remove_actor("blinder")

pre_skull: pv.PolyData = pv.read(DATA_DIR / "pre/02-skull.ply")
pl.add_mesh(pre_skull, color=BONE_COLOR, smooth_shading=True, name="mesh")
pl.save_graphic(IMG_DIR / f"pre-skull-{VIEW}.pdf")

post_face: pv.PolyData = pv.read(DATA_DIR / "post/02-face.ply")
pl.add_mesh(blinder, color="black", name="blinder")
pl.add_mesh(post_face, color=SKIN_COLOR, smooth_shading=True, name="mesh")
pl.save_graphic(IMG_DIR / f"post-face-{VIEW}.pdf")
pl.remove_actor("blinder")

post_skull: pv.PolyData = pv.read(DATA_DIR / "post/02-skull.ply")
pl.add_mesh(post_skull, color=BONE_COLOR, smooth_shading=True, name="mesh")
pl.save_graphic(IMG_DIR / f"post-skull-{VIEW}.pdf")

predict: pv.PolyData = pv.read(DATA_DIR / "eval.vtu")
pl.add_mesh(blinder, color="black", name="blinder")
pl.add_mesh(predict, scalars="loss", smooth_shading=True, name="mesh")
pl.save_graphic(IMG_DIR / f"predict-{VIEW}.pdf")
pl.remove_actor("blinder")

VIEW = "lateral"

pl.window_size = (600, 800)
pl.add_mesh(valid_pv, name="mesh", reset_camera=True)
pl.view_vector([1.0, 0.0, 0.0], viewup=[0.0, 0.0, -1.0])
pl.zoom_camera(1.7)
pl.clear_actors()
print(camera.clipping_range)
camera.clipping_range = (1e2, 1e3)

blinder: pv.PolyData = pv.Box((0.0, 300.0, 180.0, 200.0, 100.0, 120.0))
pre_face: pv.PolyData = pv.read(DATA_DIR / "pre/02-face.ply")
pl.add_mesh(pre_face, color=SKIN_COLOR, smooth_shading=True, name="mesh")
pl.add_mesh(blinder, color="black", name="blinder")
pl.save_graphic(IMG_DIR / f"pre-face-{VIEW}.pdf")
pl.remove_actor("blinder")

pre_skull: pv.PolyData = pv.read(DATA_DIR / "pre/02-skull.ply")
pl.add_mesh(pre_skull, color=BONE_COLOR, smooth_shading=True, name="mesh")
pl.save_graphic(IMG_DIR / f"pre-skull-{VIEW}.pdf")

post_face: pv.PolyData = pv.read(DATA_DIR / "post/02-face.ply")
pl.add_mesh(blinder, color="black", name="blinder")
pl.add_mesh(post_face, color=SKIN_COLOR, smooth_shading=True, name="mesh")
pl.save_graphic(IMG_DIR / f"post-face-{VIEW}.pdf")
pl.remove_actor("blinder")

post_skull: pv.PolyData = pv.read(DATA_DIR / "post/02-skull.ply")
pl.add_mesh(post_skull, color=BONE_COLOR, smooth_shading=True, name="mesh")
pl.save_graphic(IMG_DIR / f"post-skull-{VIEW}.pdf")

predict: pv.PolyData = pv.read(DATA_DIR / "eval.vtu")
pl.add_mesh(blinder, color="black", name="blinder")
pl.add_mesh(predict, scalars="loss", smooth_shading=True, name="mesh")
pl.save_graphic(IMG_DIR / f"predict-{VIEW}.pdf")
pl.remove_actor("blinder")
