import mkit.io
import trimesh

face: trimesh.Trimesh = mkit.io.load_trimesh(
    "/home/liblaf/Documents/data/template/00-face.ply"
)
skull: trimesh.Trimesh = mkit.io.load_trimesh(
    "/home/liblaf/Documents/data/template/00-skull.ply"
)
skull.faces = skull.faces[:, ::-1]
head: trimesh.Trimesh = trimesh.util.concatenate(face, skull)
head.export("head.ply")
