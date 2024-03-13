import pathlib
import subprocess
import tempfile

import meshtaichi_patcher
import taichi as ti
import trimesh


def box(relations: list[str] | None = None) -> ti.MeshInstance:
    if relations is None:
        relations = []
    with tempfile.TemporaryDirectory() as _tmpdir:
        tmpdir = pathlib.Path(_tmpdir)
        mesh_tri: trimesh.Trimesh = trimesh.creation.box()
        mesh_tri.export(tmpdir / "box.ply", encoding="ascii")
        subprocess.run(
            ["tetgen", "-p", "-q", "-O", "-z", "-V", tmpdir / "box.ply"], check=True
        )
        mesh: ti.MeshInstance = meshtaichi_patcher.load_mesh(
            str(tmpdir / "box.1.node"), relations=relations
        )
    return mesh
