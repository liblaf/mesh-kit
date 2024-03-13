import pathlib
import subprocess
from typing import no_type_check

import numpy as np
import taichi as ti
import trimesh
from trimesh import creation

from mesh_kit.linalg import cg
from mesh_kit.physics import mtm as _mtm
from mesh_kit.taichi.mesh import create

E: float = 3000
nu: float = 0.47


def test_force(tmp_path: pathlib.Path) -> None:
    ti.init(default_fp=ti.f64)
    surface: trimesh.Trimesh = creation.box()
    surface.export(tmp_path / "surface.ply", encoding="ascii")
    subprocess.run(["tetgen", "-z", tmp_path / "surface.ply"])
    mesh: ti.MeshInstance = create.box(relations=["CE", "CV", "EV", "VE"])
    mesh.verts.place({"b": ti.math.vec3})
    mtm = _mtm.MTM(mesh)
    mtm.init(E, nu)
    mtm.calc_stiffness()

    @no_type_check
    @ti.kernel
    def _x(mesh: ti.template(), x: ti.template()):
        for v in mesh.verts:
            v.x = x[v.id]

    @no_type_check
    @ti.kernel
    def _Ax(mesh: ti.template(), Ax: ti.template()):
        for v in mesh.verts:
            Ax[v.id] = v.f

    def calc_force(x: ti.MatrixField, Ax: ti.MatrixField) -> None:
        _x(mesh, x)
        mtm.calc_force()
        _Ax(mesh, Ax)

    b: ti.MatrixField = mesh.verts.get_member_field("b")
    b.fill(0)
    x: ti.MatrixField = mesh.verts.get_member_field("x")
    rng: np.random.Generator = np.random.default_rng()
    x.from_numpy(rng.random(size=(*x.shape, 3)))
    converge: bool = cg.cg(ti.linalg.LinearOperator(calc_force), b, x, tol=1e-9)
    assert converge
    np.testing.assert_allclose(x.to_numpy(), np.zeros(shape=(*x.shape, 3)), atol=1e-7)
