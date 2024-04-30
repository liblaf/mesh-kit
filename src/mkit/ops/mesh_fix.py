import pymeshfix
import pyvista as pv
import trimesh

from mkit import io as _io


def mesh_fix(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
) -> trimesh.Trimesh:
    mesh_pv: pv.PolyData = _io.as_pyvista(mesh)
    fixer = pymeshfix.MeshFix(mesh_pv)
    fixer.repair(
        verbose=verbose,
        joincomp=joincomp,
        remove_smallest_components=remove_smallest_components,
    )
    mesh_pv = fixer.mesh
    return _io.as_trimesh(mesh_pv)
