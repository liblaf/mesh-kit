import pymeshfix
import pyvista as pv
import trimesh

import mkit.io


def mesh_fix(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
) -> trimesh.Trimesh:
    mesh_pv: pv.PolyData = mkit.io.as_pyvista(mesh)
    fixer = pymeshfix.MeshFix(mesh_pv)
    fixer.repair(
        verbose=verbose,
        joincomp=joincomp,
        remove_smallest_components=remove_smallest_components,
    )
    mesh_pv = fixer.mesh
    return mkit.io.as_trimesh(mesh_pv)
