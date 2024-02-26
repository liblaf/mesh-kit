import pymeshfix
import pyvista as pv
import trimesh

from mesh_kit import trimesh as _tri


def mesh_fix(
    mesh: trimesh.Trimesh,
    *,
    verbose: bool = False,
    joincomp: bool = False,
    remove_smallest_components: bool = True,
) -> trimesh.Trimesh:
    fix: pymeshfix.MeshFix = pymeshfix.MeshFix(pv.wrap(mesh))
    fix.repair(
        verbose=verbose,
        joincomp=joincomp,
        remove_smallest_components=remove_smallest_components,
    )
    return _tri.as_trimesh(fix.mesh)
