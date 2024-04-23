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
    fixer = pymeshfix.MeshFix(pv.wrap(mesh))
    fixer.repair(
        verbose=verbose,
        joincomp=joincomp,
        remove_smallest_components=remove_smallest_components,
    )
    return _io.to_trimesh(fixer.mesh)
