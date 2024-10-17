from __future__ import annotations

import trimesh as tm

from mkit.core.trimesh._base import TriMeshBase


class TrimeshMixin(TriMeshBase):
    @property
    def trimesh(self) -> tm.Trimesh:
        return tm.Trimesh(self.points, self.faces)
