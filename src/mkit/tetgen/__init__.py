import functools
import pathlib
import subprocess
import tempfile

import meshio
from numpy import typing as npt


class TetGen(meshio.Mesh):
    def write_smesh(self, file: pathlib.Path) -> None:
        with file.open("w") as fp:
            fprint = functools.partial(print, file=fp)
            fprint("# Part 1 - node list")
            fprint(
                "# <# of points> <dimension (3)> <# of attributes> <boundary markers (0 or 1)>"
            )
            fprint(f"{len(self.points)} 3 0 0")
            fprint("# <point #> <x> <y> <z> [attributes] [boundary marker]")
            for point_id, point in enumerate(self.points):
                fprint(point_id, *point)

            fprint()
            fprint("# Part 2 - facet list")
            fprint("# <# of facets> <boundary markers (0 or 1)>")
            faces: npt.NDArray = self.get_cells_type("triangle")
            fprint(len(faces), 0)
            fprint("# <# of corners> <corner 1> ... <corner #> [boundary marker]")
            for face in faces:
                fprint(len(face), *face)

            fprint()
            fprint("# Part 3 - hole list")
            fprint("# <# of holes>")
            holes: npt.NDArray = self.field_data.get("holes", [])
            fprint(len(holes))
            fprint("# <hole #> <x> <y> <z>")
            for hole_id, hole in enumerate(holes):
                fprint(hole_id, *hole)

            fprint()
            fprint("# Part 4 - region attributes list")
            fprint("# <# of region>")
            fprint(0)

    def tetgen(self, switches: list[str] | None = None) -> meshio.Mesh:
        if switches is None:
            switches = ["-q", "-Y", "-q", "-O", "-z", "-k", "-C", "-V"]
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)
            input_file: pathlib.Path = tmpdir / "mesh.smesh"
            self.write_smesh(input_file)
            subprocess.run(["tetgen", *switches, input_file], check=True)
            return meshio.read(tmpdir / "mesh.1.vtk")


def from_meshio(mesh: meshio.Mesh) -> TetGen:
    return TetGen(points=mesh.points, cells=mesh.cells, field_data=mesh.field_data)
