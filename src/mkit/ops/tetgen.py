import functools
import pathlib
import subprocess
import tempfile
from collections.abc import Iterator
from typing import Any

import meshio
import numpy as np
from numpy import typing as npt

from mkit.typing import StrPath


def tetgen(mesh: meshio.Mesh) -> meshio.Mesh:
    """
    Args:
        mesh: input mesh

    Returns:
        tetrahedral mesh
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        input_file: pathlib.Path = tmpdir / "mesh.smesh"
        save_smesh(input_file, mesh)
        subprocess.run(
            ["tetgen", "-p", "-q", "-O", "-z", "-k", "-C", "-V", input_file],
            check=True,
        )
        tetra_mesh: meshio.Mesh = meshio.read(tmpdir / "mesh.1.vtk")
        points: npt.NDArray[np.floating] = tetra_mesh.points
        tetra: npt.NDArray[np.intp] = tetra_mesh.get_cells_type("tetra")
        faces: npt.NDArray[np.intp]
        boundary_marker: npt.NDArray[np.intp]
        faces, boundary_marker = load_face(tmpdir / "mesh.1.face")
        return meshio.Mesh(
            points=points,
            cells=[("tetra", tetra), ("triangle", faces)],
            cell_data={
                "boundary_marker": [np.zeros(len(tetra), np.intp), boundary_marker]
            },
        )


def load_face(
    file: pathlib.Path,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    lines: list[str] = list(strip_comments(file))
    # <# of faces> <boundary marker (0 or 1)>
    num_faces, has_boundary_marker = map(int, lines[0].split())
    faces: npt.NDArray[np.intp] = np.zeros((num_faces, 3), np.intp)
    boundary_marker: npt.NDArray[np.intp] = np.zeros(num_faces, np.intp)
    if has_boundary_marker:
        for line in lines[1:]:
            # <face #> <node> <node> <node> ... [boundary marker] ...
            face_id, *face, marker = map(int, line.split())
            faces[face_id] = face
            boundary_marker[face_id] = marker
        return faces, boundary_marker
    else:
        for line in lines[1:]:
            # <face #> <node> <node> <node> ... [boundary marker] ...
            face_id, *face = map(int, line.split())
            faces[face_id] = face
    return faces, boundary_marker


def save_smesh(file: StrPath, mesh: meshio.Mesh) -> None:
    file = pathlib.Path(file)
    with file.open("w") as f:
        fprint = functools.partial(print, file=f)
        fprint("# Part 1 - node list")
        fprint(
            "# <# of points> <dimension (3)> <# of attributes> <boundary markers (0 or 1)>"
        )
        if "boundary_marker" in mesh.point_data:
            fprint(f"{len(mesh.points)} 3 0 1")
            fprint("# <point #> <x> <y> <z> [attributes] [boundary marker]")
            point_boundary_marker: npt.NDArray[np.intp] = np.asarray(
                mesh.point_data["boundary_marker"]
            )
            for point_id, point in enumerate(mesh.points):
                fprint(point_id, *point, point_boundary_marker[point_id])
        else:
            fprint(f"{len(mesh.points)} 3 0 0")
            fprint("# <point #> <x> <y> <z> [attributes] [boundary marker]")
            for point_id, point in enumerate(mesh.points):
                fprint(point_id, *point)

        fprint()
        fprint("# Part 2 - facet list")
        fprint("# <# of facets> <boundary markers (0 or 1)>")
        faces: npt.NDArray[np.intp] = mesh.get_cells_type("triangle")
        if "boundary_marker" in mesh.cell_data:
            face_boundary_marker: npt.NDArray[np.intp] = mesh.get_cell_data(
                "boundary_marker", "triangle"
            )
            fprint(len(faces), 1)
            fprint("# <# of corners> <corner 1> ... <corner #> [boundary marker]")
            for face_id, face in enumerate(faces):
                fprint(len(face), *face, face_boundary_marker[face_id])
        else:
            fprint(len(faces), 0)
            fprint("# <# of corners> <corner 1> ... <corner #> [boundary marker]")
            for face in faces:
                fprint(len(face), *face)

        fprint()
        fprint("# Part 3 - hole list")
        fprint("# <# of holes>")
        holes: npt.NDArray[np.float64] = mesh.field_data.get("holes", [])
        fprint(len(holes))
        fprint("# <hole #> <x> <y> <z>")
        for hole_id, hole in enumerate(holes):
            fprint(hole_id, *hole)

        fprint()
        fprint("# Part 4 - region attributes list")
        fprint("# <# of region>")
        fprint(0)


def strip_comments(file: pathlib.Path) -> Iterator[str]:
    _: Any
    with file.open() as f:
        for line in f:
            line: str
            line, _, _ = line.partition("#")
            line = line.strip()
            if line:
                yield line
