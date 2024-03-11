import functools
import io
import pathlib

from numpy import typing as npt

from mesh_kit.io import node as _node


def faces_to_str(
    faces: npt.NDArray, *, boundary_marker: npt.NDArray | None = None
) -> str:
    fp = io.StringIO()
    fprint = functools.partial(print, file=fp)
    fprint("# <# of facets> <boundary markers (0 or 1)>")
    fprint(len(faces), 1 if boundary_marker is not None else 0)
    fprint("# <# of corners> <corner 1> ... <corner #> [boundary marker]")
    for i, face in enumerate(faces):
        fprint(len(face), *face, end="")
        if boundary_marker is not None:
            fprint("", boundary_marker[i], end="")
        fprint()
    return fp.getvalue()


def to_str(
    verts: npt.NDArray,
    faces: npt.NDArray,
    holes: npt.NDArray | None = None,
    region: npt.NDArray | None = None,
    *,
    attrs: npt.NDArray | None = None,
    vert_boundary_marker: npt.NDArray | None = None,
    face_boundary_marker: npt.NDArray | None = None,
) -> str:
    fp = io.StringIO()
    fprint = functools.partial(print, file=fp)

    fprint("# Part 1 - node list")
    fprint(_node.to_str(verts, attrs=attrs, boundary_marker=vert_boundary_marker))

    fprint("\n# Part 2 - facet list")
    fprint(faces_to_str(faces, boundary_marker=face_boundary_marker))

    fprint("\n# Part 3 - hole list")
    fprint("# <# of holes>")
    if holes is not None:
        fprint(len(holes))
        fprint("# <hole #> <x> <y> <z>")
        for i, hole in enumerate(holes):
            fprint(i, *hole)

    fprint("\n# Part 4 - region list")
    fprint("# <# of regions>")
    if region is not None:
        fprint(len(region))
        fprint("# <region #> <x> <y> <z> <region number> <region attribute>")
        raise NotImplementedError
    else:
        fprint(0)

    return fp.getvalue()


def save(
    file: pathlib.Path,
    verts: npt.NDArray,
    faces: npt.NDArray,
    holes: npt.NDArray | None = None,
    region: npt.NDArray | None = None,
    *,
    attrs: npt.NDArray | None = None,
    vert_boundary_marker: npt.NDArray | None = None,
    face_boundary_marker: npt.NDArray | None = None,
) -> None:
    file.write_text(
        to_str(
            verts,
            faces,
            holes,
            region,
            attrs=attrs,
            vert_boundary_marker=vert_boundary_marker,
            face_boundary_marker=face_boundary_marker,
        )
    )
