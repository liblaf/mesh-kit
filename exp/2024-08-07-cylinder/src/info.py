import sys

import pyvista as pv
from icecream import ic


def main() -> None:
    mesh: pv.UnstructuredGrid = pv.read(sys.argv[1])
    ic(mesh)
    ic(mesh.point_data)
    ic(mesh.cell_data)
    ic(mesh.field_data)
    ic(dict(mesh.field_data))


if __name__ == "__main__":
    main()
