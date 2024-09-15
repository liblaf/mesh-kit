import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv

from mkit.physics.preset.elastic import MODELS


def main() -> None:
    mpl.rcParams["figure.dpi"] = 300
    rest: pv.UnstructuredGrid = pv.read("data/input.vtu")
    data: dict[str, float] = {}
    for name, cfg in MODELS.items():
        mesh: pv.UnstructuredGrid = pv.read(f"data/{name}.vtu")
        mesh.warp_by_vector("solution", inplace=True, progress_bar=True)
        data[cfg.name] = mesh.volume / rest.volume
    plt.figure()
    plt.barh(data.keys(), data.values())  # pyright: ignore [reportArgumentType]
    plt.xlabel("Relative Volume Change")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig("plot/volume.png")
    plt.close()


if __name__ == "__main__":
    main()
