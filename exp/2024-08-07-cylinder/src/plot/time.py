import matplotlib as mpl
import matplotlib.pyplot as plt
import pyvista as pv
from mkit.physics.preset.elastic import MODELS


def main() -> None:
    mpl.rcParams["figure.dpi"] = 300
    data: dict[str, float] = {}
    for name, cfg in MODELS.items():
        mesh: pv.UnstructuredGrid = pv.read(f"data/{name}.vtu")
        data[cfg.name] = mesh.field_data["execution_time"].item()  # pyright: ignore [reportArgumentType]
    plt.figure()
    plt.barh(data.keys(), data.values())  # pyright: ignore [reportArgumentType]
    plt.xlabel("Execution Time (s)")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig("plot/time.png")
    plt.close()


if __name__ == "__main__":
    main()
