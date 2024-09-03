import dataclasses

from mkit.physics.energy import CellEnergy, elastic


@dataclasses.dataclass()
class Material:
    id: str
    name: str
    energy_fn: CellEnergy
    cell_data: dict[str, float]


E: float = 1e6  # Young's modulus
nu: float = 0.49  # Poisson's ratio
lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé's first parameter
G: float = E / (2 * (1 + nu))  # Shear modulus
MATERIALS: dict[str, Material] = {
    "linear": Material(
        "linear", "Linear", elastic.linear, {"mu": G, "lambda": lambda_}
    ),
    "st-venant-kirchhoff": Material(
        "st-venant-kirchhoff",
        "Saint Venant-Kirchhoff",
        elastic.saint_venant_kirchhoff,
        {"mu": G, "lambda": lambda_},
    ),
    "corotated": Material(
        "corotated",
        "Corotated (Stomakhin 2012)",
        elastic.corotated,
        {"mu": G, "lambda": lambda_},
    ),
    "neo-hookean": Material(
        "neo-hookean",
        "Neo-Hookean (Macklin 2021)",
        elastic.neo_hookean,
        {"mu": G, "lambda": lambda_},
    ),
    "stable-neo-hookean": Material(
        "stable-neo-hookean",
        "Stable Neo-Hookean (Smith 2018)",
        elastic.stable_neo_hookean,
        {"mu": G, "lambda": lambda_},
    ),
}


def presets() -> dict[str, Material]:
    return MATERIALS


def get_preset(_id: str) -> Material:
    return MATERIALS[_id]
