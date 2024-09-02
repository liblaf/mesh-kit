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
}
