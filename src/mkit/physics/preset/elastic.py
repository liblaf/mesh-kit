import dataclasses

from mkit.physics.energy import elastic
from mkit.physics.energy.abc import CellEnergy


@dataclasses.dataclass()
class Config:
    name: str
    energy_fn: CellEnergy
    params: dict[str, float]


E: float = 1e6  # Young's modulus
nu: float = 0.49  # Poisson's ratio
lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé's first parameter
G: float = E / (2 * (1 + nu))  # Shear modulus

MODELS: dict[str, Config] = {
    "linear": Config("Linear", elastic.linear, {"lambda": lambda_, "mu": G}),
    "saint_venant_kirchhoff": Config(
        "Saint Venant-Kirchhoff",
        elastic.saint_venant_kirchhoff_wiki,
        {"lambda": lambda_, "mu": G},
    ),
    "corotated": Config(
        "Corotated (Stomakhin 2012)",
        elastic.corotated_stomakhin,
        {"lambda": lambda_, "mu": G},
    ),
    "neo_hookean_macklin": Config(
        "Neo-Hookean (Macklin 2021)",
        elastic.neo_hookean_macklin,
        {"lambda": lambda_, "mu": G},
    ),
    "neo_hookean_bower": Config(
        "Neo-Hookean (Bower 2009)",
        elastic.neo_hookean_bower,
        {"lambda": lambda_, "mu": G},
    ),
    "neo_hookean_stable": Config(
        "Stable Neo-Hookean (Smith 2018)",
        elastic.neo_hookean_stable_smith,
        {"lambda": lambda_, "mu": G},
    ),
    "yeoh": Config(
        "Yeoh",
        elastic.yeoh_wiki,
        {"C10": 1.202e6, "C20": -0.057e6, "C30": 0.004e6, "C11": lambda_},
    ),
}
