"""https://en.wikipedia.org/wiki/Template:Elastic_moduli"""


def E_nu2lambda(E: float, nu: float) -> float:
    """
    Args:
        E: Young's modulus
        nu: Poisson's ratio

    Returns:
        Lamé's first parameter
    """
    return E * nu / ((1 + nu) * (1 - 2 * nu))


def E_nu2mu(E: float, nu: float) -> float:
    """
    Args:
        E: Young's modulus
        nu: Poisson's ratio

    Returns:
        Lamé's second parameter / Shear modulus
    """
    return E / (2 * (1 + nu))
