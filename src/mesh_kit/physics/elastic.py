# https://en.wikipedia.org/wiki/Template:Elastic_moduli


def E_nu2lambda(E: float, nu: float) -> float:
    return E * nu / ((1 + nu) * (1 - 2 * nu))


def E_nu2G(E: float, nu: float) -> float:
    return E / (2 * (1 + nu))
