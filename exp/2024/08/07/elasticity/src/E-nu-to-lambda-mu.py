import dataclasses

from icecream import ic
from omegaconf import OmegaConf


@dataclasses.dataclass(kw_only=True)
class Config:
    E: float = 3e3
    nu: float = 0.46


def main() -> None:
    cfg: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    ic(cfg)

    E: float = cfg.E
    nu: float = cfg.nu
    mu: float = E / (2 * (1 + nu))
    lambda_: float = E * nu / ((1 + nu) * (1 - 2 * nu))
    ic(mu)
    ic(lambda_)


if __name__ == "__main__":
    main()
