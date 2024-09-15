from pathlib import Path

import mkit


class CLIConfig(mkit.cli.CLIBaseConfig):
    dpath: Path


@mkit.cli.auto_run
def main(cfg: CLIConfig) -> None:
    dataset = mkit.io.cfg.dpath
