from pathlib import Path

import numpy as np
import pandas as pd
import pydantic
from matplotlib import pyplot as plt

import mkit


class Config(mkit.cli.BaseConfig):
    eval_dir: pydantic.DirectoryPath = Path("data/eval/stable-neo-hookean/")


EXCLUDE = ["7033231985", "7041188826", "7047578008", "7048105963", "7052847448"]
INCLUDE = [
    "7036866897",
    "7038900108",
    "7040751699",
    "7041827472",
    "7046177792",
    "7048626922",
    "7049923545",
    "7050143922",
    "7050514803",
    "7051604154",
    "7051858539",
    "7051997935",
    "7054372341",
    "7058007223",
    "7058549950",
    "7059847523",
    "7059850042",
]


def main(cfg: Config) -> None:
    summary = pd.read_csv(cfg.eval_dir / "summary.csv", dtype={"patient": str})
    summary = summary.drop(summary[summary["patient"].isin(EXCLUDE)].index)
    for c in ["mean", "95", "99", "max"]:
        plt.figure()
        plt.bar(summary["patient"], summary[f"dist ({c})"])
        plt.xlabel("Patient")
        plt.xticks(rotation=90)
        plt.ylabel(f"Distance ({c})")
        plt.title(f"Distance ({c})")
        plt.tight_layout()
        plt.savefig(cfg.eval_dir / f"dist-{c}.png")
        plt.close()


mkit.cli.auto_run()(main)
