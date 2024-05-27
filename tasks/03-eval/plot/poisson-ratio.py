import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    df: pd.DataFrame = pd.read_csv("data/poisson-ratio.csv")
    df = df[df["nu"] < 0.39]
    plt.figure()
    plt.plot(df["nu"], df["mean"])
    plt.xlabel("Poisson's ratio")
    plt.ylabel("Mean Error (mm)")
    plt.savefig("img/mean.pdf")

    plt.figure()
    plt.plot(df["nu"], df["95%"])
    plt.xlabel("Poisson's ratio")
    plt.ylabel("95% Error (mm)")
    plt.savefig("img/95.pdf")

    plt.figure()
    plt.plot(df["nu"], df["max"])
    plt.xlabel("Poisson's ratio")
    plt.ylabel("Max Error (mm)")
    plt.savefig("img/max.pdf")

    df: pd.DataFrame = pd.read_csv("data/cond.csv")
    df = df[df["nu"] < 0.38]
    plt.figure()
    plt.plot(df["nu"], df["cond"])
    plt.xlabel("Poisson's ratio")
    plt.ylabel("cond(K11)")
    plt.savefig("img/cond.pdf")


if __name__ == "__main__":
    main()
