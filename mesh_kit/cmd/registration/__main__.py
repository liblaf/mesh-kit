from typer import Typer

from mesh_kit.common.cli import run

app: Typer = Typer(name="registration")


if __name__ == "__main__":
    run(app)
