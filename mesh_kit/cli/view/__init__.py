import typer

from mesh_kit import cli as _cli
from mesh_kit.cli.view import attr as _attr
from mesh_kit.cli.view import landmarks as _landmarks
from mesh_kit.cli.view import register as _register

app: typer.Typer = typer.Typer(name="view")
app.command(name="attr")(_attr.main)
app.command(name="landmarks")(_landmarks.main)
app.command(name="register")(_register.main)


if __name__ == "__main__":
    _cli.run(app)
