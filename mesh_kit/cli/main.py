import typer

from mesh_kit import cli as _cli
from mesh_kit.cli import inspect as _inspect
from mesh_kit.cli import view as _view

app: typer.Typer = typer.Typer(name="mesh-kit")
app.add_typer(_view.app, name="view")
app.command(name="inspect")(_inspect.main)

if __name__ == "__main__":
    _cli.run(app)
