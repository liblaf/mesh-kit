default:
	ruff check --fix
	ruff format

setup: environment.yaml poetry.lock
	micromamba --yes --name "mesh-kit" create --file environment.yaml
	micromamba --name "mesh-kit" run poetry install
