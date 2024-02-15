default: ruff schemas

ruff:
	ruff check --fix
	ruff format

.PHONY: schemas
schemas: schemas/registration.json

setup: environment.yaml poetry.lock
	micromamba --yes --name "mesh-kit" create --file environment.yaml
	micromamba --name "mesh-kit" run poetry install

###############
# Auxiliaries #
###############

schemas/registration.json: mesh_kit/registration/config.py
	@ mkdir --parents --verbose "$(@D)"
	python -m mesh_kit.registration.config > "$@"
	prettier --write "$@" || true
