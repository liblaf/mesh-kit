default: fmt ruff schemas

fmt: fmt/pyproject.toml

ruff:
	ruff format
	ruff check --fix --unsafe-fixes

.PHONY: schemas
schemas: schemas/registration.json

setup: environment.yaml poetry.lock
	micromamba --yes --name "mesh-kit" create --file environment.yaml
	micromamba --name "mesh-kit" run poetry install

###############
# Auxiliaries #
###############

fmt/pyproject.toml: pyproject.toml
	toml-sort --in-place --all "$<"
	taplo format --option "reorder_keys=true" --option "reorder_arrays=true" "$<"

schemas/registration.json: mesh_kit/registration/config.py
	@ mkdir --parents --verbose "$(@D)"
	python -m mesh_kit.registration.config > "$@"
	prettier --write "$@" || true
