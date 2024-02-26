NAME     := mesh-kit
PKG_NAME := mesh_kit

default: fmt mypy ruff schemas

fmt: fmt/pyproject.toml

mypy: ruff
	mypy "."

ruff:
	ruff format
	ruff check --fix --unsafe-fixes

.PHONY: schemas
schemas: schemas/register.json

setup: environment.yaml poetry.lock
	micromamba --yes --name "$(NAME)" create --file environment.yaml
	micromamba --name "$(NAME)" run poetry install

stub:
	stubgen --include-docstrings --output "." --package "$(PKG_NAME)"

stubtest:
	stubtest "$(PKG_NAME)"

###############
# Auxiliaries #
###############

fmt/pyproject.toml: pyproject.toml
	toml-sort --in-place --all "$<"
	taplo format --option "reorder_keys=true" --option "reorder_arrays=true" "$<"

schemas/register.json: mesh_kit/register/config.py
	@ mkdir --parents --verbose "$(@D)"
	python -m mesh_kit.register.config > "$@"
	prettier --write "$@" || true
