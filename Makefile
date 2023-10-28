NAME := mesh-kit

all:

setup: environment.yml
	micromamba create --file=$< --yes
	micromamba run --name=$(NAME) poetry install
	# micromamba run --name=$(NAME) pip install trimesh[all]
