default: data/sphere/tetra.vtu

data/sphere/tetra.vtu: src/gen/sphere.py
	@ mkdir --parents --verbose "$(@D)"
	python "$<" --output "$@"
