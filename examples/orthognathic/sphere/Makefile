default: data/result/00.vtu

data/tetra.vtu: src/gen.py
	@ mkdir --parents --verbose "$(@D)"
	python "$<" --output "$@"

data/result/00.vtu: data/tetra.vtu src/solve.py
	@ rm --force --recursive "$(@D)"
	@ mkdir --parents --verbose "$(@D)"
	python src/solve.py --output-dir "$(@D)" "$<"
