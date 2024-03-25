default: data/sphere/pred/result/0.vtu

data/sphere/pred/pre-skull.ply: src/sphere/gen.py
	@ mkdir --parents --verbose "$(@D)"
	python src/sphere/gen.py --radius 0.1 "$@"

data/sphere/pred/pre-face.ply: src/sphere/gen.py
	@ mkdir --parents --verbose "$(@D)"
	python src/sphere/gen.py --radius 0.2 "$@"

data/sphere/pred/post-skull.ply: src/sphere/gen.py
	@ mkdir --parents --verbose "$(@D)"
	python src/sphere/gen.py --radius 0.1 --displacement 0.03 "$@"

data/sphere/pred/tetra.vtu: data/sphere/pred/pre-skull.ply data/sphere/pred/pre-face.ply data/sphere/pred/post-skull.ply src/tetgen.py
	@ mkdir --parents --verbose "$(@D)"
	python src/tetgen.py --pre-face "data/sphere/pred/pre-face.ply" --pre-skull "data/sphere/pred/pre-skull.ply" --post-skull "data/sphere/pred/post-skull.ply" --output "$@"

data/sphere/pred/result/0.vtu: data/sphere/pred/tetra.vtu src/solve.py
	@ rm --force --recursive "$(@D)"
	@ mkdir --parents --verbose "$(@D)"
	python src/solve.py --output-dir "$(@D)" "$<"
