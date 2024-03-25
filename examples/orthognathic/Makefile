PYTHON  := python

default: data/pred/tetra.vtu data/pred/post-face-gt.ply data/pred/post-face.ply

#####################
# Auxiliary Targets #
#####################

PATIENT  ?= $(error PATIENT is not set)
DATA_DIR := $(HOME)/Documents/data/targets/$(PATIENT)

.PHONY: data
data: data/raw

.PHONY: data/raw
data/raw: data/raw/pre-face.ply data/raw/pre-skull.ply data/raw/post-face.ply data/raw/post-skull.ply

data/raw/pre-face.ply: $(DATA_DIR)/pre/05-face.ply
data/raw/pre-skull.ply: $(DATA_DIR)/pre/05-skull.ply
data/raw/post-face.ply: $(DATA_DIR)/post/05-face.ply
data/raw/post-skull.ply: $(DATA_DIR)/post/05-skull.ply
data/raw/pre-face.ply data/raw/pre-skull.ply data/raw/post-face.ply data/raw/post-skull.ply:
	@ mkdir --parents --verbose "$(@D)"
	@ cp --archive --no-target-directory --verbose "$<" "$@"

data/pred/pre-face.ply: data/raw/pre-face.ply
data/pred/pre-skull.ply: data/raw/pre-skull.ply
data/pred/pre-face.ply data/pred/pre-skull.ply:
	@ mkdir --parents --verbose "$(@D)"
	@ cp --archive --no-target-directory --verbose "$<" "$@"

data/pred/post2pre.npz: data/raw/pre-skull.ply data/raw/post-skull.ply src/align.py
	@ mkdir --parents --verbose "$(@D)"
	$(PYTHON) src/align.py --output "$@" "data/raw/post-skull.ply" "data/raw/pre-skull.ply"

data/pred/post-skull.ply: data/raw/post-skull.ply data/pred/post2pre.npz src/transform.py
	@ mkdir --parents --verbose "$(@D)"
	$(PYTHON) src/transform.py --transform "data/pred/post2pre.npz" --output "$@" "data/raw/post-skull.ply"

data/pred/post-face-gt.ply: data/raw/post-face.ply data/pred/post2pre.npz src/transform.py
	@ mkdir --parents --verbose "$(@D)"
	$(PYTHON) src/transform.py --transform "data/pred/post2pre.npz" --output "$@" "data/raw/post-face.ply"

data/pred/tetra.vtu: data/pred/pre-face.ply data/pred/pre-skull.ply data/pred/post-skull.ply src/tetgen.py
	@ mkdir --parents --verbose "$(@D)"
	$(PYTHON) src/tetgen.py --pre-face "data/pred/pre-face.ply" --pre-skull "data/pred/pre-skull.ply" --post-skull "data/pred/post-skull.ply" --output "$@"

data/pred/result/0.vtu: data/pred/tetra.vtu src/solve.py
	@ mkdir --parents --verbose "$(@D)"
	$(PYTHON) src/solve.py --output "$(@D)" "data/pred/tetra.vtu"

data/pred/post-face.ply: data/pred/result/0.vtu src/tet2tri.py
	@ mkdir --parents --verbose "$(@D)"
	$(PYTHON) src/tet2tri.py "$<" "$@"
