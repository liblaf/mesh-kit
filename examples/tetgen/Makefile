PYTHON := python
TETGEN := tetgen -p -Y -q -A -O -z -C -V

default: data/pre.ply data/post.ply data/post-surface.ply

#####################
# Auxiliary Targets #
#####################

DATA_DIR := $(HOME)/Documents/data
TARGET_DIR := $(DATA_DIR)/targets/*

# data/pre-face.ply data/pre-skull.ply data/post-skull.ply &: create.py
# 	mkdir --parents --verbose "$(@D)"
# 	$(PYTHON) "$<" "$(@D)"

data/pre-face.ply: $(TARGET_DIR)/pre/05-face.ply
data/pre-skull.ply: $(TARGET_DIR)/pre/05-skull.ply
data/post-skull.ply: $(TARGET_DIR)/post/05-skull.ply
data/pre-face.ply data/pre-skull.ply data/post-skull.ply:
	@ mkdir --parents --verbose "$(@D)"
	@ cp --verbose "$<" "$@"

data/pre.smesh: data/pre-face.ply data/pre-skull.ply compose.py
	$(PYTHON) "compose.py" --output "$@" "data/pre-face.ply" "data/pre-skull.ply"

data/pre.1.node data/pre.1.ele &: data/pre.smesh
	$(TETGEN) "$<"

data/pre.ply: data/pre.1.node data/pre.1.ele export.py
	$(PYTHON) "export.py" --output "$@" "$<"

data/post.1.node: data/pre.1.node data/pre-skull.ply data/post-skull.ply solve.py
	$(PYTHON) "solve.py" --pre "data/pre-skull.ply" --post "data/post-skull.ply" "$<"

data/post.1.ele: data/pre.1.ele
	@ cp --verbose "$<" "$@"

data/post.1.face: data/pre.1.face
	@ cp --verbose "$<" "$@"

data/post.ply: data/post.1.node data/post.1.ele data/post.1.face export.py
	$(PYTHON) "export.py" --output "$@" "$<"

data/post-surface.ply: data/post.1.node data/post.1.ele data/post.1.face export.py
	$(PYTHON) "export.py" --no-interior --output "$@" "$<"
