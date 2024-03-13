PYTHON := python
TETGEN := tetgen -p -Y -q -A -O -z -C -V

default: data/post.ply

#####################
# Auxiliary Targets #
#####################

data/pre-face.ply data/pre-skull.ply data/post-skull.ply &: create.py
	mkdir --parents --verbose "$(@D)"
	$(PYTHON) "$<" "$(@D)"

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

data/post.ply: data/post.1.node data/post.1.ele export.py
	$(PYTHON) "export.py" --output "$@" "$<"
