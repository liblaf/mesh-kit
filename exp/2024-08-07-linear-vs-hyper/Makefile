.PHONY: all
all: plot/solution/linear.png plot/volume.png plot/time.png

.PHONY: clean
clean:
	$(RM) --recursive log
	$(RM) --recursive plot

data/input.vtu: src/gen.py
	@ mkdir --parents --verbose "$(@D)"
	python "$<"

data/linear.vtu: src/main.py data/input.vtu
	@ mkdir --parents --verbose log
	JAX_DEBUG_NANS=False python "$<"

plot/solution/linear.png: src/plot/solution.py data/linear.vtu
	@ mkdir --parents --verbose "plot/solution"
	python "$<"

plot/volume.png: src/plot/volume.py data/linear.vtu
	@ mkdir --parents --verbose "plot/solution"
	python "$<"

plot/time.png: src/plot/time.py data/linear.vtu
	@ mkdir --parents --verbose "plot/solution"
	python "$<"
