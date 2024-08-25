all: plot/tet plot/curve

clean:
	@ $(RM) --recursive plot

.PHONY: plot/tet
plot/tet:
	@ mkdir --parents --verbose plot/tet
	python src/plot/tet.py

.PHONY: plot/curve
plot/curve:
	@ mkdir --parents --verbose plot/curve
	python src/plot/curve.py
