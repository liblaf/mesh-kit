DATA_DIR  ?= $(HOME)/Documents/data
TARGET_ID ?= 120056
export DATA_DIR
export TARGET_ID

srcdir     := src
TARGET_DIR := $(DATA_DIR)/target/$(TARGET_ID)

default: $(TARGET_DIR)/ground-truth.vtu
default: $(TARGET_DIR)/predict.vtu

$(TARGET_DIR)/ground-truth.vtu: $(TARGET_DIR)/post/99-face.vtu
	@ cp --verbose "$<" "$@"

.PHONY: $(TARGET_DIR)/post/99-face.vtu
$(TARGET_DIR)/post/99-face.vtu:
	$(MAKE) --directory="../00-register" STAGE=post "$@"

.PHONY: $(TARGET_DIR)/tetra.vtu
$(TARGET_DIR)/tetra.vtu:
	$(MAKE) --directory="../01-tetgen" "$@"

$(TARGET_DIR)/predict.vtu: $(TARGET_DIR)/tetra.vtu $(srcdir)/mtm-matrix-free.py
	python "$(lastword $^)" --output="$@" "$<"

data/%.vtu: ../01-tetgen/src/create.py
	@ mkdir --parents --verbose "$(@D)"
	python "$<" --output="$@" --displacement="$*"

# displacement, Poisson's ratio, method
define run
# default: data/$(1)-$(2)-$(3).vtu
data/$(1)-$(2)-$(3).vtu: data/$(1).vtu src/mtm-$(3).py
	python "src/mtm-$(3).py" --output="$$@" --poisson-ratio="$(2)" "$$<"
endef

$(eval $(call run,0.01,0.00,matrix-free))
$(eval $(call run,0.01,0.10,matrix-free))
$(eval $(call run,0.01,0.20,matrix-free))
$(eval $(call run,0.01,0.30,matrix-free))
$(eval $(call run,0.01,0.40,matrix-free))
$(eval $(call run,0.01,0.46,matrix-free))
$(eval $(call run,0.05,0.00,matrix-free))
$(eval $(call run,0.05,0.10,matrix-free))
$(eval $(call run,0.05,0.20,matrix-free))
$(eval $(call run,0.05,0.30,matrix-free))
$(eval $(call run,0.05,0.40,matrix-free))
$(eval $(call run,0.05,0.46,matrix-free))
$(eval $(call run,0.01,0.00,mollemans))
$(eval $(call run,0.01,0.10,mollemans))
$(eval $(call run,0.01,0.20,mollemans))
$(eval $(call run,0.01,0.30,mollemans))
$(eval $(call run,0.01,0.40,mollemans))
$(eval $(call run,0.01,0.46,mollemans))
$(eval $(call run,0.05,0.00,mollemans))
$(eval $(call run,0.05,0.10,mollemans))
$(eval $(call run,0.05,0.20,mollemans))
$(eval $(call run,0.05,0.30,mollemans))
$(eval $(call run,0.05,0.40,mollemans))
$(eval $(call run,0.05,0.46,mollemans))
