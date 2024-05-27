DATA_DIR  ?= $(HOME)/Documents/data
TARGET_ID ?= 120056
export DATA_DIR
export TARGET_ID

srcdir     := src
TARGET_DIR := $(DATA_DIR)/target/$(TARGET_ID)

default: $(TARGET_DIR)/eval.vtu
# default: $(TARGET_DIR)/eval-gt.vtu

$(TARGET_DIR)/eval.vtu: $(TARGET_DIR)/predict-face.vtu $(TARGET_DIR)/post/99-face.vtu $(srcdir)/loss.py
	python "$(lastword $^)" --output="$@" "$<" "$(word 2,$^)"

$(TARGET_DIR)/eval-gt.vtu: $(TARGET_DIR)/pre/99-face.vtu $(TARGET_DIR)/post/99-face.vtu $(srcdir)/loss.py
	python "$(lastword $^)" --output="$@" "$<" "$(word 2,$^)"

$(TARGET_DIR)/predict-face.vtu: $(TARGET_DIR)/predict.vtu $(srcdir)/tet-to-tri.py
	python "$(lastword $^)" --output="$@" "$<"

$(TARGET_DIR)/predict.vtu: force
	$(MAKE) --directory="../02-simulation" "$@"

$(TARGET_DIR)/pre/99-face.vtu $(TARGET_DIR)/post/99-face.vtu: force
	$(MAKE) --directory="../00-register" "$@"

force:
