DATA_DIR  ?= $(HOME)/Documents/data
TARGET_ID ?= 120056
export DATA_DIR
export TARGET_ID

srcdir     := src
TARGET_DIR := $(DATA_DIR)/target/$(TARGET_ID)

default: $(TARGET_DIR)/post/99-face.vtu
default: $(TARGET_DIR)/tetra.vtu

# .PHONY: $(TARGET_DIR)/pre/99-face.vtu
$(TARGET_DIR)/pre/99-face.vtu: force
	$(MAKE) --directory="../00-register" STAGE=pre "$@"

# .PHONY: $(TARGET_DIR)/pre/99-mandible.vtu
$(TARGET_DIR)/pre/99-mandible.vtu: force
	$(MAKE) --directory="../00-register" STAGE=pre "$@"

# .PHONY: $(TARGET_DIR)/pre/99-maxilla.vtu
$(TARGET_DIR)/pre/99-maxilla.vtu: force
	$(MAKE) --directory="../00-register" STAGE=pre "$@"

# .PHONY: $(TARGET_DIR)/post/99-face.vtu
$(TARGET_DIR)/post/99-face.vtu: force
	$(MAKE) --directory="../00-register" STAGE=post "$@"

# .PHONY: $(TARGET_DIR)/post/99-mandible.vtu
$(TARGET_DIR)/post/99-mandible.vtu: force
	$(MAKE) --directory="../00-register" STAGE=post "$@"

# .PHONY: $(TARGET_DIR)/post/99-maxilla.vtu
$(TARGET_DIR)/post/99-maxilla.vtu: force
	$(MAKE) --directory="../00-register" STAGE=post "$@"

$(TARGET_DIR)/tetra.vtu: $(TARGET_DIR)/pre/99-face.vtu $(TARGET_DIR)/pre/99-mandible.vtu $(TARGET_DIR)/pre/99-maxilla.vtu $(TARGET_DIR)/post/99-mandible.vtu $(TARGET_DIR)/post/99-maxilla.vtu $(srcdir)/tetgen.py
	python "$(lastword $^)" --output="$@" --pre-face="$<" --pre-mandible="$(word 2,$^)" --pre-maxilla="$(word 3,$^)" --post-mandible="$(word 4,$^)" --post-maxilla="$(word 5,$^)"

force:
