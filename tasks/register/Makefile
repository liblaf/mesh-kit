DATA_DIR  ?= $(HOME)/Documents/data
TARGET_ID ?= 120056
STAGE     ?= post

srcdir       := src
TARGET_CT    := $(DATA_DIR)/CT/$(TARGET_ID)/$(STAGE)
TARGET_DIR   := $(DATA_DIR)/target/$(TARGET_ID)/$(STAGE)
TEMPLATE_DIR := $(DATA_DIR)/template

default: $(TARGET_DIR)/02-face.vtu
default: $(TARGET_DIR)/02-mandible.vtu
default: $(TARGET_DIR)/02-maxilla.vtu

############
# Template #
############

# Download template face
$(TEMPLATE_DIR)/00-face.ply:
	@ mkdir --parents --verbose "$(@D)"
	wget --output-document="$@" "https://github.com/liblaf/sculptor/raw/main/model/template/face.ply"

# Download template skull
$(TEMPLATE_DIR)/00-skull.ply:
	@ mkdir --parents --verbose "$(@D)"
	wget --output-document="$@" "https://github.com/liblaf/sculptor/raw/main/model/template/skull.ply"

# Process template face
$(TEMPLATE_DIR)/01-face.ply: $(TEMPLATE_DIR)/00-face.ply $(srcdir)/template/face/process.py
	python "$(lastword $^)" --output="$@" "$<"

# Process template skull
$(TEMPLATE_DIR)/01-skull.vtu $(TEMPLATE_DIR)/01-mandible.vtu $(TEMPLATE_DIR)/01-maxilla.vtu &: $(TEMPLATE_DIR)/00-skull.ply $(srcdir)/template/skull/process.py
	python "$(lastword $^)" --output="$(TEMPLATE_DIR)/01-skull.vtu" --mandible="$(TEMPLATE_DIR)/01-mandible.vtu" --maxilla="$(TEMPLATE_DIR)/01-maxilla.vtu" "$<"

ground: $(TEMPLATE_DIR)/00-face.ply $(TARGET_DIR)/01-face.ply $(srcdir)/template/face/ground.py
	python "$(lastword $^)" "$<" "$(word 2,$^)"

##########
# Target #
##########

# CT to face mesh
$(TARGET_DIR)/00-face.ply: $(TARGET_CT) $(srcdir)/CT2mesh.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --threshold=-200 "$<"

# CT to skull mesh
$(TARGET_DIR)/00-skull.ply: $(TARGET_CT) $(srcdir)/CT2mesh.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --threshold=200 "$<"

# simplify face
$(TARGET_DIR)/01-face.ply: $(TARGET_DIR)/00-face.ply $(srcdir)/simplify.py
	python "$(lastword $^)" --output="$@" --face-count=-1 "$<"

# simplify skull
$(TARGET_DIR)/01-skull.ply: $(TARGET_DIR)/00-skull.ply $(srcdir)/simplify.py
	python "$(lastword $^)" --output="$@" --face-count=-1 "$<"

# align template skull to target skull
$(TARGET_DIR)/02-skull.vtu $(TARGET_DIR)/template_to_target.npy &: $(TEMPLATE_DIR)/01-skull.vtu $(TARGET_DIR)/01-skull.ply $(srcdir)/align.py
	python "$(lastword $^)" --output="$(TARGET_DIR)/02-skull.vtu" --output-transform="$(TARGET_DIR)/template_to_target.npy" "$<" "$(word 2,$^)"

# align template face to target face
$(TARGET_DIR)/02-face.vtu: $(TEMPLATE_DIR)/01-face.ply $(TARGET_DIR)/01-face.ply $(TARGET_DIR)/template_to_target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" --inverse "$<" "$(word 2,$^)"

# align template mandible to target mandible
$(TARGET_DIR)/02-mandible.vtu: $(TEMPLATE_DIR)/01-mandible.vtu $(TARGET_DIR)/01-skull.ply $(TARGET_DIR)/template_to_target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" "$<" "$(word 2,$^)"

# align template maxilla to target maxilla
$(TARGET_DIR)/02-maxilla.vtu: $(TEMPLATE_DIR)/01-maxilla.vtu $(TARGET_DIR)/01-skull.ply $(TARGET_DIR)/template_to_target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" "$<" "$(word 2,$^)"

# register template mandible
$(TARGET_DIR)/04-mandible.vtu: $(TARGET_DIR)/03-mandible.vtu $(TARGET_DIR)/01-skull.ply $(srcdir)/register.py
	@ rm --force --recursive "$(TARGET_DIR)/record"
	@ mkdir --parents --verbose "$(TARGET_DIR)/record"
	python "$(lastword $^)" --output="$@" --record-dir="$(TARGET_DIR)/record" "$(word 1,$^)" "$(word 2,$^)"
