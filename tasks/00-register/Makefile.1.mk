export DATA_DIR  ?= $(HOME)/Documents/data
export TARGET_ID ?= 120056
export STAGE     ?= post

srcdir       := src
TARGET_CT    := $(DATA_DIR)/CT/$(TARGET_ID)
TARGET_DIR   := $(DATA_DIR)/target/$(TARGET_ID)
TEMPLATE_DIR := $(DATA_DIR)/template

default: $(TARGET_DIR)/pre/99-face.vtu
default: $(TARGET_DIR)/pre/99-mandible.vtu
default: $(TARGET_DIR)/pre/99-maxilla.vtu
default: $(TARGET_DIR)/post/99-face.vtu
default: $(TARGET_DIR)/post/99-mandible.vtu
default: $(TARGET_DIR)/post/99-maxilla.vtu

.SECONDARY:

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
$(TEMPLATE_DIR)/99-face.vtu: $(TEMPLATE_DIR)/00-face.ply $(srcdir)/template/face/process.py
	python "$(lastword $^)" --output="$@" "$<"

# Process template skull
$(TEMPLATE_DIR)/99-skull.vtu $(TEMPLATE_DIR)/99-mandible.vtu $(TEMPLATE_DIR)/99-maxilla.vtu &: $(TEMPLATE_DIR)/00-skull.ply $(srcdir)/template/skull/process.py
	python "$(lastword $^)" --output="$(TEMPLATE_DIR)/99-skull.vtu" --mandible="$(TEMPLATE_DIR)/99-mandible.vtu" --maxilla="$(TEMPLATE_DIR)/99-maxilla.vtu" "$<"

# Trim template face according to target face
ground: $(TEMPLATE_DIR)/00-face.ply $(TARGET_DIR)/pre/01-face.ply $(srcdir)/template/face/ground.py
	python "$(lastword $^)" "$<" "$(word 2,$^)"

$(TARGET_DIR)/pre/s-%.vtu: $(TEMPLATE_DIR)/99-%.vtu $(srcdir)/embed-landmarks.py
	python "$(lastword $^)" --output="$@" --landmarks="$(TARGET_DIR)/pre/s-$*.xyz" "$<"

$(TARGET_DIR)/post/s-%.vtu: $(TEMPLATE_DIR)/99-%.vtu $(srcdir)/embed-landmarks.py
	python "$(lastword $^)" --output="$@" --landmarks="$(TARGET_DIR)/pre/s-$*.xyz" "$<"

##########
# Target #
##########

STAGES := pre post

# ${stage} CT to ${stage} face mesh
$(TARGET_DIR)/%/00-face.ply: $(srcdir)/CT-to-mesh.py | $(TARGET_CT)/%
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --threshold=-200 "$|"

# ${stage} CT to ${stage} skull mesh
$(TARGET_DIR)/%/00-skull.ply: $(srcdir)/CT-to-mesh.py | $(TARGET_CT)/%
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --threshold=200 "$|"

# simplify ${stage} face
$(TARGET_DIR)/%/01-face.ply: $(TARGET_DIR)/%/00-face.ply $(srcdir)/simplify.py
	python "$(lastword $^)" --output="$@" --face-count=-1 "$<"

# simplify ${stage} skull
$(TARGET_DIR)/%/01-skull.ply: $(TARGET_DIR)/%/00-skull.ply $(srcdir)/simplify.py
	python "$(lastword $^)" --output="$@" --face-count=-1 "$<"

# embed ${stage} face landmarks
$(TARGET_DIR)/%/01-face.vtu: $(TARGET_DIR)/%/01-face.ply $(srcdir)/embed-landmarks.py
	python "$(lastword $^)" --output="$@" --landmarks="$(TARGET_DIR)/00-face.xyz" "$<"

# embed ${stage} mandible landmarks
$(TARGET_DIR)/%/01-skull.vtu: $(TARGET_DIR)/%/01-skull.ply $(srcdir)/embed-landmarks.py
	python "$(lastword $^)" --output="$@" --landmarks="$(TARGET_DIR)/00-mandible.xyz" "$<"

# copy pre ${face/skull}
$(TARGET_DIR)/pre/02-%.vtu: $(TARGET_DIR)/pre/01-%.vtu
	@ cp --archive --force --verbose "$<" "$@"

# align post skull to pre skull
$(TARGET_DIR)/post/02-skull.vtu $(TARGET_DIR)/post-to-pre.npy &: $(TARGET_DIR)/post/01-skull.vtu $(TARGET_DIR)/pre/01-skull.vtu $(srcdir)/align.py
	python "$(lastword $^)" --output="$(TARGET_DIR)/post/02-skull.vtu" --output-transform="$(TARGET_DIR)/post-to-pre.npy" "$<" "$(word 2,$^)"

# align post face to pre face
$(TARGET_DIR)/post/02-face.vtu: $(TARGET_DIR)/post/01-face.vtu $(TARGET_DIR)/post-to-pre.npy $(srcdir)/apply-transform.py
	python "$(lastword $^)" --output="$@" --transform="$(word 2,$^)" "$<"

# align template skull to pre skull
$(TARGET_DIR)/pre/03-skull.vtu $(TARGET_DIR)/template-to-target.npy &: $(TARGET_DIR)/pre/s-skull.vtu $(TARGET_DIR)/pre/01-skull.vtu $(srcdir)/align.py
	python "$(lastword $^)" --output="$(TARGET_DIR)/pre/03-skull.vtu" --output-transform="$(TARGET_DIR)/template-to-target.npy" --smart-initial "$<" "$(word 2,$^)"

# align template face to ${stage} face
$(TARGET_DIR)/%/03-face.vtu: $(TARGET_DIR)/%/s-face.vtu $(TARGET_DIR)/%/02-face.vtu $(TARGET_DIR)/template-to-target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" "$<" "$(word 2,$^)"

# align template mandible to ${stage} skull
$(TARGET_DIR)/%/03-mandible.vtu: $(TARGET_DIR)/%/s-mandible.vtu $(TARGET_DIR)/%/02-skull.vtu $(TARGET_DIR)/template-to-target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" "$<" "$(word 2,$^)"

# align template maxilla to ${stage} skull
$(TARGET_DIR)/%/03-maxilla.vtu: $(TARGET_DIR)/%/s-maxilla.vtu $(TARGET_DIR)/%/02-skull.vtu $(TARGET_DIR)/template-to-target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" "$<" "$(word 2,$^)"

# register template face to ${stage} face
$(TARGET_DIR)/%/99-face.vtu: $(TARGET_DIR)/%/03-face.vtu $(TARGET_DIR)/%/02-face.vtu $(srcdir)/register.py
	python "$(lastword $^)" --output="$@" "$<" "$(word 2,$^)"

# register template mandible to ${stage} skull
$(TARGET_DIR)/%/99-mandible.vtu: $(TARGET_DIR)/%/03-mandible.vtu $(TARGET_DIR)/%/02-skull.vtu $(srcdir)/register.py
	python "$(lastword $^)" --output="$@" "$<" "$(word 2,$^)"

# register template maxilla to ${stage} skull
$(TARGET_DIR)/%/99-maxilla.vtu: $(TARGET_DIR)/%/03-maxilla.vtu $(TARGET_DIR)/%/02-skull.vtu $(srcdir)/register.py
	python "$(lastword $^)" --output="$@" "$<" "$(word 2,$^)"
