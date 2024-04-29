DATA_DIR  ?= $(HOME)/Documents/data
TARGET_ID ?= 120056
STAGE     ?= post

srcdir       := src
TARGET_CT    := $(DATA_DIR)/CT/$(TARGET_ID)/$(STAGE)
TARGET_DIR   := $(DATA_DIR)/target/$(TARGET_ID)/$(STAGE)
TEMPLATE_DIR := $(DATA_DIR)/template

default: $(TARGET_DIR)/04-mandible.vtu

############
# Template #
############

$(TEMPLATE_DIR)/00-%.ply:
	@ mkdir --parents --verbose "$(@D)"
	wget --output-document "$@" "https://github.com/liblaf/sculptor/raw/main/model/template/$*.ply"

$(TEMPLATE_DIR)/01-face.vtu: $(TEMPLATE_DIR)/01-face.ply
	@ meshio convert "$<" "$@"

$(TEMPLATE_DIR)/01-skull.vtu: $(TEMPLATE_DIR)/00-skull.ply $(srcdir)/template/skull-mask-effective.py
	python "$(lastword $^)" "$<" "$@"

$(TEMPLATE_DIR)/01-mandible.vtu $(TEMPLATE_DIR)/01-maxilla.vtu &: $(TEMPLATE_DIR)/01-skull.vtu $(srcdir)/template/skull-split.py
	python "$(lastword $^)" --mandible "$(TEMPLATE_DIR)/01-mandible.vtu" --maxilla "$(TEMPLATE_DIR)/01-maxilla.vtu" "$<"

##########
# Target #
##########

# CT to skull mesh
$(TARGET_DIR)/00-skull.ply: $(TARGET_CT) $(srcdir)/CT2mesh.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --threshold 200 "$<" "$@"

# simplify
$(TARGET_DIR)/01-skull.ply: $(TARGET_DIR)/00-skull.ply $(srcdir)/simplify.py
	python "$(lastword $^)" --face-count -1 "$<" "$@"

# align template skull to target skull
$(TARGET_DIR)/02-skull.vtu $(TARGET_DIR)/template2target.npy &: $(TEMPLATE_DIR)/01-skull.vtu $(TARGET_DIR)/01-skull.ply $(srcdir)/align.py
	python "$(lastword $^)" --output-mesh "$(TARGET_DIR)/02-skull.vtu" --output-transform "$(TARGET_DIR)/template2target.npy" "$<" "$(word 2,$^)"

# align template mandible to target mandible
$(TARGET_DIR)/03-mandible.vtu: $(TEMPLATE_DIR)/01-mandible.vtu $(TARGET_DIR)/01-skull.ply $(TARGET_DIR)/template2target.npy $(srcdir)/align.py
	python "$(lastword $^)" --initial-transform "$(word 3,$^)" --output-mesh "$@" "$<" "$(word 2,$^)"

# align template maxilla to target maxilla
$(TARGET_DIR)/03-maxilla.vtu: $(TEMPLATE_DIR)/01-maxilla.vtu $(TARGET_DIR)/01-skull.ply $(TARGET_DIR)/template2target.npy $(srcdir)/align.py
	python "$(lastword $^)" --initial-transform "$(word 3,$^)" --output-mesh "$@" "$<" "$(word 2,$^)"

# register template mandible
$(TARGET_DIR)/04-mandible.vtu: $(TARGET_DIR)/03-mandible.vtu $(TARGET_DIR)/01-skull.ply $(srcdir)/register.py
	@ rm --force --recursive "$(TARGET_DIR)/record"
	@ mkdir --parents --verbose "$(TARGET_DIR)/record"
	python "$(lastword $^)" --output "$@" --record-dir "$(TARGET_DIR)/record" "$(word 1,$^)" "$(word 2,$^)"
