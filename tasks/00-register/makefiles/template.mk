DATA_DIR  ?= $(HOME)/Documents/data
TARGET_ID ?= 120056

srcdir       := src
TEMPLATE_DIR := $(DATA_DIR)/template
CT_DIR       := $(DATA_DIR)/CT/$(TARGET_ID)
TARGET_DIR   := $(DATA_DIR)/target/$(TARGET_ID)
PRE_DIR      := $(TARGET_DIR)/pre
POST_DIR     := $(TARGET_DIR)/post

FS   := face skull
SMM  := skull mandible maxilla
FSMM := face skull mandible maxilla

# Download template
$(FS:%=$(TEMPLATE_DIR)/00-%.ply): $(TEMPLATE_DIR)/00-%.ply:
	@ mkdir --parents --verbose "$(@D)"
	wget --output-document="$@" "https://github.com/liblaf/sculptor/raw/main/model/template/$*.ply"

# Process template face
$(TEMPLATE_DIR)/99-face.vtu: $(TEMPLATE_DIR)/00-face.ply $(srcdir)/template/face/process.py
	python "$(lastword $^)" --output="$@" "$<"

# Process template skull
$(SMM:%=$(TEMPLATE_DIR)/99-%.vtu) &: $(TEMPLATE_DIR)/00-skull.ply $(srcdir)/template/skull/process.py
	python "$(lastword $^)" --skull="$(TEMPLATE_DIR)/99-skull.vtu" --mandible="$(TEMPLATE_DIR)/99-mandible.vtu" --maxilla="$(TEMPLATE_DIR)/99-maxilla.vtu" "$<"

# Copy template to target pre dir
$(FSMM:%=$(TARGET_DIR)/pre/%-template.vtu): $(TARGET_DIR)/pre/%-template.vtu: $(TEMPLATE_DIR)/99-%.vtu
	@ cp --archive --force --verbose "$<" "$@"

# Copy template to target post dir
$(FSMM:%=$(TARGET_DIR)/post/%-template.vtu): $(TARGET_DIR)/post/%-template.vtu: $(TEMPLATE_DIR)/99-%.vtu
	@ cp --archive --force --verbose "$<" "$@"
