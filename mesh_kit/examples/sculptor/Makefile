DEVICE := cuda
SHELL  := bash
TARGET := 孔梓语

DATA_DIR     := data
TARGET_DIR   := $(DATA_DIR)/$(TARGET)
TEMPLATE_DIR := $(DATA_DIR)/template
CMD_DIR      := cmd

TARGET_FILES += $(TARGET_DIR)/pre/02-skull.ply
TARGET_FILES += $(TARGET_DIR)/pre/02-skull-sparse-landmarks.txt

all: $(TARGET_FILES)

###############
# Auxiliaries #
###############

CMD_DIR    := cmd
MANUAL     := bash scripts/manual.sh
MESH_FIX   := MeshFix
PYTHON     := python
ALIGN      := $(PYTHON) $(CMD_DIR)/align.py --device=$(DEVICE)
CT_TO_MESH := $(PYTHON) $(CMD_DIR)/CT_to_mesh.py
DENSIFY    := $(PYTHON) $(CMD_DIR)/densify.py

##########
# Target #
##########

$(TARGET_DIR)/pre/00-CT.nrrd:
	$(MANUAL) Export DICOM $@

$(TARGET_DIR)/post/00-CT.nrrd:
	$(MANUAL) Export DICOM $@

define CT_to_mesh
$(TARGET_DIR)/$(1)/01-$(2).ply: $(TARGET_DIR)/$(1)/00-CT.nrrd $(CMD_DIR)/CT_to_mesh.py
	$(CT_TO_MESH) --threshold=$(3) $$< $$@
endef
$(eval $(call CT_to_mesh,pre,face,0.0))
$(eval $(call CT_to_mesh,pre,skull,250.0))
$(eval $(call CT_to_mesh,post,face,0.0))
$(eval $(call CT_to_mesh,post,skull,250.0))

define manual_selection
$(TARGET_DIR)/$(1)/01-$(2)-sparse-landmarks.txt: $(TARGET_DIR)/$(1)/02-$(2).ply
	$(MANUAL) Select $$< $$@
endef
$(eval $(call manual_selection,pre,face))
$(eval $(call manual_selection,pre,skull))
$(eval $(call manual_selection,post,face))
$(eval $(call manual_selection,post,skull))

define align
TARGETS      := $(TARGET_DIR)/$(1)/02-$(2).ply
TARGETS      += $(TARGET_DIR)/$(1)/02-$(2)-sparse-landmarks.txt
REQUIREMENTS := $(TEMPLATE_DIR)/03-$(2).ply
REQUIREMENTS += $(TEMPLATE_DIR)/03-$(2)-sparse-landmarks.txt
REQUIREMENTS += $(TARGET_DIR)/$(1)/01-$(2).ply
REQUIREMENTS += $(TARGET_DIR)/$(1)/01-$(2)-sparse-landmarks.txt
$(TARGETS) &: $(REQUIREMENTS)
	$(ALIGN) \
		--source-landmarks=$(TEMPLATE_DIR)/03-$(2)-sparse-landmarks.txt \
		--target-landmarks=$(TARGET_DIR)/$(1)/01-$(2)-sparse-landmarks.txt \
		--output=$(TARGET_DIR)/$(1)/02-$(2).ply \
		--output-landmarks=$(TARGET_DIR)/$(1)/02-$(2)-sparse-landmarks.txt \
		$(TEMPLATE_DIR)/03-$(2).ply $(TARGET_DIR)/$(1)/01-$(2).ply
endef
$(eval $(call align,pre,face))
$(eval $(call align,pre,skull))
$(eval $(call align,post,face))
$(eval $(call align,post,skull))

define densify
TARGETS := $(TARGET_DIR)/$(1)/02-$(2)-dense-landmarks.txt
TARGETS += $(TARGET_DIR)/$(1)/01-$(2)-dense-landmarks.txt
REQUIREMENTS := $(TARGET_DIR)/$(1)/02-$(2).ply
REQUIREMENTS += $(TARGET_DIR)/$(1)/02-$(2)-sparse-landmarks.txt
REQUIREMENTS += $(TARGET_DIR)/$(1)/01-$(2).ply
REQUIREMENTS += $(TARGET_DIR)/$(1)/01-$(2)-sparse-landmarks.txt
$(TARGETS) &: $(REQUIREMENTS)
	$(DENSIFY) \
		--source-landmarks=$(TARGET_DIR)/$(1)/02-$(2)-sparse-landmarks.txt \
		--target-landmarks=$(TARGET_DIR)/$(1)/01-$(2)-sparse-landmarks.txt \
		--output-source-landmarks=$(TARGET_DIR)/$(1)/02-$(2)-dense-landmarks.txt \
		--output-target-landmarks=$(TARGET_DIR)/$(1)/01-$(2)-dense-landmarks.txt \
		$(TARGET_DIR)/$(1)/02-$(2).ply $(TARGET_DIR)/$(1)/01-$(2).ply
endef
$(eval $(call densify,pre,face))
$(eval $(call densify,pre,skull))
$(eval $(call densify,post,face))
$(eval $(call densify,post,skull))

############
# Template #
############

$(TEMPLATE_DIR)/00-%.ply:
	@ mkdir --parents --verbose $(@D)
	wget --output-document=$@ https://raw.githubusercontent.com/liblaf/sculptor/main/model/template/$*.ply

$(TEMPLATE_DIR)/01-%.ply: $(TEMPLATE_DIR)/00-%.ply $(CMD_DIR)/template/%.py
	$(PYTHON) $(CMD_DIR)/template/$*.py $< $@

$(TEMPLATE_DIR)/02-%.ply: $(TEMPLATE_DIR)/01-%.ply
	$(MANUAL) Edit $< $@

$(TEMPLATE_DIR)/03-%.ply: $(TEMPLATE_DIR)/02-%.ply
	$(MESH_FIX) $< $@

$(TEMPLATE_DIR)/03-%-sparse-landmarks.txt: $(TEMPLATE_DIR)/03-%.ply
	$(MANUAL) Select $< $@
