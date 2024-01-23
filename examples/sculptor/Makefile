DEVICE := cuda
SHELL  := bash
TARGET ?= TARGET

CMD_DIR      := cmd
DATA_DIR     := data
RECORDS_DIR  := $(DATA_DIR)/records
TARGET_DIR   := $(DATA_DIR)/$(TARGET)
TEMPLATE_DIR := $(DATA_DIR)/template

TARGET_FILES += $(TARGET_DIR)/pre/01-face.ply

all: $(TARGET_FILES)

check: check-pre check-post

clean:
	@ $(RM) --recursive --verbose $(RECORDS_DIR)

###############
# Auxiliaries #
###############

MANUAL       := @ bash scripts/manual.sh
MESH_FIX     := MeshFix
PYTHON       := python
ALIGN        := $(PYTHON) $(CMD_DIR)/align.py
CT_TO_MESH   := $(PYTHON) $(CMD_DIR)/CT_to_mesh.py --artifact=$(DATA_DIR)/CT_to_mesh
PLY2ASCII    := $(PYTHON) $(CMD_DIR)/ply2ascii.py
REGISTER     := $(PYTHON) $(CMD_DIR)/register.py --records=$(RECORDS_DIR)
SIMPLIFY     := $(PYTHON) $(CMD_DIR)/simplify.py --face-count=-1
TETGEN       := tetgen -p -Y -q -A -O -z -V
TETGEN_CHECK := $(TETGEN) -d -N -E -F

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

define simplify
$(TARGET_DIR)/$(1)/02-$(2).ply: $(TARGET_DIR)/$(1)/01-$(2).ply $(CMD_DIR)/simplify.py
	$(SIMPLIFY) $$< $$@
endef
$(eval $(call simplify,pre,face))
$(eval $(call simplify,pre,skull))
$(eval $(call simplify,post,face))
$(eval $(call simplify,post,skull))

define select
$(TARGET_DIR)/$(1)/02-$(2)-landmarks.txt: $(TARGET_DIR)/$(1)/02-$(2).ply
	$(MANUAL) Select $$< $$@
endef
$(eval $(call select,pre,face))
$(eval $(call select,pre,skull))
$(eval $(call select,post,face))
$(eval $(call select,post,skull))

define align
$(TARGET_DIR)/$(1)/03-$(2).ply $(TARGET_DIR)/$(1)/03-$(2)-landmarks.txt &: \
	$(TEMPLATE_DIR)/03-$(2).ply $(TEMPLATE_DIR)/03-$(2)-landmarks.txt \
	$(TARGET_DIR)/$(1)/02-$(2).ply $(TARGET_DIR)/$(1)/02-$(2)-landmarks.txt \
	$(CMD_DIR)/align.py
	$(ALIGN) --output=$(TARGET_DIR)/$(1)/03-$(2).ply $(TEMPLATE_DIR)/03-$(2).ply $(TARGET_DIR)/$(1)/02-$(2).ply
endef
$(eval $(call align,pre,face))
$(eval $(call align,pre,skull))
$(eval $(call align,post,face))
$(eval $(call align,post,skull))

define register
$(TARGET_DIR)/$(1)/04-$(2).ply: $(TARGET_DIR)/$(1)/03-$(2).ply $(TARGET_DIR)/$(1)/03-$(2)-landmarks.txt $(TARGET_DIR)/$(1)/02-$(2).ply $(TARGET_DIR)/$(1)/02-$(2)-landmarks.txt $(CMD_DIR)/register.py
	@ mkdir --parents --verbose $(RECORDS_DIR)
	$(REGISTER) --output=$$@ $(TARGET_DIR)/$(1)/03-$(2).ply $(TARGET_DIR)/$(1)/02-$(2).ply
endef
$(eval $(call register,pre,face))
$(eval $(call register,pre,skull))
$(eval $(call register,post,face))
$(eval $(call register,post,skull))

define check
$(TARGET_DIR)/$(1)/05-$(2).ply: $(TARGET_DIR)/$(1)/04-$(2).ply
	$(PLY2ASCII) $$< $$@

check-$(1)-$(2): $(TARGET_DIR)/$(1)/05-$(2).ply
	$(TETGEN_CHECK) $$<
endef
$(eval $(call check,pre,face))
$(eval $(call check,pre,skull))
$(eval $(call check,post,face))
$(eval $(call check,post,skull))
check-pre: check-pre-face check-pre-skull
check-post: check-post-face check-post-skull

############
# Template #
############

define download
$(TEMPLATE_DIR)/00-$(1).ply:
	@ mkdir --parents --verbose $$(@D)
	wget --output-document=$$@ https://raw.githubusercontent.com/liblaf/sculptor/main/model/template/$(1).ply
endef
$(eval $(call download,face))
$(eval $(call download,skull))

define pre-process
$(TEMPLATE_DIR)/01-$(1).ply: $(TEMPLATE_DIR)/00-$(1).ply $(CMD_DIR)/template/$(1).py
	$(PYTHON) $(CMD_DIR)/template/$(1).py $$< $$@
endef
$(eval $(call pre-process,face))
$(eval $(call pre-process,skull))

define edit
$(TEMPLATE_DIR)/02-$(1).ply: $(TEMPLATE_DIR)/01-$(1).ply
	$(MANUAL) Edit $$< $$@
endef
$(eval $(call edit,face))
$(eval $(call edit,skull))

define mesh-fix
$(TEMPLATE_DIR)/03-$(1).ply: $(TEMPLATE_DIR)/02-$(1).ply
	$(MESH_FIX) $$< $$@
endef
$(eval $(call mesh-fix,face))
$(eval $(call mesh-fix,skull))

define select
$(TEMPLATE_DIR)/03-$(1)-landmarks.txt: $(TEMPLATE_DIR)/03-$(1).ply
	$(MANUAL) Select $$< $$@
endef
