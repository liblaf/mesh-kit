CMD_DIR  ?= cmd
DATA_DIR ?= $(HOME)/Documents/data
TARGET   ?= $(error TARGET is not set)

TARGET_DIR   := $(DATA_DIR)/targets/$(TARGET)
TEMPLATE_DIR := $(DATA_DIR)/template

MESH_FIX := MeshFix
PYTHON   := python

CT2MESH  := $(PYTHON) $(CMD_DIR)/CT2mesh.py
ALIGN    := $(PYTHON) $(CMD_DIR)/align.py
REGISTER := $(PYTHON) $(CMD_DIR)/register.py --config data/config.json

REGISTER_DEPS := $(CMD_DIR)/register.py data/config.json

default: $(TARGET_DIR)/pre/04-face.ply
# default: $(TARGET_DIR)/pre/04-skull.ply
default: $(TARGET_DIR)/post/04-face.ply
# default: $(TARGET_DIR)/post/04-skull.ply

backup: makefiles/backup.mk
	$(MAKE) --file="$<"

recover: makefiles/recover.mk
	$(MAKE) --file="$<"

include makefiles/template.mk
include makefiles/register.mk
