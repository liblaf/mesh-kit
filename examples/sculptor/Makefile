CMD_DIR  ?= cmd
DATA_DIR ?= $(HOME)/Documents/data
TARGET   ?= $(error TARGET is not set)

TARGET_DIR   := $(DATA_DIR)/targets/$(TARGET)
TEMPLATE_DIR := $(DATA_DIR)/template

MESH_FIX := MeshFix
PYTHON   := python

CT2MESH  := $(PYTHON) $(CMD_DIR)/CT2mesh.py
ALIGN    := $(PYTHON) $(CMD_DIR)/align.py
REGISTER := $(PYTHON) $(CMD_DIR)/register.py --watertight

default: $(TARGET_DIR)/pre/04-face.ply
default: $(TARGET_DIR)/pre/04-skull.ply

include makefiles/template.mk
include makefiles/register.mk
