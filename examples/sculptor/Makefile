CMD_DIR  ?= cmd
DATA_DIR ?= $(HOME)/Documents/data
TARGET   ?= $(error TARGET is not set)

TARGET_DIR   := $(DATA_DIR)/targets/$(TARGET)
TEMPLATE_DIR := $(DATA_DIR)/template

MESH_FIX := MeshFix
PYTHON   := python

PREPROCESS_FLAGS :=
PACK             := $(PYTHON) "$(CMD_DIR)/pack.py" --mask-value 0.1
CT2MESH          := $(PYTHON) "$(CMD_DIR)/CT2mesh.py"
ALIGN            := $(PYTHON) "$(CMD_DIR)/align.py"
REGISTER         := $(PYTHON) "$(CMD_DIR)/register.py"

default: $(TARGET_DIR)/pre/05-face.ply
default: $(TARGET_DIR)/pre/05-skull.ply
default: $(TARGET_DIR)/post/05-face.ply
default: $(TARGET_DIR)/post/05-skull.ply

include makefiles/template.mk
include makefiles/register.mk
