export DATA_DIR  ?= $(HOME)/Documents/data
export TARGET_ID ?= 120056

srcdir       := src
TARGET_CT    := $(DATA_DIR)/CT/$(TARGET_ID)
TARGET_DIR   := $(DATA_DIR)/target/$(TARGET_ID)
TEMPLATE_DIR := $(DATA_DIR)/template

default: $(TARGET_DIR)/pre/99-face.vtu
default: $(TARGET_DIR)/pre/99-mandible.vtu
default: $(TARGET_DIR)/pre/99-maxilla.vtu
# default: $(TARGET_DIR)/post/99-face.vtu
# default: $(TARGET_DIR)/post/99-mandible.vtu
# default: $(TARGET_DIR)/post/99-maxilla.vtu

include makefiles/*.mk
