DATA_DIR  ?= $(HOME)/Documents/data
TARGET_ID ?= 120056
STAGE     ?= post

srcdir       := src
TARGET_CT    := $(DATA_DIR)/CT/$(TARGET_ID)/$(STAGE)
TARGET_DIR   := $(DATA_DIR)/target/$(TARGET_ID)/$(STAGE)
TEMPLATE_DIR := $(DATA_DIR)/template

default: test

test: $(TEMPLATE_DIR)/01-face.vtu $(TEMPLATE_DIR)/01-mandible.vtu $(TEMPLATE_DIR)/01-maxilla.vtu $(srcdir)/tetgen.py
	python "$(lastword $^)" --pre-face="$<" --pre-mandible="$(word 2,$^)" --pre-maxilla="$(word 3,$^)"
