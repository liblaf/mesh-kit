DATA_DIR ?= $(HOME)/Documents/data
REMOTE   ?= business:/backup/data.zip

TARGETS_DIR  := $(DATA_DIR)/targets
TEMPLATE_DIR := $(DATA_DIR)/template

backup: data.zip
	rclone copyto --progress "$<" "$(REMOTE)"

###############
# Auxiliaries #
###############

data.zip: $(wildcard $(TARGETS_DIR)/*/*/00-CT)
data.zip: $(wildcard $(TARGETS_DIR)/*/*/02-*-landmarks.txt)
data.zip: $(wildcard $(TEMPLATE_DIR)/03-*-landmarks.txt)
	@ rm --force --verbose "$@"
	cd "$(DATA_DIR)" && 7z a "$(abspath $@)" -- $(shell realpath --relative-to="$(DATA_DIR)" $^)
