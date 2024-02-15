DATA_DIR ?= $(HOME)/Documents/data
REMOTE   ?= business:/backup/data.zip

TARGETS_DIR  := $(DATA_DIR)/targets
TEMPLATE_DIR := $(DATA_DIR)/template

recover: data.zip
	@ mkdir --parents --verbose "$(DATA_DIR)"
	7z x -o"$(DATA_DIR)" "$<"

###############
# Auxiliaries #
###############

data.zip:
	rclone copyto --progress "$(REMOTE)" "$@"
