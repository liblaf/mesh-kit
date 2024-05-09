DATA_DIR  ?= $(HOME)/Documents/data
TARGET_ID ?= 120056

srcdir       := src
TEMPLATE_DIR := $(DATA_DIR)/template
CT_DIR       := $(DATA_DIR)/CT/$(TARGET_ID)
TARGET_DIR   := $(DATA_DIR)/target/$(TARGET_ID)
PRE_DIR      := $(TARGET_DIR)/pre
POST_DIR     := $(TARGET_DIR)/post

FMM  := face mandible maxilla
FS   := face skull
FSMM := face skull mandible maxilla
MM   := mandible maxilla

# CT to face mesh
$(PRE_DIR)/00-face.ply: $(CT_DIR)/pre $(srcdir)/CT-to-mesh.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --threshold="-200" "$<"

# CT to skull mesh
$(PRE_DIR)/00-skull.ply: $(CT_DIR)/pre $(srcdir)/CT-to-mesh.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --threshold="200" "$<"

# simplify face/skull
$(FS:%=$(PRE_DIR)/01-%.ply): $(PRE_DIR)/01-%.ply: $(PRE_DIR)/00-%.ply $(srcdir)/simplify.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --face-count="-1" "$<"
	@- cp --archive --force --verbose "$(<:.ply=.xyz)" "$(@:.ply=.xyz)"

# copy face/skull
$(FS:%=$(PRE_DIR)/02-%.ply): $(PRE_DIR)/02-%.ply: $(PRE_DIR)/01-%.ply
	@ cp --archive --force --verbose "$<" "$@"
	@- cp --archive --force --verbose "$(<:.ply=.xyz)" "$(@:.ply=.xyz)"

# copy mandible/maxilla
$(MM:%=$(PRE_DIR)/02-%.ply): $(PRE_DIR)/02-%.ply: $(PRE_DIR)/02-skull.ply
	@ cp --archive --force --verbose "$<" "$@"

# align template skull to target skull
$(PRE_DIR)/03-skull.vtu $(TARGET_DIR)/template-to-target.npy &: $(PRE_DIR)/skull-template.vtu $(PRE_DIR)/02-skull.ply $(srcdir)/align.py
	python "$(lastword $^)" --output="$(PRE_DIR)/03-skull.vtu" --output-transform="$(TARGET_DIR)/template-to-target.npy" --smart-initial "$<" "$(word 2,$^)"

# align template face/mandible/maxilla to target
$(FMM:%=$(PRE_DIR)/03-%.vtu): $(PRE_DIR)/03-%.vtu: $(PRE_DIR)/%-template.vtu $(PRE_DIR)/02-%.ply $(TARGET_DIR)/template-to-target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" "$<" "$(word 2,$^)"

# register template face/mandible/maxilla to target
$(FMM:%=$(PRE_DIR)/99-%.vtu): $(PRE_DIR)/99-%.vtu: $(PRE_DIR)/03-%.vtu $(PRE_DIR)/02-%.ply $(srcdir)/register.py
	python "$(lastword $^)" --output="$@" "$<" "$(word 2,$^)"
