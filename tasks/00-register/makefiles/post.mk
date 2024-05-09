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
$(POST_DIR)/00-face.ply: $(CT_DIR)/post $(srcdir)/CT-to-mesh.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --threshold="-200" "$<"

# CT to skull mesh
$(POST_DIR)/00-skull.ply: $(CT_DIR)/post $(srcdir)/CT-to-mesh.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --threshold="200" "$<"

# simplify face/skull
$(FS:%=$(POST_DIR)/01-%.ply): $(POST_DIR)/01-%.ply: $(POST_DIR)/00-%.ply $(srcdir)/simplify.py
	@ mkdir --parents --verbose "$(@D)"
	python "$(lastword $^)" --output="$@" --face-count="-1" "$<"
	@- cp --archive --force --verbose "$(<:.ply=.xyz)" "$(@:.ply=.xyz)"

# align post skull to pre skull
$(POST_DIR)/02-skull.ply $(TARGET_DIR)/post-to-pre.npy: $(POST_DIR)/01-skull.ply $(PRE_DIR)/01-skull.ply $(srcdir)/align.py
	python "$(lastword $^)" --output="$(POST_DIR)/02-skull.ply" --output-transform="$(TARGET_DIR)/post-to-pre.npy" "$<" "$(word 2,$^)"

# align post face to pre face
$(POST_DIR)/02-face.ply: $(POST_DIR)/01-face.ply $(TARGET_DIR)/post-to-pre.npy $(srcdir)/apply-transform.py
	python "$(lastword $^)" --output="$@" --transform="$(word 2,$^)" "$<"

# copy mandible/maxilla
$(MM:%=$(POST_DIR)/02-%.ply): $(POST_DIR)/02-%.ply: $(POST_DIR)/02-skull.ply
	@ cp --archive --force --verbose "$<" "$@"
	@- cp --archive --force --verbose "$(<:.ply=.xyz)" "$(@:.ply=.xyz)"

# align template skull to target skull
$(POST_DIR)/03-skull.vtu: $(POST_DIR)/skull-template.vtu $(POST_DIR)/02-skull.ply $(TARGET_DIR)/template-to-target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" "$<" "$(word 2,$^)"

# align template face/mandible/maxilla to target
$(FMM:%=$(POST_DIR)/03-%.vtu): $(POST_DIR)/03-%.vtu: $(POST_DIR)/%-template.vtu $(POST_DIR)/02-%.ply $(TARGET_DIR)/template-to-target.npy $(srcdir)/align.py
	python "$(lastword $^)" --output="$@" --initial-transform="$(word 3,$^)" "$<" "$(word 2,$^)"

# register template face/mandible/maxilla to target
$(FMM:%=$(POST_DIR)/99-%.vtu): $(POST_DIR)/99-%.vtu: $(POST_DIR)/03-%.vtu $(POST_DIR)/02-%.ply $(srcdir)/register.py
	python "$(lastword $^)" --output="$@" "$<" "$(word 2,$^)"
