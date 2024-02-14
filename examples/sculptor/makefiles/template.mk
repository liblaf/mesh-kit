TEMPLATE_DIR := $(DATA_DIR)/template

define template # COMPONENT
$(TEMPLATE_DIR)/00-$(1).ply:
	@ mkdir --parents --verbose "$$(dir $$@)"
	wget --output-document="$$@" "https://raw.githubusercontent.com/liblaf/sculptor/main/model/template/$(1).ply"

$(TEMPLATE_DIR)/01-$(1).ply: $(TEMPLATE_DIR)/00-$(1).ply $$(CMD_DIR)/template/$(1).py
	$$(PYTHON) "$$(CMD_DIR)/template/$(1).py" "$$<" "$$@"

$(TEMPLATE_DIR)/02-$(1).ply: $(TEMPLATE_DIR)/01-$(1).ply
	$$(info Manual Edit: "$$<" -> "$$@")
	@ cp --force --no-target-directory --verbose "$$<" "$$@"
	test "$$@" -nt "$$<"

$(TEMPLATE_DIR)/03-$(1).ply: $(TEMPLATE_DIR)/02-$(1).ply
	$$(MESH_FIX) "$$<" "$$@"

$(TEMPLATE_DIR)/03-$(1)-landmarks.txt: $(TEMPLATE_DIR)/03-$(1).ply
	$$(info Annotate Landmarks: "$$<")
	test -s "$$@"
	touch "$$@"
	test "$$@" -nt "$$<"
endef

$(eval $(call template,face))
$(eval $(call template,skull))
