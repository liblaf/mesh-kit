TARGET_DIR   := $(DATA_DIR)/targets/$(TARGET)
TEMPLATE_DIR := $(DATA_DIR)/template

$(TARGET_DIR)/pre/00-CT $(TARGET_DIR)/post/00-CT:
	$(info CT Data: "$@")
	test -d "$@"

define register # STAGE, COMPONENT
$$(TARGET_DIR)/$(1)/01-$(2).ply: $$(TARGET_DIR)/$(1)/00-CT
	$$(CT2MESH) --component "$(2)" "$$<" "$$@"

$$(TARGET_DIR)/$(1)/02-$(2).ply: $$(TARGET_DIR)/$(1)/01-$(2).ply
	# Simplify
	@ cp --force --no-target-directory --verbose "$$<" "$$@"

$$(TARGET_DIR)/$(1)/02-$(2)-landmarks.txt: $$(TARGET_DIR)/$(1)/02-$(2).ply
	$$(info Annotate Landmarks: "$$<")
	test -s "$$@"
	touch "$$@"
	test "$$@" -nt "$$<"

$$(TARGET_DIR)/$(1)/03-$(2).ply $$(TARGET_DIR)/$(1)/03-$(2).npz &: $$(TARGET_DIR)/$(1)/02-$(2).ply $$(TARGET_DIR)/$(1)/02-$(2)-landmarks.txt
	$$(PACK) "$$<" "$$@"

$$(TARGET_DIR)/$(1)/04-$(2).ply $$(TARGET_DIR)/$(1)/04-$(2).npz &: $$(TEMPLATE_DIR)/04-$(2).ply $$(TEMPLATE_DIR)/04-$(2).npz $$(TARGET_DIR)/$(1)/03-$(2).ply $$(TARGET_DIR)/$(1)/03-$(2).npz
	$$(ALIGN) --output "$$(TARGET_DIR)/$(1)/04-$(2).ply" "$$(TEMPLATE_DIR)/04-$(2).ply" "$$(TARGET_DIR)/$(1)/03-$(2).ply"

$$(TARGET_DIR)/$(1)/05-$(2).ply $$(TARGET_DIR)/$(1)/05-$(2).npz &: $$(TARGET_DIR)/$(1)/04-$(2).ply $$(TARGET_DIR)/$(1)/04-$(2).npz $$(TARGET_DIR)/$(1)/03-$(2).ply $$(TARGET_DIR)/$(1)/03-$(2).npz $(REGISTER_DEPS)
	@ rm --force --recursive --verbose "$$(dir $$@)/register-$(2)"
	@ mkdir --parents --verbose "$$(dir $$@)/register-$(2)"
	$$(REGISTER) --output "$$@" --config "config/$(2).json" --record-dir "$$(dir $$@)/register-$(2)" "$$(TARGET_DIR)/$(1)/04-$(2).ply" "$$(TARGET_DIR)/$(1)/03-$(2).ply"
endef

$(eval $(call register,pre,face))
$(eval $(call register,pre,skull))
$(eval $(call register,post,face))
$(eval $(call register,post,skull))
