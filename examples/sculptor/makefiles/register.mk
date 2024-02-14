TARGET_DIR   := $(DATA_DIR)/$(TARGET)
TEMPLATE_DIR := $(DATA_DIR)/template

$(TARGET_DIR)/pre/00-CT:
$(TARGET_DIR)/post/00-CT:
	$(info CT Data: "$@")
	test -d "$@"

define register # STAGE, COMPONENT
$(TARGET_DIR)/$(1)/01-$(2).ply: $(TARGET_DIR)/$(1)/00-CT
	$(CT2MESH) --component "$(2)" "$$<" "$$@"

$(TARGET_DIR)/$(1)/02-$(2).ply: $(TARGET_DIR)/$(1)/01-$(2).ply
	# Simplify
	@ cp --force --no-target-directory --verbose "$$<" "$$@"

$(TARGET_DIR)/$(1)/02-$(2)-landmarks.txt: $(TARGET_DIR)/$(1)/02-$(2).ply
	$$(info Annotate Landmarks: "$$<")
	test -s "$$@"
	touch "$$@"
	test "$$@" -nt "$$<"

$(TARGET_DIR)/$(1)/03-$(2).ply $(TARGET_DIR)/$(1)/03-$(2)-landmarks.txt &: $$(TEMPLATE_DIR)/03-$(2).ply $$(TEMPLATE_DIR)/03-$(2)-landmarks.txt $(TARGET_DIR)/$(1)/02-$(2).ply $(TARGET_DIR)/$(1)/02-$(2)-landmarks.txt
	$(ALIGN) --output "$(TARGET_DIR)/$(1)/03-$(2).ply" "$$(TEMPLATE_DIR)/03-$(2).ply" "$(TARGET_DIR)/$(1)/02-$(2).ply"

$(TARGET_DIR)/$(1)/04-$(2).ply: $(TARGET_DIR)/$(1)/03-$(2).ply $(TARGET_DIR)/$(1)/03-$(2)-landmarks.txt $(TARGET_DIR)/$(1)/02-$(2).ply $(TARGET_DIR)/$(1)/02-$(2)-landmarks.txt
	@ rm --force --recursive --verbose "$$(dir $$@)/register"
	@ mkdir --parents --verbose "$$(dir $$@)/register"
	$(REGISTER) --output "$$@" --record-dir "$$(dir $$@)/register" "$(TARGET_DIR)/$(1)/03-$(2).ply" "$(TARGET_DIR)/$(1)/02-$(2).ply"
endef

$(eval $(call register,pre,face))
$(eval $(call register,pre,skull))
$(eval $(call register,post,face))
$(eval $(call register,post,skull))
