default:

data/%.vtu: ../01-tetgen/src/create.py
	@ mkdir --parents --verbose "$(@D)"
	python "$<" --output="$@" --displacement="$*"

# displacement, Poisson's ratio, method
define run
default: data/$(1)-$(2)-$(3).vtu
data/$(1)-$(2)-$(3).vtu: data/$(1).vtu src/mtm-$(3).py
	python "src/mtm-$(3).py" --output="$$@" --poisson-ratio="$(2)" "$$<"
endef

$(eval $(call run,0.01,0.00,matrix-free))
$(eval $(call run,0.01,0.10,matrix-free))
$(eval $(call run,0.01,0.20,matrix-free))
$(eval $(call run,0.01,0.30,matrix-free))
$(eval $(call run,0.01,0.40,matrix-free))
$(eval $(call run,0.01,0.46,matrix-free))
$(eval $(call run,0.05,0.00,matrix-free))
$(eval $(call run,0.05,0.10,matrix-free))
$(eval $(call run,0.05,0.20,matrix-free))
$(eval $(call run,0.05,0.30,matrix-free))
$(eval $(call run,0.05,0.40,matrix-free))
$(eval $(call run,0.05,0.46,matrix-free))
$(eval $(call run,0.01,0.00,mollemans))
$(eval $(call run,0.01,0.10,mollemans))
$(eval $(call run,0.01,0.20,mollemans))
$(eval $(call run,0.01,0.30,mollemans))
$(eval $(call run,0.01,0.40,mollemans))
$(eval $(call run,0.01,0.46,mollemans))
$(eval $(call run,0.05,0.00,mollemans))
$(eval $(call run,0.05,0.10,mollemans))
$(eval $(call run,0.05,0.20,mollemans))
$(eval $(call run,0.05,0.30,mollemans))
$(eval $(call run,0.05,0.40,mollemans))
$(eval $(call run,0.05,0.46,mollemans))
