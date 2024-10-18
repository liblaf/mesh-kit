default:

include makefiles/*.mk

.PHONY: gen-init
gen-init:
	@ bash scripts/gen-init.sh
