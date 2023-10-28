BIN_DIR := $(HOME)/.local/bin

SYSTEM  != python -c "import platform; print(platform.system().lower())"
MACHINE != python -c "import platform; print(platform.machine().lower())"

ifeq ($(SYSTEM), windows)
  EXE := .exe
else
  EXE :=
endif

all: MeshFix tetgen

MeshFix: $(BIN_DIR)/MeshFix$(EXE)

tetgen: $(BIN_DIR)/tetgen$(EXE)

#####################
# Auxiliary Targets #
#####################

$(BIN_DIR)/%$(EXE):
	wget --output-document=$@ https://github.com/liblaf/$*-mirror/releases/latest/download/$*-$(SYSTEM)-$(MACHINE)$(EXE)
	@ chmod --verbose +x $@
