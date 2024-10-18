GEN_TYPING_TARGETS += src/mkit/typing/array/__init__.pyi
GEN_TYPING_TARGETS += src/mkit/typing/array/_bool.py
GEN_TYPING_TARGETS += src/mkit/typing/array/_float.py
GEN_TYPING_TARGETS += src/mkit/typing/array/_integer.py
GEN_TYPING_TARGETS += src/mkit/typing/jax/__init__.pyi
GEN_TYPING_TARGETS += src/mkit/typing/jax/_bool.py
GEN_TYPING_TARGETS += src/mkit/typing/jax/_export.py
GEN_TYPING_TARGETS += src/mkit/typing/jax/_float.py
GEN_TYPING_TARGETS += src/mkit/typing/jax/_integer.py
GEN_TYPING_TARGETS += src/mkit/typing/numpy/__init__.pyi
GEN_TYPING_TARGETS += src/mkit/typing/numpy/_bool.py
GEN_TYPING_TARGETS += src/mkit/typing/numpy/_export.py
GEN_TYPING_TARGETS += src/mkit/typing/numpy/_float.py
GEN_TYPING_TARGETS += src/mkit/typing/numpy/_integer.py
GEN_TYPING_TARGETS += src/mkit/typing/torch/__init__.pyi
GEN_TYPING_TARGETS += src/mkit/typing/torch/_bool.py
GEN_TYPING_TARGETS += src/mkit/typing/torch/_export.py
GEN_TYPING_TARGETS += src/mkit/typing/torch/_float.py
GEN_TYPING_TARGETS += src/mkit/typing/torch/_integer.py

.PHONY: $(GEN_TYPING_TARGETS)
gen-typing: $(GEN_TYPING_TARGETS)

# ----------------------------- Auxiliary Targets ---------------------------- #

$(GEN_TYPING_TARGETS): src/mkit/%: templates/%.jinja scripts/gen-typing.py
	@ python scripts/gen-typing.py --output "$@" "$<"
	@ ruff check "$@"
