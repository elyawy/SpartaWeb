# SpartaWeb — Emscripten build
# Prerequisites: source emsdk_env.sh before running make
#
# Output: sparta.js + sparta.wasm  (ES6 module)
# Usage:  make          — release build
#         make debug    — unoptimised + source maps
#         make clean

EMCC      := em++
OUTDIR    := dist
OUT       := $(OUTDIR)/sparta.js

SRC       := spartaBindings.cpp \
             ../libs/MSAStats/src/MsaStatsCalculator.cpp \
             ../libs/MSAStats/src/read_seqs.cpp

# Adjust paths if your layout differs
PHYLOLIB  := ../libs/Sailfish-backend/libs/emPhylolib/libPhylolib.a
INCLUDES  := -I../libs \
             -I../libs/eigen

CXXFLAGS  := -std=c++17 \
             $(INCLUDES)

# -sMODULARIZE + EXPORT_ES6 → ES6 module  (import createSparta from './sparta.js')
# -sALLOW_MEMORY_GROWTH     → heap can grow; simulation sizes vary
# -lembind                  → Embind runtime
# -sEXPORT_NAME             → module factory function name in JS
EMFLAGS   := -lembind \
             -sMODULARIZE=1 \
             -sEXPORT_ES6=1 \
             -sEXPORT_NAME=createSparta \
             -sALLOW_MEMORY_GROWTH=1 \
             -sSTACK_SIZE=8388608 \
             -sENVIRONMENT=web \
             -sNO_DISABLE_EXCEPTION_CATCHING

# ── Release ──────────────────────────────────────────────────────────────────
.PHONY: all
all: $(OUTDIR) $(OUT)

$(OUT): $(SRC) $(PHYLOLIB)
	$(EMCC) $(CXXFLAGS) -O3 $(SRC) $(PHYLOLIB) $(EMFLAGS) -o $(OUT)

# ── Debug (no optimisation, DWARF source maps) ────────────────────────────────
.PHONY: debug
debug: $(OUTDIR)
	$(EMCC) $(CXXFLAGS) -O0 -g $(SRC) $(PHYLOLIB) $(EMFLAGS) \
	    -sASSERTIONS=2 \
	    -o $(OUTDIR)/sparta.debug.js

# ── Helpers ───────────────────────────────────────────────────────────────────
$(OUTDIR):
	mkdir -p $(OUTDIR)

.PHONY: clean
clean:
	rm -rf $(OUTDIR)
