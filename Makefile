TARGET      := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
HDRS        := $(wildcard *.h *.cuh)
CC          := nvcc

GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM_ARCHS   := 75 86 89 $(GPU_ARCH)
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
NATIVE_GENCODE := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

# --- A/B EXPERIMENT TOGGLE (f1 branch only; delete when the A/B concludes) --------------
# f1 builds the HOST_ONLY_FOUND_STOP treatment, so a plain `make` -- and therefore both
# bench_ab.sh's per-branch build AND proof.py -- exercises the flag-ON kernel (in-kernel
# found-flag polls removed; host stops scheduling between launches). main carries no such
# flag and is the baseline arm. If it wins, bake the behavior into the source and delete
# this; if it loses, it dies with the branch. See the HOST_ONLY_FOUND_STOP note in
# CUDACyclone.cu. Override at the CLI with `make AB_FLAGS=` to force a baseline build here.
AB_FLAGS   ?= -DHOST_ONLY_FOUND_STOP

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE) $(AB_FLAGS)
CXXFLAGS   := -std=c++17

LDFLAGS    := -lcudadevrt -cudart=static

.PHONY: all clean ptxinfo sass resusage

all: $(TARGET)

# NOTE: every compile below now lists $(HDRS) as a prerequisite, so editing any header
# forces a rebuild -- no more silent stale builds requiring `make clean` (the old trap).

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cu $(HDRS) third_party/RCKangaroo/RCGpuUtils.h
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

# ---- Phase 0: codegen inspection (no effect on the shipped binary) --------------------
# Surface what ptxas actually emitted so perf decisions (noinline, register budget,
# constant folding) are made on evidence rather than inference. See phase0-inspect.sh for
# a one-command wrapper that extracts the key signals.

# Verbose ptxas resource report (registers/thread, spill stores/loads, stack frame) printed
# during a native-arch build for every kernel + non-inlined device function.
ptxinfo: $(SRC) $(HDRS)
	$(CC) -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(NATIVE_GENCODE) $(CXXFLAGS) $(AB_FLAGS) \
	      -Xptxas -v $(SRC) -o CUDACyclone-ptxinfo $(LDFLAGS)
	@rm -f CUDACyclone-ptxinfo

# Per-kernel resource usage read back from the built fat binary (authoritative).
resusage: $(TARGET)
	cuobjdump -res-usage $(TARGET)

# Full SASS disassembly of the built fat binary.
sass: $(TARGET)
	cuobjdump -sass $(TARGET)

clean:
	rm -f $(TARGET) CUDACyclone-ptxinfo $(OBJ)
