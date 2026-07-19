TARGET      := CUDACyclone
# P3.0: CUDAHash.cu is NOT compiled standalone -- it is #included into CUDACyclone.cu so the
# whole device program is ONE TU and can build with rdc=false (see NVCC_FLAGS). Listing it here
# too would emit a duplicate getHash160 device symbol at link.
SRC         := CUDACyclone.cu
OBJ         := $(SRC:.cu=.o)
HDRS        := $(wildcard *.h *.cuh)
CC          := nvcc

GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM_ARCHS   := 75 86 89 $(GPU_ARCH)
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
NATIVE_GENCODE := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

# P3.0: -rdc=true REMOVED. It existed solely to device-link the cross-TU getHash160 call; with
# CUDAHash.cu #included into CUDACyclone.cu there are no cross-TU device calls left, so separate
# compilation (and its conservative ABI-stable calling convention) is no longer needed. Dropping
# it lets ptxas optimize hash+kernel together and use an intra-module register ABI for the
# still-__noinline__ getHash160. -lcudadevrt goes with it: the device runtime is only required
# for rdc device linking (no dynamic parallelism / device-side launch in this code).
NVCC_FLAGS := -O3 -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17

LDFLAGS    := -cudart=static

.PHONY: all clean ptxinfo sass resusage

all: $(TARGET)

# NOTE: every compile below now lists $(HDRS) as a prerequisite, so editing any header
# forces a rebuild -- no more silent stale builds requiring `make clean` (the old trap).

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

# P3.0: CUDAHash.cu is now an #included TU member, but HDRS only globs *.h/*.cuh -- so it must be
# listed explicitly or editing the hash would silently NOT rebuild CUDACyclone.o (the project's
# long-standing stale-build trap, in a new disguise).
%.o: %.cu $(HDRS) CUDAHash.cu third_party/RCKangaroo/RCGpuUtils.h
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

# ---- Phase 0: codegen inspection (no effect on the shipped binary) --------------------
# Surface what ptxas actually emitted so perf decisions (noinline, register budget,
# constant folding) are made on evidence rather than inference. See phase0-inspect.sh for
# a one-command wrapper that extracts the key signals.

# Verbose ptxas resource report (registers/thread, spill stores/loads, stack frame) printed
# during a native-arch build for every kernel + non-inlined device function.
# P3.0: -rdc=true dropped here too. This target MUST mirror NVCC_FLAGS' build model -- if it
# still passed -rdc=true it would report registers/spill for a DIFFERENT compilation than the
# shipped binary, i.e. the gate would lie.
ptxinfo: $(SRC) $(HDRS) CUDAHash.cu
	$(CC) -O3 -use_fast_math --ptxas-options=-O3 $(NATIVE_GENCODE) $(CXXFLAGS) \
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
