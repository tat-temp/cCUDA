TARGET      := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
HDRS        := $(wildcard *.h *.cuh)
CC          := nvcc

GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM_ARCHS   := 75 86 89 $(GPU_ARCH)
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
NATIVE_GENCODE := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17

LDFLAGS    := -lcudadevrt -cudart=static

.PHONY: all clean ecgen shaonly ptxinfo sass resusage legacy

all: $(TARGET)

# NOTE: every compile below now lists $(HDRS) as a prerequisite, so editing any header
# forces a rebuild -- no more silent stale builds requiring `make clean` (the old trap).

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

# EC-generation-only benchmark: the identical kernel built with -DEC_GEN_ONLY, which
# skips SHA-256/RIPEMD-160 + the address match (a tiny XOR sink keeps the EC math live).
ecgen: CUDACyclone-ecgen
CUDACyclone-ecgen: $(SRC) $(HDRS)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -DEC_GEN_ONLY $(SRC) -o $@ $(LDFLAGS)

# SHA-only benchmark: same kernel that hashes with SHA-256 but skips RIPEMD-160 (-DSHA_ONLY),
# to split the hashing cost. Compare full vs shaonly vs ecgen.
shaonly: CUDACyclone-shaonly
CUDACyclone-shaonly: $(SRC) $(HDRS)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -DSHA_ONLY $(SRC) -o $@ $(LDFLAGS)

%.o: %.cu $(HDRS) third_party/RCKangaroo/RCGpuUtils.h
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

# Legacy field build: CUDACyclone's own JeanLucPons-lineage _ModMult/_ModSqr/_ModInv instead of
# the default RCKangaroo ops (compiles NO third_party/RCKangaroo code in). ~8.5% slower on the
# RTX 5090; kept for reference and A/B against the default. See CUDAMath.h / LICENSE.
legacy: CUDACyclone-legacy
CUDACyclone-legacy: $(SRC) $(HDRS)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -DUSE_CYCLONE_FIELD $(SRC) -o $@ $(LDFLAGS)

# ---- Phase 0: codegen inspection (no effect on the shipped binary) --------------------
# Surface what ptxas actually emitted so perf decisions (noinline, register budget,
# constant folding) are made on evidence rather than inference. See phase0-inspect.sh for
# a one-command wrapper that extracts the key signals.

# Verbose ptxas resource report (registers/thread, spill stores/loads, stack frame) printed
# during a native-arch build for every kernel + non-inlined device function.
ptxinfo: $(SRC) $(HDRS)
	$(CC) -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(NATIVE_GENCODE) $(CXXFLAGS) \
	      -Xptxas -v $(SRC) -o CUDACyclone-ptxinfo $(LDFLAGS)
	@rm -f CUDACyclone-ptxinfo

# Per-kernel resource usage read back from the built fat binary (authoritative).
resusage: $(TARGET)
	cuobjdump -res-usage $(TARGET)

# Full SASS disassembly of the built fat binary.
sass: $(TARGET)
	cuobjdump -sass $(TARGET)

clean:
	rm -f $(TARGET) CUDACyclone-legacy CUDACyclone-ecgen CUDACyclone-shaonly \
	      CUDACyclone-ptxinfo $(OBJ)
