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

.PHONY: all clean ecgen shaonly ptxinfo sass resusage ecbench rckfield rckinv rckall crfield crfieldD

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

%.o: %.cu $(HDRS)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

# ---- RCKangaroo EC-field A/B (branch rck-ec-ab) ---------------------------------------
# See EC_AB_README.md. Compares RCKangaroo's field ops (third_party/RCKangaroo, GPLv3)
# against CUDACyclone's own. ec_backend.cuh is the shim; -DUSE_RCK_FIELD / -DUSE_RCK_INV
# swap them into the production kernel with zero call-site changes.
RCK_HDR := third_party/RCKangaroo/RCGpuUtils.h ec_backend.cuh

# Standalone per-op correctness (vs Python KATs) + throughput microbench. No hashing.
ecbench: CUDACyclone-ecbench
CUDACyclone-ecbench: ecbench.cu $(HDRS) $(RCK_HDR)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) ecbench.cu -o $@ $(LDFLAGS)

# Full search binary, RCKangaroo mul+sqr swapped in (the throughput lever).
rckfield: CUDACyclone-rckfield
CUDACyclone-rckfield: $(SRC) $(HDRS) $(RCK_HDR)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -DUSE_RCK_FIELD $(SRC) -o $@ $(LDFLAGS)

# Full search binary, RCKangaroo safegcd inversion swapped in (amortized ~1/batch).
rckinv: CUDACyclone-rckinv
CUDACyclone-rckinv: $(SRC) $(HDRS) $(RCK_HDR)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -DUSE_RCK_INV $(SRC) -o $@ $(LDFLAGS)

# Full search binary, RCKangaroo mul+sqr+inv all swapped in.
rckall: CUDACyclone-rckall
CUDACyclone-rckall: $(SRC) $(HDRS) $(RCK_HDR)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -DUSE_RCK_FIELD -DUSE_RCK_INV $(SRC) -o $@ $(LDFLAGS)

# Full search binary, CLEAN-ROOM 32-bit multiply swapped in (cr_field.cuh, license-clean
# reproduction of the RCK mul/sqr win; baseline inverse kept). Add CR_NOINLINE=1 to test
# the __noinline__ fallback if the register check shows spills.
CR_DEFS := -DUSE_CR_FIELD $(if $(CR_NOINLINE),-DCR_NOINLINE,)
# crfield  = clean-room operand-scan multiply (prodA, default)
crfield: CUDACyclone-crfield
CUDACyclone-crfield: $(SRC) $(HDRS) cr_field.cuh
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(CR_DEFS) $(SRC) -o $@ $(LDFLAGS)
# crfieldD = clean-room Comba multiply (prodD)
crfieldD: CUDACyclone-crfieldD
CUDACyclone-crfieldD: $(SRC) $(HDRS) cr_field.cuh
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(CR_DEFS) -DCR_USE_D $(SRC) -o $@ $(LDFLAGS)

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
	rm -f $(TARGET) CUDACyclone-ecgen CUDACyclone-shaonly CUDACyclone-ptxinfo \
	      CUDACyclone-ecbench CUDACyclone-rckfield CUDACyclone-rckinv CUDACyclone-rckall \
	      CUDACyclone-crfield CUDACyclone-crfieldD $(OBJ)
