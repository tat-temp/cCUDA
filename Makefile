TARGET      := CUDACyclone
# P3.0: CUDAHash.cu is NOT compiled standalone -- it is #included into CUDACyclone.cu so the
# whole device program is ONE TU and can build with rdc=false (see NVCC_FLAGS). Listing it here
# too would emit a duplicate getHash160 device symbol at link.
SRC         := CUDACyclone.cu
OBJ         := $(SRC:.cu=.o)
HDRS        := $(wildcard *.h *.cuh)
CC          := nvcc

# Native arch, auto-detected. `?=` so it can be set explicitly on a box where nvidia-smi is
# absent or not on PATH:  make GPU_ARCH=120
GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')

# HARD FAIL on an undetected GPU_ARCH rather than building something subtly wrong.
# WITHOUT this guard the failure mode is backwards: an empty GPU_ARCH degrades SM_ARCHS to
# the still-VALID list "75 86 89", so `make all` SUCCEEDS and ships a binary with no cubin
# for the arch it will actually run on -- which then falls back to slow PTX JIT at load --
# while `make ptxinfo`/`gate` fail loudly on a malformed "-gencode arch=compute_,code=sm_".
# i.e. the thing that ships wrong succeeds and the diagnostic errors. Guarded so targets
# that need no compiler still work on a GPU-less box.
NOARCH_OK := clean fieldtest
ifneq ($(filter-out $(NOARCH_OK),$(or $(MAKECMDGOALS),all)),)
  ifeq ($(strip $(GPU_ARCH)),)
    $(error could not detect GPU compute capability (nvidia-smi missing, off PATH, or returned nothing). Pass it explicitly: make GPU_ARCH=120)
  endif
endif

# Dedupe the native arch against the baked-in list: on an sm_75/86/89 box the naive
# "75 86 89 $(GPU_ARCH)" yields a DUPLICATE -gencode, which nvcc rejects outright.
# filter-out is used rather than $(sort) deliberately -- $(sort) would also REORDER the
# list lexicographically ("120 75 86 89"), whereas this appends only when needed and so
# emits a byte-identical command line to the previous Makefile on any non-{75,86,89} GPU.
BASE_ARCHS := 75 86 89
SM_ARCHS   := $(strip $(BASE_ARCHS) $(filter-out $(BASE_ARCHS),$(GPU_ARCH)))
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
NATIVE_GENCODE := -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)

# P3.0: -rdc=true REMOVED. It existed solely to device-link the cross-TU getHash160 call; with
# CUDAHash.cu #included into CUDACyclone.cu there are no cross-TU device calls left, so separate
# compilation (and its conservative ABI-stable calling convention) is no longer needed. Dropping
# it lets ptxas optimize hash+kernel together and use an intra-module register ABI for the
# still-__noinline__ getHash160. -lcudadevrt goes with it: the device runtime is only required
# for rdc device linking (no dynamic parallelism / device-side launch in this code).
NVCC_FLAGS := -O3 -use_fast_math --ptxas-options=-O3 $(GENCODE)
# -pthread is required now that the host side spawns a std::thread (worker/reporter split).
# On glibc >= 2.34 libpthread is folded into libc and this is a no-op; below that, omitting it
# fails at link or throws std::system_error on first thread construction. Host-side only --
# it does not affect device codegen, so `make gate` numbers are unchanged by it.
CXXFLAGS   := -std=c++17 -Xcompiler -pthread

LDFLAGS    := -cudart=static -Xcompiler -pthread

.PHONY: all clean ptxinfo gate sass resusage fieldtest

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

# ---- Codegen inspection (no effect on the shipped binary) -----------------------------
# Surface what ptxas actually emitted so perf decisions (noinline, register budget,
# constant folding) are made on evidence rather than inference. `ptxinfo` PRINTS the
# numbers for a human; `gate` CHECKS them and fails the build. Prefer `gate` in any
# workflow where nobody is guaranteed to read the output.

# Native-arch verbose compile, shared by `ptxinfo` and `gate` so the two can never drift.
# This MUST mirror NVCC_FLAGS' build model -- if it still passed -rdc=true (dropped in P3.0)
# it would report registers/spill for a DIFFERENT compilation than the shipped binary, i.e.
# the gate would lie. Native-arch-only is deliberate for that same fidelity reason;
# `make resusage` (cuobjdump -res-usage on the fat binary) is the all-arch cross-check.
PTXAS_V_BUILD = $(CC) -O3 -use_fast_math --ptxas-options=-O3 $(NATIVE_GENCODE) $(CXXFLAGS) \
                -Xptxas -v $(SRC) -o CUDACyclone-ptxinfo $(LDFLAGS)

# Register ceiling = regfile per SM / (threads per block * blocks per SM)
#                  = 65536 / (256 * 2) = 128, from __launch_bounds__(256,2) on the hot
# kernel. Crossing it forces ptxas to spill or to drop occupancy below the RAW-optimal
# 16 warps/SM -- both measured losses. Keep this in sync with the launch_bounds.
HOT_KERNEL  := kernel_point_add_and_check_oneinv
REG_CEILING := 128
GATE_LOG    := ptxas-gate.log

# Verbose ptxas resource report (registers/thread, spill stores/loads, stack frame) printed
# during a native-arch build for every kernel + non-inlined device function.
ptxinfo: $(SRC) $(HDRS) CUDAHash.cu
	$(PTXAS_V_BUILD)
	@rm -f CUDACyclone-ptxinfo

# Machine-checked resource gate: exits NONZERO on any spill, or if the hot kernel crosses
# the register ceiling. Run it before any benchmark -- both checks are clock-independent,
# so they stay valid on a busy or throttled box where a keys/s A/B is not.
#
# WHY THIS EXISTS: a spill is semantically transparent -- hashes stay bit-identical, so
# proof.py passes unchanged and cannot catch one. Before this target the only thing
# standing between a regression and a silent 5%+ loss was a human remembering to read
# `make ptxinfo` output.
#
# SCOPE -- read this before trusting a PASS: "0 bytes spill stores/loads" is NOT "no
# local-memory traffic". ptxas counts an explicit __device__ local array (the hot kernel's
# 16 KB subp[MAX_BATCH_SIZE/2][4]) toward the STACK FRAME but never toward spill, so the
# frame is legitimately ~16 KB here while spills are zero. Frame size is ALLOCATION;
# LDL/STL in SASS is TRAFFIC. They are not interchangeable. This gate covers spill and
# registers only -- use `make sass` and count LDL/STL by address region for traffic.
gate: $(SRC) $(HDRS) CUDAHash.cu
	@echo "== native-arch build with -Xptxas -v =="
	@$(PTXAS_V_BUILD) 2> $(GATE_LOG) || { echo "GATE FAIL: build error --"; cat $(GATE_LOG); rm -f $(GATE_LOG); exit 1; }
	@rm -f CUDACyclone-ptxinfo
	@echo "== resource report =="
	@grep -E "Compiling entry|Function properties|bytes stack frame|Used [0-9]+ registers" $(GATE_LOG) || true
	@echo "== spill check =="
	@if grep "bytes spill" $(GATE_LOG) | grep -qv "0 bytes spill stores, 0 bytes spill loads"; then \
	   echo "GATE FAIL: spill detected --"; \
	   grep "bytes spill" $(GATE_LOG) | grep -v "0 bytes spill stores, 0 bytes spill loads"; \
	   rm -f $(GATE_LOG); exit 1; \
	 fi
	@echo "  ok: 0 spill stores / 0 spill loads in every reported function"
	@echo "== register check ($(HOT_KERNEL) <= $(REG_CEILING)) =="
	@regs=$$(awk '/Compiling entry function/{h=index($$0,"$(HOT_KERNEL)")>0} h&&/Used [0-9]+ registers/{if(match($$0,/Used [0-9]+/)){print substr($$0,RSTART+5,RLENGTH-5);exit}}' $(GATE_LOG)); \
	 rm -f $(GATE_LOG); \
	 if [ -z "$$regs" ]; then echo "GATE FAIL: $(HOT_KERNEL) not found in ptxas output"; exit 1; fi; \
	 if [ "$$regs" -gt "$(REG_CEILING)" ]; then \
	   echo "GATE FAIL: $(HOT_KERNEL) uses $$regs registers, ceiling is $(REG_CEILING)"; exit 1; \
	 fi; \
	 echo "  ok: $(HOT_KERNEL) uses $$regs registers (ceiling $(REG_CEILING))"
	@echo "GATE PASS"

# ---- Field-math equivalence gate (host-only: no GPU, no nvcc) -------------------------
# Proves the split-column cores in field_split.cuh are BIT-IDENTICAL to the RCKangaroo
# cores they replace, by compiling the shipped function bodies against a host emulation of
# the PTX carry primitives. Runs anywhere a C++ compiler exists, so it gates the port
# before a GPU box is ever rented -- and unlike proof.py it isolates the field math, so a
# failure points straight at the multiply rather than at the whole pipeline.
#
# Bit-equality (not just congruence) is the right bar here: the two paths share a
# byte-identical reduction tail, so equal 512-bit products mean the split path emits the
# same lazy representative and nothing downstream can observe the swap.
PYTHON          ?= python3
HOSTCXX         ?= g++
FIELDTEST_ITERS ?= 2000000

fieldtest: tests/field_split_equiv.cpp tests/field_split_include.cpp field_split.cuh \
           tests/ptx_host.h tests/extract_field.py third_party/RCKangaroo/RCGpuUtils.h
	$(PYTHON) tests/extract_field.py
	$(HOSTCXX) -O2 -fno-strict-aliasing -I tests -o tests/field_split_include tests/field_split_include.cpp
	$(HOSTCXX) -O2 -fno-strict-aliasing -I tests -o tests/field_split_equiv   tests/field_split_equiv.cpp
	./tests/field_split_include
	./tests/field_split_equiv $(FIELDTEST_ITERS)

# Per-kernel resource usage read back from the built fat binary (authoritative).
resusage: $(TARGET)
	cuobjdump -res-usage $(TARGET)

# Full SASS disassembly of the built fat binary.
sass: $(TARGET)
	cuobjdump -sass $(TARGET)

clean:
	rm -f $(TARGET) CUDACyclone-ptxinfo $(GATE_LOG) $(OBJ)
	rm -f tests/field_split_equiv tests/field_split_include
	rm -f tests/field_split_equiv.exe tests/field_split_include.exe
