TARGET      := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
CC          := nvcc

GPU_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
SM_ARCHS   := 75 86 89 $(GPU_ARCH)
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17

LDFLAGS    := -lcudadevrt -cudart=static

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

# EC-generation-only benchmark: the identical kernel built with -DEC_GEN_ONLY, which
# skips SHA-256/RIPEMD-160 + the address match (a tiny XOR sink keeps the EC math live).
# Use `make clean && make ecgen` after header edits (Makefile doesn't track header deps).
ecgen: CUDACyclone-ecgen
CUDACyclone-ecgen: $(SRC)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -DEC_GEN_ONLY $(SRC) -o $@ $(LDFLAGS)

# SHA-only benchmark: same kernel that hashes with SHA-256 but skips RIPEMD-160 (-DSHA_ONLY),
# to split the hashing cost. Compare full vs shaonly vs ecgen.
shaonly: CUDACyclone-shaonly
CUDACyclone-shaonly: $(SRC)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -DSHA_ONLY $(SRC) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) CUDACyclone-ecgen CUDACyclone-shaonly $(OBJ)

