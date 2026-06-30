#ifndef CUDA_HASH_CUH
#define CUDA_HASH_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <cstring>

__device__ void getHash160_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint32_t out5[5]);
#endif
