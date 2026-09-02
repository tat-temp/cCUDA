// ec_backend.cuh — secp256k1 field backend.
//
// Product cores are the split-column fused-MAC ones in field_split.cuh; the reduction tail and
// InvModP are RetiredCoder's (third_party/RCKangaroo/RCGpuUtils.h, GPLv3), so a distributed
// binary is a GPLv3 derivative — see LICENSE.
//
// Representation contract:
//   * 256-bit values are 4x uint64_t, little-endian (limb[0] least significant).
//   * rmul/rsqr take inputs in [0,P) and return a value congruent mod P in [0,2^256) — "lazy",
//     not necessarily canonical. Harmless here: a non-canonical X would need a canonical form
//     below ~2^32 (~2^-224 of the space), which no real pubkey hits. Inputs at or above P are
//     outside the contract: the single-fold tail can leave a non-congruent residue.
//   * In-place aliasing (res==a and/or res==b) is safe: all inputs are read into locals first.
//   * rinv inverts a 256-bit value in place and touches a 9th u32 word, so callers must back it
//     with at least uint64_t[5] (as CUDAMath.h's fieldInv does). Output is canonical [0,P).
#pragma once

#include <cstdint>

namespace rck {

typedef unsigned long long u64;
typedef unsigned int       u32;
typedef int                i32;

// The PTX-asm helper macros and P_* constants defined inside these headers are file-global
// rather than namespaced, but do not collide with CUDAMath.h's (UADDO/MADDC/...).
#include "third_party/RCKangaroo/RCGpuUtils.h"
#include "field_split.cuh"

// r = a * b (mod P)
__device__ __forceinline__ void rmul(uint64_t* r, const uint64_t* a, const uint64_t* b)
{
    MulModP_split((u64*)r, (u64*)a, (u64*)b);
}

// r = r * a (mod P)  (2-arg form used by CUDACyclone's `_ModMult(inverse, subp[0])`)
__device__ __forceinline__ void rmul(uint64_t* r, const uint64_t* a)
{
    MulModP_split((u64*)r, (u64*)r, (u64*)a);
}

// r = a^2 (mod P)
__device__ __forceinline__ void rsqr(uint64_t* r, const uint64_t* a)
{
    SqrModP_split((u64*)r, (u64*)a);
}

// r = a^-1 (mod P), in place
__device__ __forceinline__ void rinv(uint64_t* r)
{
    InvModP((u32*)r);
}

} // namespace rck
