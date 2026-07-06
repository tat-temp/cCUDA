// ec_backend.cuh — RCKangaroo field-arithmetic backend shim.
//
// Purpose: expose RetiredCoder's secp256k1 field primitives (MulModP / SqrModP /
// InvModP / SubModP / AddModP / NegModP, from third_party/RCKangaroo/RCGpuUtils.h)
// under a private namespace, wired to CUDACyclone's _ModMult / _ModSqr / _ModInv
// call sites in CUDAMath.h. This is the sole field backend (the older
// JeanLucPons-lineage ops it was originally A/B'd against have been removed).
//
// The vendored header is GPLv3 (c) 2024 RetiredCoder, so any distributed binary is
// a GPLv3 derivative — see LICENSE.
//
// Representation contract (verified against both libraries):
//   * 256-bit values are 4x uint64_t, little-endian (limb[0] = least significant).
//   * MulModP/SqrModP take arbitrary [0,2^256) inputs and return a value congruent
//     mod P in [0,2^256) — NOT necessarily canonical (< P). This is a "lazy" convention:
//     values stay congruent mod P at every internal step. The emitted representative may
//     be v or v+P only for a value whose canonical form is < ~2^32 (a ~2^-224 fraction),
//     since v+P must still fit in 256 bits. Harmless here: the hash is taken on the
//     canonical X for every real key (a non-canonical X would need canonical form < 2^32,
//     ~2^-224, which no real pubkey hits). -DRCK_CANON forces canonical [0,P) if exact
//     byte-parity is ever required.
//   * In-place aliasing res==a and/or res==b is safe for MulModP/SqrModP: both read
//     all inputs into locals before writing res. (RCKangaroo's own KernelA relies on
//     MulModP(inverse, inverse, tmp).)
//   * InvModP((u32*)v) inverts a 256-bit value held in v[0..7] (u32 view of a
//     uint64_t[4]); it touches up to v[8] (9th u32 word), so callers must back it
//     with at least a uint64_t[5] (as CUDACyclone's fieldInv does). Output is
//     canonical [0,P). Input may be lazy (>=P, <2^256), as in KernelA.
#pragma once

#include <cstdint>

namespace rck {

// RCGpuUtils.h needs only these typedefs from RCKangaroo's defs.h; everything else
// it uses (PTX add/mul macros, P_0/P_123/P_INV32, CUDA intrinsics) it defines itself.
typedef unsigned long long u64;
typedef long long          i64;
typedef unsigned int       u32;
typedef int                i32;
typedef unsigned short     u16;
typedef short              i16;
typedef unsigned char      u8;
typedef char               i8;

// NOTE: the PTX-asm helper macros (add_cc_64, mul_wide_32, ...) and the P_* constants
// defined inside this header are preprocessor macros — they are file-global, not
// namespaced — but their names do not collide with CUDAMath.h's (UADDO/MADDC/...).
#include "third_party/RCKangaroo/RCGpuUtils.h"

// Fully reduce a lazily-reduced value r in [0,2^256) to canonical [0,P). Because
// r < 2^256 < 2P, a single conditional subtract of P suffices.
__device__ __forceinline__ void field_canon(u64* r)
{
    u64 t0, t1, t2, t3, br;
    sub_cc_64 (t0, r[0], P_0);
    subc_cc_64(t1, r[1], P_123);
    subc_cc_64(t2, r[2], P_123);
    subc_cc_64(t3, r[3], P_123);
    subc_64   (br, 0ull, 0ull);      // br == 0  <=>  no borrow  <=>  r >= P
    if (br == 0ull) { r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3; }
}

// ---- Uniform wrappers matching CUDACyclone call conventions --------------------

// r = a * b (mod P)
__device__ __forceinline__ void rmul(uint64_t* r, const uint64_t* a, const uint64_t* b)
{
    MulModP((u64*)r, (u64*)a, (u64*)b);
#ifdef RCK_CANON
    field_canon((u64*)r);
#endif
}

// r = r * a (mod P)  (2-arg form used by CUDACyclone's `_ModMult(inverse, subp[0])`)
__device__ __forceinline__ void rmul(uint64_t* r, const uint64_t* a)
{
    MulModP((u64*)r, (u64*)r, (u64*)a);
#ifdef RCK_CANON
    field_canon((u64*)r);
#endif
}

// r = a^2 (mod P)
__device__ __forceinline__ void rsqr(uint64_t* r, const uint64_t* a)
{
    SqrModP((u64*)r, (u64*)a);
#ifdef RCK_CANON
    field_canon((u64*)r);
#endif
}

// r = a^-1 (mod P), in place. r must be backed by uint64_t[5] (InvModP writes word 8).
__device__ __forceinline__ void rinv(uint64_t* r)
{
    InvModP((u32*)r);
}

// r = a - b (mod P)
__device__ __forceinline__ void rsub(uint64_t* r, const uint64_t* a, const uint64_t* b)
{
    SubModP((u64*)r, (u64*)a, (u64*)b);
}

// r = a + b (mod P)
__device__ __forceinline__ void radd(uint64_t* r, const uint64_t* a, const uint64_t* b)
{
    AddModP((u64*)r, (u64*)a, (u64*)b);
}

// r = -r (mod P), in place
__device__ __forceinline__ void rneg(uint64_t* r)
{
    NegModP((u64*)r);
}

} // namespace rck
