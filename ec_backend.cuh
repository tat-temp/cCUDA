// ec_backend.cuh — RCKangaroo field-arithmetic backend shim (A/B experiment).
//
// Purpose: expose RetiredCoder's secp256k1 field primitives (MulModP / SqrModP /
// InvModP / SubModP / AddModP / NegModP, from third_party/RCKangaroo/RCGpuUtils.h)
// under a private namespace so they can be A/B-benchmarked and drop-in-swapped
// against CUDACyclone's own JeanLucPons-lineage ops (_ModMult / _ModSqr / _ModInv
// / ModSub256 ... in CUDAMath.h).
//
// The vendored header is GPLv3 (c) 2024 RetiredCoder. Pulling it into a build has
// license implications for any distributed binary — this file exists for local
// benchmarking on the experiment branch only.
//
// Representation contract (verified against both libraries):
//   * 256-bit values are 4x uint64_t, little-endian (limb[0] = least significant).
//   * MulModP/SqrModP take arbitrary [0,2^256) inputs and return a value congruent
//     mod P in [0,2^256) — NOT necessarily canonical (< P). This is the SAME "lazy"
//     convention CUDAMath.h's _ModMult/_ModSqr use, so the two are interchangeable
//     AS RESIDUES mod P at every internal step. They are NOT guaranteed to pick the
//     same non-canonical representative: for a value whose canonical form is < ~2^32
//     (a ~2^-224 fraction) one backend may emit v and the other v+P. This is harmless
//     — the baseline shares the identical ~2^-224 non-canonical tail at the raw-limb
//     hash boundary — but it means the two binaries are byte-identical only up to that
//     tail. -DRCK_CANON forces canonical [0,P) if exact byte-parity is ever required.
//   * In-place aliasing res==a and/or res==b is safe for MulModP/SqrModP: both read
//     all inputs into locals before writing res. (RCKangaroo's own KernelA relies on
//     MulModP(inverse, inverse, tmp).)
//   * InvModP((u32*)v) inverts a 256-bit value held in v[0..7] (u32 view of a
//     uint64_t[4]); it touches up to v[8] (9th u32 word), so callers must back it
//     with at least a uint64_t[5] — exactly like CUDACyclone's _ModInv. Output is
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
