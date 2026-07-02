// cr_field.cuh — clean-room 32-bit-limb secp256k1 field multiply (v2).
//
// Reproduces RCKangaroo's measured +7% multiply win WITHOUT its GPLv3 source, so the
// result is mergeable under CUDACyclone's own license. Two ingredients, both license-clean:
//   * the 256x256 -> 512 product is computed with the NATIVE 32x32->64 multiplier
//     (`mul.wide.u32`, emitted explicitly so ptxas cannot fall back to a 64-bit mul.lo),
//     from the standard operand-scanning schoolbook (public algorithm); and
//   * the secp256k1 reduction reuses CUDACyclone's OWN `UMultSpecial` fold (defined in
//     CUDAMath.h) — the project's existing code, not RetiredC's.
// No line of RCKangaroo's RCGpuUtils.h was consulted.
//
// v1 note: the first cut reduced with __uint128_t (which pulls in 64-bit mul.lo/hi.u64)
// and measured cr/cyc 0.73x — SLOWER than the 64-bit baseline. v2 removes all 64-bit
// multiplies: the product is guaranteed 32-bit-wide and the reduction is the baseline's.
//
// This header is included from CUDAMath.h AFTER its UADD*/UMULLO/UMULHI/UMultSpecial/NBBLOCK
// macros are defined, and reuses them; it is therefore NOT standalone (unlike ec_backend.cuh).
// Output is lazily reduced (congruent mod p, in [0, 2^256)) — identical convention to
// _ModMult, so it is a drop-in at every call site. Product correctness is trivial (schoolbook
// = a*b); the reduction is the baseline's, already proven by proof.py 848/848.
#pragma once

#include <cstdint>

#ifdef CR_NOINLINE
#define CR_INLINE __noinline__
#else
#define CR_INLINE __forceinline__
#endif

namespace cr {

// native 32x32 -> 64 multiply (the whole point: one hardware multiplier op, not a
// synthesized 64-bit mul). Pure/non-volatile so ptxas can schedule/fuse freely.
__device__ __forceinline__ uint64_t mw(uint32_t a, uint32_t b){
    uint64_t r; asm("mul.wide.u32 %0, %1, %2;" : "=l"(r) : "r"(a), "r"(b)); return r;
}

// rr[0..3] = a[0..3] * b[0..3] (mod p, lazy). Safe when rr aliases a and/or b: both
// operands are fully consumed into the product before rr is written.
__device__ CR_INLINE void mul(uint64_t* rr, const uint64_t* a, const uint64_t* b){
    const uint32_t* A = (const uint32_t*)a;   // 8 LE 32-bit limbs, indexed in place
    const uint32_t* B = (const uint32_t*)b;

    // ---- 256x256 -> 512, operand-scanning, native mul.wide.u32 ----
    __align__(8) uint32_t t[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) t[k] = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t carry = 0u;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t m = mw(A[i], B[j]) + (uint64_t)t[i + j] + (uint64_t)carry;
            t[i + j] = (uint32_t)m;
            carry    = (uint32_t)(m >> 32);
        }
        t[i + 8] = carry;
    }

    // ---- secp reduction: reuse CUDACyclone's own 512->320->256 fold (license-clean) ----
    uint64_t* r512 = (uint64_t*)t;            // 8x u64 view of the 512-bit product
    uint64_t red[NBBLOCK];
    uint64_t ah, al;
    UMultSpecial(red, (r512 + 4));            // red = high256 * 0x1000003D1  (up to 320-bit)
    UADDO1(r512[0], red[0]);
    UADDC1(r512[1], red[1]);
    UADDC1(r512[2], red[2]);
    UADDC1(r512[3], red[3]);
    UADD1(red[4], 0ULL);                      // capture the carry out of the low add
    UMULLO(al, red[4], 0x1000003D1ULL);
    UMULHI(ah, red[4], 0x1000003D1ULL);
    UADDO(rr[0], r512[0], al);
    UADDC(rr[1], r512[1], ah);
    UADDC(rr[2], r512[2], 0ULL);
    UADD (rr[3], r512[3], 0ULL);
}

// 2-arg in-place form: rr = rr * a  (mod p). Aliasing-safe (see mul above).
__device__ CR_INLINE void mul(uint64_t* rr, const uint64_t* a) { mul(rr, rr, a); }

// rr = a^2 (mod p, lazy). Squaring is ~2% in isolation and not the lever, so it reuses
// the general multiply rather than a dedicated (fewer-partial-products) squaring.
__device__ CR_INLINE void sqr(uint64_t* rr, const uint64_t* a) { mul(rr, a, a); }

} // namespace cr
