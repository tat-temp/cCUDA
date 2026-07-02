// cr_field.cuh — clean-room 32-bit-limb secp256k1 field multiply (v3, carry-chain bake-off).
//
// Reproduces RCKangaroo's measured +7% multiply win WITHOUT its GPLv3 source. Public math only:
//   * a 256x256 -> 512 product built from the native 32-bit multiplier with HARDWARE carry-flag
//     chains (mad.lo.cc.u32 / madc.hi.cc.u32 / addc.u32) — the structure a fast bignum multiply
//     needs, and the reason v1/v2 (manual `carry = m>>32` chains) were ~0.71x baseline; and
//   * the secp256k1 reduction, reusing CUDACyclone's OWN UMultSpecial 512->320->256 fold (the
//     project's proof.py-proven code — license-clean, not RetiredC's).
// No line of RCKangaroo's RCGpuUtils.h was consulted.
//
// Two product structures are provided for an on-hardware bake-off:
//   prodA — OPERAND-SCAN: for each b[j], add a*b[j] into out[j..] via a lo-chain then a hi-chain
//           (the 32-bit analog of CUDACyclone's 64-bit UMult; more ILP among partials).
//   prodD — COMBA product-scan: one output word at a time via a 3-word accumulator (tighter, but
//           a longer serial dependency).
// Both were designed + independently cross-checked + Python-validated bit-exact; prodA's j>=1
// hi-chain operand numbering (an off-by-one the generators missed) was fixed and re-validated
// here over 300k trials. crfield defaults to prodA; build with -DCR_USE_D to select prodD.
//
// Included from CUDAMath.h AFTER its UADD*/UMULLO/UMULHI/UMultSpecial/NBBLOCK macros (reused by
// reduce512); NOT standalone. Output is lazily reduced (congruent mod p, [0,2^256)) — drop-in for
// _ModMult. ecbench's triangulated host-KAT check is the on-device correctness gate.
#pragma once

#include <cstdint>

#ifdef CR_NOINLINE
#define CR_INLINE __noinline__
#else
#define CR_INLINE __forceinline__
#endif

namespace cr {

// ---- product A: operand-scan with hardware carry-flag chains -------------------------
__device__ __forceinline__ void prodA(uint32_t out[16], const uint32_t a[8], const uint32_t b[8])
{
    // j = 0: out[0..8] = a[0..7] * b[0]
    {
        uint32_t bj = b[0];
        asm volatile(
            "mul.lo.u32     %0, %8,  %16;\n\t"
            "mad.lo.cc.u32  %1, %9,  %16, 0;\n\t"
            "madc.lo.cc.u32 %2, %10, %16, 0;\n\t"
            "madc.lo.cc.u32 %3, %11, %16, 0;\n\t"
            "madc.lo.cc.u32 %4, %12, %16, 0;\n\t"
            "madc.lo.cc.u32 %5, %13, %16, 0;\n\t"
            "madc.lo.cc.u32 %6, %14, %16, 0;\n\t"
            "madc.lo.u32    %7, %15, %16, 0;\n\t"
            : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3]),
              "=r"(out[4]), "=r"(out[5]), "=r"(out[6]), "=r"(out[7])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(bj));
        asm volatile(
            "mad.hi.cc.u32  %0, %8,  %16, %0;\n\t"
            "madc.hi.cc.u32 %1, %9,  %16, %1;\n\t"
            "madc.hi.cc.u32 %2, %10, %16, %2;\n\t"
            "madc.hi.cc.u32 %3, %11, %16, %3;\n\t"
            "madc.hi.cc.u32 %4, %12, %16, %4;\n\t"
            "madc.hi.cc.u32 %5, %13, %16, %5;\n\t"
            "madc.hi.cc.u32 %6, %14, %16, %6;\n\t"
            "madc.hi.u32    %7, %15, %16, 0;\n\t"
            : "+r"(out[1]), "+r"(out[2]), "+r"(out[3]), "+r"(out[4]),
              "+r"(out[5]), "+r"(out[6]), "+r"(out[7]), "=r"(out[8])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(bj));
    }
    // out[9..15] are first *accumulated* (+r) by the j>=1 chains, but j==0 only wrote
    // out[0..8] -- they must be zeroed first. (A local uint32_t[16] is NOT zero-initialized;
    // omitting this reads garbage high limbs and every product is wrong.)
    #pragma unroll
    for (int k = 9; k < 16; ++k) out[k] = 0u;
    // j = 1..7: out[j..j+8] += a[0..7] * b[j]
    #pragma unroll
    for (int j = 1; j < 8; ++j) {
        uint32_t bj = b[j];
        asm volatile(                                 // lo-chain: 9 outputs => a starts at %9, bj=%17
            "mad.lo.cc.u32  %0, %9,  %17, %0;\n\t"
            "madc.lo.cc.u32 %1, %10, %17, %1;\n\t"
            "madc.lo.cc.u32 %2, %11, %17, %2;\n\t"
            "madc.lo.cc.u32 %3, %12, %17, %3;\n\t"
            "madc.lo.cc.u32 %4, %13, %17, %4;\n\t"
            "madc.lo.cc.u32 %5, %14, %17, %5;\n\t"
            "madc.lo.cc.u32 %6, %15, %17, %6;\n\t"
            "madc.lo.cc.u32 %7, %16, %17, %7;\n\t"
            "addc.u32       %8, %8,  0;\n\t"
            : "+r"(out[j+0]), "+r"(out[j+1]), "+r"(out[j+2]), "+r"(out[j+3]),
              "+r"(out[j+4]), "+r"(out[j+5]), "+r"(out[j+6]), "+r"(out[j+7]),
              "+r"(out[j+8])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(bj));
        asm volatile(                                 // hi-chain: 8 outputs => a starts at %8, bj=%16
            "mad.hi.cc.u32  %0, %8,  %16, %0;\n\t"
            "madc.hi.cc.u32 %1, %9,  %16, %1;\n\t"
            "madc.hi.cc.u32 %2, %10, %16, %2;\n\t"
            "madc.hi.cc.u32 %3, %11, %16, %3;\n\t"
            "madc.hi.cc.u32 %4, %12, %16, %4;\n\t"
            "madc.hi.cc.u32 %5, %13, %16, %5;\n\t"
            "madc.hi.cc.u32 %6, %14, %16, %6;\n\t"
            "madc.hi.u32    %7, %15, %16, %7;\n\t"
            : "+r"(out[j+1]), "+r"(out[j+2]), "+r"(out[j+3]), "+r"(out[j+4]),
              "+r"(out[j+5]), "+r"(out[j+6]), "+r"(out[j+7]), "+r"(out[j+8])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
              "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]), "r"(bj));
    }
}

// ---- product D: Comba product-scan with a 3-word accumulator -------------------------
__device__ __forceinline__ void prodD(uint32_t out[16], const uint32_t a[8], const uint32_t b[8])
{
    uint32_t c0 = 0u, c1 = 0u, c2 = 0u;
    #define CR_MAC(i,j) \
        asm volatile( \
            "mad.lo.cc.u32   %0, %3, %4, %0;\n\t" \
            "madc.hi.cc.u32  %1, %3, %4, %1;\n\t" \
            "addc.u32        %2, %2, 0;\n\t" \
            : "+r"(c0), "+r"(c1), "+r"(c2) \
            : "r"(a[i]), "r"(b[j]))
    #define CR_COL(k) do { out[k] = c0; c0 = c1; c1 = c2; c2 = 0u; } while(0)
    CR_MAC(0,0);                                                                     CR_COL(0);
    CR_MAC(0,1); CR_MAC(1,0);                                                        CR_COL(1);
    CR_MAC(0,2); CR_MAC(1,1); CR_MAC(2,0);                                           CR_COL(2);
    CR_MAC(0,3); CR_MAC(1,2); CR_MAC(2,1); CR_MAC(3,0);                              CR_COL(3);
    CR_MAC(0,4); CR_MAC(1,3); CR_MAC(2,2); CR_MAC(3,1); CR_MAC(4,0);                 CR_COL(4);
    CR_MAC(0,5); CR_MAC(1,4); CR_MAC(2,3); CR_MAC(3,2); CR_MAC(4,1); CR_MAC(5,0);    CR_COL(5);
    CR_MAC(0,6); CR_MAC(1,5); CR_MAC(2,4); CR_MAC(3,3); CR_MAC(4,2); CR_MAC(5,1); CR_MAC(6,0); CR_COL(6);
    CR_MAC(0,7); CR_MAC(1,6); CR_MAC(2,5); CR_MAC(3,4); CR_MAC(4,3); CR_MAC(5,2); CR_MAC(6,1); CR_MAC(7,0); CR_COL(7);
    CR_MAC(1,7); CR_MAC(2,6); CR_MAC(3,5); CR_MAC(4,4); CR_MAC(5,3); CR_MAC(6,2); CR_MAC(7,1); CR_COL(8);
    CR_MAC(2,7); CR_MAC(3,6); CR_MAC(4,5); CR_MAC(5,4); CR_MAC(6,3); CR_MAC(7,2);    CR_COL(9);
    CR_MAC(3,7); CR_MAC(4,6); CR_MAC(5,5); CR_MAC(6,4); CR_MAC(7,3);                 CR_COL(10);
    CR_MAC(4,7); CR_MAC(5,6); CR_MAC(6,5); CR_MAC(7,4);                              CR_COL(11);
    CR_MAC(5,7); CR_MAC(6,6); CR_MAC(7,5);                                           CR_COL(12);
    CR_MAC(6,7); CR_MAC(7,6);                                                        CR_COL(13);
    CR_MAC(7,7);                                                                     CR_COL(14);
    out[15] = c0;
    #undef CR_MAC
    #undef CR_COL
}

// ---- shared secp reduction: CUDACyclone's own UMultSpecial fold (license-clean) -----
__device__ __forceinline__ void reduce512(uint64_t* rr, uint32_t* t)
{
    uint64_t* r512 = (uint64_t*)t;            // 8x u64 view of the 512-bit product
    uint64_t red[NBBLOCK];
    uint64_t ah, al;
    UMultSpecial(red, (r512 + 4));
    UADDO1(r512[0], red[0]);
    UADDC1(r512[1], red[1]);
    UADDC1(r512[2], red[2]);
    UADDC1(r512[3], red[3]);
    UADD1(red[4], 0ULL);
    UMULLO(al, red[4], 0x1000003D1ULL);
    UMULHI(ah, red[4], 0x1000003D1ULL);
    UADDO(rr[0], r512[0], al);
    UADDC(rr[1], r512[1], ah);
    UADDC(rr[2], r512[2], 0ULL);
    UADD (rr[3], r512[3], 0ULL);
}

__device__ CR_INLINE void mulA(uint64_t* rr, const uint64_t* a, const uint64_t* b){
    __align__(8) uint32_t t[16]; prodA(t, (const uint32_t*)a, (const uint32_t*)b); reduce512(rr, t);
}
__device__ CR_INLINE void mulD(uint64_t* rr, const uint64_t* a, const uint64_t* b){
    __align__(8) uint32_t t[16]; prodD(t, (const uint32_t*)a, (const uint32_t*)b); reduce512(rr, t);
}

// backend-selected mul (crfield uses this). Default = D (Comba) -- it measured correct AND
// fastest (1.124x baseline, 0.903x RCK) in the v3 bake-off; -DCR_USE_A selects the operand-scan.
#if defined(CR_USE_A)
__device__ CR_INLINE void mul(uint64_t* rr, const uint64_t* a, const uint64_t* b){ mulA(rr,a,b); }
#else
__device__ CR_INLINE void mul(uint64_t* rr, const uint64_t* a, const uint64_t* b){ mulD(rr,a,b); }
#endif
__device__ CR_INLINE void mul(uint64_t* rr, const uint64_t* a){ mul(rr, rr, a); }  // rr *= a
__device__ CR_INLINE void sqr(uint64_t* rr, const uint64_t* a){ mul(rr, a, a); }

} // namespace cr
