// cr_field.cuh — clean-room 32-bit-limb secp256k1 field multiply.
//
// Purpose: reproduce the measured +7% end-to-end win from RCKangaroo's 32-bit MulModP
// WITHOUT using its GPLv3 source, so the result is mergeable under CUDACyclone's own
// license. This file is written from public mathematics only:
//   * schoolbook 256x256 -> 512 multiplication via 32x32->64 (`mul.wide.u32`), and
//   * the secp256k1 pseudo-Mersenne reduction: p = 2^256 - 2^32 - 977, hence
//     2^256 == C (mod p) with C = 0x1000003D1, so a 512-bit product H*2^256 + L
//     reduces as L + H*C, folded twice plus a final carry.
// No line of RetiredCoder's RCGpuUtils.h was consulted for the implementation; the
// exact u64-limb carry logic below was validated bit-exact against a Python reference
// over ~1.5M random/edge/square trials before being written.
//
// Design notes vs. the earlier failed Phase-3 32-bit attempt (which spilled -9%/-16%):
//   * operands are NOT split into separate u32[8] arrays — they are indexed in place
//     via `(const uint32_t*)a`, so the multiply's live set is just the 16-word product
//     accumulator, not product + two operand copies (the extra 16 regs that likely
//     pushed Phase 3 past the 128-reg (256,2) occupancy cap and forced spills).
//   * output is lazily reduced (congruent mod p, in [0, 2^256)), matching CUDACyclone's
//     `_ModMult` convention, so it is a drop-in at every call site.
//
// If the register check on hardware still shows spills, build with -DCR_NOINLINE to move
// the multiply's working set out of the kernel's live register count (a __noinline__
// device call keeps its regs off the caller — the same mechanism that protects
// getHash160_33_from_limbs). That trades call overhead for occupancy.
#pragma once

#include <cstdint>

#ifdef CR_NOINLINE
#define CR_INLINE __noinline__
#else
#define CR_INLINE __forceinline__
#endif

namespace cr {

// rr[0..3] = a[0..3] * b[0..3]  (mod p, lazy). Safe when rr aliases a and/or b: both
// operands are fully consumed into the product before rr is written.
__device__ CR_INLINE void mul(uint64_t* rr, const uint64_t* a, const uint64_t* b)
{
    const uint32_t* A = (const uint32_t*)a;   // 8 LE 32-bit limbs, indexed in place
    const uint32_t* B = (const uint32_t*)b;

    // ---- 256x256 -> 512, operand-scanning; (uint64_t)A[i]*B[j] lowers to mul.wide.u32 ----
    __align__(8) uint32_t t[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) t[k] = 0u;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t carry = 0u;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint64_t m = (uint64_t)A[i] * (uint64_t)B[j] + (uint64_t)t[i + j] + (uint64_t)carry;
            t[i + j] = (uint32_t)m;
            carry    = (uint32_t)(m >> 32);
        }
        t[i + 8] = carry;
    }
    const uint64_t* P = (const uint64_t*)t;    // 8x u64 view of the 512-bit product
    const uint64_t  C = 0x1000003D1ULL;         // 2^256 mod p

    // ---- fold #1: hc = P[hi] * C   (256 x 64 -> up to 320-bit) ----
    uint64_t hc0, hc1, hc2, hc3, hc4;
    { __uint128_t cur = 0, tt;
      tt = (__uint128_t)P[4] * C + cur; hc0 = (uint64_t)tt; cur = tt >> 64;
      tt = (__uint128_t)P[5] * C + cur; hc1 = (uint64_t)tt; cur = tt >> 64;
      tt = (__uint128_t)P[6] * C + cur; hc2 = (uint64_t)tt; cur = tt >> 64;
      tt = (__uint128_t)P[7] * C + cur; hc3 = (uint64_t)tt; cur = tt >> 64;
      hc4 = (uint64_t)cur; }

    // ---- add to low: acc[0..3], acc4 = P[lo] + hc ----
    uint64_t a0, a1, a2, a3, a4;
    { __uint128_t cur = 0, tt;
      tt = (__uint128_t)P[0] + hc0 + cur; a0 = (uint64_t)tt; cur = tt >> 64;
      tt = (__uint128_t)P[1] + hc1 + cur; a1 = (uint64_t)tt; cur = tt >> 64;
      tt = (__uint128_t)P[2] + hc2 + cur; a2 = (uint64_t)tt; cur = tt >> 64;
      tt = (__uint128_t)P[3] + hc3 + cur; a3 = (uint64_t)tt; cur = tt >> 64;
      a4 = hc4 + (uint64_t)cur; }             // acc4 < ~2^34

    // ---- fold #2: acc4 * C  (< 2^67) into acc[0..3], then a final carry fold ----
    __uint128_t mc = (__uint128_t)a4 * C;
    uint64_t mc0 = (uint64_t)mc, mc1 = (uint64_t)(mc >> 64);
    uint64_t r0, r1, r2, r3; __uint128_t cur = 0, tt;
    tt = (__uint128_t)a0 + mc0;       r0 = (uint64_t)tt; cur = tt >> 64;
    tt = (__uint128_t)a1 + mc1 + cur; r1 = (uint64_t)tt; cur = tt >> 64;
    tt = (__uint128_t)a2 + cur;       r2 = (uint64_t)tt; cur = tt >> 64;
    tt = (__uint128_t)a3 + cur;       r3 = (uint64_t)tt; cur = tt >> 64;
    if ((uint64_t)cur) {              // bit-256 carry -> += C (rarely carries once more)
        tt = (__uint128_t)r0 + C;   r0 = (uint64_t)tt; cur = tt >> 64;
        tt = (__uint128_t)r1 + cur; r1 = (uint64_t)tt; cur = tt >> 64;
        tt = (__uint128_t)r2 + cur; r2 = (uint64_t)tt; cur = tt >> 64;
        tt = (__uint128_t)r3 + cur; r3 = (uint64_t)tt; cur = tt >> 64;
        if ((uint64_t)cur) r0 += C;   // r0 was tiny here; cannot carry further
    }
    rr[0] = r0; rr[1] = r1; rr[2] = r2; rr[3] = r3;
}

// 2-arg in-place form: rr = rr * a  (mod p). Aliasing-safe (see mul above).
__device__ CR_INLINE void mul(uint64_t* rr, const uint64_t* a) { mul(rr, rr, a); }

// rr = a^2 (mod p, lazy). Squaring is ~2% in isolation and not the lever, so it reuses
// the general multiply rather than a dedicated (fewer-partial-products) squaring.
__device__ CR_INLINE void sqr(uint64_t* rr, const uint64_t* a) { mul(rr, a, a); }

} // namespace cr
