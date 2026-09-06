#include "CUDAHash.cuh"
#include <cstdio>
#include <cstdint>
#include <stdint.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstring>

__device__ __forceinline__ uint32_t ror32(uint32_t x, int n)
{
#if __CUDA_ARCH__ >= 350
    return __funnelshift_r(x, x, n);
#else
    return (x >> n) | (x << (32 - n));
#endif
}

// a ^ b ^ c. Left to ptxas as plain C: it folds this to one LOP3 (0x96) AND is free to
// fuse it across op boundaries (the RIPEMD-160 f-function audit proved forced lop3.b32
// asm BLOCKS that fusion, costing +2 LOP3). Inverse-test: does unpinning SHA help too?
__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c){
    return a ^ b ^ c;
}

// Sigma0/Sigma1 with the common rotate hoisted OUT of the xor. Rotation is GF(2)-linear, so
//     ror(a,k) ^ ror(b,k) ^ ror(c,k)  ==  ror(a ^ b ^ c, k)
// and with k=2 / k=6 the inner amounts collapse to 13-2=11, 22-2=20 / 11-6=5, 25-6=19. This is
// an identity, not an approximation: agreement on the 32 basis vectors is a COMPLETE proof over
// all 2^32 inputs (host-checked, plus 500k random values and the usual edge set, 0 mismatches).
//
// WHY: instruction COUNT is unchanged (3 SHF + 1 LOP3 either way). The bet is placement -- the
// surviving rotate now sits at the OUTSIDE of the expression, adjacent to the T1/T2 accumulation
// in SHA_RND, where ptxas can fold it into the add as LEA.HI:
//     {3 SHF, 1 LOP3, 1 IADD} -> {2 SHF, 1 LOP3, 1 LEA}   x 126 live sites
// It only pays if LEA is NOT on the ~1/3-rate logic/shift unit that binds this kernel; two
// independent reviews argued it is. SCREEN BEFORE BENCHING: `make sass | grep -c SHF` must drop
// by ~126. If it does not, the fusion did not fire -- revert, spend no bench slot.
__device__ __forceinline__ uint32_t bigS0(uint32_t x) { return ror32(xor3(x, ror32(x, 11), ror32(x, 20)), 2); }
__device__ __forceinline__ uint32_t bigS1(uint32_t x) { return ror32(xor3(x, ror32(x,  5), ror32(x, 19)), 6); }
__device__ __forceinline__ uint32_t smallS0(uint32_t x){ return xor3(ror32(x, 7), ror32(x, 18), (x >> 3)); }
__device__ __forceinline__ uint32_t smallS1(uint32_t x){ return xor3(ror32(x,17), ror32(x, 19), (x >>10)); }

// Ch = (x&y)^(~x&z) -> LOP3 0xCA;  Maj = (x&y)|(x&z)|(y&z) -> 0xE8. Plain C: ptxas folds
// each to one LOP3 and may fuse across boundaries (see xor3 note / RIPEMD f-function audit).
__device__ __forceinline__ uint32_t Ch (uint32_t x,uint32_t y,uint32_t z){
    return (x & y) ^ (~x & z);
}
__device__ __forceinline__ uint32_t Maj(uint32_t x,uint32_t y,uint32_t z){
    return (x & y) | (x & z) | (y & z);
}

// SHA-256 round constants. `static constexpr` makes every K[t] (all indices are
// literal in the fully-unrolled rounds) a compile-time constant EXPRESSION, so each
// is materialized as an immediate operand with NO memory access — no LDC (constant
// bank) and no LDG (global load). It also makes K a *front-end* constant, so the
// IV-seed round-0 fold can absorb K[0] into the constant chain.
// NOTE: Phase-0 SASS already showed the ORIGINAL `__device__ __constant__` form folds
// K to immediates too (hash fn LDC=3, NOT ~64 per-round loads), so this is mainly a
// belt-and-suspenders guarantee + the front-end-constant benefit. Values UNCHANGED
// => bit-exact. VERIFY on the 5090 with `make sass` (SHA rounds must carry zero K
// loads). If `static constexpr` ever fails to compile on the toolchain, fall back to
// `__device__ __constant__` (Phase-0-proven to fold) — NOT `__device__ static const`,
// a global that can emit an LDG if not folded.
static constexpr uint32_t K[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
};

// SHA-256 IV is written as literals straight into the state at the single seed
// site (SHA256_33_from_limbs) so ptxas sees compile-time inputs to round 0 and can
// constant-fold it, instead of reloading the IV from __constant__ memory (LDC).
// The former __constant__ IV[8] + SHA256Initialize() are gone; values are unchanged.
// --- Fully hand-unrolled SHA-256 (branch-free) -------------------------------------
// Replaces the `#pragma unroll 64` loop + `if (t >= 16)` with 64 straight-line rounds.
// Uses name-rotated working registers (the caller shifts a..h by one slot each round,
// so there are no `h=g; g=f; ...` value moves) over an in-place 16-word rolling
// schedule. Each round mutates only its `d` (d += T1) and `h` (h = T1 + T2) arguments.
// The round lines + schedule updates were machine-generated and verified bit-exact vs
// the prior loop and vs hashlib over 40000 random single-block messages.
#define SHA_RND(a,b,c,d,e,f,g,h, kt, wt) do { \
    uint32_t T1 = (h) + bigS1(e) + Ch(e,f,g) + (kt) + (wt); \
    uint32_t T2 = bigS0(a) + Maj(a,b,c); \
    (d) += T1; \
    (h) = T1 + T2; \
} while (0)

// Specialized for the fixed 33-byte compressed-pubkey message: only the 9 data words
// M[0..8] vary per key; the padding tail is compile-time constant (w[9..14]=0) and the
// length field is fixed (w[15] = 33*8 = 264). The caller no longer materializes those
// 7 words, and the schedule words w[16..63] are computed in-place below (never passed in).
__device__ __forceinline__ void SHA256Transform(uint32_t state[8], const uint32_t M[9])
{
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    uint32_t w[16];
    w[ 0] = M[ 0];
    w[ 1] = M[ 1];
    w[ 2] = M[ 2];
    w[ 3] = M[ 3];
    w[ 4] = M[ 4];
    w[ 5] = M[ 5];
    w[ 6] = M[ 6];
    w[ 7] = M[ 7];
    w[ 8] = M[ 8];
    w[ 9] = 0u;
    w[10] = 0u;
    w[11] = 0u;
    w[12] = 0u;
    w[13] = 0u;
    w[14] = 0u;
    w[15] = 33u * 8u;   // 264: message-length field for the fixed 33-byte input

    SHA_RND(a,b,c,d,e,f,g,h, K[ 0], w[ 0]);
    SHA_RND(h,a,b,c,d,e,f,g, K[ 1], w[ 1]);
    SHA_RND(g,h,a,b,c,d,e,f, K[ 2], w[ 2]);
    SHA_RND(f,g,h,a,b,c,d,e, K[ 3], w[ 3]);
    SHA_RND(e,f,g,h,a,b,c,d, K[ 4], w[ 4]);
    SHA_RND(d,e,f,g,h,a,b,c, K[ 5], w[ 5]);
    SHA_RND(c,d,e,f,g,h,a,b, K[ 6], w[ 6]);
    SHA_RND(b,c,d,e,f,g,h,a, K[ 7], w[ 7]);
    SHA_RND(a,b,c,d,e,f,g,h, K[ 8], w[ 8]);
    SHA_RND(h,a,b,c,d,e,f,g, K[ 9], w[ 9]);
    SHA_RND(g,h,a,b,c,d,e,f, K[10], w[10]);
    SHA_RND(f,g,h,a,b,c,d,e, K[11], w[11]);
    SHA_RND(e,f,g,h,a,b,c,d, K[12], w[12]);
    SHA_RND(d,e,f,g,h,a,b,c, K[13], w[13]);
    SHA_RND(c,d,e,f,g,h,a,b, K[14], w[14]);
    SHA_RND(b,c,d,e,f,g,h,a, K[15], w[15]);
    w[ 0] += smallS1(w[14]) + w[ 9] + smallS0(w[ 1]);
    SHA_RND(a,b,c,d,e,f,g,h, K[16], w[ 0]);
    w[ 1] += smallS1(w[15]) + w[10] + smallS0(w[ 2]);
    SHA_RND(h,a,b,c,d,e,f,g, K[17], w[ 1]);
    w[ 2] += smallS1(w[ 0]) + w[11] + smallS0(w[ 3]);
    SHA_RND(g,h,a,b,c,d,e,f, K[18], w[ 2]);
    w[ 3] += smallS1(w[ 1]) + w[12] + smallS0(w[ 4]);
    SHA_RND(f,g,h,a,b,c,d,e, K[19], w[ 3]);
    w[ 4] += smallS1(w[ 2]) + w[13] + smallS0(w[ 5]);
    SHA_RND(e,f,g,h,a,b,c,d, K[20], w[ 4]);
    w[ 5] += smallS1(w[ 3]) + w[14] + smallS0(w[ 6]);
    SHA_RND(d,e,f,g,h,a,b,c, K[21], w[ 5]);
    w[ 6] += smallS1(w[ 4]) + w[15] + smallS0(w[ 7]);
    SHA_RND(c,d,e,f,g,h,a,b, K[22], w[ 6]);
    w[ 7] += smallS1(w[ 5]) + w[ 0] + smallS0(w[ 8]);
    SHA_RND(b,c,d,e,f,g,h,a, K[23], w[ 7]);
    w[ 8] += smallS1(w[ 6]) + w[ 1] + smallS0(w[ 9]);
    SHA_RND(a,b,c,d,e,f,g,h, K[24], w[ 8]);
    w[ 9] += smallS1(w[ 7]) + w[ 2] + smallS0(w[10]);
    SHA_RND(h,a,b,c,d,e,f,g, K[25], w[ 9]);
    w[10] += smallS1(w[ 8]) + w[ 3] + smallS0(w[11]);
    SHA_RND(g,h,a,b,c,d,e,f, K[26], w[10]);
    w[11] += smallS1(w[ 9]) + w[ 4] + smallS0(w[12]);
    SHA_RND(f,g,h,a,b,c,d,e, K[27], w[11]);
    w[12] += smallS1(w[10]) + w[ 5] + smallS0(w[13]);
    SHA_RND(e,f,g,h,a,b,c,d, K[28], w[12]);
    w[13] += smallS1(w[11]) + w[ 6] + smallS0(w[14]);
    SHA_RND(d,e,f,g,h,a,b,c, K[29], w[13]);
    w[14] += smallS1(w[12]) + w[ 7] + smallS0(w[15]);
    SHA_RND(c,d,e,f,g,h,a,b, K[30], w[14]);
    w[15] += smallS1(w[13]) + w[ 8] + smallS0(w[ 0]);
    SHA_RND(b,c,d,e,f,g,h,a, K[31], w[15]);
    w[ 0] += smallS1(w[14]) + w[ 9] + smallS0(w[ 1]);
    SHA_RND(a,b,c,d,e,f,g,h, K[32], w[ 0]);
    w[ 1] += smallS1(w[15]) + w[10] + smallS0(w[ 2]);
    SHA_RND(h,a,b,c,d,e,f,g, K[33], w[ 1]);
    w[ 2] += smallS1(w[ 0]) + w[11] + smallS0(w[ 3]);
    SHA_RND(g,h,a,b,c,d,e,f, K[34], w[ 2]);
    w[ 3] += smallS1(w[ 1]) + w[12] + smallS0(w[ 4]);
    SHA_RND(f,g,h,a,b,c,d,e, K[35], w[ 3]);
    w[ 4] += smallS1(w[ 2]) + w[13] + smallS0(w[ 5]);
    SHA_RND(e,f,g,h,a,b,c,d, K[36], w[ 4]);
    w[ 5] += smallS1(w[ 3]) + w[14] + smallS0(w[ 6]);
    SHA_RND(d,e,f,g,h,a,b,c, K[37], w[ 5]);
    w[ 6] += smallS1(w[ 4]) + w[15] + smallS0(w[ 7]);
    SHA_RND(c,d,e,f,g,h,a,b, K[38], w[ 6]);
    w[ 7] += smallS1(w[ 5]) + w[ 0] + smallS0(w[ 8]);
    SHA_RND(b,c,d,e,f,g,h,a, K[39], w[ 7]);
    w[ 8] += smallS1(w[ 6]) + w[ 1] + smallS0(w[ 9]);
    SHA_RND(a,b,c,d,e,f,g,h, K[40], w[ 8]);
    w[ 9] += smallS1(w[ 7]) + w[ 2] + smallS0(w[10]);
    SHA_RND(h,a,b,c,d,e,f,g, K[41], w[ 9]);
    w[10] += smallS1(w[ 8]) + w[ 3] + smallS0(w[11]);
    SHA_RND(g,h,a,b,c,d,e,f, K[42], w[10]);
    w[11] += smallS1(w[ 9]) + w[ 4] + smallS0(w[12]);
    SHA_RND(f,g,h,a,b,c,d,e, K[43], w[11]);
    w[12] += smallS1(w[10]) + w[ 5] + smallS0(w[13]);
    SHA_RND(e,f,g,h,a,b,c,d, K[44], w[12]);
    w[13] += smallS1(w[11]) + w[ 6] + smallS0(w[14]);
    SHA_RND(d,e,f,g,h,a,b,c, K[45], w[13]);
    w[14] += smallS1(w[12]) + w[ 7] + smallS0(w[15]);
    SHA_RND(c,d,e,f,g,h,a,b, K[46], w[14]);
    w[15] += smallS1(w[13]) + w[ 8] + smallS0(w[ 0]);
    SHA_RND(b,c,d,e,f,g,h,a, K[47], w[15]);
    w[ 0] += smallS1(w[14]) + w[ 9] + smallS0(w[ 1]);
    SHA_RND(a,b,c,d,e,f,g,h, K[48], w[ 0]);
    w[ 1] += smallS1(w[15]) + w[10] + smallS0(w[ 2]);
    SHA_RND(h,a,b,c,d,e,f,g, K[49], w[ 1]);
    w[ 2] += smallS1(w[ 0]) + w[11] + smallS0(w[ 3]);
    SHA_RND(g,h,a,b,c,d,e,f, K[50], w[ 2]);
    w[ 3] += smallS1(w[ 1]) + w[12] + smallS0(w[ 4]);
    SHA_RND(f,g,h,a,b,c,d,e, K[51], w[ 3]);
    w[ 4] += smallS1(w[ 2]) + w[13] + smallS0(w[ 5]);
    SHA_RND(e,f,g,h,a,b,c,d, K[52], w[ 4]);
    w[ 5] += smallS1(w[ 3]) + w[14] + smallS0(w[ 6]);
    SHA_RND(d,e,f,g,h,a,b,c, K[53], w[ 5]);
    w[ 6] += smallS1(w[ 4]) + w[15] + smallS0(w[ 7]);
    SHA_RND(c,d,e,f,g,h,a,b, K[54], w[ 6]);
    w[ 7] += smallS1(w[ 5]) + w[ 0] + smallS0(w[ 8]);
    SHA_RND(b,c,d,e,f,g,h,a, K[55], w[ 7]);
    w[ 8] += smallS1(w[ 6]) + w[ 1] + smallS0(w[ 9]);
    SHA_RND(a,b,c,d,e,f,g,h, K[56], w[ 8]);
    w[ 9] += smallS1(w[ 7]) + w[ 2] + smallS0(w[10]);
    SHA_RND(h,a,b,c,d,e,f,g, K[57], w[ 9]);
    w[10] += smallS1(w[ 8]) + w[ 3] + smallS0(w[11]);
    SHA_RND(g,h,a,b,c,d,e,f, K[58], w[10]);
    w[11] += smallS1(w[ 9]) + w[ 4] + smallS0(w[12]);
    SHA_RND(f,g,h,a,b,c,d,e, K[59], w[11]);
    w[12] += smallS1(w[10]) + w[ 5] + smallS0(w[13]);
    SHA_RND(e,f,g,h,a,b,c,d, K[60], w[12]);
    w[13] += smallS1(w[11]) + w[ 6] + smallS0(w[14]);
    SHA_RND(d,e,f,g,h,a,b,c, K[61], w[13]);
    w[14] += smallS1(w[12]) + w[ 7] + smallS0(w[15]);
    SHA_RND(c,d,e,f,g,h,a,b, K[62], w[14]);
    w[15] += smallS1(w[13]) + w[ 8] + smallS0(w[ 0]);
    SHA_RND(b,c,d,e,f,g,h,a, K[63], w[15]);

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}
#undef SHA_RND
// RIPEMD-160 IV is written as literals at its single seed site
// (RIPEMD160_from_SHA256_state). These values were already literals, so this is a
// pure code-tidy — no codegen change — kept for symmetry with the SHA-256 seed.

#define ROL(x,n) ((x>>(32-n))|(x<<n))
#define f1(x, y, z) (x ^ y ^ z)
#define f2(x, y, z) ((x & y) | (~x & z))
#define f3(x, y, z) ((x | ~y) ^ z)
#define f4(x, y, z) ((x & z) | (~z & y))
#define f5(x, y, z) (x ^ (y | ~z))

#define RPRound(a,b,c,d,e,f,x,k,r) \
  u = a + f + x + k; \
  a = ROL(u, r) + e; \
  c = ROL(c, 10);

#define R11(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x5A827999ul, r)
#define R31(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCul, r)
#define R51(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eul, r)
#define R12(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6ul, r)
#define R22(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124ul, r)
#define R32(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3ul, r)
#define R42(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9ul, r)
#define R52(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)

// FULL=false computes ONLY hash160 word 2 and skips the 7 rounds that cannot affect it.
//
// The final combine below is  h[2] = IV3 + e1 + a2.  Tracing the name rotation (each round
// writes its `a` slot and rotates its `c` slot by 10), `e1` is last WRITTEN at round 76 and
// only ROL10'd afterwards (at round 78); `a2` is last written at round 75 and only ROL10'd
// afterwards (at round 77). So h[2] == IV3 + ROL(e1@76,10) + ROL(a2@75,10), and line-1 rounds
// 77/78/79 plus line-2 rounds 76/77/78/79 are dead apart from those two rotates -- 153 of 160
// rounds, i.e. 7*(1 LOP3 + 2 SHF) - 2 SHF = 19 fewer ops on the sm_120 logic/shift pipe that
// binds this kernel, plus 8 IADDs from the collapsed combine.
//
// Word 2 is the CHEAPEST of the five: h[0] needs line 1 through round 78, h[1] through 77,
// h[2] only through 76. Any of the five is an equally selective 32-bit filter, so the hot path
// takes the cheap one and the ~2^-32 rare path recomputes the full digest (getHash160_33_*).
//
// VERIFIED BIT-EXACT: getHash160_w2_from_limbs() == getHash160_33_from_limbs().w[2] over 2e6
// random x + both prefixes, plus the 02||Gx known-answer test (hash160 751e76e8...f1433bd6);
// separately, 3e6 random/edge blocks through both instantiations of this transform. Method:
// compile THIS FILE on the host with -D__device__= -D__forceinline__=inline -D__noinline__=
// and stubs for __byte_perm/__funnelshift_r, then compare the two entry points.
//
// RE-RUN THAT CHECK IF THE ROUND LIST BELOW IS EVER EDITED. The trim's correctness rests on
// WHERE each register is last written, not on the digest being right: move a round and the
// FULL path can still be a correct RIPEMD-160 while word 2 silently stops matching it.
template<bool FULL>
__device__ __forceinline__ void RIPEMD160Transform(uint32_t s[5], uint32_t* w)
{
    uint32_t u;
    uint32_t a1 = s[0], b1 = s[1], c1 = s[2], d1 = s[3], e1 = s[4];
    uint32_t a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;

    R11(a1, b1, c1, d1, e1, w[0], 11);
	R12(a2, b2, c2, d2, e2, w[5], 8);
	R11(e1, a1, b1, c1, d1, w[1], 14);
	R12(e2, a2, b2, c2, d2, w[14], 9);
	R11(d1, e1, a1, b1, c1, w[2], 15);
	R12(d2, e2, a2, b2, c2, w[7], 9);
	R11(c1, d1, e1, a1, b1, w[3], 12);
	R12(c2, d2, e2, a2, b2, w[0], 11);
	R11(b1, c1, d1, e1, a1, w[4], 5);
	R12(b2, c2, d2, e2, a2, w[9], 13);
	R11(a1, b1, c1, d1, e1, w[5], 8);
	R12(a2, b2, c2, d2, e2, w[2], 15);
	R11(e1, a1, b1, c1, d1, w[6], 7);
	R12(e2, a2, b2, c2, d2, w[11], 15);
	R11(d1, e1, a1, b1, c1, w[7], 9);
	R12(d2, e2, a2, b2, c2, w[4], 5);
	R11(c1, d1, e1, a1, b1, w[8], 11);
	R12(c2, d2, e2, a2, b2, w[13], 7);
	R11(b1, c1, d1, e1, a1, w[9], 13);
	R12(b2, c2, d2, e2, a2, w[6], 7);
	R11(a1, b1, c1, d1, e1, w[10], 14);
	R12(a2, b2, c2, d2, e2, w[15], 8);
	R11(e1, a1, b1, c1, d1, w[11], 15);
	R12(e2, a2, b2, c2, d2, w[8], 11);
	R11(d1, e1, a1, b1, c1, w[12], 6);
	R12(d2, e2, a2, b2, c2, w[1], 14);
	R11(c1, d1, e1, a1, b1, w[13], 7);
	R12(c2, d2, e2, a2, b2, w[10], 14);
	R11(b1, c1, d1, e1, a1, w[14], 9);
	R12(b2, c2, d2, e2, a2, w[3], 12);
	R11(a1, b1, c1, d1, e1, w[15], 8);
	R12(a2, b2, c2, d2, e2, w[12], 6);

	R21(e1, a1, b1, c1, d1, w[7], 7);
	R22(e2, a2, b2, c2, d2, w[6], 9);
	R21(d1, e1, a1, b1, c1, w[4], 6);
	R22(d2, e2, a2, b2, c2, w[11], 13);
	R21(c1, d1, e1, a1, b1, w[13], 8);
	R22(c2, d2, e2, a2, b2, w[3], 15);
	R21(b1, c1, d1, e1, a1, w[1], 13);
	R22(b2, c2, d2, e2, a2, w[7], 7);
	R21(a1, b1, c1, d1, e1, w[10], 11);
	R22(a2, b2, c2, d2, e2, w[0], 12);
	R21(e1, a1, b1, c1, d1, w[6], 9);
	R22(e2, a2, b2, c2, d2, w[13], 8);
	R21(d1, e1, a1, b1, c1, w[15], 7);
	R22(d2, e2, a2, b2, c2, w[5], 9);
	R21(c1, d1, e1, a1, b1, w[3], 15);
	R22(c2, d2, e2, a2, b2, w[10], 11);
	R21(b1, c1, d1, e1, a1, w[12], 7);
	R22(b2, c2, d2, e2, a2, w[14], 7);
	R21(a1, b1, c1, d1, e1, w[0], 12);
	R22(a2, b2, c2, d2, e2, w[15], 7);
	R21(e1, a1, b1, c1, d1, w[9], 15);
	R22(e2, a2, b2, c2, d2, w[8], 12);
	R21(d1, e1, a1, b1, c1, w[5], 9);
	R22(d2, e2, a2, b2, c2, w[12], 7);
	R21(c1, d1, e1, a1, b1, w[2], 11);
	R22(c2, d2, e2, a2, b2, w[4], 6);
	R21(b1, c1, d1, e1, a1, w[14], 7);
	R22(b2, c2, d2, e2, a2, w[9], 15);
	R21(a1, b1, c1, d1, e1, w[11], 13);
	R22(a2, b2, c2, d2, e2, w[1], 13);
	R21(e1, a1, b1, c1, d1, w[8], 12);
	R22(e2, a2, b2, c2, d2, w[2], 11);

	R31(d1, e1, a1, b1, c1, w[3], 11);
	R32(d2, e2, a2, b2, c2, w[15], 9);
	R31(c1, d1, e1, a1, b1, w[10], 13);
	R32(c2, d2, e2, a2, b2, w[5], 7);
	R31(b1, c1, d1, e1, a1, w[14], 6);
	R32(b2, c2, d2, e2, a2, w[1], 15);
	R31(a1, b1, c1, d1, e1, w[4], 7);
	R32(a2, b2, c2, d2, e2, w[3], 11);
	R31(e1, a1, b1, c1, d1, w[9], 14);
	R32(e2, a2, b2, c2, d2, w[7], 8);
	R31(d1, e1, a1, b1, c1, w[15], 9);
	R32(d2, e2, a2, b2, c2, w[14], 6);
	R31(c1, d1, e1, a1, b1, w[8], 13);
	R32(c2, d2, e2, a2, b2, w[6], 6);
	R31(b1, c1, d1, e1, a1, w[1], 15);
	R32(b2, c2, d2, e2, a2, w[9], 14);
	R31(a1, b1, c1, d1, e1, w[2], 14);
	R32(a2, b2, c2, d2, e2, w[11], 12);
	R31(e1, a1, b1, c1, d1, w[7], 8);
	R32(e2, a2, b2, c2, d2, w[8], 13);
	R31(d1, e1, a1, b1, c1, w[0], 13);
	R32(d2, e2, a2, b2, c2, w[12], 5);
	R31(c1, d1, e1, a1, b1, w[6], 6);
	R32(c2, d2, e2, a2, b2, w[2], 14);
	R31(b1, c1, d1, e1, a1, w[13], 5);
	R32(b2, c2, d2, e2, a2, w[10], 13);
	R31(a1, b1, c1, d1, e1, w[11], 12);
	R32(a2, b2, c2, d2, e2, w[0], 13);
	R31(e1, a1, b1, c1, d1, w[5], 7);
	R32(e2, a2, b2, c2, d2, w[4], 7);
	R31(d1, e1, a1, b1, c1, w[12], 5);
	R32(d2, e2, a2, b2, c2, w[13], 5);

	R41(c1, d1, e1, a1, b1, w[1], 11);
	R42(c2, d2, e2, a2, b2, w[8], 15);
	R41(b1, c1, d1, e1, a1, w[9], 12);
	R42(b2, c2, d2, e2, a2, w[6], 5);
	R41(a1, b1, c1, d1, e1, w[11], 14);
	R42(a2, b2, c2, d2, e2, w[4], 8);
	R41(e1, a1, b1, c1, d1, w[10], 15);
	R42(e2, a2, b2, c2, d2, w[1], 11);
	R41(d1, e1, a1, b1, c1, w[0], 14);
	R42(d2, e2, a2, b2, c2, w[3], 14);
	R41(c1, d1, e1, a1, b1, w[8], 15);
	R42(c2, d2, e2, a2, b2, w[11], 14);
	R41(b1, c1, d1, e1, a1, w[12], 9);
	R42(b2, c2, d2, e2, a2, w[15], 6);
	R41(a1, b1, c1, d1, e1, w[4], 8);
	R42(a2, b2, c2, d2, e2, w[0], 14);
	R41(e1, a1, b1, c1, d1, w[13], 9);
	R42(e2, a2, b2, c2, d2, w[5], 6);
	R41(d1, e1, a1, b1, c1, w[3], 14);
	R42(d2, e2, a2, b2, c2, w[12], 9);
	R41(c1, d1, e1, a1, b1, w[7], 5);
	R42(c2, d2, e2, a2, b2, w[2], 12);
	R41(b1, c1, d1, e1, a1, w[15], 6);
	R42(b2, c2, d2, e2, a2, w[13], 9);
	R41(a1, b1, c1, d1, e1, w[14], 8);
	R42(a2, b2, c2, d2, e2, w[9], 12);
	R41(e1, a1, b1, c1, d1, w[5], 6);
	R42(e2, a2, b2, c2, d2, w[7], 5);
	R41(d1, e1, a1, b1, c1, w[6], 5);
	R42(d2, e2, a2, b2, c2, w[10], 15);
	R41(c1, d1, e1, a1, b1, w[2], 12);
	R42(c2, d2, e2, a2, b2, w[14], 8);

	R51(b1, c1, d1, e1, a1, w[4], 9);
	R52(b2, c2, d2, e2, a2, w[12], 8);
	R51(a1, b1, c1, d1, e1, w[0], 15);
	R52(a2, b2, c2, d2, e2, w[15], 5);
	R51(e1, a1, b1, c1, d1, w[5], 5);
	R52(e2, a2, b2, c2, d2, w[10], 12);
	R51(d1, e1, a1, b1, c1, w[9], 11);
	R52(d2, e2, a2, b2, c2, w[4], 9);
	R51(c1, d1, e1, a1, b1, w[7], 6);
	R52(c2, d2, e2, a2, b2, w[1], 12);
	R51(b1, c1, d1, e1, a1, w[12], 8);
	R52(b2, c2, d2, e2, a2, w[5], 5);
	R51(a1, b1, c1, d1, e1, w[2], 13);
	R52(a2, b2, c2, d2, e2, w[8], 14);
	R51(e1, a1, b1, c1, d1, w[10], 12);
	R52(e2, a2, b2, c2, d2, w[7], 6);
	R51(d1, e1, a1, b1, c1, w[14], 5);
	R52(d2, e2, a2, b2, c2, w[6], 8);
	R51(c1, d1, e1, a1, b1, w[1], 12);
	R52(c2, d2, e2, a2, b2, w[2], 13);
	R51(b1, c1, d1, e1, a1, w[3], 13);
	R52(b2, c2, d2, e2, a2, w[13], 6);
	R51(a1, b1, c1, d1, e1, w[8], 14);
	R52(a2, b2, c2, d2, e2, w[14], 5);
	R51(e1, a1, b1, c1, d1, w[11], 11);   // t=76: LAST write of e1
    if constexpr (FULL) {
	R52(e2, a2, b2, c2, d2, w[0], 15);
	R51(d1, e1, a1, b1, c1, w[6], 8);
	R52(d2, e2, a2, b2, c2, w[3], 13);
	R51(c1, d1, e1, a1, b1, w[15], 5);
	R52(c2, d2, e2, a2, b2, w[9], 11);
	R51(b1, c1, d1, e1, a1, w[13], 6);
	R52(b2, c2, d2, e2, a2, w[11], 11);
    } else {
        a2 = ROL(a2, 10);   // sole surviving effect of line-2 round 77 (its c slot is a2)
        e1 = ROL(e1, 10);   // sole surviving effect of line-1 round 78 (its c slot is e1)
    }

    if constexpr (FULL) {
        uint32_t t = s[0];
        s[0] = s[1] + c1 + d2;
        s[1] = s[2] + d1 + e2;
        s[2] = s[3] + e1 + a2;
        s[3] = s[4] + a1 + b2;
        s[4] = t + b1 + c2;
    } else {
        s[2] = s[3] + e1 + a2;   // s[3] still holds the IV3 seed; s[0]/s[1]/s[4] left untouched
    }
}


__device__ __forceinline__ uint32_t bswap32(uint32_t x){
    return __byte_perm(x, 0, 0x0123);   // reverse the 4 bytes in one PRMT
}
// SHA-256 message build for the fixed 33-byte compressed pubkey. The high/low 32-bit
// halves of each little-endian X limb are ALREADY big-endian-ordered when read as u32
// (v>>32 puts X's most-significant byte in the u32 MSB), so e0..e7 need no per-byte bswap.
// e0 = top 32 bits of X ... e7 = low 32 bits. The 33-byte message is [prefix] ++ BE(X),
// so each SHA word is a 1-byte-shifted window across e0..e7 -- one PRMT (__byte_perm) each:
//   M[j] = (e[j-1] << 24) | (e[j] >> 8)  ==  __byte_perm(e[j], e[j-1], 0x4321)
// Replaces the per-byte pack_be4 shift/mask chains; verified bit-exact vs the old build
// over 2e6 random + edge inputs (host emulation of __byte_perm).
__device__ __forceinline__ void SHA256_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint32_t out_state[16]){
    const uint32_t e0 = (uint32_t)(x_be_limbs[3] >> 32), e1 = (uint32_t)x_be_limbs[3];
    const uint32_t e2 = (uint32_t)(x_be_limbs[2] >> 32), e3 = (uint32_t)x_be_limbs[2];
    const uint32_t e4 = (uint32_t)(x_be_limbs[1] >> 32), e5 = (uint32_t)x_be_limbs[1];
    const uint32_t e6 = (uint32_t)(x_be_limbs[0] >> 32), e7 = (uint32_t)x_be_limbs[0];
    // Only the 9 data words are built here; SHA256Transform bakes in the constant
    // padding tail (w[9..14]=0) and length word (w[15]=264) itself.
    uint32_t M[9];
    M[0] = __byte_perm(e0, (uint32_t)prefix02_03, 0x4321);   // [prefix, X.b0, X.b1, X.b2]
    M[1] = __byte_perm(e1, e0, 0x4321);
    M[2] = __byte_perm(e2, e1, 0x4321);
    M[3] = __byte_perm(e3, e2, 0x4321);
    M[4] = __byte_perm(e4, e3, 0x4321);
    M[5] = __byte_perm(e5, e4, 0x4321);
    M[6] = __byte_perm(e6, e5, 0x4321);
    M[7] = __byte_perm(e7, e6, 0x4321);
    M[8] = __byte_perm(e7, 0x00000080u, 0x0455);             // [X.b31, 0x80, 0x00, 0x00]
    uint32_t st[8];
    // SHA-256 IV as literals (was SHA256Initialize copying from __constant__ IV[8]).
    // Seeding st[] with immediates lets ptxas constant-fold round 0; SHA256Transform
    // and its final `state[i] += var` are untouched, so st[i] is bit-identical.
    st[0] = 0x6a09e667u;
    st[1] = 0xbb67ae85u;
    st[2] = 0x3c6ef372u;
    st[3] = 0xa54ff53au;
    st[4] = 0x510e527fu;
    st[5] = 0x9b05688cu;
    st[6] = 0x1f83d9abu;
    st[7] = 0x5be0cd19u;
    SHA256Transform(st, M);
    out_state[0]=bswap32(st[0]); out_state[1]=bswap32(st[1]); out_state[2]=bswap32(st[2]); out_state[3]=bswap32(st[3]);
    out_state[4]=bswap32(st[4]); out_state[5]=bswap32(st[5]); out_state[6]=bswap32(st[6]); out_state[7]=bswap32(st[7]);
}

// The 16-word RIPEMD-160 block is the 8 SHA-256 words plus a compile-time-constant tail.
// Both entry points below share it; keep the two padding writes identical.
#define RIPEMD160_PAD_TAIL(w) do { \
    (w)[8]  = 0x00000080u; \
    (w)[9]=0u; (w)[10]=0u; (w)[11]=0u; (w)[12]=0u; (w)[13]=0u; \
    (w)[14] = 256u; \
    (w)[15] = 0u; \
} while (0)

// Hot path: hash160 word 2 only, via the 153-round RIPEMD160Transform<false>. Returning one
// u32 instead of a 5-word H160 also narrows the by-value ABI at the __noinline__ boundary --
// see CUDAHash.cuh for why that boundary's register cost is worth watching.
__device__ __forceinline__ uint32_t RIPEMD160_w2_from_SHA256_state(uint32_t sha_state_le[16])
{
    RIPEMD160_PAD_TAIL(sha_state_le);
    uint32_t s[5];
    s[0] = 0x67452301u;
    s[1] = 0xEFCDAB89u;
    s[2] = 0x98BADCFEu;
    s[3] = 0x10325476u;   // IV3: the only seed word the trimmed combine reads
    s[4] = 0xC3D2E1F0u;
    RIPEMD160Transform<false>(s, sha_state_le);
    return s[2];          // s[0]/s[1]/s[3]/s[4] are dead here and DCE away
}

__device__ __forceinline__ void RIPEMD160_from_SHA256_state(uint32_t sha_state_le[16],
                                                            uint32_t out5[5])
{
    RIPEMD160_PAD_TAIL(sha_state_le);

    // RIPEMD-160 IV as literals, written straight into out5 — which doubles as the
    // RIPEMD state. RIPEMD160Transform reads these at entry, leaves them untouched
    // through the 80 rounds, then overwrites them with the final digest, so the old
    // s[5] scratch + 5-word copy-back were redundant. (out5 and sha_state_le are
    // distinct buffers, so no aliasing.) out5[i] is hash160 word i, little-endian.
    out5[0] = 0x67452301u;
    out5[1] = 0xEFCDAB89u;
    out5[2] = 0x98BADCFEu;
    out5[3] = 0x10325476u;
    out5[4] = 0xC3D2E1F0u;
    RIPEMD160Transform<true>(out5, sha_state_le);
}

// HOT PATH. Same by-value ABI, but the result is one u32 (hash160 word 2) instead of five,
// and the RIPEMD-160 inside is the 153-round trim. This is the ONLY hash call on the scanning
// path; getHash160_33_from_limbs below runs only after a 32-bit filter hit (~2^-32).
__device__ __noinline__ uint32_t getHash160_w2_from_limbs(uint8_t prefix02_03, U256 x)
{
    uint32_t sha_state[16];
    SHA256_33_from_limbs(prefix02_03, x.v, sha_state);
    return RIPEMD160_w2_from_SHA256_state(sha_state);
}

// 2-WIDE HOT PATH. Two independent points, one CALL -- see CUDAHash.cuh for the measured
// rationale (latency-bound at 16 warps; two separate __noinline__ calls are hard-serialized by
// CALL/RET, so block2 could never fill block1's dependency stalls).
//
// BIT-EXACT BY CONSTRUCTION: this calls the SAME __forceinline__ SHA256_33_from_limbs and
// RIPEMD160_w2_from_SHA256_state bodies the 1-wide path uses, on the same inputs, so
// getHash160_w2_x2(pA,xA,pB,xB) == { getHash160_w2_from_limbs(pA,xA),
// getHash160_w2_from_limbs(pB,xB) } identically. No new hash arithmetic is introduced and the
// round lists are untouched, so the word-2 trim's "where each register is last written" proof
// still holds unchanged. The ONLY new thing here is the scheduling scope.
//
// ORDER IS THE WHOLE POINT: both SHA chains first (adjacent and mutually independent), then both
// RIPEMD chains. Writing SHA(A);RIPEMD(A);SHA(B);RIPEMD(B) instead would recreate the serial
// shape -- A's RIPEMD depends on A's SHA, so nothing independent would sit adjacent to fill the
// stalls, and the merge would buy nothing over the two calls it replaces.
//
// Each chain keeps its OWN 16-word state array: SHA writes [0..7] and RIPEMD160_PAD_TAIL writes
// [8..15] of the same buffer (exactly as in the 1-wide path), so the two must not share one.
__device__ __noinline__ Hash2 getHash160_w2_x2(uint8_t prefixA, U256 xA, uint8_t prefixB, U256 xB)
{
    uint32_t shaA[16], shaB[16];
    SHA256_33_from_limbs(prefixA, xA.v, shaA);
    SHA256_33_from_limbs(prefixB, xB.v, shaB);

    Hash2 r;
    r.w2a = RIPEMD160_w2_from_SHA256_state(shaA);
    r.w2b = RIPEMD160_w2_from_SHA256_state(shaB);
    return r;
}

// BY-VALUE ABI (x in by value, hash160 out by value) -- see CUDAHash.cuh for the measured effect
// and why this must not be reverted to pointers. __noinline__ is deliberate: the CALL is kept,
// only its ABI changed. The body is unchanged from the pointer version, so the hash is bit-exact.
// COLD PATH ONLY since the word-2 filter landed: called under `pref &&`, i.e. ~2^-32 of keys.
__device__ __noinline__ H160 getHash160_33_from_limbs(uint8_t prefix02_03, U256 x)
{
    uint32_t sha_state[16];
    SHA256_33_from_limbs(prefix02_03, x.v, sha_state);   // x arrives in regs; callee is forceinline
    H160 h;
    RIPEMD160_from_SHA256_state(sha_state, h.w);         // forceinline -> SROA keeps h.w in regs
    return h;
}
