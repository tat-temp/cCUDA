// Host emulation of CUDAHash.cu, per the method recorded in the file's own comments:
//   compile with -D__device__= -D__forceinline__=inline -D__noinline__= plus stubs for
//   __byte_perm / __funnelshift_r, then compare entry points.
//
// Proves, on the host:
//   1. KAT: getHash160_w2_from_limbs(0x02, Gx) == word 2 of hash160(02||Gx)
//      (privkey 1 -> 751e76e8199196d454941c45d1b3a323f1433bd6 -> word2 = 0x451c9454)
//   2. getHash160_w2_x2(pA,xA,pB,xB) == { w2(pA,xA), w2(pB,xB) }  -- the new 2-wide entry
//   3. getHash160_w2_from_limbs(p,x) == getHash160_33_from_limbs(p,x).w[2]  -- the trim invariant

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

// ---- CUDA intrinsic stubs -----------------------------------------------------------------
// __byte_perm(x,y,s): bytes are selected from the 64-bit value {y,x} (x = low word). Each of the
// 4 nibbles of s picks a byte index 0..7 (low nibble -> result byte 0). Index bit 3 set means
// "replicate the sign bit of byte (index&7)".
static inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s)
{
    unsigned long long tmp = ((unsigned long long)y << 32) | (unsigned long long)x;
    unsigned int result = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned int sel = (s >> (4 * i)) & 0xF;
        unsigned char byte;
        unsigned char b = (unsigned char)((tmp >> (8 * (sel & 7))) & 0xFF);
        if (sel & 0x8) byte = (b & 0x80) ? 0xFF : 0x00;
        else           byte = b;
        result |= ((unsigned int)byte) << (8 * i);
    }
    return result;
}

static inline unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int shift)
{
    unsigned long long v = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    return (unsigned int)(v >> (shift & 31));
}

#include "CUDAHash.cu"

// -------------------------------------------------------------------------------------------
static U256 mk(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3)
{
    U256 r; r.v[0] = l0; r.v[1] = l1; r.v[2] = l2; r.v[3] = l3; return r;
}

int main()
{
    int fail = 0;

    // ---- 1. Known-answer test: compressed pubkey of privkey 1 is 02||Gx ----
    U256 Gx = mk(0x59f2815b16f81798ULL, 0x029bfcdb2dce28d9ULL,
                 0x55a06295ce870b07ULL, 0x79be667ef9dcbbacULL);
    const uint32_t KAT_W2 = 0x451c9454u;   // bytes 8..11 of 751e76e8...f1433bd6, little-endian
    uint32_t kat = getHash160_w2_from_limbs(0x02, Gx);
    if (kat != KAT_W2) { printf("FAIL KAT: got %08x want %08x\n", kat, KAT_W2); ++fail; }
    else                printf("PASS KAT   : w2(02||Gx) = %08x\n", kat);

    H160 kat5 = getHash160_33_from_limbs(0x02, Gx);
    const uint32_t want5[5] = {0xe8761e75u, 0xd4969119u, 0x451c9454u, 0x23a3b3d1u, 0xd63b43f1u};
    for (int i = 0; i < 5; ++i)
        if (kat5.w[i] != want5[i]) { printf("FAIL KAT5 word %d: %08x != %08x\n", i, kat5.w[i], want5[i]); ++fail; }
    if (!fail) printf("PASS KAT5  : full hash160 of 02||Gx matches\n");

    // ---- 2 & 3. randomized equivalence ----
    std::mt19937_64 rng(0xC0FFEEULL);
    const long N = 300000;
    long checked = 0;
    for (long n = 0; n < N; ++n) {
        U256 a = mk(rng(), rng(), rng(), rng());
        U256 b = mk(rng(), rng(), rng(), rng());
        uint8_t pA = (rng() & 1) ? 0x03 : 0x02;
        uint8_t pB = (rng() & 1) ? 0x03 : 0x02;

        uint32_t w2a = getHash160_w2_from_limbs(pA, a);
        uint32_t w2b = getHash160_w2_from_limbs(pB, b);
        Hash2 h2 = getHash160_w2_x2(pA, a, pB, b);

        if (h2.w2a != w2a || h2.w2b != w2b) {
            printf("FAIL x2 at n=%ld: (%08x,%08x) != (%08x,%08x)\n", n, h2.w2a, h2.w2b, w2a, w2b);
            ++fail; break;
        }
        // the word-2 trim invariant, on a subset (the full path is much more expensive)
        if ((n & 0x3FF) == 0) {
            H160 full = getHash160_33_from_limbs(pA, a);
            if (full.w[2] != w2a) {
                printf("FAIL trim at n=%ld: full.w[2]=%08x w2=%08x\n", n, full.w[2], w2a);
                ++fail; break;
            }
        }
        ++checked;
    }
    if (!fail) printf("PASS x2    : %ld random pairs, getHash160_w2_x2 == two 1-wide calls\n", checked);
    if (!fail) printf("PASS trim  : sampled full.w[2] == w2 on %ld cases\n", checked / 1024 + 1);

    // ---- edge cases: aliasing-ish and extremes ----
    U256 zero = mk(0, 0, 0, 0);
    U256 ones = mk(~0ULL, ~0ULL, ~0ULL, ~0ULL);
    struct { U256 x; uint8_t p; } edges[] = {
        {zero, 0x02}, {zero, 0x03}, {ones, 0x02}, {ones, 0x03}, {Gx, 0x03},
    };
    for (int i = 0; i < 5 && !fail; ++i)
        for (int j = 0; j < 5 && !fail; ++j) {
            Hash2 h2 = getHash160_w2_x2(edges[i].p, edges[i].x, edges[j].p, edges[j].x);
            uint32_t ea = getHash160_w2_from_limbs(edges[i].p, edges[i].x);
            uint32_t eb = getHash160_w2_from_limbs(edges[j].p, edges[j].x);
            if (h2.w2a != ea || h2.w2b != eb) { printf("FAIL edge (%d,%d)\n", i, j); ++fail; }
        }
    if (!fail) printf("PASS edges : 25 edge combinations\n");

    // ---- same-input-both-lanes (the singleton-routing shape, and a self-consistency check) ----
    for (int i = 0; i < 5 && !fail; ++i) {
        Hash2 h2 = getHash160_w2_x2(edges[i].p, edges[i].x, edges[i].p, edges[i].x);
        if (h2.w2a != h2.w2b) { printf("FAIL dup lane %d: %08x != %08x\n", i, h2.w2a, h2.w2b); ++fail; }
    }
    if (!fail) printf("PASS dup   : identical inputs give identical lanes\n");

    printf(fail ? "\nRESULT: FAIL (%d)\n" : "\nRESULT: ALL PASS\n", fail);
    return fail ? 1 : 0;
}
