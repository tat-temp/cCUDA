// Host correctness gate for the split-column field cores (no GPU, no nvcc).
//
// Includes the REAL field_split.cuh and runs it against a host emulation of the PTX carry
// primitives, checking:
//   T1  mul512_split / sqr512_split == schoolbook 512-bit product, bit-for-bit
//   T2  MulModP_split / SqrModP_split congruent to a*b (mod P) vs an independent bignum
//
// T1 is the strong one: an exact 512-bit product plus RCKangaroo's unchanged reduction tail is
// what makes the split cores a drop-in for the cores they replaced.
//
//   g++ -O2 -fno-strict-aliasing -I tests -o field_split_test tests/field_split_test.cpp
//   ./field_split_test [iterations]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

#include "ptx_host.h"
#include "field_extract_rck.inc"   // mul_256_by_P0inv, the tail's one dependency
#include "../field_split.cuh"

#if defined(mad_lo_cc) || defined(madc_hi_cc) || defined(madc_lo_cc)
#error "field_split.cuh leaked its short-form macro aliases"
#endif

static const u64 K_FOLD = 0x1000003D1ull;   // P = 2^256 - K_FOLD

static void mul512_ref(u32* p, const u32* a, const u32* b)
{
    memset(p, 0, 16 * sizeof(u32));
    for (int i = 0; i < 8; i++) {
        u64 carry = 0;
        for (int j = 0; j < 8; j++) {
            u64 t = (u64)a[i] * b[j] + p[i + j] + carry;
            p[i + j] = (u32)t;
            carry = t >> 32;
        }
        p[i + 8] = (u32)carry;
    }
}

// Fold a 512-bit value to canonical [0,P) using v == lo + hi*K (mod P).
static void modp_ref(u32* out8, const u32* p16)
{
    u32 v[24];
    memset(v, 0, sizeof(v));
    memcpy(v, p16, 16 * sizeof(u32));

    int pass = 0;
    for (; pass < 16; pass++) {
        bool hi_zero = true;
        for (int i = 8; i < 24; i++) if (v[i]) { hi_zero = false; break; }
        if (hi_zero) break;

        u32 hi[16], lo[8];
        memcpy(lo, v, 8 * sizeof(u32));
        memset(hi, 0, sizeof(hi));
        for (int i = 8; i < 24; i++) hi[i - 8] = v[i];

        memset(v, 0, sizeof(v));
        u64 carry = 0;
        for (int i = 0; i < 16; i++) {
            u64 t = (u64)hi[i] * (u32)(K_FOLD & 0xFFFFFFFFu) + carry;
            v[i] = (u32)t;
            carry = t >> 32;
        }
        v[16] = (u32)carry;
        carry = 0;
        for (int i = 0; i < 16; i++) {          // K_FOLD's bit 32 -> add hi << 32
            u64 t = (u64)v[i + 1] + (u64)hi[i] * (u32)(K_FOLD >> 32) + carry;
            v[i + 1] = (u32)t;
            carry = t >> 32;
        }
        v[17] += (u32)carry;
        carry = 0;
        for (int i = 0; i < 8; i++) {
            u64 t = (u64)v[i] + lo[i] + carry;
            v[i] = (u32)t;
            carry = t >> 32;
        }
        for (int i = 8; i < 24 && carry; i++) {
            u64 t = (u64)v[i] + carry;
            v[i] = (u32)t;
            carry = t >> 32;
        }
    }
    if (pass == 16) { fprintf(stderr, "modp_ref: fold did not converge\n"); abort(); }

    static const u32 P[8] = { 0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                              0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
    u32 t[8];
    u64 borrow = 0;
    for (int i = 0; i < 8; i++) {
        u64 d = (u64)v[i] - P[i] - borrow;
        t[i] = (u32)d;
        borrow = (d >> 32) ? 1 : 0;
    }
    memcpy(out8, borrow ? v : t, 8 * sizeof(u32));
}

static void canon256(u32* out8, const u32* in8)
{
    u32 p16[16];
    memset(p16, 0, sizeof(p16));
    memcpy(p16, in8, 8 * sizeof(u32));
    modp_ref(out8, p16);
}

static bool lt_P(const u32* v)
{
    static const u32 P[8] = { 0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                              0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
    for (int i = 7; i >= 0; i--) if (v[i] != P[i]) return v[i] < P[i];
    return false;
}

static int fails = 0;

static void note_fail(const char* what, const u32* a, const u32* b)
{
    if (++fails > 5) return;
    printf("  FAIL %s\n    a = ", what);
    for (int i = 7; i >= 0; i--) printf("%08x", a[i]);
    if (b) { printf("\n    b = "); for (int i = 7; i >= 0; i--) printf("%08x", b[i]); }
    printf("\n");
}

int main(int argc, char** argv)
{
    long iters = (argc > 1) ? atol(argv[1]) : 2000000;
    std::mt19937_64 rng(0xC0FFEEULL);

    // Edges are where carry propagation actually breaks: 0, 1, P-1, P, P+1, 2^256-1.
    static const u64 EDGE[][4] = {
        { 0, 0, 0, 0 },
        { 1, 0, 0, 0 },
        { 0xFFFFFFFEFFFFFC2Eull, ~0ull, ~0ull, ~0ull },
        { 0xFFFFFFFEFFFFFC2Full, ~0ull, ~0ull, ~0ull },
        { ~0ull, ~0ull, ~0ull, ~0ull },
        { 0xFFFFFFFEFFFFFC30ull, ~0ull, ~0ull, ~0ull },
        { 0, 0, 0, 0x8000000000000000ull },
        { ~0ull, 0, 0, 0 },
    };
    const int NEDGE = (int)(sizeof(EDGE) / sizeof(EDGE[0]));

    printf("field_split_test: %ld random iterations + %d^2 edge pairs\n", iters, NEDGE);
    long n = 0;

    for (long it = 0; it < iters + NEDGE * NEDGE; it++) {
        u64 A[4], B[4];
        if (it < NEDGE * NEDGE) {
            memcpy(A, EDGE[it / NEDGE], sizeof(A));
            memcpy(B, EDGE[it % NEDGE], sizeof(B));
        } else {
            for (int i = 0; i < 4; i++) { A[i] = rng(); B[i] = rng(); }
            if ((it & 15) == 0) { A[3] |= 0xFFFFFFFF00000000ull; B[3] |= 0xFFFFFFFF00000000ull; }
        }
        n++;

        // T1: raw product cores vs schoolbook
        u32 p_new[16], p_ref[16], q_new[16], q_ref[16];
        mul512_split(p_new, (const u32*)A, (const u32*)B);
        mul512_ref(p_ref, (const u32*)A, (const u32*)B);
        if (memcmp(p_new, p_ref, sizeof(p_ref)) != 0) note_fail("T1 mul512_split", (u32*)A, (u32*)B);

        sqr512_split(q_new, (const u32*)A);
        mul512_ref(q_ref, (const u32*)A, (const u32*)A);
        if (memcmp(q_new, q_ref, sizeof(q_ref)) != 0) note_fail("T1 sqr512_split", (u32*)A, NULL);

        // T2: full field ops congruent mod P. Only meaningful for canonical inputs -- at or
        // above P the (unchanged) RCKangaroo tail's single fold can leave a non-congruent
        // residue, which is a documented limit of the tail, not of these cores.
        u64 a[4], b[4], r[4], s[4];
        u32 want[8], got[8];

        memcpy(a, A, sizeof(a)); memcpy(b, B, sizeof(b)); MulModP_split(r, a, b);
        memcpy(a, A, sizeof(a)); SqrModP_split(s, a);

        if (lt_P((const u32*)A) && lt_P((const u32*)B)) {
            modp_ref(want, p_ref);
            canon256(got, (const u32*)r);
            if (memcmp(want, got, sizeof(want)) != 0) note_fail("T2 MulModP_split", (u32*)A, (u32*)B);
        }
        if (lt_P((const u32*)A)) {
            modp_ref(want, q_ref);
            canon256(got, (const u32*)s);
            if (memcmp(want, got, sizeof(want)) != 0) note_fail("T2 SqrModP_split", (u32*)A, NULL);
        }
    }

    printf("pairs tested : %ld\n", n);
    if (fails) { printf("RESULT: %d FAILURES\n", fails); return 1; }
    printf("RESULT: ALL PASS (T1 T2)\n");
    return 0;
}
