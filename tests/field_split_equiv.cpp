// Host equivalence gate for the split-column field cores (no GPU required).
//
// Builds the REAL shipped bodies of RCKangaroo's MulModP/SqrModP and this tree's
// MulModP_split/SqrModP_split (extracted verbatim by tests/extract_field.py) against a
// faithful emulation of the PTX carry-flag primitives, then checks:
//
//   T1  MulModP_split(a,b) == MulModP(a,b)     bit-for-bit, over random + edge inputs
//   T2  SqrModP_split(a)   == SqrModP(a)       bit-for-bit
//   T3  mul512_split / sqr512_split == schoolbook 512-bit product, bit-for-bit
//   T4  both paths are congruent to a*b (mod P) against an independent bignum reference
//   T5  instruments t32[9] (the word upstream's tail tweak assumes is <= 2)
//
// T1/T2 are the gate that matters: the two implementations share a byte-identical
// reduction tail, so bit-equality of the product cores means the split path returns the
// SAME lazy representative, not merely a congruent one -- nothing downstream can observe
// the swap. T4 is the independent check that catches a fault common to both.
//
//   g++ -O2 -fno-strict-aliasing -o field_split_equiv tests/field_split_equiv.cpp
//   ./field_split_equiv [iterations]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <random>

#include "ptx_host.h"

#include "field_extract_rck.inc"
#include "field_extract_split.inc"

// ---------------------------------------------------------------------------
// Independent bignum reference (plain C, no carry-flag trickery).
// ---------------------------------------------------------------------------
static const u64 K_FOLD = 0x1000003D1ull;  // P = 2^256 - K_FOLD

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

static void sqr512_ref(u32* p, const u32* a) { mul512_ref(p, a, a); }

// Fold a 512-bit value down to canonical [0,P) using v == lo + hi*K (mod P).
static void modp_ref(u32* out8, const u32* p16)
{
    u32 v[24];
    memset(v, 0, sizeof(v));
    memcpy(v, p16, 16 * sizeof(u32));

    // Fold until nothing remains above bit 256. Each pass shrinks the high part by ~2^223,
    // so this converges in three; the cap is only a runaway guard and is asserted below.
    int pass = 0;
    for (; pass < 16; pass++) {
        bool hi_zero = true;
        for (int i = 8; i < 24; i++) if (v[i]) { hi_zero = false; break; }
        if (hi_zero) break;

        u32 hi[16], lo[8];
        memcpy(lo, v, 8 * sizeof(u32));
        memset(hi, 0, sizeof(hi));
        for (int i = 8; i < 24; i++) hi[i - 8] = v[i];

        // v = lo + hi * K
        memset(v, 0, sizeof(v));
        u64 carry = 0;
        for (int i = 0; i < 16; i++) {
            u64 t = (u64)hi[i] * (u32)(K_FOLD & 0xFFFFFFFFu) + carry;
            v[i] = (u32)t;
            carry = t >> 32;
        }
        v[16] = (u32)carry;
        carry = 0;
        for (int i = 0; i < 16; i++) {           // K_FOLD's bit 32 -> add hi << 32
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

    // v < 2^256 now; one conditional subtract of P gives canonical.
    u32 P[8] = { 0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
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

// ---------------------------------------------------------------------------
static int fails = 0;
static u32 max_t32_9 = 0;

static void note_fail(const char* what, const u32* a, const u32* b)
{
    if (++fails > 5) return;
    printf("  FAIL %s\n    a = ", what);
    for (int i = 7; i >= 0; i--) printf("%08x", a[i]);
    if (b) { printf("\n    b = "); for (int i = 7; i >= 0; i--) printf("%08x", b[i]); }
    printf("\n");
}

// true if the 256-bit little-endian value is canonical (< P)
static bool lt_P(const u32* v)
{
    static const u32 P[8] = { 0xFFFFFC2Fu, 0xFFFFFFFEu, 0xFFFFFFFFu, 0xFFFFFFFFu,
                              0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu };
    for (int i = 7; i >= 0; i--) {
        if (v[i] != P[i]) return v[i] < P[i];
    }
    return false;  // equal to P
}

static void dump8(const char* tag, const u32* v)
{
    printf("    %-10s ", tag);
    for (int i = 7; i >= 0; i--) printf("%08x", v[i]);
    printf("\n");
}

// Recompute the tail's t32[9] for the instrumentation in T5.
static void probe_t32_9(const u32* prod16)
{
    u64 tmp[5];
    memset(tmp, 0, sizeof(tmp));
    mul_256_by_P0inv((u32*)tmp, (u32*)(prod16 + 8));
    u64 buff[4];
    memcpy(buff, prod16, 4 * sizeof(u64));
    add_cc_64(buff[0], buff[0], tmp[0]);
    addc_cc_64(buff[1], buff[1], tmp[1]);
    addc_cc_64(buff[2], buff[2], tmp[2]);
    addc_cc_64(buff[3], buff[3], tmp[3]);
    addc_64(tmp[4], tmp[4], 0ull);
    u32 w = ((u32*)tmp)[9];
    if (w > max_t32_9) max_t32_9 = w;
}

int main(int argc, char** argv)
{
    long iters = (argc > 1) ? atol(argv[1]) : 2000000;
    std::mt19937_64 rng(0xC0FFEEULL);

    // Edge cases first, then random. Edges: 0, 1, P-1, P, 2^256-1, and near-P values,
    // which are where a carry-propagation bug in a product core actually shows up.
    static const u64 EDGE[][4] = {
        { 0, 0, 0, 0 },
        { 1, 0, 0, 0 },
        { 0xFFFFFFFEFFFFFC2Eull, ~0ull, ~0ull, ~0ull },  // P-1
        { 0xFFFFFFFEFFFFFC2Full, ~0ull, ~0ull, ~0ull },  // P
        { ~0ull, ~0ull, ~0ull, ~0ull },                  // 2^256-1
        { 0xFFFFFFFEFFFFFC30ull, ~0ull, ~0ull, ~0ull },  // P+1
        { 0, 0, 0, 0x8000000000000000ull },
        { ~0ull, 0, 0, 0 },
    };
    const int NEDGE = (int)(sizeof(EDGE) / sizeof(EDGE[0]));

    long n_pairs = 0, n_noncanon = 0, n_noncanon_diverge = 0;
    printf("field_split_equiv: %ld random iterations + %d^2 edge pairs\n", iters, NEDGE);

    for (long it = 0; it < iters + NEDGE * NEDGE; it++) {
        u64 A[4], B[4];
        if (it < NEDGE * NEDGE) {
            memcpy(A, EDGE[it / NEDGE], sizeof(A));
            memcpy(B, EDGE[it % NEDGE], sizeof(B));
        } else {
            for (int i = 0; i < 4; i++) { A[i] = rng(); B[i] = rng(); }
            // Bias a slice of the space toward the top of the range, where carries chain.
            if ((it & 15) == 0) { A[3] |= 0xFFFFFFFF00000000ull; B[3] |= 0xFFFFFFFF00000000ull; }
        }
        n_pairs++;

        u64 r_ref[4], r_new[4], s_ref[4], s_new[4];
        u64 a_copy[4], b_copy[4];

        // T1: MulModP vs MulModP_split (inputs copied -- the originals take non-const u64*)
        memcpy(a_copy, A, sizeof(A)); memcpy(b_copy, B, sizeof(B));
        MulModP(r_ref, a_copy, b_copy);
        memcpy(a_copy, A, sizeof(A)); memcpy(b_copy, B, sizeof(B));
        MulModP_split(r_new, a_copy, b_copy);
        if (memcmp(r_ref, r_new, sizeof(r_ref)) != 0) note_fail("T1 MulModP_split != MulModP", (u32*)A, (u32*)B);

        // T2: SqrModP vs SqrModP_split
        memcpy(a_copy, A, sizeof(A));
        SqrModP(s_ref, a_copy);
        memcpy(a_copy, A, sizeof(A));
        SqrModP_split(s_new, a_copy);
        if (memcmp(s_ref, s_new, sizeof(s_ref)) != 0) note_fail("T2 SqrModP_split != SqrModP", (u32*)A, NULL);

        // T3: raw 512-bit product cores vs schoolbook
        u32 p_new[16], p_ref[16], q_new[16], q_ref[16];
        mul512_split(p_new, (const u32*)A, (const u32*)B);
        mul512_ref(p_ref, (const u32*)A, (const u32*)B);
        if (memcmp(p_new, p_ref, sizeof(p_ref)) != 0) note_fail("T3 mul512_split != schoolbook", (u32*)A, (u32*)B);

        sqr512_split(q_new, (const u32*)A);
        sqr512_ref(q_ref, (const u32*)A);
        if (memcmp(q_new, q_ref, sizeof(q_ref)) != 0) note_fail("T3 sqr512_split != schoolbook", (u32*)A, NULL);

        // T4: congruence to a*b (mod P) against the independent reference.
        //
        // Checked for BOTH implementations. RCKangaroo's reduction tail assumes bounded
        // (canonical) inputs -- feed it a >= P and the single fold it performs can leave a
        // non-congruent residue. That is a property of the tail this port does not touch, so
        // it is required to hold for canonical inputs and merely COUNTED (old vs new, which
        // T1/T2 already pin as identical) for out-of-range ones.
        const bool canonical = lt_P((const u32*)A) && lt_P((const u32*)B);
        u32 want[8], got_old[8], got_new[8];

        modp_ref(want, p_ref);
        canon256(got_old, (const u32*)r_ref);
        canon256(got_new, (const u32*)r_new);
        bool ok_old = (memcmp(want, got_old, sizeof(want)) == 0);
        bool ok_new = (memcmp(want, got_new, sizeof(want)) == 0);
        if (canonical) {
            if (!ok_new) note_fail("T4 mul(split) not congruent mod P", (u32*)A, (u32*)B);
            if (!ok_old) note_fail("T4 mul(RCK) not congruent mod P", (u32*)A, (u32*)B);
        } else {
            n_noncanon++;
            if (!ok_old || !ok_new) n_noncanon_diverge++;
            if (ok_old != ok_new) note_fail("T4 mul: split and RCK disagree out of range", (u32*)A, (u32*)B);
        }

        modp_ref(want, q_ref);
        canon256(got_old, (const u32*)s_ref);
        canon256(got_new, (const u32*)s_new);
        ok_old = (memcmp(want, got_old, sizeof(want)) == 0);
        ok_new = (memcmp(want, got_new, sizeof(want)) == 0);
        if (lt_P((const u32*)A)) {
            if (!ok_new) note_fail("T4 sqr(split) not congruent mod P", (u32*)A, NULL);
            if (!ok_old) note_fail("T4 sqr(RCK) not congruent mod P", (u32*)A, NULL);
        } else if (ok_old != ok_new) {
            note_fail("T4 sqr: split and RCK disagree out of range", (u32*)A, NULL);
        }

        // T5: instrument the word upstream's tail tweak assumes is small
        probe_t32_9(p_ref);
        probe_t32_9(q_ref);
    }

    printf("pairs tested : %ld  (%ld with a or b >= P; %ld of those left a non-congruent\n"
           "               residue -- identically in BOTH implementations, see T4 note)\n",
           n_pairs, n_noncanon, n_noncanon_diverge);
    printf("max t32[9]   : %u  (upstream tail tweak assumes <= 2; 977*%u = %llu)\n",
           max_t32_9, max_t32_9, (unsigned long long)max_t32_9 * 977ull);
    if (fails) { printf("RESULT: %d FAILURES\n", fails); return 1; }
    printf("RESULT: ALL PASS (T1 T2 T3 T4)\n");
    return 0;
}
