// Compile-and-run check of field_split.cuh AS INCLUDED (no GPU required).
//
// field_split_equiv.cpp exercises the extracted function BODIES, which would still pass if
// the header's macro-alias block or its trailing #undefs were broken. This test includes the
// real header the way ec_backend.cuh does -- after the RCKangaroo dependencies -- so a typo in
// the mad_lo_cc/madc_hi_cc/madc_lo_cc aliases, a missing #pragma once, or a stray #undef is a
// build failure here rather than an nvcc failure later on the GPU box.
//
// It then re-checks a handful of vectors against the RCKangaroo cores, so the header-included
// build is confirmed to behave identically to the extracted-body build.
//
//   g++ -O2 -fno-strict-aliasing -I tests -o field_split_include tests/field_split_include.cpp
#include <cstdio>
#include <cstring>
#include <random>

#include "ptx_host.h"

// RCKangaroo dependencies first (field_split.cuh calls mul_256_by_P0inv), exactly as
// ec_backend.cuh arranges it by including RCGpuUtils.h ahead of the port.
#include "field_extract_rck.inc"

// ...then the REAL header, unmodified.
#include "../field_split.cuh"

// The header must leave no macro residue behind for later includes.
#if defined(mad_lo_cc) || defined(madc_hi_cc) || defined(madc_lo_cc)
#error "field_split.cuh leaked its short-form macro aliases"
#endif

int main()
{
    std::mt19937_64 rng(0xABCDEFULL);
    int bad = 0;
    const long N = 200000;

    for (long i = 0; i < N; i++) {
        u64 A[4], B[4], a[4], b[4], r_ref[4], r_new[4], s_ref[4], s_new[4];
        for (int k = 0; k < 4; k++) { A[k] = rng(); B[k] = rng(); }

        memcpy(a, A, sizeof(a)); memcpy(b, B, sizeof(b)); MulModP(r_ref, a, b);
        memcpy(a, A, sizeof(a)); memcpy(b, B, sizeof(b)); MulModP_split(r_new, a, b);
        if (memcmp(r_ref, r_new, sizeof(r_ref)) != 0) bad++;

        memcpy(a, A, sizeof(a)); SqrModP(s_ref, a);
        memcpy(a, A, sizeof(a)); SqrModP_split(s_new, a);
        if (memcmp(s_ref, s_new, sizeof(s_ref)) != 0) bad++;
    }

    printf("field_split_include: header compiles as included; %ld vectors, %d mismatches\n", N, bad);
    if (bad) { printf("RESULT: FAIL\n"); return 1; }
    printf("RESULT: PASS\n");
    return 0;
}
