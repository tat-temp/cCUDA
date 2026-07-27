// CUDACyclone — GPU secp256k1 Bitcoin-puzzle solver.
// SPDX-License-Identifier: GPL-3.0-or-later
// The default build's secp256k1 field arithmetic is RCKangaroo (c) 2024 RetiredCoder, GPLv3
// (third_party/RCKangaroo/); the EC math lineage is VanitySearch (c) Jean-Luc Pons, GPLv3.
// This program is therefore distributed under the GNU GPL v3 — see LICENSE.

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>

#include "CUDAMath.h"
#include "sha256.h"
// P3.0 (single-TU + rdc=false): pull the getHash160 DEFINITION into this TU rather than the
// CUDAHash.cuh declaration. getHash160_33_from_limbs is the ONLY __noinline__ device fn in
// CUDAHash.cu (every helper there is __forceinline__ and inlines into it), so it was the sole
// cross-TU device call and the sole reason this build needed -rdc=true. In one TU with rdc=false
// ptxas emits an INTRA-MODULE CALL.REL with a custom register ABI in place of the cross-TU
// CALL.ABS.NOINC, which is pinned to the conservative stack-heavy ABI-stable convention.
// CUDAHash.cu is dropped from Makefile SRC so it is not also compiled standalone.
#include "CUDAHash.cu"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// The kernel does NOT poll the global found-flag inside its hot loops. The finding warp
// publishes its result via atomicCAS + __threadfence_system + atomicExch(FOUND_READY); the host
// reads the flag between launches (see the launch loop in main) and stops scheduling further
// work once it is set. A find is a single terminal event, so the one in-flight launch simply
// runs to completion -- at most one launch of wasted compute per run. Keeping no in-kernel poll
// also keeps every launch on its normal per-thread state write-back path (no early-return-
// before-writeback, the historical prefix-skip hazard).

// Rare path (~2^-32 of keys): a word-2 filter hit. Recompute the full 160-bit digest with the
// untrimmed hash and compare all five words. Kept __forceinline__ so each call site keeps the
// exact `bool full = pref && ...` shape it had when the filter was word 0 -- the found path's
// loop structure is load-bearing (see the 2026-07-22 found-path-refactor regression).
__device__ __forceinline__ bool hash160_full_match(uint8_t prefix02_03, U256 x,
                                                   const uint32_t target_w[5])
{
    H160 h5 = getHash160_33_from_limbs(prefix02_03, x);
    return hash160_matches_full(h5.w, target_w);
}

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check_oneinv(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;

    // Filter on hash160 WORD 2, not word 0: word 2 is the cheapest of the five to produce
    // (its RIPEMD-160 inputs are final 7 rounds earlier -- see CUDAHash.cu). Any single word
    // is an equally selective 32-bit filter, so this is a free choice.
    const uint32_t target_prefix = c_target_words[2];

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
    { const uint64_t idx = gid*4 + 0; x1[0] = Px[idx]; y1[0] = Py[idx]; S[0] = start_scalars[idx]; }
    { const uint64_t idx = gid*4 + 1; x1[1] = Px[idx]; y1[1] = Py[idx]; S[1] = start_scalars[idx]; }
    { const uint64_t idx = gid*4 + 2; x1[2] = Px[idx]; y1[2] = Py[idx]; S[2] = start_scalars[idx]; }
    { const uint64_t idx = gid*4 + 3; x1[3] = Px[idx]; y1[3] = Py[idx]; S[3] = start_scalars[idx]; }
    uint64_t rem[4];
    rem[0] = counts256[gid*4 + 0];
    rem[1] = counts256[gid*4 + 1];
    rem[2] = counts256[gid*4 + 2];
    rem[3] = counts256[gid*4 + 3];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
        Rx[gid*4+0] = x1[0]; Ry[gid*4+0] = y1[0];
        Rx[gid*4+1] = x1[1]; Ry[gid*4+1] = y1[1];
        Rx[gid*4+2] = x1[2]; Ry[gid*4+2] = y1[2];
        Rx[gid*4+3] = x1[3]; Ry[gid*4+3] = y1[3];
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        {
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            uint32_t hw2 = getHash160_w2_from_limbs(prefix, u256_of(x1));   // by-value ABI: 1 reg out
            bool pref = (hw2 == target_prefix);
            if (__any_sync(full_mask, pref)) {
                bool full = pref && hash160_full_match(prefix, u256_of(x1), c_target_words);
                if (full) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->scalar[0]=S[0]; d_found_result->scalar[1]=S[1]; d_found_result->scalar[2]=S[2]; d_found_result->scalar[3]=S[3];
                        d_found_result->Rx[0]=x1[0]; d_found_result->Rx[1]=x1[1]; d_found_result->Rx[2]=x1[2]; d_found_result->Rx[3]=x1[3];
                        d_found_result->Ry[0]=y1[0]; d_found_result->Ry[1]=y1[1]; d_found_result->Ry[2]=y1[2]; d_found_result->Ry[3]=y1[3];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                // Abandon the launch only on a real full-hash160 match; a bare 32-bit prefix
                // collision must keep scanning, else the skipped write-back drops keys.
                if (__any_sync(full_mask, full)) { __syncwarp(full_mask); WARP_FLUSH_HASHES(); return; }
            }
        }

        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

        // Read the __constant__ operand straight into the subtract instead of copying it into the
        // destination first. ModSub256 takes uint64_t* and __constant__ is a memory space, not a
        // const qualifier, so c_Jx decays and binds directly. Same pattern at every dx site below.
        ModSub256(acc, c_Jx, x1);
        subp[half-1][0] = acc[0]; subp[half-1][1] = acc[1]; subp[half-1][2] = acc[2]; subp[half-1][3] = acc[3];

        for (int i = half - 2; i >= 0; --i) {
            ModSub256(tmp, &c_Gx[(size_t)(i+1)*4], x1);
            _ModMult(acc, acc, tmp);
            subp[i][0] = acc[0]; subp[i][1] = acc[1]; subp[i][2] = acc[2]; subp[i][3] = acc[3];
        }

        // inverse MUST stay uint64_t[5] even though only 4 limbs are ever read: InvModP writes
        // res[8] (RCGpuUtils.h:529), the low half of inverse[4] -- a [4] declaration is a 4-byte
        // OOB store into whatever the allocator put next. See ec_backend.cuh:93.
        uint64_t inverse[5];
        ModSub256(inverse, c_Gx, x1);      // d0 = c_Gx[0] - x1, straight into inverse[0..3]
        _ModMult(inverse, subp[0]);
        // No zero-init of inverse[4] here: InvModP sets res[8]=0 itself BEFORE its first read of
        // res[0..7] (RCGpuUtils.h:529-531), and res[9] (the high half) is never read or written --
        // every downstream op on res is 288-bit, res[0..8]. After this, inverse is read as 4 limbs
        // only (the running _ModMult at each loop tail, and the final lam multiply).
        _ModInv(inverse);

        for (int i = 0; i < half - 1; ++i) {
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
                px_i[0]=c_Gx[(size_t)i*4+0]; px_i[1]=c_Gx[(size_t)i*4+1]; px_i[2]=c_Gx[(size_t)i*4+2]; px_i[3]=c_Gx[(size_t)i*4+3];
                py_i[0]=c_Gy[(size_t)i*4+0]; py_i[1]=c_Gy[(size_t)i*4+1]; py_i[2]=c_Gy[(size_t)i*4+2]; py_i[3]=c_Gy[(size_t)i*4+3];

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256_2(px3, px3, x1, px_i);   // px3 = lam^2 - x1 - px_i (fused, one reduction)

                ModSub256(s, x1, px3); 
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint32_t hw2 = getHash160_w2_from_limbs(odd?0x03:0x02, u256_of(px3));
                bool pref = (hw2 == target_prefix);
                if (__any_sync(full_mask, pref)) {
                    bool full = pref && hash160_full_match(odd?0x03:0x02, u256_of(px3), c_target_words);
                    if (full) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; fs[0]=S[0]; fs[1]=S[1]; fs[2]=S[2]; fs[3]=S[3];
                            uint64_t addv=(uint64_t)(i+1);
                            { uint64_t old=fs[0]; fs[0]=old+addv; addv=(fs[0]<old)?1ull:0ull; }
                            { uint64_t old=fs[1]; fs[1]=old+addv; addv=(fs[1]<old)?1ull:0ull; }
                            { uint64_t old=fs[2]; fs[2]=old+addv; addv=(fs[2]<old)?1ull:0ull; }
                            { uint64_t old=fs[3]; fs[3]=old+addv; addv=(fs[3]<old)?1ull:0ull; }
                            d_found_result->scalar[0]=fs[0]; d_found_result->scalar[1]=fs[1]; d_found_result->scalar[2]=fs[2]; d_found_result->scalar[3]=fs[3];
                            d_found_result->Rx[0]=px3[0]; d_found_result->Rx[1]=px3[1]; d_found_result->Rx[2]=px3[2]; d_found_result->Rx[3]=px3[3];
                           
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
                            d_found_result->Ry[0]=y3[0]; d_found_result->Ry[1]=y3[1]; d_found_result->Ry[2]=y3[2]; d_found_result->Ry[3]=y3[3];
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    // Abandon the launch only on a real full-hash160 match; a bare 32-bit prefix
                    // collision must keep scanning, else the skipped write-back drops keys.
                    if (__any_sync(full_mask, full)) { __syncwarp(full_mask); WARP_FLUSH_HASHES(); return; }
                }
            }

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
                px_i[0]=c_Gx[(size_t)i*4+0]; px_i[1]=c_Gx[(size_t)i*4+1]; px_i[2]=c_Gx[(size_t)i*4+2]; px_i[3]=c_Gx[(size_t)i*4+3];
                py_i[0]=c_Gy[(size_t)i*4+0]; py_i[1]=c_Gy[(size_t)i*4+1]; py_i[2]=c_Gy[(size_t)i*4+2]; py_i[3]=c_Gy[(size_t)i*4+3];
                ModNeg256(py_i, py_i); 

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256_2(px3, px3, x1, px_i);   // px3 = lam^2 - x1 - px_i (fused, one reduction)

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint32_t hw2 = getHash160_w2_from_limbs(odd?0x03:0x02, u256_of(px3));
                bool pref = (hw2 == target_prefix);
                if (__any_sync(full_mask, pref)) {
                    bool full = pref && hash160_full_match(odd?0x03:0x02, u256_of(px3), c_target_words);
                    if (full) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; fs[0]=S[0]; fs[1]=S[1]; fs[2]=S[2]; fs[3]=S[3];
                            uint64_t sub=(uint64_t)(i+1);
                            { uint64_t old=fs[0]; fs[0]=old-sub; sub=(old<sub)?1ull:0ull; }
                            { uint64_t old=fs[1]; fs[1]=old-sub; sub=(old<sub)?1ull:0ull; }
                            { uint64_t old=fs[2]; fs[2]=old-sub; sub=(old<sub)?1ull:0ull; }
                            { uint64_t old=fs[3]; fs[3]=old-sub; sub=(old<sub)?1ull:0ull; }
                            d_found_result->scalar[0]=fs[0]; d_found_result->scalar[1]=fs[1]; d_found_result->scalar[2]=fs[2]; d_found_result->scalar[3]=fs[3];
                            d_found_result->Rx[0]=px3[0]; d_found_result->Rx[1]=px3[1]; d_found_result->Rx[2]=px3[2]; d_found_result->Rx[3]=px3[3];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
                            d_found_result->Ry[0]=y3[0]; d_found_result->Ry[1]=y3[1]; d_found_result->Ry[2]=y3[2]; d_found_result->Ry[3]=y3[3];
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    // Abandon the launch only on a real full-hash160 match; a bare 32-bit prefix
                    // collision must keep scanning, else the skipped write-back drops keys.
                    if (__any_sync(full_mask, full)) { __syncwarp(full_mask); WARP_FLUSH_HASHES(); return; }
                }
            }

            uint64_t gxmi[4];
            ModSub256(gxmi, &c_Gx[(size_t)i*4], x1);
            _ModMult(inverse, inverse, gxmi);
        }

        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4];
            uint64_t px_i[4], py_i[4];
            px_i[0]=c_Gx[(size_t)i*4+0]; px_i[1]=c_Gx[(size_t)i*4+1]; px_i[2]=c_Gx[(size_t)i*4+2]; px_i[3]=c_Gx[(size_t)i*4+3];
            py_i[0]=c_Gy[(size_t)i*4+0]; py_i[1]=c_Gy[(size_t)i*4+1]; py_i[2]=c_Gy[(size_t)i*4+2]; py_i[3]=c_Gy[(size_t)i*4+3];
            ModNeg256(py_i, py_i);

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256_2(px3, px3, x1, px_i);   // px3 = lam^2 - x1 - px_i (fused, one reduction)

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, y1, &odd);

            uint32_t hw2 = getHash160_w2_from_limbs(odd?0x03:0x02, u256_of(px3));
            bool pref = (hw2 == target_prefix);
            if (__any_sync(full_mask, pref)) {
                bool full = pref && hash160_full_match(odd?0x03:0x02, u256_of(px3), c_target_words);
                if (full) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; fs[0]=S[0]; fs[1]=S[1]; fs[2]=S[2]; fs[3]=S[3];
                        uint64_t sub=(uint64_t)half;
                        { uint64_t old=fs[0]; fs[0]=old-sub; sub=(old<sub)?1ull:0ull; }
                        { uint64_t old=fs[1]; fs[1]=old-sub; sub=(old<sub)?1ull:0ull; }
                        { uint64_t old=fs[2]; fs[2]=old-sub; sub=(old<sub)?1ull:0ull; }
                        { uint64_t old=fs[3]; fs[3]=old-sub; sub=(old<sub)?1ull:0ull; }
                        d_found_result->scalar[0]=fs[0]; d_found_result->scalar[1]=fs[1]; d_found_result->scalar[2]=fs[2]; d_found_result->scalar[3]=fs[3];
                        d_found_result->Rx[0]=px3[0]; d_found_result->Rx[1]=px3[1]; d_found_result->Rx[2]=px3[2]; d_found_result->Rx[3]=px3[3];
                        uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
                        d_found_result->Ry[0]=y3[0]; d_found_result->Ry[1]=y3[1]; d_found_result->Ry[2]=y3[2]; d_found_result->Ry[3]=y3[3];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                // Abandon the launch only on a real full-hash160 match; a bare 32-bit prefix
                // collision must keep scanning, else the skipped write-back drops keys.
                if (__any_sync(full_mask, full)) { __syncwarp(full_mask); WARP_FLUSH_HASHES(); return; }
            }

            uint64_t last_dx[4];
            ModSub256(last_dx, &c_Gx[(size_t)i*4], x1);
            _ModMult(inverse, inverse, last_dx);
        }

        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
            ModSub256(Jy_minus_y1, c_Jy, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            uint64_t Jx_local[4]; Jx_local[0]=c_Jx[0]; Jx_local[1]=c_Jx[1]; Jx_local[2]=c_Jx[2]; Jx_local[3]=c_Jx[3];
            ModSub256_2(x3, x3, x1, Jx_local);   // x3 = lam^2 - x1 - Jx (fused, one reduction)

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

            x1[0] = x3[0]; y1[0] = y3[0];
            x1[1] = x3[1]; y1[1] = y3[1];
            x1[2] = x3[2]; y1[2] = y3[2];
            x1[3] = x3[3]; y1[3] = y3[3];
        }

        {
            uint64_t addv=(uint64_t)B;
            { uint64_t old=S[0]; S[0]=old+addv; addv=(S[0]<old)?1ull:0ull; }
            { uint64_t old=S[1]; S[1]=old+addv; addv=(S[1]<old)?1ull:0ull; }
            { uint64_t old=S[2]; S[2]=old+addv; addv=(S[2]<old)?1ull:0ull; }
            { uint64_t old=S[3]; S[3]=old+addv; addv=(S[3]<old)?1ull:0ull; }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        local_hashes += (unsigned int)B; MAYBE_WARP_FLUSH();  // count the whole batch at once (B | 65536 keeps the 64Ki flush cadence)
        ++batches_done;
    }

    Rx[gid*4+0] = x1[0]; Ry[gid*4+0] = y1[0]; counts256[gid*4+0] = rem[0]; start_scalars[gid*4+0] = S[0];
    Rx[gid*4+1] = x1[1]; Ry[gid*4+1] = y1[1]; counts256[gid*4+1] = rem[1]; start_scalars[gid*4+1] = S[1];
    Rx[gid*4+2] = x1[2]; Ry[gid*4+2] = y1[2]; counts256[gid*4+2] = rem[2]; start_scalars[gid*4+2] = S[2];
    Rx[gid*4+3] = x1[3]; Ry[gid*4+3] = y1[3]; counts256[gid*4+3] = rem[3]; start_scalars[gid*4+3] = S[3];
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

// hexToLE64/hexToHash160/formatHex256/ld_from_u256 come from CUDAUtils.h,
// decode_p2pkh_address from sha256.h, formatCompressedPubHex from CUDAUtils.h, and
// scalarMulKernelBase from CUDAMath.h/CUDAStructures.h -- all included above, so the
// former local re-declarations here were redundant.
int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string target_hash_hex, range_hex, address_b58;
    // Defaults = the configuration this project actually benchmarks. The former 128,8 was never
    // exercised by any A/B (every benchmark passes --grid explicitly) and left roughly half the
    // card on the floor: at batches_per_sm=8, userUpper = SM*8*256 caps threadsTotal at 2^18 =
    // 1024 blocks against 340 resident (__launch_bounds__(256,2) on 170 SMs) = 3.01 waves ->
    // ceil 4 -> ~75% launch utilization. At 512 the same arithmetic gives 65,536 blocks over
    // 193 waves = 99.87%.
    //
    // batch size 1024 over 512: MEASURED +0.601% keys/cycle (5 reps) and +0.484% (6 reps,
    // balanced), pooled +0.537% +- 0.063 over 11 reps, t=+8.49, total separation in both runs,
    // clock flat at a pinned 600 W, A/A null clean at -0.088%. The only B-dependent term is the
    // one safegcd InvModP per batch; mul/key, sqr/key, ModSub/key and subp[] traffic (32 B/key)
    // are exactly B-invariant, and the subp frame is sized by MAX_BATCH_SIZE rather than B, so
    // B=1024 costs no extra registers, frame or occupancy on any card.
    //
    // batches_per_sm=512 rests on the wave arithmetic above plus the campaign's entire measured
    // history (every A/B since the start ran at 512); it has NOT been A/B'd against 8 directly.
    uint32_t runtime_points_batch_size = 1024;
    uint32_t runtime_batches_per_sm    = 512;
    uint32_t slices_per_launch         = 64;

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        char* endp=nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp) return false;
        endp=nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp) return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--target-hash160" && i + 1 < argc) target_hash_hex = argv[++i];
        else if (arg == "--address"        && i + 1 < argc) address_b58     = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
        }
        else if (arg == "--slices" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<20)) {
                std::cerr << "Error: --slices must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [--grid A,B] [--slices N]\n";
        return EXIT_FAILURE;
    }
    if (!target_hash_hex.empty() && !address_b58.empty()) {
        std::cerr << "Error: provide either --address or --target-hash160, not both.\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58, target_hash160)) {
            std::cerr << "Error: invalid P2PKH address\n"; return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n"; return EXIT_FAILURE;
        }
    }

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (kernel limit).\n";
        return EXIT_FAILURE;
    }

    uint64_t range_len[4]; sub256(range_end, range_start, range_len); add256_u64(range_len, 1ull, range_len);

    auto is_zero_256 = [](const uint64_t a[4])->bool { return (a[0]|a[1]|a[2]|a[3]) == 0ull; };
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if (is_zero_256(a)) return false;
        uint64_t am1[4]; uint64_t borrow = 1ull;
        { uint64_t v = a[0] - borrow; borrow = (a[0] < borrow) ? 1ull : 0ull; am1[0] = v; }
        { uint64_t v = a[1] - borrow; borrow = (a[1] < borrow) ? 1ull : 0ull; am1[1] = v; }
        { uint64_t v = a[2] - borrow; borrow = (a[2] < borrow) ? 1ull : 0ull; am1[2] = v; }
        { uint64_t v = a[3] - borrow; borrow = (a[3] < borrow) ? 1ull : 0ull; am1[3] = v; }
        uint64_t and0=a[0]&am1[0], and1=a[1]&am1[1], and2=a[2]&am1[2], and3=a[3]&am1[3];
        return (and0|and1|and2|and3)==0ull;
    };
    if (!is_power_of_two_256(range_len)) {
        std::cerr << "Error: range length (end - start + 1) must be a power of two.\n"; return EXIT_FAILURE;
    }
    uint64_t len_minus1[4];
    {   uint64_t borrow=1ull;
        { uint64_t v=range_len[0]-borrow; borrow=(range_len[0]<borrow)?1ull:0ull; len_minus1[0]=v; }
        { uint64_t v=range_len[1]-borrow; borrow=(range_len[1]<borrow)?1ull:0ull; len_minus1[1]=v; }
        { uint64_t v=range_len[2]-borrow; borrow=(range_len[2]<borrow)?1ull:0ull; len_minus1[2]=v; }
        { uint64_t v=range_len[3]-borrow; borrow=(range_len[3]<borrow)?1ull:0ull; len_minus1[3]=v; }
    }
    {   uint64_t and0 = range_start[0] & len_minus1[0];
        uint64_t and1 = range_start[1] & len_minus1[1];
        uint64_t and2 = range_start[2] & len_minus1[2];
        uint64_t and3 = range_start[3] & len_minus1[3];
        if ((and0|and1|and2|and3) != 0ull) {
            std::cerr << "Error: start must be aligned to the range length.\n"; return EXIT_FAILURE;
        }
    }

    // Enable mapped (zero-copy) host allocations before the CUDA context is created (this is
    // the first CUDA call in main). Lets the host poll loop read the found-flag and hash
    // counter straight from pinned host memory instead of a cudaMemcpy per poll. No-op on the
    // UVA systems this targets (mapped alloc is always available there); harmless if it fails.
    (void)cudaSetDeviceFlags(cudaDeviceMapHost);
    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int threadsPerBlock=256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock=prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock=32;

    const uint64_t bytesPerThread = 2ull*4ull*sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    uint64_t q_div_batch[4], r_div_batch = 0ull;
    divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull) {
        // Reachable on a short range now that the default batch is 1024 rather than 128, so say
        // how to fix it instead of only what is wrong.
        std::cerr << "Error: range length must be divisible by batch size (" << runtime_points_batch_size << ").\n"
                  << "       Pass a smaller points-per-batch, e.g. --grid 128,512, or widen the range.\n";
        return EXIT_FAILURE;
    }
    bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
    uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
    if (!q_fits_u64) { std::cerr << "Error: total batches too large for u64.\n"; return EXIT_FAILURE; }

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    auto pick_threads_total = [&](uint64_t upper)->uint64_t {
        if (upper < (uint64_t)threadsPerBlock) return 0ull;
        uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
        uint64_t q = total_batches_u64;
        while (t >= (uint64_t)threadsPerBlock) {
            if ((q % t) == 0ull) return t;
            t -= (uint64_t)threadsPerBlock;
        }
        return 0ull;
    };

    uint64_t upper = maxThreadsByMem;
    if (total_batches_u64 < upper) upper = total_batches_u64;
    if (userUpper         < upper) upper = userUpper;

    uint64_t threadsTotal = pick_threads_total(upper);
    if (threadsTotal == 0ull) {
        std::cerr << "Error: failed to pick threadsTotal satisfying divisibility.\n";
        return EXIT_FAILURE;
    }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    uint64_t per_thread_cnt[4]; uint64_t r_u64 = 0ull;
    divmod_256_by_u64(range_len, threadsTotal, per_thread_cnt, r_u64);
    if (r_u64 != 0ull) { std::cerr << "Internal error: range_len not divisible by threadsTotal.\n"; return EXIT_FAILURE; }
    {   uint64_t qq[4], rr=0ull;
        divmod_256_by_u64(per_thread_cnt, (uint64_t)runtime_points_batch_size, qq, rr);
        if (rr != 0ull) { std::cerr << "Internal error: per-thread count is not a multiple of batch size.\n"; return EXIT_FAILURE; }
    }

    uint64_t* h_counts256     = nullptr;
    uint64_t* h_start_scalars = nullptr;
    cudaHostAlloc(&h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    for (uint64_t i = 0; i < threadsTotal; ++i) {
        h_counts256[i*4+0] = per_thread_cnt[0];
        h_counts256[i*4+1] = per_thread_cnt[1];
        h_counts256[i*4+2] = per_thread_cnt[2];
        h_counts256[i*4+3] = per_thread_cnt[3];
    }

    const uint32_t B = runtime_points_batch_size;
    const uint32_t half = B >> 1;
    {
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t Sc[4]; add256_u64(cur, (uint64_t)half, Sc); 
            h_start_scalars[i*4+0] = Sc[0];
            h_start_scalars[i*4+1] = Sc[1];
            h_start_scalars[i*4+2] = Sc[2];
            h_start_scalars[i*4+3] = Sc[3];

            uint64_t next[4]; add256(cur, per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    }

    {
        uint32_t target_words[5];
        target_words[0] = (uint32_t)target_hash160[ 0] | ((uint32_t)target_hash160[ 1] << 8) | ((uint32_t)target_hash160[ 2] << 16) | ((uint32_t)target_hash160[ 3] << 24);
        target_words[1] = (uint32_t)target_hash160[ 4] | ((uint32_t)target_hash160[ 5] << 8) | ((uint32_t)target_hash160[ 6] << 16) | ((uint32_t)target_hash160[ 7] << 24);
        target_words[2] = (uint32_t)target_hash160[ 8] | ((uint32_t)target_hash160[ 9] << 8) | ((uint32_t)target_hash160[10] << 16) | ((uint32_t)target_hash160[11] << 24);
        target_words[3] = (uint32_t)target_hash160[12] | ((uint32_t)target_hash160[13] << 8) | ((uint32_t)target_hash160[14] << 16) | ((uint32_t)target_hash160[15] << 24);
        target_words[4] = (uint32_t)target_hash160[16] | ((uint32_t)target_hash160[17] << 8) | ((uint32_t)target_hash160[18] << 16) | ((uint32_t)target_hash160[19] << 24);
        cudaMemcpyToSymbol(c_target_words, target_words, sizeof(target_words));
    }

    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr, *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    int *d_found_flag=nullptr; FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr; unsigned int *d_any_left=nullptr;
    // Zero-copy (pinned, mapped) host views of the found-flag and hash counter: the host poll
    // loop reads *host_found / *h_hashes directly (no cudaMemcpy per poll); the kernel writes
    // through the matching device pointers (d_found_flag / d_hashes_accum) that
    // cudaHostGetDevicePointer maps onto the SAME pinned pages.
    int *host_found=nullptr;
    unsigned long long *h_hashes=nullptr;

    auto ck = [](cudaError_t e, const char* msg){
        if (e != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    ck(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_start_scalars)");
    ck(cudaMalloc(&d_Px,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Px)");
    ck(cudaMalloc(&d_Py,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Py)");
    ck(cudaMalloc(&d_Rx,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Rx)");
    ck(cudaMalloc(&d_Ry,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Ry)");
    ck(cudaMalloc(&d_counts256,    threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_counts256)");
    ck(cudaHostAlloc((void**)&host_found, sizeof(int), cudaHostAllocMapped),                   "cudaHostAlloc(host_found)");
    ck(cudaHostGetDevicePointer((void**)&d_found_flag, host_found, 0),                         "cudaHostGetDevicePointer(found)");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)),                 "cudaMalloc(d_found_result)");
    ck(cudaHostAlloc((void**)&h_hashes, sizeof(unsigned long long), cudaHostAllocMapped),      "cudaHostAlloc(h_hashes)");
    ck(cudaHostGetDevicePointer((void**)&d_hashes_accum, h_hashes, 0),                         "cudaHostGetDevicePointer(hashes)");
    ck(cudaMalloc(&d_any_left,     sizeof(unsigned int)),                "cudaMalloc(d_any_left)");

    ck(cudaMemcpy(d_start_scalars, h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start_scalars");
    ck(cudaMemcpy(d_counts256,     h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts256");
    *host_found = FOUND_NONE;   // mapped host writes, visible to the kernel launched later
    *h_hashes   = 0ull;

    {
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
        ck(cudaGetLastError(), "scalarMulKernelBase launch");
    }

    {
        uint64_t* h_scalars_half = nullptr;
        cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
        ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_scalars_half)");
        ck(cudaMalloc(&d_Gx_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gx_half)");
        ck(cudaMalloc(&d_Gy_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gy_half)");
        ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy half scalars");

        int blocks_scal = (int)((half + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(half) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(half) launch");

        uint64_t* h_Gx_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        uint64_t* h_Gy_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx_half");
        ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy_half");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

        cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
        cudaFreeHost(h_scalars_half);
        std::free(h_Gx_half); std::free(h_Gy_half);
    }
    {
        uint64_t* h_scalarB = nullptr;
        cudaHostAlloc(&h_scalarB, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalarB, 0, 4 * sizeof(uint64_t));
        h_scalarB[0] = (uint64_t)B;

        uint64_t *d_scalarB=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
        ck(cudaMalloc(&d_scalarB, 4 * sizeof(uint64_t)), "cudaMalloc(d_scalarB)");
        ck(cudaMalloc(&d_Jx,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jx)");
        ck(cudaMalloc(&d_Jy,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jy)");
        ck(cudaMemcpy(d_scalarB, h_scalarB, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalarB");

        scalarMulKernelBase<<<1, 1>>>(d_scalarB, d_Jx, d_Jy, 1);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(B) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(B) launch");

        uint64_t hJx[4], hJy[4];
        ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
        ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
        ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");

        cudaFree(d_scalarB); cudaFree(d_Jx); cudaFree(d_Jy);
        cudaFreeHost(h_scalarB);
    }

    size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
    size_t usedB = totalB - freeB;
    double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "SM"                << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(20) << "ThreadsPerBlock"   << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(20) << "Blocks"            << " : " << (int)(threadsTotal / (uint64_t)threadsPerBlock) << "\n";
    std::cout << std::left << std::setw(20) << "Points batch size" << " : " << B << "\n";
    std::cout << std::left << std::setw(20) << "Batches/SM"        << " : " << runtime_batches_per_sm << "\n";
    std::cout << std::left << std::setw(20) << "Batches/launch"    << " : " << slices_per_launch << " (per thread)\n";
    std::cout << std::left << std::setw(20) << "Memory utilization"<< " : "
              << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)usedB) << " / " << human_bytes((double)totalB) << ")\n";
    std::cout << "------------------------------------------------------- \n";
    std::cout << std::left << std::setw(20) << "Total threads"     << " : " << (uint64_t)threadsTotal << "\n\n";
    std::cout << "======== Phase-1: BruteForce ==========================\n";

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");

    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv, cudaFuncCachePreferL1);

    auto t0 = std::chrono::high_resolution_clock::now();

    // ---- Host thread split ---------------------------------------------------------------
    // THREE threads of execution, with one hard ownership rule:
    //   * tWorker   (spawned) owns EVERY CUDA call for the duration of the search, plus the
    //     d_Px/d_Rx ping-pong buffers.
    //   * tListener (spawned) owns SIGINT translation and stdout, and makes NO CUDA call.
    //   * main blocks in the joins and touches neither stdout nor CUDA until both are gone.
    // That rule is what makes teardown provably safe: join() returns only after the thread's
    // lambda has returned, so no CUDA call can be in flight or issued afterwards, and it
    // supplies the happens-before that publishes worker_cuda_err / launch_error /
    // completed_all and the mapped-page writes to main.
    //
    // stdout stays single-owner throughout: tListener alone prints between the spawns and the
    // joins, so the Ctrl+C banner and the \r progress line cannot interleave and corrupt each
    // other. main resumes printing only after tListener.join().
    //
    // The listener polls at 1 ms. A signal handler may not safely do anything but assign to a
    // volatile sig_atomic_t, so "listening" for Ctrl+C is necessarily polling the flag the
    // handler wrote, and this interval is what bounds how long the user waits to SEE the banner.
    // It does NOT bound how long the search takes to STOP: the worker tests g_stop only at slice
    // boundaries (multi-second at the default slices_per_launch), so the poll governs feedback
    // latency, not shutdown latency. The progress line is gated on its own 1 s elapsed check, so
    // raising the poll rate does not change how much this thread prints.
    std::atomic<bool> g_stop{false};
    std::atomic<bool> completed_all{false};
    std::atomic<bool> launch_error{false};
    cudaError_t worker_cuda_err = cudaSuccess;   // plain; published to main by tWorker.join()
    static_assert(std::atomic<bool>::is_always_lock_free, "g_stop must be lock-free");

    // NOTE: ck() calls std::exit() and MUST NOT be used inside tWorker -- std::exit from a
    // spawned thread runs atexit handlers and static destructors, tearing down the statically
    // linked CUDA runtime while main is still live. Every in-loop CUDA call below therefore
    // captures its error into worker_cuda_err instead of exiting. ck() stays correct for the
    // setup calls above, which all run on main before any thread exists.
    auto worker = [&]() {
        while (!g_stop.load(std::memory_order_relaxed)) {
            unsigned int zeroU = 0u;
            cudaError_t e = cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, streamKernel);
            if (e != cudaSuccess) { worker_cuda_err = e; launch_error.store(true); break; }

            kernel_point_add_and_check_oneinv<<<blocks, threadsPerBlock, 0, streamKernel>>>(
                d_Px, d_Py, d_Rx, d_Ry,
                d_start_scalars, d_counts256,
                threadsTotal,
                B,
                slices_per_launch,
                d_found_flag, d_found_result,
                d_hashes_accum,
                d_any_left
            );
            cudaError_t launchErr = cudaGetLastError();
            if (launchErr != cudaSuccess) { worker_cuda_err = launchErr; launch_error.store(true); break; }

            // Wait shape kept BYTE-FOR-BYTE as it was pre-split (query + 1 ms sleep, then the
            // stream sync). This commit moves WHERE code runs, never HOW it waits -- swapping in
            // a bare blocking cudaStreamSynchronize would change host busy-state and the
            // resulting package-power shift would be misread as a threading effect.
            bool found_now = false;
            while (true) {
                if (*(volatile int*)host_found == FOUND_READY) { found_now = true; break; }

                cudaError_t qs = cudaStreamQuery(streamKernel);
                if (qs == cudaSuccess) break;
                if (qs != cudaErrorNotReady) {
                    worker_cuda_err = qs; (void)cudaGetLastError();
                    launch_error.store(true); found_now = true; break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            cudaError_t se = cudaStreamSynchronize(streamKernel);
            if (se != cudaSuccess) { worker_cuda_err = se; launch_error.store(true); break; }

            // Stop BEFORE the swap. One swap per COMPLETED launch is what keeps each thread's
            // point in lockstep with its scalar; breaking out after a swap is the exact shape of
            // the historical prefix-collision key-skip bug (fixed in PR#5).
            //
            // This tests g_stop and deliberately NOT g_sigint. Ctrl+C therefore reaches the
            // worker only via tListener's 1 ms poll, i.e. up to 1 ms late. At the default
            // slices_per_launch a launch is multi-second, so that is invisible; with a very
            // small slice count an extra launch may complete before the stop lands (harmless --
            // parity holds on each, so no key range is skipped). Reading g_sigint here would
            // erase even that, but would let the worker set g_stop and exit before tListener had
            // a chance to print the interrupt banner, which is the silent-exit defect this
            // design exists to avoid. Latency is the cheaper cost.
            if (found_now || g_stop.load(std::memory_order_relaxed)) break;

            unsigned int h_any = 0u;
            cudaError_t e2 = cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            if (e2 != cudaSuccess) { worker_cuda_err = e2; launch_error.store(true); break; }

            std::swap(d_Px, d_Rx);
            std::swap(d_Py, d_Ry);

            if (h_any == 0u) { completed_all.store(true); break; }
        }
        g_stop.store(true);   // wake tListener out of its poll loop on EVERY exit path
    };

    // tListener: SIGINT listener + reporter. Deliberately contains no CUDA call.
    auto listener = [&]() {
        auto tLast = t0;
        unsigned long long lastHashes = 0ull;
        const long double total_keys_ld = ld_from_u256(range_len);   // hoisted out of the print
        bool announced_sigint = false;

        while (!g_stop.load(std::memory_order_relaxed)) {
            // Sole g_sigint -> g_stop translation point in the program. Keeping it here (rather
            // than also testing g_sigint in the worker) is what guarantees the banner prints
            // exactly once: the worker cannot race ahead and stop the search silently.
            if (g_sigint && !announced_sigint) {
                announced_sigint = true;
                std::cerr << "\n[Ctrl+C] Interrupt received. Finishing current kernel slice and exiting...\n";
                g_stop.store(true);
            }

            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 1.0) {
                unsigned long long hashes_now = *(volatile unsigned long long*)h_hashes;  // zero-copy read (no cudaMemcpy)
                double delta = (double)(hashes_now - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                long double prog = total_keys_ld > 0.0L ? ((long double)hashes_now / total_keys_ld) * 100.0L : 0.0L;
                if (prog > 100.0L) prog = 100.0L;
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
                          << " Mkeys/s | Count: " << hashes_now
                          << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %";
                std::cout.flush();
                lastHashes = hashes_now; tLast = now;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        std::cout.flush();
    };

    std::thread tWorker(worker);
    std::thread tListener(listener);

    // tWorker first: it is the thread that sets g_stop on every exit path, so tListener is
    // guaranteed to be within one 1 ms poll of returning by the time this join lands. Joining
    // in the other order would work too, but this way main is never blocked behind the thread
    // that only prints.
    tWorker.join();
    tListener.join();

    // MUST stay after the join. The worker's error-exit paths break out WITHOUT a stream sync,
    // so device work may still be queued; this is what drains it before the frees below.
    cudaDeviceSynchronize();
    std::cout << "\n";

    int h_found_flag = *(volatile int*)host_found;   // zero-copy read (kernel published via __threadfence_system + atomicExch)

    int exit_code = EXIT_SUCCESS;

    // Precedence: found > cuda-error > Ctrl+C > exhausted > terminated.
    // The launch_error branch is NEW and fixes a real bug that predates this refactor: a kernel
    // launch failure (or a cudaStreamQuery error) used to print at most a message, break the
    // loop, and fall through to "TERMINATED" with exit_code still EXIT_SUCCESS -- i.e. a CUDA
    // failure returned 0 and any script driving this binary saw a clean run.
    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read found_result");
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
    } else if (launch_error.load()) {
        std::cerr << "======== CUDA ERROR ===================================\n";
        std::cerr << "Search aborted by a CUDA error: " << cudaGetErrorString(worker_cuda_err) << "\n";
        exit_code = EXIT_FAILURE;
    } else if (g_sigint) {
        std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
        std::cout << "Search was interrupted by user. Partial progress above.\n";
        exit_code = 130;
    } else if (completed_all.load()) {
        std::cout << "======== KEY NOT FOUND (exhaustive) ===================\n";
        std::cout << "Target hash160 was not found within the specified range.\n";
    } else {
        std::cout << "======== TERMINATED ===================================\n";
    }

    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py); cudaFree(d_Rx); cudaFree(d_Ry);
    cudaFree(d_counts256); cudaFree(d_found_result); cudaFree(d_any_left);
    cudaFreeHost(host_found); cudaFreeHost(h_hashes);   // d_found_flag/d_hashes_accum are mapped views, not cudaMalloc'd
    cudaStreamDestroy(streamKernel);

    if (h_start_scalars) cudaFreeHost(h_start_scalars);
    if (h_counts256)     cudaFreeHost(h_counts256);

    return exit_code;
}
