// ec_backend.cuh — secp256k1 field backend.
//
// MulModP/SqrModP use split-column fused-MAC 256x256 -> 512 product cores over RetiredCoder's
// reduction tail; InvModP is his safegcd inverse (third_party/RCKangaroo/RCGpuUtils.h, GPLv3),
// so a distributed binary is a GPLv3 derivative — see LICENSE.
//
// Blackwell fuses an adjacent mad.lo.cc.u32 / madc.hi.cc.u32 pair into one IMAD.WIDE.U32.X, but
// only when the accumulator pair is 64-bit ALIGNED. Splitting the limb products by column parity
// (even -> e[], odd -> o[], then p = e + (o << 32)) makes every pair aligned so the fusion sticks;
// the wide-multiply count is unchanged, the carry plumbing around it is what drops. Ported from
// tat-temp/cCUDAm@9671279 (branch f5).
//
// Representation contract:
//   * 256-bit values are 4x uint64_t, little-endian (limb[0] least significant).
//   * rmul/rsqr take inputs in [0,P) and return a value congruent mod P in [0,2^256) — "lazy",
//     not necessarily canonical. Harmless here: a non-canonical X would need a canonical form
//     below ~2^32 (~2^-224 of the space), which no real pubkey hits. Inputs at or above P are
//     outside the contract: the single-fold tail can leave a non-congruent residue.
//   * In-place aliasing (res==a and/or res==b) is safe: all inputs are read into locals first.
//   * rinv inverts a 256-bit value in place and touches a 9th u32 word, so callers must back it
//     with at least uint64_t[5] (as CUDAMath.h's fieldInv does). Output is canonical [0,P).
#pragma once

#include <cstdint>

namespace rck {

typedef unsigned long long u64;
typedef unsigned int       u32;
typedef int                i32;

// The PTX-asm helper macros and P_* constants defined inside this header are file-global rather
// than namespaced, but do not collide with CUDAMath.h's (UADDO/MADDC/...).
#include "third_party/RCKangaroo/RCGpuUtils.h"

// The product cores below are GENERATED — do not hand-edit between the markers.
// Regenerate with tools/fieldmath (see its README); correctness oracles live there too.
// Upstream's macro spelling, aliased onto the _32 macros RCGpuUtils.h already defines.
#define mad_lo_cc(res, a, b, c)   mad_lo_cc_32(res, a, b, c)
#define madc_hi_cc(res, a, b, c)  madc_hi_cc_32(res, a, b, c)
#define madc_lo_cc(res, a, b, c)  madc_lo_cc_32(res, a, b, c)

// ---- BEGIN GENERATED (tools/fieldmath) ----
// generated: even/odd column-split 256x256 -> 512, fused MAC form
__device__ __forceinline__ void mul512_split(uint32_t* p, const uint32_t* a, const uint32_t* b)
{
	uint32_t e[16], o[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) { e[i] = 0; o[i] = 0; }
	mad_lo_cc(e[6], a[5], b[1], e[6]);
	madc_hi_cc(e[7], a[5], b[1], e[7]);
	madc_lo_cc(e[8], a[6], b[2], e[8]);
	madc_hi_cc(e[9], a[6], b[2], e[9]);
	addc_32(e[10], e[10], 0);
	mad_lo_cc(e[6], a[6], b[0], e[6]);
	madc_hi_cc(e[7], a[6], b[0], e[7]);
	madc_lo_cc(e[8], a[7], b[1], e[8]);
	madc_hi_cc(e[9], a[7], b[1], e[9]);
	addc_32(e[10], e[10], 0);
	mad_lo_cc(e[4], a[3], b[1], e[4]);
	madc_hi_cc(e[5], a[3], b[1], e[5]);
	madc_lo_cc(e[6], a[3], b[3], e[6]);
	madc_hi_cc(e[7], a[3], b[3], e[7]);
	madc_lo_cc(e[8], a[4], b[4], e[8]);
	madc_hi_cc(e[9], a[4], b[4], e[9]);
	madc_lo_cc(e[10], a[6], b[4], e[10]);
	madc_hi_cc(e[11], a[6], b[4], e[11]);
	addc_32(e[12], e[12], 0);
	mad_lo_cc(e[4], a[4], b[0], e[4]);
	madc_hi_cc(e[5], a[4], b[0], e[5]);
	madc_lo_cc(e[6], a[4], b[2], e[6]);
	madc_hi_cc(e[7], a[4], b[2], e[7]);
	madc_lo_cc(e[8], a[5], b[3], e[8]);
	madc_hi_cc(e[9], a[5], b[3], e[9]);
	madc_lo_cc(e[10], a[7], b[3], e[10]);
	madc_hi_cc(e[11], a[7], b[3], e[11]);
	addc_32(e[12], e[12], 0);
	mad_lo_cc(e[2], a[1], b[1], e[2]);
	madc_hi_cc(e[3], a[1], b[1], e[3]);
	madc_lo_cc(e[4], a[1], b[3], e[4]);
	madc_hi_cc(e[5], a[1], b[3], e[5]);
	madc_lo_cc(e[6], a[1], b[5], e[6]);
	madc_hi_cc(e[7], a[1], b[5], e[7]);
	madc_lo_cc(e[8], a[2], b[6], e[8]);
	madc_hi_cc(e[9], a[2], b[6], e[9]);
	madc_lo_cc(e[10], a[4], b[6], e[10]);
	madc_hi_cc(e[11], a[4], b[6], e[11]);
	madc_lo_cc(e[12], a[6], b[6], e[12]);
	madc_hi_cc(e[13], a[6], b[6], e[13]);
	addc_32(e[14], e[14], 0);
	mad_lo_cc(e[2], a[2], b[0], e[2]);
	madc_hi_cc(e[3], a[2], b[0], e[3]);
	madc_lo_cc(e[4], a[2], b[2], e[4]);
	madc_hi_cc(e[5], a[2], b[2], e[5]);
	madc_lo_cc(e[6], a[2], b[4], e[6]);
	madc_hi_cc(e[7], a[2], b[4], e[7]);
	madc_lo_cc(e[8], a[3], b[5], e[8]);
	madc_hi_cc(e[9], a[3], b[5], e[9]);
	madc_lo_cc(e[10], a[5], b[5], e[10]);
	madc_hi_cc(e[11], a[5], b[5], e[11]);
	madc_lo_cc(e[12], a[7], b[5], e[12]);
	madc_hi_cc(e[13], a[7], b[5], e[13]);
	addc_32(e[14], e[14], 0);
	mad_lo_cc(e[0], a[0], b[0], e[0]);
	madc_hi_cc(e[1], a[0], b[0], e[1]);
	madc_lo_cc(e[2], a[0], b[2], e[2]);
	madc_hi_cc(e[3], a[0], b[2], e[3]);
	madc_lo_cc(e[4], a[0], b[4], e[4]);
	madc_hi_cc(e[5], a[0], b[4], e[5]);
	madc_lo_cc(e[6], a[0], b[6], e[6]);
	madc_hi_cc(e[7], a[0], b[6], e[7]);
	madc_lo_cc(e[8], a[1], b[7], e[8]);
	madc_hi_cc(e[9], a[1], b[7], e[9]);
	madc_lo_cc(e[10], a[3], b[7], e[10]);
	madc_hi_cc(e[11], a[3], b[7], e[11]);
	madc_lo_cc(e[12], a[5], b[7], e[12]);
	madc_hi_cc(e[13], a[5], b[7], e[13]);
	madc_lo_cc(e[14], a[7], b[7], e[14]);
	madc_hi_cc(e[15], a[7], b[7], e[15]);
	mad_lo_cc(o[6], a[6], b[1], o[6]);
	madc_hi_cc(o[7], a[6], b[1], o[7]);
	addc_32(o[8], o[8], 0);
	mad_lo_cc(o[6], a[7], b[0], o[6]);
	madc_hi_cc(o[7], a[7], b[0], o[7]);
	addc_32(o[8], o[8], 0);
	mad_lo_cc(o[4], a[4], b[1], o[4]);
	madc_hi_cc(o[5], a[4], b[1], o[5]);
	madc_lo_cc(o[6], a[4], b[3], o[6]);
	madc_hi_cc(o[7], a[4], b[3], o[7]);
	madc_lo_cc(o[8], a[6], b[3], o[8]);
	madc_hi_cc(o[9], a[6], b[3], o[9]);
	addc_32(o[10], o[10], 0);
	mad_lo_cc(o[4], a[5], b[0], o[4]);
	madc_hi_cc(o[5], a[5], b[0], o[5]);
	madc_lo_cc(o[6], a[5], b[2], o[6]);
	madc_hi_cc(o[7], a[5], b[2], o[7]);
	madc_lo_cc(o[8], a[7], b[2], o[8]);
	madc_hi_cc(o[9], a[7], b[2], o[9]);
	addc_32(o[10], o[10], 0);
	mad_lo_cc(o[2], a[2], b[1], o[2]);
	madc_hi_cc(o[3], a[2], b[1], o[3]);
	madc_lo_cc(o[4], a[2], b[3], o[4]);
	madc_hi_cc(o[5], a[2], b[3], o[5]);
	madc_lo_cc(o[6], a[2], b[5], o[6]);
	madc_hi_cc(o[7], a[2], b[5], o[7]);
	madc_lo_cc(o[8], a[4], b[5], o[8]);
	madc_hi_cc(o[9], a[4], b[5], o[9]);
	madc_lo_cc(o[10], a[6], b[5], o[10]);
	madc_hi_cc(o[11], a[6], b[5], o[11]);
	addc_32(o[12], o[12], 0);
	mad_lo_cc(o[2], a[3], b[0], o[2]);
	madc_hi_cc(o[3], a[3], b[0], o[3]);
	madc_lo_cc(o[4], a[3], b[2], o[4]);
	madc_hi_cc(o[5], a[3], b[2], o[5]);
	madc_lo_cc(o[6], a[3], b[4], o[6]);
	madc_hi_cc(o[7], a[3], b[4], o[7]);
	madc_lo_cc(o[8], a[5], b[4], o[8]);
	madc_hi_cc(o[9], a[5], b[4], o[9]);
	madc_lo_cc(o[10], a[7], b[4], o[10]);
	madc_hi_cc(o[11], a[7], b[4], o[11]);
	addc_32(o[12], o[12], 0);
	mad_lo_cc(o[0], a[0], b[1], o[0]);
	madc_hi_cc(o[1], a[0], b[1], o[1]);
	madc_lo_cc(o[2], a[0], b[3], o[2]);
	madc_hi_cc(o[3], a[0], b[3], o[3]);
	madc_lo_cc(o[4], a[0], b[5], o[4]);
	madc_hi_cc(o[5], a[0], b[5], o[5]);
	madc_lo_cc(o[6], a[0], b[7], o[6]);
	madc_hi_cc(o[7], a[0], b[7], o[7]);
	madc_lo_cc(o[8], a[2], b[7], o[8]);
	madc_hi_cc(o[9], a[2], b[7], o[9]);
	madc_lo_cc(o[10], a[4], b[7], o[10]);
	madc_hi_cc(o[11], a[4], b[7], o[11]);
	madc_lo_cc(o[12], a[6], b[7], o[12]);
	madc_hi_cc(o[13], a[6], b[7], o[13]);
	addc_32(o[14], o[14], 0);
	mad_lo_cc(o[0], a[1], b[0], o[0]);
	madc_hi_cc(o[1], a[1], b[0], o[1]);
	madc_lo_cc(o[2], a[1], b[2], o[2]);
	madc_hi_cc(o[3], a[1], b[2], o[3]);
	madc_lo_cc(o[4], a[1], b[4], o[4]);
	madc_hi_cc(o[5], a[1], b[4], o[5]);
	madc_lo_cc(o[6], a[1], b[6], o[6]);
	madc_hi_cc(o[7], a[1], b[6], o[7]);
	madc_lo_cc(o[8], a[3], b[6], o[8]);
	madc_hi_cc(o[9], a[3], b[6], o[9]);
	madc_lo_cc(o[10], a[5], b[6], o[10]);
	madc_hi_cc(o[11], a[5], b[6], o[11]);
	madc_lo_cc(o[12], a[7], b[6], o[12]);
	madc_hi_cc(o[13], a[7], b[6], o[13]);
	addc_32(o[14], o[14], 0);
	// p = e + (o << 32)
	add_cc_32(p[1], e[1], o[0]);
	addc_cc_32(p[2], e[2], o[1]);
	addc_cc_32(p[3], e[3], o[2]);
	addc_cc_32(p[4], e[4], o[3]);
	addc_cc_32(p[5], e[5], o[4]);
	addc_cc_32(p[6], e[6], o[5]);
	addc_cc_32(p[7], e[7], o[6]);
	addc_cc_32(p[8], e[8], o[7]);
	addc_cc_32(p[9], e[9], o[8]);
	addc_cc_32(p[10], e[10], o[9]);
	addc_cc_32(p[11], e[11], o[10]);
	addc_cc_32(p[12], e[12], o[11]);
	addc_cc_32(p[13], e[13], o[12]);
	addc_cc_32(p[14], e[14], o[13]);
	addc_cc_32(p[15], e[15], o[14]);
	p[0] = e[0];
}

// generated: split-column fused-MAC 256-bit square -> 512
__device__ __forceinline__ void sqr512_split(uint32_t* p, const uint32_t* a)
{
	uint32_t e[16], o[16];
	#pragma unroll
	for (int i = 0; i < 16; i++) { e[i] = 0; o[i] = 0; }
	mad_lo_cc(e[6], a[2], a[4], e[6]);
	madc_hi_cc(e[7], a[2], a[4], e[7]);
	madc_lo_cc(e[8], a[3], a[5], e[8]);
	madc_hi_cc(e[9], a[3], a[5], e[9]);
	addc_32(e[10], e[10], 0);
	mad_lo_cc(e[4], a[1], a[3], e[4]);
	madc_hi_cc(e[5], a[1], a[3], e[5]);
	madc_lo_cc(e[6], a[1], a[5], e[6]);
	madc_hi_cc(e[7], a[1], a[5], e[7]);
	madc_lo_cc(e[8], a[2], a[6], e[8]);
	madc_hi_cc(e[9], a[2], a[6], e[9]);
	madc_lo_cc(e[10], a[4], a[6], e[10]);
	madc_hi_cc(e[11], a[4], a[6], e[11]);
	addc_32(e[12], e[12], 0);
	mad_lo_cc(e[2], a[0], a[2], e[2]);
	madc_hi_cc(e[3], a[0], a[2], e[3]);
	madc_lo_cc(e[4], a[0], a[4], e[4]);
	madc_hi_cc(e[5], a[0], a[4], e[5]);
	madc_lo_cc(e[6], a[0], a[6], e[6]);
	madc_hi_cc(e[7], a[0], a[6], e[7]);
	madc_lo_cc(e[8], a[1], a[7], e[8]);
	madc_hi_cc(e[9], a[1], a[7], e[9]);
	madc_lo_cc(e[10], a[3], a[7], e[10]);
	madc_hi_cc(e[11], a[3], a[7], e[11]);
	madc_lo_cc(e[12], a[5], a[7], e[12]);
	madc_hi_cc(e[13], a[5], a[7], e[13]);
	addc_32(e[14], e[14], 0);
	mad_lo_cc(o[6], a[3], a[4], o[6]);
	madc_hi_cc(o[7], a[3], a[4], o[7]);
	addc_32(o[8], o[8], 0);
	mad_lo_cc(o[4], a[2], a[3], o[4]);
	madc_hi_cc(o[5], a[2], a[3], o[5]);
	madc_lo_cc(o[6], a[2], a[5], o[6]);
	madc_hi_cc(o[7], a[2], a[5], o[7]);
	madc_lo_cc(o[8], a[4], a[5], o[8]);
	madc_hi_cc(o[9], a[4], a[5], o[9]);
	addc_32(o[10], o[10], 0);
	mad_lo_cc(o[2], a[1], a[2], o[2]);
	madc_hi_cc(o[3], a[1], a[2], o[3]);
	madc_lo_cc(o[4], a[1], a[4], o[4]);
	madc_hi_cc(o[5], a[1], a[4], o[5]);
	madc_lo_cc(o[6], a[1], a[6], o[6]);
	madc_hi_cc(o[7], a[1], a[6], o[7]);
	madc_lo_cc(o[8], a[3], a[6], o[8]);
	madc_hi_cc(o[9], a[3], a[6], o[9]);
	madc_lo_cc(o[10], a[5], a[6], o[10]);
	madc_hi_cc(o[11], a[5], a[6], o[11]);
	addc_32(o[12], o[12], 0);
	mad_lo_cc(o[0], a[0], a[1], o[0]);
	madc_hi_cc(o[1], a[0], a[1], o[1]);
	madc_lo_cc(o[2], a[0], a[3], o[2]);
	madc_hi_cc(o[3], a[0], a[3], o[3]);
	madc_lo_cc(o[4], a[0], a[5], o[4]);
	madc_hi_cc(o[5], a[0], a[5], o[5]);
	madc_lo_cc(o[6], a[0], a[7], o[6]);
	madc_hi_cc(o[7], a[0], a[7], o[7]);
	madc_lo_cc(o[8], a[2], a[7], o[8]);
	madc_hi_cc(o[9], a[2], a[7], o[9]);
	madc_lo_cc(o[10], a[4], a[7], o[10]);
	madc_hi_cc(o[11], a[4], a[7], o[11]);
	madc_lo_cc(o[12], a[6], a[7], o[12]);
	madc_hi_cc(o[13], a[6], a[7], o[13]);
	addc_32(o[14], o[14], 0);
	// p = 2*(e + (o << 32))
	add_cc_32(p[1], e[1], o[0]);
	addc_cc_32(p[2], e[2], o[1]);
	addc_cc_32(p[3], e[3], o[2]);
	addc_cc_32(p[4], e[4], o[3]);
	addc_cc_32(p[5], e[5], o[4]);
	addc_cc_32(p[6], e[6], o[5]);
	addc_cc_32(p[7], e[7], o[6]);
	addc_cc_32(p[8], e[8], o[7]);
	addc_cc_32(p[9], e[9], o[8]);
	addc_cc_32(p[10], e[10], o[9]);
	addc_cc_32(p[11], e[11], o[10]);
	addc_cc_32(p[12], e[12], o[11]);
	addc_cc_32(p[13], e[13], o[12]);
	addc_cc_32(p[14], e[14], o[13]);
	addc_cc_32(p[15], e[15], o[14]);
	p[0] = e[0];
	add_cc_32(p[0], p[0], p[0]);
	addc_cc_32(p[1], p[1], p[1]);
	addc_cc_32(p[2], p[2], p[2]);
	addc_cc_32(p[3], p[3], p[3]);
	addc_cc_32(p[4], p[4], p[4]);
	addc_cc_32(p[5], p[5], p[5]);
	addc_cc_32(p[6], p[6], p[6]);
	addc_cc_32(p[7], p[7], p[7]);
	addc_cc_32(p[8], p[8], p[8]);
	addc_cc_32(p[9], p[9], p[9]);
	addc_cc_32(p[10], p[10], p[10]);
	addc_cc_32(p[11], p[11], p[11]);
	addc_cc_32(p[12], p[12], p[12]);
	addc_cc_32(p[13], p[13], p[13]);
	addc_cc_32(p[14], p[14], p[14]);
	addc_cc_32(p[15], p[15], p[15]);
	// += diagonal squares a[i]^2 at column 2i
	mad_lo_cc(p[0], a[0], a[0], p[0]);
	madc_hi_cc(p[1], a[0], a[0], p[1]);
	madc_lo_cc(p[2], a[1], a[1], p[2]);
	madc_hi_cc(p[3], a[1], a[1], p[3]);
	madc_lo_cc(p[4], a[2], a[2], p[4]);
	madc_hi_cc(p[5], a[2], a[2], p[5]);
	madc_lo_cc(p[6], a[3], a[3], p[6]);
	madc_hi_cc(p[7], a[3], a[3], p[7]);
	madc_lo_cc(p[8], a[4], a[4], p[8]);
	madc_hi_cc(p[9], a[4], a[4], p[9]);
	madc_lo_cc(p[10], a[5], a[5], p[10]);
	madc_hi_cc(p[11], a[5], a[5], p[11]);
	madc_lo_cc(p[12], a[6], a[6], p[12]);
	madc_hi_cc(p[13], a[6], a[6], p[13]);
	madc_lo_cc(p[14], a[7], a[7], p[14]);
	madc_hi_cc(p[15], a[7], a[7], p[15]);
}
// ---- END GENERATED ----

#undef mad_lo_cc
#undef madc_hi_cc
#undef madc_lo_cc

// r = a * b (mod P). Tail from "fast mod P" down is RCKangaroo's, verbatim.
__device__ __forceinline__ void MulModP(u64* res, u64* val1, u64* val2)
{
	u64 buff[8], tmp[5], tmp2[2], tmp3;
//calc 512 bits
	mul512_split((u32*)buff, (const u32*)val1, (const u32*)val2);
//fast mod P
	mul_256_by_P0inv((u32*)tmp, (u32*)(buff + 4));
	add_cc_64(buff[0], buff[0], tmp[0]);
	addc_cc_64(buff[1], buff[1], tmp[1]);
	addc_cc_64(buff[2], buff[2], tmp[2]);
	addc_cc_64(buff[3], buff[3], tmp[3]);
	addc_64(tmp[4], tmp[4], 0ull);
//see mul_256_by_P0inv for details
	u32* t32 = (u32*)tmp;
	u32* a32 = (u32*)tmp2;
	u32* k = (u32*)&tmp3;
	mul_wide_32(tmp2[0], t32[8], P_INV32);
	mul_wide_32(tmp3, t32[9], P_INV32);
	add_cc_32(a32[1], a32[1], k[0]);
	addc_32(a32[2], k[1], 0); //we cannot get carry here for a32[3]
	add_cc_32(a32[1], a32[1], t32[8]);
	addc_cc_32(a32[2], a32[2], t32[9]);
	addc_32(a32[3], 0, 0);

	add_cc_64(res[0], buff[0], tmp2[0]);
	addc_cc_64(res[1], buff[1], tmp2[1]);
	addc_cc_64(res[2], buff[2], 0ull);
	addc_64(res[3], buff[3], 0ull);
}

// r = a^2 (mod P)
__device__ __forceinline__ void SqrModP(u64* res, u64* val)
{
	u64 buff[8], tmp[5], tmp2[2], tmp3;
//calc 512 bits
	sqr512_split((u32*)buff, (const u32*)val);
//fast mod P
	mul_256_by_P0inv((u32*)tmp, (u32*)(buff + 4));
	add_cc_64(buff[0], buff[0], tmp[0]);
	addc_cc_64(buff[1], buff[1], tmp[1]);
	addc_cc_64(buff[2], buff[2], tmp[2]);
	addc_cc_64(buff[3], buff[3], tmp[3]);
	addc_64(tmp[4], tmp[4], 0ull);
//see mul_256_by_P0inv for details
	u32* t32 = (u32*)tmp;
	u32* a32 = (u32*)tmp2;
	u32* k = (u32*)&tmp3;
	mul_wide_32(tmp2[0], t32[8], P_INV32);
	mul_wide_32(tmp3, t32[9], P_INV32);
	add_cc_32(a32[1], a32[1], k[0]);
	addc_32(a32[2], k[1], 0); //we cannot get carry here for a32[3]
	add_cc_32(a32[1], a32[1], t32[8]);
	addc_cc_32(a32[2], a32[2], t32[9]);
	addc_32(a32[3], 0, 0);

	add_cc_64(res[0], buff[0], tmp2[0]);
	addc_cc_64(res[1], buff[1], tmp2[1]);
	addc_cc_64(res[2], buff[2], 0ull);
	addc_64(res[3], buff[3], 0ull);
}

// ---- Wrappers matching CUDACyclone's call conventions ---------------------------

// r = a * b (mod P)
__device__ __forceinline__ void rmul(uint64_t* r, const uint64_t* a, const uint64_t* b)
{
    MulModP((u64*)r, (u64*)a, (u64*)b);
}

// r = r * a (mod P)  (2-arg form used by CUDACyclone's `_ModMult(inverse, subp[0])`)
__device__ __forceinline__ void rmul(uint64_t* r, const uint64_t* a)
{
    MulModP((u64*)r, (u64*)r, (u64*)a);
}

// r = a^2 (mod P)
__device__ __forceinline__ void rsqr(uint64_t* r, const uint64_t* a)
{
    SqrModP((u64*)r, (u64*)a);
}

// r = a^-1 (mod P), in place
__device__ __forceinline__ void rinv(uint64_t* r)
{
    InvModP((u32*)r);
}

} // namespace rck
