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

// a ^ b ^ c folded into one LOP3 (truth table 0x96). ptxas usually does this anyway;
// writing it explicitly guarantees the single instruction.
__device__ __forceinline__ uint32_t xor3(uint32_t a, uint32_t b, uint32_t c){
#if __CUDA_ARCH__ >= 500
    uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r;
#else
    return a ^ b ^ c;
#endif
}

__device__ __forceinline__ uint32_t bigS0(uint32_t x) { return xor3(ror32(x, 2), ror32(x, 13), ror32(x, 22)); }
__device__ __forceinline__ uint32_t bigS1(uint32_t x) { return xor3(ror32(x, 6), ror32(x, 11), ror32(x, 25)); }
__device__ __forceinline__ uint32_t smallS0(uint32_t x){ return xor3(ror32(x, 7), ror32(x, 18), (x >> 3)); }
__device__ __forceinline__ uint32_t smallS1(uint32_t x){ return xor3(ror32(x,17), ror32(x, 19), (x >>10)); }

// Ch = (x&y)^(~x&z)  -> LOP3 truth table 0xCA;  Maj = (x&y)|(x&z)|(y&z) -> 0xE8. One instruction each.
__device__ __forceinline__ uint32_t Ch (uint32_t x,uint32_t y,uint32_t z){
#if __CUDA_ARCH__ >= 500
    uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xCA;" : "=r"(r) : "r"(x), "r"(y), "r"(z)); return r;
#else
    return (x & y) ^ (~x & z);
#endif
}
__device__ __forceinline__ uint32_t Maj(uint32_t x,uint32_t y,uint32_t z){
#if __CUDA_ARCH__ >= 500
    uint32_t r; asm("lop3.b32 %0, %1, %2, %3, 0xE8;" : "=r"(r) : "r"(x), "r"(y), "r"(z)); return r;
#else
    return (x & y) | (x & z) | (y & z);
#endif
}

__device__ __constant__ uint32_t K[64] = {
    0x428A2F98,0x71374491,0xB5C0FBCF,0xE9B5DBA5,0x3956C25B,0x59F111F1,0x923F82A4,0xAB1C5ED5,
    0xD807AA98,0x12835B01,0x243185BE,0x550C7DC3,0x72BE5D74,0x80DEB1FE,0x9BDC06A7,0xC19BF174,
    0xE49B69C1,0xEFBE4786,0x0FC19DC6,0x240CA1CC,0x2DE92C6F,0x4A7484AA,0x5CB0A9DC,0x76F988DA,
    0x983E5152,0xA831C66D,0xB00327C8,0xBF597FC7,0xC6E00BF3,0xD5A79147,0x06CA6351,0x14292967,
    0x27B70A85,0x2E1B2138,0x4D2C6DFC,0x53380D13,0x650A7354,0x766A0ABB,0x81C2C92E,0x92722C85,
    0xA2BFE8A1,0xA81A664B,0xC24B8B70,0xC76C51A3,0xD192E819,0xD6990624,0xF40E3585,0x106AA070,
    0x19A4C116,0x1E376C08,0x2748774C,0x34B0BCB5,0x391C0CB3,0x4ED8AA4A,0x5B9CCA4F,0x682E6FF3,
    0x748F82EE,0x78A5636F,0x84C87814,0x8CC70208,0x90BEFFFA,0xA4506CEB,0xBEF9A3F7,0xC67178F2
};

__device__ __constant__ uint32_t IV[8] = {
    0x6a09e667ul,0xbb67ae85ul,0x3c6ef372ul,0xa54ff53aul,
    0x510e527ful,0x9b05688cul,0x1f83d9ab ,0x5be0cd19ul
};
__device__ __forceinline__ void SHA256Initialize(uint32_t s[8])
{
#pragma unroll
    for (int i = 0; i < 8; i++) s[i] = IV[i];
}
__device__ __forceinline__ void SHA256Transform(uint32_t state[8], uint32_t W_in[64])
{
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) w[i] = W_in[i];

#pragma unroll 64
    for (int t = 0; t < 64; ++t) {
        if (t >= 16) {
            uint32_t s0 = smallS0(w[(t + 1)  & 15]);
            uint32_t s1 = smallS1(w[(t + 14) & 15]);
            uint32_t newW = w[t & 15] + s1 + w[(t + 9) & 15] + s0;
            w[t & 15] = newW;
        }
        uint32_t Wt = w[t & 15];
        uint32_t T1 = h + bigS1(e) + Ch(e, f, g) + K[t] + Wt;
        uint32_t T2 = bigS0(a) + Maj(a, b, c);

        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}
__device__ __forceinline__ void RIPEMD160Initialize(uint32_t s[5])
{

	s[0] = 0x67452301ul;
	s[1] = 0xEFCDAB89ul;
	s[2] = 0x98BADCFEul;
	s[3] = 0x10325476ul;
	s[4] = 0xC3D2E1F0ul;

}

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
	R51(e1, a1, b1, c1, d1, w[11], 11);
	R52(e2, a2, b2, c2, d2, w[0], 15);
	R51(d1, e1, a1, b1, c1, w[6], 8);
	R52(d2, e2, a2, b2, c2, w[3], 13);
	R51(c1, d1, e1, a1, b1, w[15], 5);
	R52(c2, d2, e2, a2, b2, w[9], 11);
	R51(b1, c1, d1, e1, a1, w[13], 6);
	R52(b2, c2, d2, e2, a2, w[11], 11);

    uint32_t t = s[0];
    s[0] = s[1] + c1 + d2;
    s[1] = s[2] + d1 + e2;
    s[2] = s[3] + e1 + a2;
    s[3] = s[4] + a1 + b2;
    s[4] = t + b1 + c2;
}


__device__ __forceinline__ uint32_t bswap32(uint32_t x){
    return __byte_perm(x, 0, 0x0123);   // reverse the 4 bytes in one PRMT
}
__device__ __forceinline__ uint32_t pack_be4(uint8_t a,uint8_t b,uint8_t c,uint8_t d){
    return ((uint32_t)a<<24)|((uint32_t)b<<16)|((uint32_t)c<<8)|((uint32_t)d);
}
__device__ __forceinline__ void SHA256_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint32_t out_state[16]){
    const uint64_t v3 = x_be_limbs[3];
    const uint64_t v2 = x_be_limbs[2];
    const uint64_t v1 = x_be_limbs[1];
    const uint64_t v0 = x_be_limbs[0];
    uint32_t M[16];
    M[0] = pack_be4(prefix02_03, (uint8_t)(v3>>56), (uint8_t)(v3>>48), (uint8_t)(v3>>40));
    M[1] = pack_be4((uint8_t)(v3>>32), (uint8_t)(v3>>24), (uint8_t)(v3>>16), (uint8_t)(v3>>8));
    M[2] = pack_be4((uint8_t)(v3>>0), (uint8_t)(v2>>56), (uint8_t)(v2>>48), (uint8_t)(v2>>40));
    M[3] = pack_be4((uint8_t)(v2>>32), (uint8_t)(v2>>24), (uint8_t)(v2>>16), (uint8_t)(v2>>8));
    M[4] = pack_be4((uint8_t)(v2>>0), (uint8_t)(v1>>56), (uint8_t)(v1>>48), (uint8_t)(v1>>40));
    M[5] = pack_be4((uint8_t)(v1>>32), (uint8_t)(v1>>24), (uint8_t)(v1>>16), (uint8_t)(v1>>8));
    M[6] = pack_be4((uint8_t)(v1>>0), (uint8_t)(v0>>56), (uint8_t)(v0>>48), (uint8_t)(v0>>40));
    M[7] = pack_be4((uint8_t)(v0>>32), (uint8_t)(v0>>24), (uint8_t)(v0>>16), (uint8_t)(v0>>8));
    M[8] = pack_be4((uint8_t)(v0>>0), 0x80u, 0x00u, 0x00u);
#pragma unroll
    for(int i=9;i<16;++i) M[i]=0;
    M[15] = 33u*8u;
    uint32_t st[8];
    SHA256Initialize(st);
    SHA256Transform(st, M);
#pragma unroll
    for(int i=0;i<8;++i) out_state[i]= bswap32(st[i]);
}

__device__ __forceinline__ void RIPEMD160_from_SHA256_state(uint32_t sha_state_le[16],
                                                            uint32_t out5[5])
{
    sha_state_le[8]  = 0x00000080u;
#pragma unroll
    for(int i=9;i<14;++i) sha_state_le[i]=0;
    sha_state_le[14] = 256u;
    sha_state_le[15] = 0u;

    uint32_t s[5];
    RIPEMD160Initialize(s);
    RIPEMD160Transform(s, sha_state_le);
    // out5[i] == load_u32_le(old ripemd20 + 4*i): the LE byte serialization was a no-op
    // round-trip, so the state words are exactly the words the comparison reads back.
#pragma unroll
    for (int i = 0; i < 5; ++i) out5[i] = s[i];
}

__device__ __noinline__ void getHash160_33_from_limbs(uint8_t prefix02_03,
                                                      const uint64_t x_be_limbs[4],
                                                      uint32_t out5[5])
{
    uint32_t sha_state[16];
    SHA256_33_from_limbs(prefix02_03, x_be_limbs, sha_state);
#ifndef SHA_ONLY
    RIPEMD160_from_SHA256_state(sha_state, out5);
#else
    // SHA-only benchmark (-DSHA_ONLY): skip RIPEMD-160; emit the SHA-256 words directly
    // so the SHA-256 work stays live and out5 is still written.
#pragma unroll
    for (int i = 0; i < 5; ++i) out5[i] = sha_state[i];
#endif
}
