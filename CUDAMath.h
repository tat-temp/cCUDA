// SPDX-License-Identifier: GPL-3.0-or-later  (see LICENSE; default build links GPLv3 RCKangaroo)
//
// Field-arithmetic backend. The secp256k1 field ops (_ModMult / _ModSqr / _ModInv) are
// RetiredCoder's RCKangaroo implementations (32-bit-limb MulModP/SqrModP + safegcd InvModP,
// third_party/RCKangaroo/, GPLv3) -- measured ~+8.5% end-to-end on RTX 5090 over the older
// JeanLucPons-lineage field math this project used to carry (removed). See ec_backend.cuh
// and LICENSE.
#include "ec_backend.cuh"


#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) );

__device__ void ModSub256isOdd(uint64_t* a, uint64_t* b, uint8_t* parity) {    //no need to compute py, we need only parity

    uint64_t t;   
    uint64_t T[4]; 

    USUBO(T[0], a[0], b[0]);   
    USUBC(T[1], a[1], b[1]);   
    USUBC(T[2], a[2], b[2]);   
    USUBC(T[3], a[3], b[3]);  

    USUB(t, 0ULL, 0ULL);  // borrow

    *parity = (T[0] & 1) ^ (t & 1);  // LSB of T[0] and LSB of t -> parity od sub
}



__device__ void ModNeg256(uint64_t *r,uint64_t *a) {

  uint64_t t[4];
  USUBO(t[0],0ULL,a[0]);
  USUBC(t[1],0ULL,a[1]);
  USUBC(t[2],0ULL,a[2]);
  USUBC(t[3],0ULL,a[3]);
  UADDO(r[0],t[0],0xFFFFFFFEFFFFFC2FULL);
  UADDC(r[1],t[1],0xFFFFFFFFFFFFFFFFULL);
  UADDC(r[2],t[2],0xFFFFFFFFFFFFFFFFULL);
  UADD(r[3],t[3],0xFFFFFFFFFFFFFFFFULL);

}

__device__ void ModSub256(uint64_t *r,uint64_t *a,uint64_t *b) {

    uint64_t borrow;
    uint64_t p[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                      0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };

    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(borrow, 0ULL, 0ULL); 

    if (borrow) {
        UADDO1(r[0], p[0]);
        UADDC1(r[1], p[1]);
        UADDC1(r[2], p[2]);
        UADD1(r[3], p[3]);
    }
}
__device__ void ModSub256(uint64_t* r,uint64_t* b) {

    uint64_t borrow;
    uint64_t p[4] = { 0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                      0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };

    USUBO1(r[0], b[0]);
    USUBC1(r[1], b[1]);
    USUBC1(r[2], b[2]);
    USUBC1(r[3], b[3]);
    USUB(borrow, 0ULL, 0ULL);

    if (borrow) {
        UADDO1(r[0], p[0]);
        UADDC1(r[1], p[1]);
        UADDC1(r[2], p[2]);
        UADD1(r[3], p[3]);
    }
}

// Fused (a - b - c) mod p for the secp256k1 prime, inputs a,b,c in [0,p).
// Two subtract chains, then a SINGLE reduction. Since p = 2^256 - K (K = 0x1000003D1) the raw
// a-b-c lands in (-2p, p) with a total borrow count B in {0,1,2}, and a-b-c == r - B*2^256
// == r - B*K (mod p), where r is the 256-bit wrapped result. Subtracting B*K then one conditional
// +p brings it back to [0,p). NOTE: "add p once per borrow" is INCORRECT at the boundary (e.g.
// a=0,b=p-1,c=2), which is why the B*K form + fix-up is required -- verified vs Python reference.
// r may alias a (each a[i] is read before r[i] is written); c and b must not alias r.
__device__ void ModSub256_2(uint64_t* r, const uint64_t* a, const uint64_t* b, const uint64_t* c) {

    uint64_t brw1, brw2, brw3;

    // r = a - b
    USUBO(r[0], a[0], b[0]);
    USUBC(r[1], a[1], b[1]);
    USUBC(r[2], a[2], b[2]);
    USUBC(r[3], a[3], b[3]);
    USUB(brw1, 0ULL, 0ULL);            // 0 or 0xFFFFFFFFFFFFFFFF

    // r = r - c
    USUBO1(r[0], c[0]);
    USUBC1(r[1], c[1]);
    USUBC1(r[2], c[2]);
    USUBC1(r[3], c[3]);
    USUB(brw2, 0ULL, 0ULL);

    // a-b-c == r - B*K (mod p), B = (#borrows) in {0,1,2}
    uint64_t BK = ((brw1 & 1ULL) + (brw2 & 1ULL)) * 0x1000003D1ULL;
    USUBO1(r[0], BK);
    USUBC1(r[1], 0ULL);
    USUBC1(r[2], 0ULL);
    USUBC1(r[3], 0ULL);
    USUB(brw3, 0ULL, 0ULL);

    if (brw3) {                        // r - B*K underflowed -> add p once
        UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL);
        UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL);
        UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL);
        UADD1(r[3], 0xFFFFFFFFFFFFFFFFULL);
    }
}

__device__ __forceinline__ void _ModInv(uint64_t* R){ rck::rinv(R); }

__device__ __forceinline__ void _ModMult(uint64_t* r, uint64_t* a, uint64_t* b){ rck::rmul(r,a,b); }
__device__ __forceinline__ void _ModMult(uint64_t* r, uint64_t* a){ rck::rmul(r,a); }
__device__ __forceinline__ void _ModSqr(uint64_t* rp, const uint64_t* up){ rck::rsqr(rp,up); }

__device__ void fieldInv(const uint64_t in[4], uint64_t out[4]) {
    uint64_t t[5];
    t[0] = in[0];
    t[1] = in[1];
    t[2] = in[2];
    t[3] = in[3];
    t[4] = 0;
    _ModInv(t);
    out[0] = t[0];
    out[1] = t[1];
    out[2] = t[2];
    out[3] = t[3];
}

// --- Secp256k1 block (point mult, doubling etc with a few helpers) -----------------------------

static __device__ const uint64_t SECP_P_LE[4] = {
    0xFFFFFFFEFFFFFC2FULL, 
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL  
};

static __device__ const uint64_t SECP_GX_LE[4] = {
    0x59f2815b16f81798ULL,
    0x029bfcdb2dce28d9ULL,
    0x55a06295ce870b07ULL,
    0x79be667ef9dcbbacULL
};
static __device__ const uint64_t SECP_GY_LE[4] = {
    0x9c47d08ffb10d4b8ULL,
    0xfd17b448a6855419ULL,
    0x5da4fbfc0e1108a8ULL,
    0x483ada7726a3c465ULL
};

__device__ __forceinline__ void fieldCopy(const uint64_t a[4], uint64_t out[4]) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
}

__device__ __forceinline__ bool fieldIsZero(const uint64_t a[4]) {
    return ( (a[0] | a[1] | a[2] | a[3]) == 0ULL );
}

__device__ void fieldAdd(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    __uint128_t t = 0;
    uint64_t c = 0;

    t = (__uint128_t)a[0] + b[0];
    out[0] = (uint64_t)t;
    c = (uint64_t)(t >> 64);

    t = (__uint128_t)a[1] + b[1] + c;
    out[1] = (uint64_t)t;
    c = (uint64_t)(t >> 64);

    t = (__uint128_t)a[2] + b[2] + c;
    out[2] = (uint64_t)t;
    c = (uint64_t)(t >> 64);

    t = (__uint128_t)a[3] + b[3] + c;
    out[3] = (uint64_t)t;
    c = (uint64_t)(t >> 64); 

    if (c || (out[3] > SECP_P_LE[3]) || 
        (out[3] == SECP_P_LE[3] && out[2] > SECP_P_LE[2]) || 
        (out[3] == SECP_P_LE[3] && out[2] == SECP_P_LE[2] && out[1] > SECP_P_LE[1]) || 
        (out[3] == SECP_P_LE[3] && out[2] == SECP_P_LE[2] && out[1] == SECP_P_LE[1] && out[0] >= SECP_P_LE[0])) {

        __uint128_t tb;
        uint64_t borrow = 0;
        tb = (__uint128_t)out[0] - SECP_P_LE[0];
        out[0] = (uint64_t)tb;
        borrow = (tb > 0xFFFFFFFFFFFFFFFFULL) ? 1 : 0;

        tb = (__uint128_t)out[1] - SECP_P_LE[1] - borrow;
        out[1] = (uint64_t)tb;
        borrow = (tb > 0xFFFFFFFFFFFFFFFFULL) ? 1 : 0;

        tb = (__uint128_t)out[2] - SECP_P_LE[2] - borrow;
        out[2] = (uint64_t)tb;
        borrow = (tb > 0xFFFFFFFFFFFFFFFFULL) ? 1 : 0;

        tb = (__uint128_t)out[3] - SECP_P_LE[3] - borrow;
        out[3] = (uint64_t)tb;
    }
}

__device__ void fieldSub(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    __int128_t t;
    uint64_t borrow = 0;

    t = (__int128_t)a[0] - b[0];
    out[0] = (uint64_t)t; borrow = (t < 0);

    t = (__int128_t)a[1] - b[1] - borrow;
    out[1] = (uint64_t)t; borrow = (t < 0);

    t = (__int128_t)a[2] - b[2] - borrow;
    out[2] = (uint64_t)t; borrow = (t < 0);

    t = (__int128_t)a[3] - b[3] - borrow;
    out[3] = (uint64_t)t; borrow = (t < 0);

    if (borrow) {
        __uint128_t tu;
        uint64_t carry = 0;
        tu = (__uint128_t)out[0] + SECP_P_LE[0];
        out[0] = (uint64_t)tu; carry = (uint64_t)(tu >> 64);

        tu = (__uint128_t)out[1] + SECP_P_LE[1] + carry;
        out[1] = (uint64_t)tu; carry = (uint64_t)(tu >> 64);

        tu = (__uint128_t)out[2] + SECP_P_LE[2] + carry;
        out[2] = (uint64_t)tu; carry = (uint64_t)(tu >> 64);

        tu = (__uint128_t)out[3] + SECP_P_LE[3] + carry;
        out[3] = (uint64_t)tu;
    }
}

__device__ __forceinline__ void fieldMul(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    _ModMult(out, (uint64_t*)a, (uint64_t*)b);
}
__device__ __forceinline__ void fieldSqr(const uint64_t a[4], uint64_t out[4]) {
    _ModSqr(out, a);
}

// --- Simple EC affine coordinates (we don't need fast implementation of point mult) ----------------

struct ECPointA {
    uint64_t X[4];
    uint64_t Y[4];
    bool infinity;
};

__device__ __forceinline__ void pointSetInfinity(ECPointA &P) {
    P.infinity = true;
    P.X[0]=P.X[1]=P.X[2]=P.X[3]=0ULL;
    P.Y[0]=P.Y[1]=P.Y[2]=P.Y[3]=0ULL;
}
__device__ __forceinline__ void pointSetG(ECPointA &P) {
    pointSetInfinity(P); 
    P.infinity = false;
    P.X[0] = SECP_GX_LE[0];
    P.X[1] = SECP_GX_LE[1];
    P.X[2] = SECP_GX_LE[2];
    P.X[3] = SECP_GX_LE[3];
    P.Y[0] = SECP_GY_LE[0];
    P.Y[1] = SECP_GY_LE[1];
    P.Y[2] = SECP_GY_LE[2];
    P.Y[3] = SECP_GY_LE[3];
}
__device__ void pointDoubleAffine(const ECPointA &P, ECPointA &R) {
    if (P.infinity) { pointSetInfinity(R); return; }

    uint64_t x2[4], two_x2[4], three_x2[4];
    uint64_t denom[4], invDen[4], lambda[4];

    fieldSqr(P.X, x2);
    fieldAdd(x2, x2, two_x2);
    fieldAdd(two_x2, x2, three_x2);

    fieldAdd(P.Y, P.Y, denom);
    fieldInv(denom, invDen);

    fieldMul(three_x2, invDen, lambda);

    uint64_t lambda2[4], twoX[4], newX[4];
    fieldSqr(lambda, lambda2);
    fieldAdd(P.X, P.X, twoX);
    fieldSub(lambda2, twoX, newX);

    uint64_t tmp[4], prod[4], newY[4];
    fieldSub(P.X, newX, tmp);
    fieldMul(lambda, tmp, prod);
    fieldSub(prod, P.Y, newY);

    fieldCopy(newX, R.X);
    fieldCopy(newY, R.Y);
    R.infinity = false;
}

__device__ void pointAddAffine(const ECPointA &P, const ECPointA &Q, ECPointA &R) {
    if (P.infinity) { R = Q; return; }
    if (Q.infinity) { R = P; return; }

    bool sameX = (P.X[0]==Q.X[0] && P.X[1]==Q.X[1] && P.X[2]==Q.X[2] && P.X[3]==Q.X[3]);
    bool sameY = (P.Y[0]==Q.Y[0] && P.Y[1]==Q.Y[1] && P.Y[2]==Q.Y[2] && P.Y[3]==Q.Y[3]);

    if (sameX && sameY) {
        pointDoubleAffine(P, R);
        return;
    }

    if (sameX && !sameY) {
        pointSetInfinity(R);
        return;
    }

    uint64_t dx[4], dy[4], invdx[4], lambda[4], lambda2[4];
    uint64_t tmp1[4], prod[4], newX[4], newY[4];

    fieldSub(Q.X, P.X, dx);     // dx = x2 - x1
    fieldSub(Q.Y, P.Y, dy);     // dy = y2 - y1

    fieldInv(dx, invdx);        // invdx = 1/dx
    fieldMul(dy, invdx, lambda);// lambda = dy * invdx = (y2 - y1) / (x2 - x1)

    // x3 = lambda^2 - x1 - x2
    fieldSqr(lambda, lambda2);
    fieldSub(lambda2, P.X, tmp1);   // tmp1 = lambda^2 - x1
    fieldSub(tmp1, Q.X, newX);      // newX = lambda^2 - x1 - x2

    // y3 = lambda*(x1 - x3) - y1
    fieldSub(P.X, newX, tmp1);      // tmp1 = x1 - x3
    fieldMul(lambda, tmp1, prod);   // prod = lambda * (x1 - x3)
    fieldSub(prod, P.Y, newY);      // newY = prod - y1

    fieldCopy(newX, R.X);
    fieldCopy(newY, R.Y);
    R.infinity = false;
}

__device__ void scalarMulBaseAffine(const uint64_t scalar_le[4], uint64_t outX[4], uint64_t outY[4]) {
    ECPointA R;
    pointSetInfinity(R);

    int msb = -1;
    if      (scalar_le[3] != 0) msb = 3 * 64 + 63 - __clzll(scalar_le[3]);
    else if (scalar_le[2] != 0) msb = 2 * 64 + 63 - __clzll(scalar_le[2]);
    else if (scalar_le[1] != 0) msb = 1 * 64 + 63 - __clzll(scalar_le[1]);
    else if (scalar_le[0] != 0) msb = 0 * 64 + 63 - __clzll(scalar_le[0]);

    if (msb == -1) {
        // scalar == 0 -> infinity
        outX[0]=outX[1]=outX[2]=outX[3]=0ULL;
        outY[0]=outY[1]=outY[2]=outY[3]=0ULL;
        return;
    }

    for (int bi = msb; bi >= 0; --bi) {
        // R = 2*R
        if (!R.infinity) {
            ECPointA tmpD;
            pointDoubleAffine(R, tmpD);
            R = tmpD;
        }
        // if bit == 1, R = R + G
        int limb = bi >> 6;
        int shift = bi & 63;
        uint64_t bit = (scalar_le[limb] >> shift) & 1ULL;
        if (bit) {
            ECPointA Gp;
            pointSetG(Gp);
            if (R.infinity) {
                R = Gp;
            } else {
                ECPointA tmpA;
                pointAddAffine(R, Gp, tmpA);
                R = tmpA;
            }
        }
    }

    if (R.infinity) {
        outX[0]=outX[1]=outX[2]=outX[3]=0ULL;
        outY[0]=outY[1]=outY[2]=outY[3]=0ULL;
    } else {
        fieldCopy(R.X, outX);
        fieldCopy(R.Y, outY);
    }
}

__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const uint64_t* scalar = scalars_in + idx*4;
    uint64_t* outx = outX + idx*4;
    uint64_t* outy = outY + idx*4;

    scalarMulBaseAffine(scalar, outx, outy);
}


