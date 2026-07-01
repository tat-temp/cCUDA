// ecbench.cu — standalone A/B for secp256k1 field arithmetic:
//   CUDACyclone (_ModMult/_ModSqr/_ModInv/ModSub256..., CUDAMath.h, JeanLucPons lineage)
//        vs
//   RCKangaroo (MulModP/SqrModP/InvModP/SubModP..., third_party/RCKangaroo, via ec_backend.cuh)
//
// Two phases:
//   1) CORRECTNESS — every op is checked against an INDEPENDENT host reference
//      (arbitrary-precision-style reduce, anchored to Python-computed KATs). A result
//      is trusted only when host-ref == Cyclone == RCK (triangulation). Also reports
//      how many outputs are non-canonical (>= P) to decide whether the hash boundary
//      needs an extra reduce.
//   2) THROUGHPUT — dependent op-chains (DCE-proof, written to a global sink), many
//      warps in flight so it is throughput- not latency-bound, mirroring the real
//      kernel. Reports Mops/s for each backend and the rck/cyclone ratio. An "ecstep"
//      mix reproduces the per-key op blend of the production point-add loop.
//
// Build:  make ecbench     (see Makefile)     Run: ./CUDACyclone-ecbench [reps] [scale]
//
// NOTE: no GPU is needed to READ this, but it must be built + run on the target card.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "CUDAMath.h"      // CUDACyclone backend: _ModMult / _ModSqr / _ModInv / ModSub256 / ModNeg256
#include "ec_backend.cuh"  // RCKangaroo backend:  rck::rmul / rsqr / rinv / rsub / rneg
#include "cr_field.cuh"    // clean-room backend:  cr::mul / cr::sqr

// ------------------------------------------------------------------ KAT / constants
static const uint64_t P_LIMBS[4] = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                                    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
// KAT (secp256k1 field, p=2^256-2^32-977), Python-generated, seed=1337
static const uint64_t KAT_A[4]   = {0xECEFE37B9E250D03ULL, 0xB5BAB1CD888417A5ULL, 0x922BADB05DA83CFFULL, 0xBB5D75B895F628F2ULL};
static const uint64_t KAT_B[4]   = {0xC6737B8B2A6A7B5FULL, 0x5531AE6DD30A286EULL, 0xA28718E5623A7A75ULL, 0x5C1ED35FCA2410FDULL};
static const uint64_t KAT_MUL[4] = {0x9F051D785673749CULL, 0x1A992A581E5B0775ULL, 0x94A5B7B26565DBEFULL, 0xCFE7823B896F0EA4ULL};
static const uint64_t KAT_SQR[4] = {0xA5CC33F33E57E11FULL, 0x748A44A405B07FB8ULL, 0x0AF87A79CA875C39ULL, 0x034919768219B27BULL};
static const uint64_t KAT_INV[4] = {0x062EFBCD36525A34ULL, 0x78D63B3489ACA83EULL, 0x31BA2785F7B51C89ULL, 0x9095E357121F5793ULL};
static const uint64_t KAT_SUB[4] = {0x267C67F073BA91A4ULL, 0x6089035FB579EF37ULL, 0xEFA494CAFB6DC28AULL, 0x5F3EA258CBD217F4ULL};
static const uint64_t KAT_ADD[4] = {0xB3635F07C88F8C33ULL, 0x0AEC603B5B8E4014ULL, 0x34B2C695BFE2B775ULL, 0x177C4918601A39F0ULL};

// ============================================================ HOST reference (generic)
// Bulletproof, implementation-independent 512-bit mod p via shift/subtract. Slow but
// only used to validate a modest number of samples; anchored by KAT self-tests.
static int  cmp_n(const uint64_t* a, const uint64_t* b, int n){
    for(int i=n-1;i>=0;--i){ if(a[i]!=b[i]) return a[i]<b[i]?-1:1; } return 0;
}
static void sub_n(uint64_t* a, const uint64_t* b, int n){ // a -= b (a>=b)
    unsigned __int128 br=0;
    for(int i=0;i<n;++i){ unsigned __int128 t=(unsigned __int128)a[i]-b[i]-br; a[i]=(uint64_t)t; br=(t>>64)?1:0; }
}
static void shr1_n(uint64_t* a, int n){ // a >>= 1
    uint64_t carry=0; for(int i=n-1;i>=0;--i){ uint64_t nc=a[i]&1ULL; a[i]=(a[i]>>1)|(carry<<63); carry=nc; }
}
static void mul256(const uint64_t a[4], const uint64_t b[4], uint64_t r[8]){
    for(int i=0;i<8;++i) r[i]=0;
    for(int i=0;i<4;++i){
        unsigned __int128 carry=0;
        for(int j=0;j<4;++j){
            unsigned __int128 t=(unsigned __int128)a[i]*b[j]+r[i+j]+carry;
            r[i+j]=(uint64_t)t; carry=t>>64;
        }
        r[i+4]+=(uint64_t)carry;
    }
}
static void reduce512_modp(uint64_t T[8], uint64_t out[4]){
    uint64_t M[8]={0}; for(int i=0;i<4;++i) M[i+4]=P_LIMBS[i];   // M = p << 256
    for(int s=256;s>=0;--s){ if(cmp_n(T,M,8)>=0) sub_n(T,M,8); if(s) shr1_n(M,8); }
    for(int i=0;i<4;++i) out[i]=T[i];
}
static void host_mulmod(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]){
    uint64_t T[8]; mul256(a,b,T); reduce512_modp(T,out);
}
static void host_submod(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]){
    unsigned __int128 br=0; uint64_t r[4];
    for(int i=0;i<4;++i){ unsigned __int128 t=(unsigned __int128)a[i]-b[i]-br; r[i]=(uint64_t)t; br=(t>>64)?1:0; }
    if(br){ unsigned __int128 c=0; for(int i=0;i<4;++i){ unsigned __int128 t=(unsigned __int128)r[i]+P_LIMBS[i]+c; r[i]=(uint64_t)t; c=t>>64; } }
    for(int i=0;i<4;++i) out[i]=r[i];
}
static void host_addmod(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]){
    unsigned __int128 c=0; uint64_t r[4];
    for(int i=0;i<4;++i){ unsigned __int128 t=(unsigned __int128)a[i]+b[i]+c; r[i]=(uint64_t)t; c=t>>64; }
    if(c || cmp_n(r,P_LIMBS,4)>=0) sub_n(r,P_LIMBS,4);
    for(int i=0;i<4;++i) out[i]=r[i];
}
static bool ge_p(const uint64_t a[4]){ return cmp_n(a,P_LIMBS,4)>=0; }
static void canon(uint64_t a[4]){ if(ge_p(a)) sub_n(a,P_LIMBS,4); }  // a<2^256<2p -> one subtract
static bool eq4(const uint64_t a[4], const uint64_t b[4]){ return !memcmp(a,b,4*sizeof(uint64_t)); }

// ================================================================= DEVICE op kernels
// One-shot correctness kernels: read inputs, write outputs (lazy reduction preserved).
__global__ void k_mul_cyc(const uint64_t* a, const uint64_t* b, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x[4],y[4],r[4];
    #pragma unroll
    for(int k=0;k<4;++k){ x[k]=a[4*i+k]; y[k]=b[4*i+k]; }
    _ModMult(r,x,y);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=r[k];
}
__global__ void k_mul_rck(const uint64_t* a, const uint64_t* b, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x[4],y[4],r[4];
    #pragma unroll
    for(int k=0;k<4;++k){ x[k]=a[4*i+k]; y[k]=b[4*i+k]; }
    rck::rmul(r,x,y);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=r[k];
}
__global__ void k_sqr_cyc(const uint64_t* a, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x[4],r[4];
    #pragma unroll
    for(int k=0;k<4;++k) x[k]=a[4*i+k];
    _ModSqr(r,x);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=r[k];
}
__global__ void k_sqr_rck(const uint64_t* a, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x[4],r[4];
    #pragma unroll
    for(int k=0;k<4;++k) x[k]=a[4*i+k];
    rck::rsqr(r,x);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=r[k];
}
__global__ void k_mul_cr(const uint64_t* a, const uint64_t* b, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x[4],y[4],r[4];
    #pragma unroll
    for(int k=0;k<4;++k){ x[k]=a[4*i+k]; y[k]=b[4*i+k]; }
    cr::mul(r,x,y);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=r[k];
}
__global__ void k_sqr_cr(const uint64_t* a, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x[4],r[4];
    #pragma unroll
    for(int k=0;k<4;++k) x[k]=a[4*i+k];
    cr::sqr(r,x);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=r[k];
}
__global__ void k_inv_cyc(const uint64_t* a, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t t[5]; t[4]=0;
    #pragma unroll
    for(int k=0;k<4;++k) t[k]=a[4*i+k];
    _ModInv(t);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=t[k];
}
__global__ void k_inv_rck(const uint64_t* a, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t t[5]; t[4]=0;
    #pragma unroll
    for(int k=0;k<4;++k) t[k]=a[4*i+k];
    rck::rinv(t);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=t[k];
}
__global__ void k_sub_cyc(const uint64_t* a, const uint64_t* b, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x[4],y[4];
    #pragma unroll
    for(int k=0;k<4;++k){ x[k]=a[4*i+k]; y[k]=b[4*i+k]; }
    ModSub256(x,x,y);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=x[k];
}
__global__ void k_sub_rck(const uint64_t* a, const uint64_t* b, uint64_t* o, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x[4],y[4],r[4];
    #pragma unroll
    for(int k=0;k<4;++k){ x[k]=a[4*i+k]; y[k]=b[4*i+k]; }
    rck::rsub(r,x,y);
    #pragma unroll
    for(int k=0;k<4;++k) o[4*i+k]=r[k];
}

// ---------------------------------------------------- throughput (dependent chains)
__global__ void t_mul_cyc(const uint64_t* seed, const uint64_t* kk, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t a[4],c[4];
    #pragma unroll
    for(int k=0;k<4;++k){ a[k]=seed[4*i+k]; c[k]=kk[k]; }
    #pragma unroll 1
    for(int it=0; it<iters; ++it) _ModMult(a,a,c);
    sink[i]=a[0]^a[1]^a[2]^a[3];
}
__global__ void t_mul_rck(const uint64_t* seed, const uint64_t* kk, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t a[4],c[4];
    #pragma unroll
    for(int k=0;k<4;++k){ a[k]=seed[4*i+k]; c[k]=kk[k]; }
    #pragma unroll 1
    for(int it=0; it<iters; ++it) rck::rmul(a,a,c);
    sink[i]=a[0]^a[1]^a[2]^a[3];
}
__global__ void t_sqr_cyc(const uint64_t* seed, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t a[4];
    #pragma unroll
    for(int k=0;k<4;++k) a[k]=seed[4*i+k];
    #pragma unroll 1
    for(int it=0; it<iters; ++it){ uint64_t r[4]; _ModSqr(r,a); a[0]=r[0];a[1]=r[1];a[2]=r[2];a[3]=r[3]; }
    sink[i]=a[0]^a[1]^a[2]^a[3];
}
__global__ void t_sqr_rck(const uint64_t* seed, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t a[4];
    #pragma unroll
    for(int k=0;k<4;++k) a[k]=seed[4*i+k];
    #pragma unroll 1
    for(int it=0; it<iters; ++it){ uint64_t r[4]; rck::rsqr(r,a); a[0]=r[0];a[1]=r[1];a[2]=r[2];a[3]=r[3]; }
    sink[i]=a[0]^a[1]^a[2]^a[3];
}
__global__ void t_inv_cyc(const uint64_t* seed, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t t[5]; t[4]=0;
    #pragma unroll
    for(int k=0;k<4;++k) t[k]=seed[4*i+k];
    #pragma unroll 1
    for(int it=0; it<iters; ++it){ t[4]=0; _ModInv(t); }
    sink[i]=t[0]^t[1]^t[2]^t[3];
}
__global__ void t_inv_rck(const uint64_t* seed, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t t[5]; t[4]=0;
    #pragma unroll
    for(int k=0;k<4;++k) t[k]=seed[4*i+k];
    #pragma unroll 1
    for(int it=0; it<iters; ++it){ t[4]=0; rck::rinv(t); }
    sink[i]=t[0]^t[1]^t[2]^t[3];
}
// "ecstep": one shared-inverse +/- twin per iteration == the real kernel's per-2-key blend:
// 6 mul + 2 sqr + 9 sub + 1 neg (3 mul : 1 sqr per key), matching kernel_point_add_and_check_oneinv.
// Sub/neg go through CUDACyclone's ModSub256/ModNeg256 in BOTH variants, because no shipped RCK
// target swaps sub/neg (only mul/sqr under -DUSE_RCK_FIELD) -- so this predicts rckfield, not a
// phantom "RCK sub" config. MUL/SQR are the only difference; both _ModMult and rck::rmul provide
// the 2-arg (r*=a) and 3-arg (r=a*b) overloads the macro relies on.
#define EC_TWIN_BODY(MUL, SQR) \
    uint64_t s[4], t[4], lam[4], px3[4], ya[4], gyn[4]; \
    MUL(dxi, gx);                                   /* inverse-share evolve (prefix/unwind stand-in) */ \
    ModSub256(s, gy, y1);   MUL(lam, s, dxi);       /* + branch: 2 mul, 1 sqr, 4 sub */ \
    SQR(px3, lam);          ModSub256(px3, px3, x1); ModSub256(px3, px3, gx); \
    ModSub256(t, x1, px3);  MUL(ya, t, lam); \
    gyn[0]=gy[0];gyn[1]=gy[1];gyn[2]=gy[2];gyn[3]=gy[3]; ModNeg256(gyn, gyn); \
    ModSub256(s, gyn, y1);  MUL(lam, s, dxi);       /* - branch: 2 mul, 1 sqr, 4 sub, 1 neg */ \
    SQR(t, lam);            ModSub256(t, t, x1);     ModSub256(t, t, gx); \
    ModSub256(s, x1, t);    MUL(s, s, lam); \
    MUL(dxi, dxi, gy);                              /* running-inverse unwind mul => 6 mul, 2 sqr total */ \
    x1[0]=px3[0];x1[1]=px3[1];x1[2]=px3[2];x1[3]=px3[3]; \
    y1[0]=ya[0]^s[0];y1[1]=ya[1]^s[1];y1[2]=ya[2]^s[2];y1[3]=ya[3]^s[3];  /* keep both branches live */

__global__ void t_ec_cyc(const uint64_t* seed, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x1[4],y1[4],dxi[4],gx[4],gy[4];
    #pragma unroll
    for(int k=0;k<4;++k){ x1[k]=seed[4*i+k]; y1[k]=seed[4*i+((k+1)&3)]; dxi[k]=seed[4*i+((k+2)&3)]|1ULL; }
    #pragma unroll
    for(int k=0;k<4;++k){ gx[k]=SECP_GX_LE[k]; gy[k]=SECP_GY_LE[k]; }
    #pragma unroll 1
    for(int it=0; it<iters; ++it){ EC_TWIN_BODY(_ModMult, _ModSqr) }
    sink[i]=x1[0]^x1[1]^x1[2]^x1[3]^y1[0]^dxi[0];
}
__global__ void t_ec_rck(const uint64_t* seed, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x1[4],y1[4],dxi[4],gx[4],gy[4];
    #pragma unroll
    for(int k=0;k<4;++k){ x1[k]=seed[4*i+k]; y1[k]=seed[4*i+((k+1)&3)]; dxi[k]=seed[4*i+((k+2)&3)]|1ULL; }
    #pragma unroll
    for(int k=0;k<4;++k){ gx[k]=SECP_GX_LE[k]; gy[k]=SECP_GY_LE[k]; }
    #pragma unroll 1
    for(int it=0; it<iters; ++it){ EC_TWIN_BODY(rck::rmul, rck::rsqr) }
    sink[i]=x1[0]^x1[1]^x1[2]^x1[3]^y1[0]^dxi[0];
}
__global__ void t_mul_cr(const uint64_t* seed, const uint64_t* kk, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t a[4],c[4];
    #pragma unroll
    for(int k=0;k<4;++k){ a[k]=seed[4*i+k]; c[k]=kk[k]; }
    #pragma unroll 1
    for(int it=0; it<iters; ++it) cr::mul(a,a,c);
    sink[i]=a[0]^a[1]^a[2]^a[3];
}
__global__ void t_ec_cr(const uint64_t* seed, uint64_t* sink, int N, int iters){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    uint64_t x1[4],y1[4],dxi[4],gx[4],gy[4];
    #pragma unroll
    for(int k=0;k<4;++k){ x1[k]=seed[4*i+k]; y1[k]=seed[4*i+((k+1)&3)]; dxi[k]=seed[4*i+((k+2)&3)]|1ULL; }
    #pragma unroll
    for(int k=0;k<4;++k){ gx[k]=SECP_GX_LE[k]; gy[k]=SECP_GY_LE[k]; }
    #pragma unroll 1
    for(int it=0; it<iters; ++it){ EC_TWIN_BODY(cr::mul, cr::sqr) }
    sink[i]=x1[0]^x1[1]^x1[2]^x1[3]^y1[0]^dxi[0];
}
#undef EC_TWIN_BODY

// ============================================================================ driver
#define CK(x) do{ cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA error %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} }while(0)

static void self_test_reference(){
    uint64_t r[4];
    host_mulmod(KAT_A,KAT_B,r); if(!eq4(r,KAT_MUL)){ printf("FATAL: host ref mul != KAT\n"); exit(2);}
    host_mulmod(KAT_A,KAT_A,r); if(!eq4(r,KAT_SQR)){ printf("FATAL: host ref sqr != KAT\n"); exit(2);}
    host_submod(KAT_A,KAT_B,r); if(!eq4(r,KAT_SUB)){ printf("FATAL: host ref sub != KAT\n"); exit(2);}
    host_addmod(KAT_A,KAT_B,r); if(!eq4(r,KAT_ADD)){ printf("FATAL: host ref add != KAT\n"); exit(2);}
    host_mulmod(KAT_INV,KAT_A,r); uint64_t one[4]={1,0,0,0};
    if(!eq4(r,one)){ printf("FATAL: host ref inv*a != 1\n"); exit(2);}
    printf("[ref] host reference self-test vs Python KATs: OK\n");
}

int main(int argc, char** argv){
    int reps  = (argc>1)? atoi(argv[1]) : 5;
    int scale = (argc>2)? atoi(argv[2]) : 1;
    if(reps<1) reps=1; if(scale<1) scale=1;

    self_test_reference();

    int dev=0; cudaDeviceProp prop{}; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop,dev));
    printf("[dev] %s  (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // -- correctness data --------------------------------------------------------
    const int N = 1<<16;
    std::mt19937_64 rng(0xC0FFEEULL);
    std::vector<uint64_t> ha(4*N), hb(4*N);
    for(int i=0;i<N;++i){
        for(int k=0;k<4;++k){ ha[4*i+k]=rng(); hb[4*i+k]=rng(); }
        canon(&ha[4*i]); canon(&hb[4*i]);
        if((ha[4*i]|ha[4*i+1]|ha[4*i+2]|ha[4*i+3])==0) ha[4*i]=1; // avoid 0 for inv
    }
    uint64_t *da,*db,*dout; CK(cudaMalloc(&da,4*N*8)); CK(cudaMalloc(&db,4*N*8)); CK(cudaMalloc(&dout,4*N*8));
    CK(cudaMemcpy(da,ha.data(),4*N*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db,hb.data(),4*N*8,cudaMemcpyHostToDevice));
    std::vector<uint64_t> hoc(4*N), hor(4*N);
    int tpb=256, bl=(N+tpb-1)/tpb;

    // Compare two already-copied-back device outputs (hoc=cyclone, hor=rck) against the
    // host reference. mode: 0=mul (ref=a*b), 1=sub (ref=a-b), 2=sqr (ref=a*a).
    auto report = [&](const char* name, int mode){
        int badc=0,badr=0,ncc=0,ncr=0;
        for(int i=0;i<N;++i){
            uint64_t ref[4], cc[4], cr[4];
            if(mode==0)      host_mulmod(&ha[4*i],&hb[4*i],ref);
            else if(mode==1) host_submod(&ha[4*i],&hb[4*i],ref);
            else             host_mulmod(&ha[4*i],&ha[4*i],ref);
            memcpy(cc,&hoc[4*i],32); memcpy(cr,&hor[4*i],32);
            if(ge_p(cc)) ncc++; if(ge_p(cr)) ncr++;
            canon(cc); canon(cr);
            if(!eq4(cc,ref)) badc++;
            if(!eq4(cr,ref)) badr++;
        }
        printf("[chk] %-7s N=%d  cyclone_vs_ref=%s  rck_vs_ref=%s  (mismatch cyc=%d rck=%d)  non-canonical>=P: cyc=%d rck=%d\n",
               name, N, badc?"FAIL":"OK", badr?"FAIL":"OK", badc, badr, ncc, ncr);
    };

    // mulmod
    k_mul_cyc<<<bl,tpb>>>(da,db,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hoc.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
    k_mul_rck<<<bl,tpb>>>(da,db,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hor.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
    report("mulmod",0);
    // submod
    k_sub_cyc<<<bl,tpb>>>(da,db,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hoc.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
    k_sub_rck<<<bl,tpb>>>(da,db,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hor.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
    report("submod",1);
    // sqrmod
    k_sqr_cyc<<<bl,tpb>>>(da,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hoc.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
    k_sqr_rck<<<bl,tpb>>>(da,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hor.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
    report("sqrmod",2);
    // invmod: check inv(a)*a == 1 via host mul
    {
        k_inv_cyc<<<bl,tpb>>>(da,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hoc.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
        k_inv_rck<<<bl,tpb>>>(da,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hor.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
        int badc=0,badr=0,agree=0; uint64_t one[4]={1,0,0,0};
        for(int i=0;i<N;++i){ uint64_t pc[4],pr[4]; host_mulmod(&hoc[4*i],&ha[4*i],pc); host_mulmod(&hor[4*i],&ha[4*i],pr);
            if(!eq4(pc,one))badc++; if(!eq4(pr,one))badr++;
            if(eq4(&hoc[4*i],&hor[4*i]))agree++; }
        printf("[chk] %-7s N=%d  cyclone(inv*a==1)=%s  rck(inv*a==1)=%s  (fail cyc=%d rck=%d)  identical-bits cyc==rck: %d/%d\n",
               "invmod",N,badc?"FAIL":"OK",badr?"FAIL":"OK",badc,badr,agree,N);
    }
    // clean-room backend (cr::mul / cr::sqr) vs the same independent host reference
    {
        k_mul_cr<<<bl,tpb>>>(da,db,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hoc.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
        int bad=0,nc=0;
        for(int i=0;i<N;++i){ uint64_t ref[4],cc[4]; host_mulmod(&ha[4*i],&hb[4*i],ref); memcpy(cc,&hoc[4*i],32); if(ge_p(cc))nc++; canon(cc); if(!eq4(cc,ref))bad++; }
        printf("[chk] %-7s N=%d  cr_vs_ref=%s  (mismatch %d)  non-canonical>=P: %d\n","mul(cr)",N,bad?"FAIL":"OK",bad,nc);
        k_sqr_cr<<<bl,tpb>>>(da,dout,N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hoc.data(),dout,4*N*8,cudaMemcpyDeviceToHost));
        int bad2=0;
        for(int i=0;i<N;++i){ uint64_t ref[4],cc[4]; host_mulmod(&ha[4*i],&ha[4*i],ref); memcpy(cc,&hoc[4*i],32); canon(cc); if(!eq4(cc,ref))bad2++; }
        printf("[chk] %-7s N=%d  cr_vs_ref=%s  (mismatch %d)\n","sqr(cr)",N,bad2?"FAIL":"OK",bad2);
    }
    // Lazy-INPUT edge: random inputs land in [0,P), so the [P,2^256) reduction band (width ~2^32)
    // is never exercised above. Feed a = P + r (r in [1, 2^256-P)) so both ops must reduce a
    // non-canonical input -- exactly what the kernel does between ops. ref uses a's residue r.
    {
        const int M = 4096;
        std::vector<uint64_t> la(4*M), lb(4*M);
        for(int i=0;i<M;++i){
            uint64_t r = (rng() % 0x1000003D0ULL) + 1ULL;             // r in [1, 2^256-P)
            uint64_t a[4]={P_LIMBS[0],P_LIMBS[1],P_LIMBS[2],P_LIMBS[3]};
            unsigned __int128 t=(unsigned __int128)a[0]+r; a[0]=(uint64_t)t; uint64_t c=(uint64_t)(t>>64);
            a[1]+=c; if(a[1]<c){ a[2]++; if(a[2]==0) a[3]++; }        // a = P + r  (in [P,2^256))
            for(int k=0;k<4;++k){ la[4*i+k]=a[k]; lb[4*i+k]=rng(); }
            canon(&lb[4*i]);
        }
        CK(cudaMemcpy(da,la.data(),4*M*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(db,lb.data(),4*M*8,cudaMemcpyHostToDevice));
        int blM=(M+tpb-1)/tpb;
        k_mul_cyc<<<blM,tpb>>>(da,db,dout,M); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hoc.data(),dout,4*M*8,cudaMemcpyDeviceToHost));
        k_mul_rck<<<blM,tpb>>>(da,db,dout,M); CK(cudaGetLastError()); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(hor.data(),dout,4*M*8,cudaMemcpyDeviceToHost));
        int badc=0,badr=0;
        for(int i=0;i<M;++i){ uint64_t ref[4],cc[4],cr[4]; host_mulmod(&la[4*i],&lb[4*i],ref);
            memcpy(cc,&hoc[4*i],32); memcpy(cr,&hor[4*i],32); canon(cc); canon(cr);
            if(!eq4(cc,ref))badc++; if(!eq4(cr,ref))badr++; }
        printf("[chk] %-7s M=%d  cyclone_vs_ref=%s  rck_vs_ref=%s  (mismatch cyc=%d rck=%d)  [inputs in [P,2^256)]\n",
               "mul/lazy",M,badc?"FAIL":"OK",badr?"FAIL":"OK",badc,badr);
    }
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dout));

    // ------------------------------------------------------------- throughput ----
    int T_tpb=256, T_bl=prop.multiProcessorCount*16;
    long long Nthreads=(long long)T_bl*T_tpb;
    uint64_t *tseed,*tk,*tsink; CK(cudaMalloc(&tseed,Nthreads*4*8)); CK(cudaMalloc(&tk,4*8)); CK(cudaMalloc(&tsink,Nthreads*8));
    { std::vector<uint64_t> s(Nthreads*4); for(long long i=0;i<Nthreads*4;++i) s[i]=rng()|1ULL;
      CK(cudaMemcpy(tseed,s.data(),Nthreads*4*8,cudaMemcpyHostToDevice));
      CK(cudaMemcpy(tk,KAT_B,4*8,cudaMemcpyHostToDevice)); }
    // ecstep uses SECP_GX_LE/SECP_GY_LE (from CUDAMath.h) as fixed non-trivial field constants.

    cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    auto timed = [&](const char* label, double opsPerIter, int iters, auto launch){
        launch(iters); CK(cudaDeviceSynchronize());               // warmup
        double best=1e30;
        for(int r=0;r<reps;++r){
            CK(cudaEventRecord(e0)); launch(iters); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms=0; CK(cudaEventElapsedTime(&ms,e0,e1)); if(ms<best) best=ms;
        }
        double ops=(double)Nthreads*iters*opsPerIter;
        double mops=ops/(best*1e3);
        return mops;
    };

    int itMul=2000*scale, itSqr=2000*scale, itInv=64*scale, itEc=400*scale;
    printf("--- throughput  (threads=%lld, reps=%d, best-of; Mops/s, higher=better) ---\n", Nthreads, reps);

    double m1=timed("mul_cyc",1,itMul,[&](int it){ t_mul_cyc<<<T_bl,T_tpb>>>(tseed,tk,tsink,(int)Nthreads,it); });
    double m2=timed("mul_rck",1,itMul,[&](int it){ t_mul_rck<<<T_bl,T_tpb>>>(tseed,tk,tsink,(int)Nthreads,it); });
    printf("mulmod   cyclone %8.1f   rck %8.1f   rck/cyc %.3fx\n", m1, m2, m2/m1);

    double s1=timed("sqr_cyc",1,itSqr,[&](int it){ t_sqr_cyc<<<T_bl,T_tpb>>>(tseed,tsink,(int)Nthreads,it); });
    double s2=timed("sqr_rck",1,itSqr,[&](int it){ t_sqr_rck<<<T_bl,T_tpb>>>(tseed,tsink,(int)Nthreads,it); });
    printf("sqrmod   cyclone %8.1f   rck %8.1f   rck/cyc %.3fx\n", s1, s2, s2/s1);

    double v1=timed("inv_cyc",1,itInv,[&](int it){ t_inv_cyc<<<T_bl,T_tpb>>>(tseed,tsink,(int)Nthreads,it); });
    double v2=timed("inv_rck",1,itInv,[&](int it){ t_inv_rck<<<T_bl,T_tpb>>>(tseed,tsink,(int)Nthreads,it); });
    printf("invmod   cyclone %8.1f   rck %8.1f   rck/cyc %.3fx\n", v1, v2, v2/v1);

    double e_1=timed("ec_cyc",1,itEc,[&](int it){ t_ec_cyc<<<T_bl,T_tpb>>>(tseed,tsink,(int)Nthreads,it); });
    double e_2=timed("ec_rck",1,itEc,[&](int it){ t_ec_rck<<<T_bl,T_tpb>>>(tseed,tsink,(int)Nthreads,it); });
    printf("ecstep   cyclone %8.1f   rck %8.1f   rck/cyc %.3fx   (6mul+2sqr per +/- twin; sub via Cyclone)\n", e_1, e_2, e_2/e_1);

    double mc=timed("mul_cr",1,itMul,[&](int it){ t_mul_cr<<<T_bl,T_tpb>>>(tseed,tk,tsink,(int)Nthreads,it); });
    double ec=timed("ec_cr", 1,itEc,[&](int it){ t_ec_cr <<<T_bl,T_tpb>>>(tseed,tsink,(int)Nthreads,it); });
    printf("clean-room mul %8.1f  cr/cyc %.3fx  cr/rck %.3fx   |   ecstep %8.1f  cr/cyc %.3fx  cr/rck %.3fx\n",
           mc, mc/m1, mc/m2, ec, ec/e_1, ec/e_2);

    printf("\nInterpretation: ecstep reproduces the kernel's per-key blend (~3 mul : 1 sqr; sub NOT\n"
           "swapped, matching rckfield) so ecstep rck/cyc APPROXIMATES the mul/sqr-limited end-to-end\n"
           "ratio -- but the AUTHORITATIVE number is ab_ec.sh's end-to-end Mkeys/s (it also captures\n"
           "register pressure / occupancy, which is what sank the earlier 32-bit multiply). Inversion is\n"
           "amortized ~1 per batch, so invmod barely moves the total.\n");
    return 0;
}
