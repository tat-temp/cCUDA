// DIFFERENTIAL TEST: old (main) kernel vs new (2-wide) kernel, same stub field ops, real hash.
//
// The field ops are replaced by deterministic mixers, so the "points" are not on the curve --
// but BOTH kernels see the identical mixers, so any difference in (a) which candidates get
// hashed, in what order, or (b) what is written to d_found_result on a match, is a real
// control-flow divergence introduced by the restructure. That is exactly the property the
// numeric tests cannot check and the GPU-side `make proof` would only check end-to-end.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define __launch_bounds__(...)
#define __constant__

struct Dim3Stub { unsigned x, y, z; };
static Dim3Stub blockIdx  = {0, 0, 0};
static Dim3Stub blockDim  = {256, 1, 1};
static Dim3Stub threadIdx = {0, 0, 0};

static inline int  __any_sync(unsigned, bool p) { return p ? 1 : 0; }
static inline void __syncwarp(unsigned) {}
static inline void __threadfence_system() {}
static inline int  atomicCAS(int* a, int cmp, int val) { int o = *a; if (o == cmp) *a = val; return o; }
static inline int  atomicExch(int* a, int val) { int o = *a; *a = val; return o; }
static inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) { unsigned long long o = *a; *a += v; return o; }
static inline unsigned int       atomicAdd(unsigned int* a, unsigned int v) { unsigned int o = *a; *a += v; return o; }
static inline unsigned long long __shfl_down_sync(unsigned, unsigned long long v, int, int = 32) { return v; }

static inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s)
{
    unsigned long long tmp = ((unsigned long long)y << 32) | (unsigned long long)x;
    unsigned int r = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned int sel = (s >> (4 * i)) & 0xF;
        unsigned char b = (unsigned char)((tmp >> (8 * (sel & 7))) & 0xFF);
        unsigned char by = (sel & 0x8) ? ((b & 0x80) ? 0xFF : 0x00) : b;
        r |= ((unsigned int)by) << (8 * i);
    }
    return r;
}
static inline unsigned int __funnelshift_r(unsigned int lo, unsigned int hi, unsigned int sh)
{
    unsigned long long v = ((unsigned long long)hi << 32) | (unsigned long long)lo;
    return (unsigned int)(v >> (sh & 31));
}

#include "CUDAHash.cu"

#define WARP_SIZE 32
#define FOUND_NONE  0
#define FOUND_LOCK  1
#define FOUND_READY 2
struct FoundResult { uint64_t scalar[4]; uint64_t Rx[4]; uint64_t Ry[4]; };
#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif

__constant__ uint32_t c_target_words[5];
__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE / 2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE / 2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

static inline bool hash160_matches_full(const uint32_t h5[5], const uint32_t t[5])
{ for (int i = 0; i < 5; ++i) if (h5[i] != t[i]) return false; return true; }
static inline bool ge256_u64(const uint64_t a[4], uint64_t b)
{ if (a[3] | a[2] | a[1]) return true; return a[0] >= b; }
static inline void sub256_u64_inplace(uint64_t a[4], uint64_t dec)
{
    uint64_t br = (a[0] < dec) ? 1ull : 0ull; a[0] -= dec;
    { uint64_t t = a[1]; a[1] = t - br; br = (t < br) ? 1ull : 0ull; }
    { uint64_t t = a[2]; a[2] = t - br; br = (t < br) ? 1ull : 0ull; }
    { uint64_t t = a[3]; a[3] = t - br; br = (t < br) ? 1ull : 0ull; }
}
static inline unsigned long long warp_reduce_add_ull(unsigned long long v) { return v; }

// ---- deterministic, alias-safe stand-ins for the PTX field ops -----------------------------
static inline uint64_t mix64(uint64_t z)
{
    z += 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static void ModSub256isOdd(uint64_t* a, uint64_t* b, uint8_t* parity)
{ *parity = (uint8_t)((mix64(a[0] ^ b[0])) & 1); }
static void ModNeg256(uint64_t* r, uint64_t* a)
{ for (int i = 0; i < 4; ++i) r[i] = ~a[i] + 1u; }
static void ModSub256(uint64_t* r, uint64_t* a, uint64_t* b)
{ for (int i = 0; i < 4; ++i) r[i] = a[i] - b[i]; }
static void ModSub256(uint64_t* r, uint64_t* b)
{ for (int i = 0; i < 4; ++i) r[i] -= b[i]; }
static void ModSub256_2(uint64_t* r, const uint64_t* a, const uint64_t* b, const uint64_t* c)
{ for (int i = 0; i < 4; ++i) r[i] = a[i] - b[i] - c[i]; }
static void _ModInv(uint64_t* R)
{ uint64_t t[4]; for (int i = 0; i < 4; ++i) t[i] = mix64(R[i] + (uint64_t)i * 7 + 1);
  for (int i = 0; i < 4; ++i) R[i] = t[i]; }
static void _ModMult(uint64_t* r, uint64_t* a, uint64_t* b)
{ uint64_t t[4]; for (int i = 0; i < 4; ++i) t[i] = mix64(a[i] ^ mix64(b[(i + 1) & 3] + (uint64_t)i));
  for (int i = 0; i < 4; ++i) r[i] = t[i]; }
static void _ModMult(uint64_t* r, uint64_t* a) { _ModMult(r, r, a); }
static void _ModSqr(uint64_t* rp, const uint64_t* up)
{ uint64_t t[4]; for (int i = 0; i < 4; ++i) t[i] = mix64(up[i] + mix64(up[(i + 2) & 3]));
  for (int i = 0; i < 4; ++i) rp[i] = t[i]; }

// ---- candidate logging hooks ----------------------------------------------------------------
struct Rec { uint8_t p; uint64_t x[4]; };
static std::vector<Rec>* g_log = nullptr;
static void logrec(uint8_t p, const U256& x)
{ if (g_log) { Rec r; r.p = p; for (int i = 0; i < 4; ++i) r.x[i] = x.v[i]; g_log->push_back(r); } }

static uint32_t HOOK_w2(uint8_t p, U256 x) { logrec(p, x); return getHash160_w2_from_limbs(p, x); }
static Hash2 HOOK_w2x2(uint8_t pA, U256 xA, uint8_t pB, U256 xB)
{ logrec(pA, xA); logrec(pB, xB); return getHash160_w2_x2(pA, xA, pB, xB); }

__device__ __forceinline__ bool hash160_full_match(uint8_t prefix02_03, U256 x, const uint32_t target_w[5])
{ H160 h5 = getHash160_33_from_limbs(prefix02_03, x); return hash160_matches_full(h5.w, target_w); }

#include "kernel_old.inc"
#include "kernel_new.inc"

// ---- driver ----------------------------------------------------------------------------------
static const uint32_t B_TEST = 16;          // even power of two
static const uint64_t THREADS = 1;
static const uint32_t MAXBATCH = 3;

struct RunOut {
    std::vector<Rec> log;
    int flag = FOUND_NONE;
    FoundResult res{};
    uint64_t Rx[4]{}, Ry[4]{}, scal[4]{}, cnt[4]{};
    unsigned long long hashes = 0;
    unsigned int anyleft = 0;
};

static void setup_tables()
{
    for (uint32_t k = 0; k < B_TEST / 2; ++k)
        for (int j = 0; j < 4; ++j) {
            c_Gx[k * 4 + j] = mix64(0x1000 + k * 4 + j);
            c_Gy[k * 4 + j] = mix64(0x2000 + k * 4 + j);
        }
    for (int j = 0; j < 4; ++j) { c_Jx[j] = mix64(0x3000 + j); c_Jy[j] = mix64(0x4000 + j); }
}

template <typename KFn>
static void run(KFn kfn, const uint32_t target[5], RunOut& out)
{
    uint64_t Px[4], Py[4], Rx[4] = {0}, Ry[4] = {0}, S[4], cnt[4];
    for (int j = 0; j < 4; ++j) {
        Px[j] = mix64(0x5000 + j);
        Py[j] = mix64(0x6000 + j);
        S[j]  = (j == 0) ? 100000ull : 0ull;
        cnt[j] = (j == 0) ? (uint64_t)B_TEST * 3ull : 0ull;
    }
    int flag = FOUND_NONE;
    FoundResult fr{};
    unsigned long long hashes = 0;
    unsigned int anyleft = 0;
    for (int i = 0; i < 5; ++i) c_target_words[i] = target[i];

    out.log.clear();
    g_log = &out.log;
    kfn(Px, Py, Rx, Ry, S, cnt, THREADS, B_TEST, MAXBATCH, &flag, &fr, &hashes, &anyleft);
    g_log = nullptr;

    out.flag = flag; out.res = fr; out.hashes = hashes; out.anyleft = anyleft;
    for (int j = 0; j < 4; ++j) { out.Rx[j] = Rx[j]; out.Ry[j] = Ry[j]; out.scal[j] = S[j]; out.cnt[j] = cnt[j]; }
}

static bool same_log(const std::vector<Rec>& a, const std::vector<Rec>& b, size_t n, const char* tag)
{
    for (size_t i = 0; i < n; ++i) {
        if (a[i].p != b[i].p || memcmp(a[i].x, b[i].x, 32) != 0) {
            printf("  FAIL %s: candidate %zu differs\n", tag, i);
            return false;
        }
    }
    return true;
}

static bool same_found(const RunOut& a, const RunOut& b)
{
    if (a.flag != b.flag) { printf("  FAIL: flag %d != %d\n", a.flag, b.flag); return false; }
    if (memcmp(a.res.scalar, b.res.scalar, 32) != 0) { printf("  FAIL: found scalar differs\n"); return false; }
    if (memcmp(a.res.Rx, b.res.Rx, 32) != 0) { printf("  FAIL: found Rx differs\n"); return false; }
    if (memcmp(a.res.Ry, b.res.Ry, 32) != 0) { printf("  FAIL: found Ry differs\n"); return false; }
    return true;
}

int main()
{
    setup_tables();
    int fail = 0;

    // ---- Test 1: no match anywhere -> identical candidate streams and identical end state ----
    const uint32_t no_match[5] = {0xDEADBEEFu, 1, 2, 3, 4};
    RunOut o1, n1;
    run(kernel_old, no_match, o1);
    run(kernel_new, no_match, n1);

    printf("Test 1 (no match): old hashed %zu candidates, new hashed %zu\n", o1.log.size(), n1.log.size());
    if (o1.log.size() != n1.log.size()) { printf("  FAIL: candidate COUNT differs\n"); ++fail; }
    else if (!same_log(o1.log, n1.log, o1.log.size(), "stream")) ++fail;
    else printf("  PASS: identical candidate stream (prefix + x), in order\n");

    if (memcmp(o1.Rx, n1.Rx, 32) || memcmp(o1.Ry, n1.Ry, 32) ||
        memcmp(o1.scal, n1.scal, 32) || memcmp(o1.cnt, n1.cnt, 32)) {
        printf("  FAIL: end-of-launch write-back differs\n"); ++fail;
    } else printf("  PASS: identical write-back (Rx, Ry, start_scalars, counts)\n");
    if (o1.hashes != n1.hashes || o1.anyleft != n1.anyleft) {
        printf("  FAIL: hash counter/any_left differ (%llu/%u vs %llu/%u)\n",
               o1.hashes, o1.anyleft, n1.hashes, n1.anyleft); ++fail;
    } else printf("  PASS: identical hash counter (%llu) and any_left (%u)\n", o1.hashes, o1.anyleft);

    // ---- Test 2: plant each candidate in turn as the target; both must find the same key ----
    printf("\nTest 2 (planted targets): %zu candidates\n", o1.log.size());
    int planted = 0, agreed = 0;
    for (size_t c = 0; c < o1.log.size(); ++c) {
        U256 x; for (int j = 0; j < 4; ++j) x.v[j] = o1.log[c].x[j];
        H160 h = getHash160_33_from_limbs(o1.log[c].p, x);
        RunOut o2, n2;
        run(kernel_old, h.w, o2);
        run(kernel_new, h.w, n2);
        ++planted;
        if (o2.flag != FOUND_READY) { printf("  note: candidate %zu did not trigger old kernel\n", c); }
        if (!same_found(o2, n2)) { printf("  (at planted candidate %zu)\n", c); ++fail; break; }
        ++agreed;
    }
    printf("  planted %d targets, %d agreed on flag+scalar+Rx+Ry\n", planted, agreed);
    if (planted == agreed && planted > 0) printf("  PASS: found-path equivalence\n");

    printf(fail ? "\nRESULT: FAIL (%d)\n" : "\nRESULT: ALL PASS\n", fail);
    return fail ? 1 : 0;
}
