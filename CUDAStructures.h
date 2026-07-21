#define WARP_SIZE 32
#define FOUND_NONE  0
#define FOUND_LOCK  1
#define FOUND_READY 2

// The device records only WHERE the hit was, never the point itself: the host re-derives the
// public key from the recovered private key (see the FOUND_READY branch in main).
//
// Dropping Rx/Ry is not just a size saving -- it deletes a whole y3 recomputation from three of
// the four found sites. Those sites already compute s = (x1 - px3) * lam, but ModSub256isOdd
// consumes s and DISCARDS the difference, so each site had to redo ModSub/ModMult/ModSub purely
// to refill Ry.
//
// `scalar` is the thread's CURRENT batch centre: this kernel advances S by B at the bottom of
// every batch (the carry chain just before the write-back) and persists it, so at hit time S is
// the centre of the batch being scanned and the key is exactly S + offset. NOTE this differs from
// the f1-all3 branch, whose S0 is the constant FIRST-batch centre and whose host therefore needs
// S0 + (span - rem) + offset. Do not port that formula here -- the extra term yields a wrong key
// that still prints as a well-formed 64-hex-digit number, and nothing validates it.
struct FoundResult {
    uint64_t scalar[4];   // S: current batch centre at hit time
    int32_t  offset;      // signed intra-batch delta: 0 = centre(x1), +(i+1) = +iG, -(i+1) = -iG, -half = last
};

// Target hash160 as five little-endian 32-bit words (word i = LE load of target bytes [4i..4i+3]).
// Hash comparison runs in word space, so candidate hashes are never serialized to bytes.
__device__ __constant__ uint32_t c_target_words[5];

__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);
