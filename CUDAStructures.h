#define WARP_SIZE 32
#define FOUND_NONE  0
#define FOUND_LOCK  1
#define FOUND_READY 2

struct FoundResult {
    uint64_t scalar[4];
    uint64_t Rx[4];
    uint64_t Ry[4];
};

// Target hash160 as five little-endian 32-bit words (word i = LE load of target bytes [4i..4i+3]).
// Hash comparison runs in word space, so candidate hashes are never serialized to bytes.
__device__ __constant__ uint32_t c_target_words[5];

__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);
