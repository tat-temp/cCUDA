#ifndef CUDA_HASH_CUH
#define CUDA_HASH_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <cstring>

// getHash160 uses a BY-VALUE ABI across the __noinline__ boundary: the 256-bit x-coordinate goes
// in as a register-passed struct and the 5-word hash160 comes back the same way. This is not a
// style choice -- it is the single largest codegen win of the campaign, and the mechanism is
// specific:
//
// A POINTER argument forces its pointee to have an addressable stack home, which in turn drags
// the field ops that PRODUCE that value (_ModSqr + ModSub256_2 building px3) through local memory
// instead of registers. Measured on f1-all3, sm_120, local ops (STL/LDL) by address region:
//
//                        f48ae08   +P3.0 (rdc=false)   +P3.1 (this, by-value)   mole
//   kernel body            185            49                   20                --
//   hash body                9(generic)   45                    0                 0
//   stack frame           288 B         288 B                 40 B              32 B
//   registers               122           124                  122              ~120
//
// i.e. ~5x less DYNAMIC local traffic (~56k -> ~10k ops/batch at B=1024), matching mole's profile
// on both hot terms -- without the ground-up register-lean field rewrite the campaign twice
// declined. Registers went DOWN, not up: ptxas saves more by not maintaining stack addresses than
// value-passing costs.
//
// getHash160 deliberately stays __noinline__. We want the CALL kept and only its ABI changed --
// inlining would stack the ~R64 hash working set onto the kernel's R122 and breach the 128-reg /
// 16-warp budget. Paired with the single-TU / rdc=false build (see Makefile) so ptxas emits an
// intra-module CALL.REL with a custom register convention rather than the conservative,
// stack-heavy cross-TU ABI-stable one.
//
// On the f1-all3 kernel the residual 40-byte frame is exactly inverse[5] (5 x u64) -- the last
// pointer-escape, because _ModInv(uint64_t*) is still __noinline__. Mole carries 32 B likewise.
//
// ⚠ THE ABSOLUTE FRAME NUMBERS ABOVE ARE f1-all3-SPECIFIC -- do not read them as a gate on every
// kernel. A kernel that declares a large LOCAL array has that array in its stack frame too: main's
// `uint64_t subp[MAX_BATCH_SIZE/2][4]` is 16 KB and dominates the frame, so main reads ~16 KB and
// that is CORRECT, not a regression. The portable regression signal is the DELTA across a change
// (and the STL/LDL count by address region), never the absolute frame size.
//
// DO NOT revert either struct to a pointer without re-reading `make ptxinfo`: the addressability
// spill returns (on f1-all3 the frame jumped back toward 288 B) and the hot-loop traffic with it.

struct H160 { uint32_t w[5]; };   // 5-word hash160, returned in registers
struct U256 { uint64_t v[4]; };   // 256-bit x-coordinate, passed in registers

// Pack 4 limbs into the by-value carrier. __forceinline__, so this is register moves, not a copy.
__device__ __forceinline__ U256 u256_of(const uint64_t x[4]) {
    U256 r; r.v[0] = x[0]; r.v[1] = x[1]; r.v[2] = x[2]; r.v[3] = x[3]; return r;
}

__device__ H160 getHash160_33_from_limbs(uint8_t prefix02_03, U256 x);

#endif
