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
// On the RETIRED f1-all3 kernel the residual 40-byte frame was exactly inverse[5] (5 x u64) -- the
// last pointer-escape, because _ModInv was __noinline__ THERE. Mole carried 32 B likewise.
// THIS NO LONGER DESCRIBES THIS SOURCE: _ModInv is __forceinline__ (CUDAMath.h:138), so inverse[5]
// stays in registers and the frame is exactly subp[] (16 KB) and nothing else. Kept because the
// dependency still bites: if _ModInv or InvModP is ever made __noinline__ again, inverse becomes
// addressable, lands in the frame, and the zero-init elided at CUDACyclone.cu:~195 turns back into
// real STLs. Recheck both together.
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
struct Hash2 { uint32_t w2a, w2b; };   // two hash160 word-2 values, returned in registers

// Pack 4 limbs into the by-value carrier. __forceinline__, so this is register moves, not a copy.
__device__ __forceinline__ U256 u256_of(const uint64_t x[4]) {
    U256 r; r.v[0] = x[0]; r.v[1] = x[1]; r.v[2] = x[2]; r.v[3] = x[3]; return r;
}

// Hot path: hash160 word 2 only (the cheapest of the five to produce -- see the trim rationale
// on RIPEMD160Transform in CUDAHash.cu). Returning one register instead of five also narrows
// the by-value ABI described above, in the same direction that won PR#15.
__device__ uint32_t getHash160_w2_from_limbs(uint8_t prefix02_03, U256 x);

// ---- 2-WIDE HOT PATH: the ILP lever -------------------------------------------------------
// Hashes TWO independent points in ONE __noinline__ call.
//
// WHY THIS EXISTS. Nsight (sm_120, CUDA 13.3) measures the kernel as LATENCY-bound, not
// pipe-bound: ALU pipe 46.9%, issue slots 47.68%, IPC 2.17 of 4, and no-eligible-warp 45.74%.
// The hash is 73% of all dynamic instructions and each SHA-256 is a 64-round serial dependency
// chain. A warp issues IN ORDER with no speculation across CALL/RET, so the main loop's two
// independent points (P+iG and P-iG) -- which used to make two separate __noinline__ calls --
// were HARD-SERIALIZED: block2's SHA could not begin until block1's RET, and block1's stalls
// could only ever be filled by OTHER warps, never by block2 of the same warp.
//
// Putting both chains inside ONE callee gives ptxas a single scheduling scope holding two
// independent dependency chains, so when chain A stalls on its RAW chain, chain B's next round
// is issuable. That is intra-warp ILP bought without adding warps (24 warps was measured WORSE:
// it spills at the 128-reg ceiling and saturates the ALU pipe).
//
// WHY __noinline__ IS LOAD-BEARING HERE. Fully INLINING the 1-wide hash into the kernel was
// measured at -6.95%: it made four inlined copies (9824 -> 18704 insns) and spilled 384/600 B
// against the 128-register ceiling. That failure was about paying for ILP out of the KERNEL's
// register budget. Keeping this callee __noinline__ confines its ~2x working set to its own
// frame (SHA and RIPEMD run sequentially inside, so the frame is max(), not sum) while the
// kernel still sees exactly one CALL. Do NOT make this __forceinline__.
//
// EXPECTED MAGNITUDE IS TEMPERED: RIPEMD-160 already exposes 2-way ILP internally (its two
// independent lines a1..e1 / a2..e2) and SHA has ~2-way intra-round ILP, so this widens roughly
// 2-way -> 4-way, not serial -> parallel. The largest marginal gain is SHA's cross-round chain.
//
// SCREEN BEFORE BENCHING (this is a SCHEDULING change, not a count change):
//   * `make gate`  -- kernel <= 128 regs and 0 spill in EVERY function, this callee included.
//   * `make sass`  -- the two chains must appear INTERLEAVED in the callee (alternating register
//                     clusters per round). If ptxas emitted all of chain A then all of chain B,
//                     the ILP did NOT fire and this is a null -- escalate to hand-interleaving
//                     the round lists before spending a bench slot.
//   * ncu          -- Executed IPC must RISE from 2.17 and no-eligible-warp must FALL from
//                     45.74%. Both flat => it compiled but did nothing.
__device__ Hash2 getHash160_w2_x2(uint8_t prefixA, U256 xA, uint8_t prefixB, U256 xB);

// Cold path: the full 160-bit digest, for confirming a word-2 filter hit (~2^-32 of keys).
__device__ H160 getHash160_33_from_limbs(uint8_t prefix02_03, U256 x);

#endif
