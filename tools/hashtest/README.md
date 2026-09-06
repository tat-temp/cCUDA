# tools/hashtest — host-side verification for the hash and the hot kernel

**No GPU, no nvcc, no CUDA toolkit required.** A C++17 host compiler and Python are enough.

```bash
bash tools/hashtest/run.sh                  # working tree vs `main`
BASE_REF=cfd9c3b bash tools/hashtest/run.sh # pin the baseline commit
```

This exists because two of this project's checks cannot be done any other way:

* **A wrong hash runs at exactly the same speed**, so no keys/s A/B can catch one. Only an
  equivalence check can.
* **`proof.py` is end-to-end**, so when it fails it does not localise the fault, and it needs
  hardware. These tests isolate the hash and the kernel's control flow, and run anywhere.

It also exists because the campaign has repeatedly lost its verification tooling
(the split-column correctness harness was deleted the same day it was written, and its results
survive only in a memory note). Keep this directory alive.

## What each test proves

### `test_x2.cpp` — the real hash, compiled for the host

Compiles `CUDAHash.cu` unchanged with `-D__device__= -D__forceinline__=inline -D__noinline__=`
plus stubs for `__byte_perm` / `__funnelshift_r` (the method recorded in `CUDAHash.cu`'s own
comments). Then:

1. **Known-answer test** — `hash160(02||Gx)` must be `751e76e8199196d454941c45d1b3a323f1433bd6`
   (the pubkey of private key 1), checked as all five words and as word 2 alone (`0x451c9454`).
   This is what makes the emulation trustworthy, `__byte_perm` included.
2. **2-wide equivalence** — over 300k random pairs plus edge cases,
   `getHash160_w2_x2(pA,xA,pB,xB) == { getHash160_w2_from_limbs(pA,xA), getHash160_w2_from_limbs(pB,xB) }`.
3. **Word-2 trim invariant** — `getHash160_w2_from_limbs(p,x) == getHash160_33_from_limbs(p,x).w[2]`.
   Re-run this whenever the RIPEMD round list is touched: the trim's correctness rests on *where*
   each register is last written, so a moved round can still be a correct RIPEMD-160 while word 2
   silently stops matching the full digest.

### `difftest.cpp` — kernel control-flow differential

`extract2.py` pulls the hot kernel out of the baseline ref *and* the working tree, renames them,
and redirects their hash call sites to logging wrappers. Both are then compiled into one binary
over **identical deterministic stand-ins** for the PTX-asm field ops (g++ cannot assemble PTX, and
the real field math is not what this test is about). Because both kernels see the same mixers, any
difference is a control-flow divergence introduced by the change. It checks:

* the **candidate stream** — same count, same order, same `(prefix, x)` for every hashed point;
* the **end-of-launch write-back** — `Rx`, `Ry`, `start_scalars`, `counts256`, hash counter, `any_left`;
* the **found path** — every candidate is planted as the target in turn, and both kernels must
  agree on the found flag and on the recovered `scalar`, `Rx` and `Ry`.

That last one is the important one: the found path is this project's historical bug locus (the
prefix-collision key-skip, PR#5), it is reached about 2^-32 of the time, and a restructure there
is exactly what a throughput benchmark cannot see.

## What this does NOT cover — GPU box only

| check | command | why it matters here |
|---|---|---|
| register / spill budget | `make gate` | the decisive feasibility gate; ≤128 regs and 0 spill in **every** function |
| did the interleave fire? | `make sass` | the two chains must be interleaved, not chain-A-then-chain-B |
| scheduler effect | `ncu` | Executed IPC must rise from 2.17, no-eligible-warp must fall from 45.74% |
| end-to-end recovery | `make proof` | planted key on real hardware |
| throughput | `bench_ab.sh` | interleaved ABAB, `GRID=1024,512`, ≥6 reps |

## Notes

* `kernel_old.inc` / `kernel_new.inc` and the `.exe` files are generated; they are gitignored.
* After a change merges, `BASE_REF=main` compares the tree against itself and passes trivially.
  Pin `BASE_REF` to the pre-change commit to keep the test meaningful.
