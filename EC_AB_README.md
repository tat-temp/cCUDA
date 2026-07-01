# RCKangaroo vs CUDACyclone — EC field-arithmetic A/B

Branch `rck-ec-ab`. This experiment swaps **RetiredCoder's RCKangaroo** secp256k1 field
primitives in for CUDACyclone's own and measures the effect, both per-op and end-to-end.

## What is actually being compared

Both projects use the *same* EC algorithm — batched affine point-add amortized over **one**
Montgomery inversion per batch (prefix-product of differences → single inverse → unwind).
RCKangaroo's `KernelA` and CUDACyclone's `kernel_point_add_and_check_oneinv` are structurally
identical. So "EC logic" reduces to the **field primitives**, and only those differ:

| op        | CUDACyclone (`CUDAMath.h`)            | RCKangaroo (`third_party/RCKangaroo/RCGpuUtils.h`) |
|-----------|--------------------------------------|----------------------------------------------------|
| mul       | `_ModMult` — 64-bit limbs (`mad.hi`) | `MulModP` — **32-bit** limbs (`mul.wide.u32`)      |
| sqr       | `_ModSqr` — 64-bit limbs             | `SqrModP` — 32-bit limbs, hand-scheduled           |
| inverse   | `_ModInv` — JeanLucPons 62-bit divstep binary-GCD | `InvModP` — **safegcd** (Bernstein–Yang), 30-bit divsteps on 288-bit/32-bit words |
| add/sub   | `ModSub256`/`ModNeg256`              | `SubModP`/`AddModP`/`NegModP` (equivalent)         |

Operation mix per key in the hot loop is dominated by **mul + sqr**; the **inverse is
amortized to ~1 call per batch** of `B` keys (B up to 1024). So:
* `MulModP`/`SqrModP` are the real throughput lever.
* `InvModP` barely moves end-to-end Mkeys/s no matter how much faster it is in isolation.

## ⚠️ Prior result to keep in mind

The perf roadmap already recorded that a **32-bit-limb multiply lost decisively (-9% / -16%)**
end-to-end on this 5090 — most likely register-pressure / occupancy, not raw op latency.
`MulModP` *is* a 32-bit-limb multiply (a much more carefully scheduled one), so there is a real
prior that `rckfield` loses end-to-end. The safegcd `InvModP` is the more likely isolated win,
but it is amortized. Treat a headline win as surprising and re-verify it.

## The swap mechanism (zero kernel edits)

`CUDAMath.h` routes the ops through the RCKangaroo backend under compile flags — the kernel's
call sites are untouched:

* `-DUSE_RCK_FIELD` → `_ModMult` + `_ModSqr` become `MulModP` / `SqrModP`
* `-DUSE_RCK_INV`   → `_ModInv` becomes `InvModP`

**`ModSub256` / `ModNeg256` (add/sub/neg) are NOT swapped by any target** — the kernel always
uses CUDACyclone's, deliberately isolating the mul/sqr (and separately inv) lever. `ecbench`
validates `rck::rsub` for completeness only; it is not in any shipped binary.

`ec_backend.cuh` is the shim (wraps RCKangaroo's ops in `namespace rck`, uniform signatures).

Both libraries use the same **lazy reduction** convention (outputs congruent mod P in
`[0, 2^256)`, not necessarily `< P`), so the two are interchangeable **as residues mod P** at
every internal step. They may pick *different* non-canonical representatives for the ~2⁻²²⁴
fraction of values whose canonical form is `< ~2^32` (one emits `v`, the other `v+P`) — harmless,
because the baseline shares that identical ~2⁻²²⁴ non-canonical tail at the raw-limb hash boundary.
An optional `-DRCK_CANON` forces RCKangaroo's mul/sqr to canonical `[0,P)` if exact byte-parity is
ever wanted; it is **off by default** because CUDACyclone doesn't canonicalize either (fair A/B).
`ecbench`'s `non-canonical>=P` counters are ~2⁻²²⁴ and thus effectively always `0` at `N=2^16`
(a sampling gap, not a proof); the `mul/lazy` row instead exercises the reduction on inputs
actually in `[P, 2^256)`, which is the meaningful edge.

## Build targets (`make …`)

| target      | binary                    | meaning                                    |
|-------------|---------------------------|--------------------------------------------|
| *(default)* | `CUDACyclone`             | baseline (CUDACyclone field ops)           |
| `rckfield`  | `CUDACyclone-rckfield`    | RCKangaroo mul + sqr                        |
| `rckinv`    | `CUDACyclone-rckinv`      | RCKangaroo inverse only                     |
| `rckall`    | `CUDACyclone-rckall`      | RCKangaroo mul + sqr + inverse             |
| `crfield`   | `CUDACyclone-crfield`     | **clean-room** 32-bit mul + sqr (license-clean; see below) |
| `ecbench`   | `CUDACyclone-ecbench`     | standalone per-op correctness + microbench |

## Clean-room multiply (`cr_field.cuh`, license-clean)

`rckfield` proved RCKangaroo's 32-bit multiply is worth **+7%** end-to-end — but it's GPLv3.
`cr_field.cuh` is a from-scratch reimplementation of *that idea* (256×256 via `mul.wide.u32`
+ the public secp256k1 pseudo-Mersenne reduction, folded with CUDACyclone's own convention),
written without consulting RetiredC's source, so it can merge under this repo's license. It
routes through `-DUSE_CR_FIELD` exactly like the RCK backend. The multiply's limb-carry logic
was validated bit-exact against a Python reference over ~1.5M trials before being written.

The make-or-break question is **register pressure**: the earlier Phase-3 32-bit attempt *spilled*
(−9%/−16%) because it held split operand arrays past the 128-reg `(256,2)` cap. `cr_field.cuh`
indexes operands in place to avoid that. `ab_ec.sh` prints `crfield`'s `cuobjdump -res-usage`
right after building — **want `REG≤128, LOCAL:0, no spills`**. If it spills, rebuild with
`CR_NOINLINE=1 make crfield` (moves the multiply's registers off the kernel via `__noinline__`)
and re-measure. `ecbench` reports `cr/cyc` and `cr/rck` so you can see how close the clean-room
multiply gets to RCKangaroo's in isolation; `ab_ec.sh` includes `crfield` in the end-to-end A/B.
The clean-room does **not** reimplement the safegcd inverse (a separate ~+0.7%); `crfield` keeps
CUDACyclone's inverse, targeting the ~+7% multiply win.

## How to run (on the RTX 5090 box)

```bash
git fetch origin && git checkout rck-ec-ab && git reset --hard origin/rck-ec-ab   # avoid stale build

# One shot: build, gate correctness, run the interleaved throughput A/B.
bash ab_ec.sh 6 30            # 6 windows x 30 s per variant

# Or piecemeal:
make ecbench && ./CUDACyclone-ecbench      # per-op: correctness (vs KATs) + Mops/s + rck/cyc ratio
make rckall  && python3 proof.py -c ./CUDACyclone-rckall -r 8000000000:ffffffffff --grid 512,512
```

`ab_ec.sh` **aborts before timing** if either correctness gate fails, so a reported throughput
number always corresponds to a backend that actually finds keys.

### Reading the output

* **`ab_ec.sh`'s end-to-end Mkeys/s is the authoritative number** — it alone captures register
  pressure / occupancy, which is what sank the earlier 32-bit multiply (not raw op latency). It
  prints, per variant, mean Mkeys/s, CV%, `%` vs baseline, and Welch `t` vs baseline.
* `ecbench` `ecstep rck/cyc` *approximates* the mul/sqr-limited end-to-end ratio: it reproduces
  the real per-key blend (~3·mul + 1·sqr; ≈3.5·mul amortized once the prefix-product and
  running-inverse-unwind muls are counted) with sub/neg via CUDACyclone's ops (matching `rckfield`),
  and excludes the amortized inverse. Treat it as a fast directional predictor, not the verdict.
  `|t| < ~2` ⇒ within run-to-run noise. Negative `vs base%` ⇒ RCKangaroo is slower end-to-end.

## License note ⚠️

`third_party/RCKangaroo/` is **GPLv3, © 2024 RetiredCoder** (notice preserved). Compiling it into
a distributed CUDACyclone binary makes that binary a GPLv3 derivative. This branch is for **local
benchmarking**; do not merge the vendored code into a differently-licensed release without
resolving the license. Nuke it with `git branch -D rck-ec-ab` + `rm -rf third_party` if the
experiment is a dead end.
