# Split-column field math

The 256x256 -> 512-bit product cores used by `MulModP_split` / `SqrModP_split` in
`../../field_split.cuh` are GENERATED. Do not hand-edit them there -- edit the generator
and re-emit.

Blackwell fuses an adjacent `mad.lo.cc.u32` / `madc.hi.cc.u32` pair into one `IMAD.WIDE.U32.X`
(multiply + 64-bit accumulate + carry in/out), but only when the accumulator pair is 64-bit
aligned. Splitting the limb products by column parity makes every pair aligned, so every
fusion sticks.

    python3 gen.py    | tr -d '\r' > split.inc      # mul512_split
    python3 gensqr.py | tr -d '\r' > sqrsplit.inc   # sqr512_split
    python3 verify_split.py split.inc               # replays the emitted stream vs Python bignums
    python3 verify_sqr.py   sqrsplit.inc

`tr -d '\r'` matters on Windows checkouts (`core.autocrlf=true`), otherwise the emitted
`.inc` picks up CRLF and no longer matches the committed copy byte-for-byte.

Then paste the two `.inc` bodies into `field_split.cuh` between the BEGIN/END GENERATED
markers. The reduction tail there is NOT generated -- it is RCKangaroo's, unchanged.

The Python oracles cover 0, 2^256-1 squared, and random pairs. For the end-to-end
bit-exactness gate against the RCKangaroo cores this replaces, see
`../../tests/field_split_equiv.cpp` (host, no GPU required).

After any toolkit bump, re-check that `IMAD.WIDE.U32.X` still appears in the SASS -- the
fusion is a ptxas peephole, verified upstream on CUDA 13.0 / sm_120.

## Provenance

Generators and Python oracles are taken verbatim from `tat-temp/cCUDAm` branch `f5`
(commit `9671279`, "Split-column fused-MAC cores for mul_mod and sqr_mod"), so upstream
fixes port across cleanly. Only this README differs (paths + the CRLF note).
