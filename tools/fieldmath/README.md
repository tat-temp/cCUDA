# Split-column field math

The product cores in `../../field_split.cuh` are GENERATED. Do not hand-edit them there --
edit the generator and re-emit.

    python3 gen.py    | tr -d '\r' > split.inc      # mul512_split
    python3 gensqr.py | tr -d '\r' > sqrsplit.inc   # sqr512_split
    python3 verify_split.py split.inc               # replays the stream vs Python bignums
    python3 verify_sqr.py   sqrsplit.inc
    python3 emit_header.py                          # splice into field_split.cuh

`tr -d '\r'` matters on Windows checkouts (`core.autocrlf=true`). End-to-end correctness is
`make fieldtest`.

After a toolkit bump, re-check that `IMAD.WIDE.U32.X` still appears in the SASS -- the fusion
is a ptxas peephole, verified upstream on CUDA 13.0 / sm_120.

Generators and oracles are verbatim from `tat-temp/cCUDAm@9671279` (branch `f5`) so upstream
fixes port cleanly; only this README differs.
