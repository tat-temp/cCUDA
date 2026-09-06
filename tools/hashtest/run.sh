#!/usr/bin/env bash
# Host-side verification for the hash and the hot kernel's control flow. NO GPU REQUIRED --
# this runs anywhere a C++17 host compiler exists, which is the point: it is the only check
# available on a box without nvcc, and it covers what `make proof` cannot isolate.
#
#   bash tools/hashtest/run.sh            # compare working tree against `main`
#   BASE_REF=cfd9c3b bash tools/hashtest/run.sh
#
# What each test proves:
#   test_x2  -- the REAL hash, compiled for the host: a known-answer test against the published
#               hash160 of privkey 1, then 300k random pairs showing getHash160_w2_x2(A,B) is
#               bit-identical to two separate getHash160_w2_from_limbs calls, plus the word-2
#               trim invariant (w2 == full digest's word 2).
#   difftest -- the baseline kernel and the working-tree kernel side by side over identical
#               deterministic stand-ins for the PTX field ops. Proves they hash the SAME
#               candidates in the SAME order, write the same end-of-launch state, and -- with
#               every candidate planted as the target in turn -- agree on the found flag,
#               scalar, Rx and Ry. That is the found-path regression the perf A/B cannot see
#               (a wrong hash runs at exactly the same speed).
#
# NOT covered here (GPU box only): register/spill budget (`make gate`), whether ptxas actually
# interleaved the two chains (`make sass`), IPC/no-eligible-warp (ncu), and end-to-end recovery
# on real hardware (`make proof`).
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CXX="${CXX:-g++}"
BASE_REF="${BASE_REF:-main}"

FLAGS="-O1 -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function"
DEFS="-D__device__= -D__forceinline__=inline -D__noinline__= -D__global__="
INCS="-I$HERE -I$REPO"

# Only some branches carry a 2-wide hash entry point. Detect it rather than forking the harness:
# a tree that does not touch the hash still wants the KAT, because difftest's result rests on the
# host emulation being faithful.
if grep -q 'getHash160_w2_x2' "$REPO/CUDAHash.cuh"; then
    DEFS="$DEFS -DHAVE_HASH2_X2"
    echo "   (tree has getHash160_w2_x2 -- 2-wide equivalence enabled)"
fi

echo "== 1/2: hash equivalence (real hash on the host) =="
"$CXX" $FLAGS $DEFS $INCS "$HERE/test_x2.cpp" -o "$HERE/test_x2.exe"
"$HERE/test_x2.exe"

echo
echo "== 2/2: kernel control-flow differential vs $BASE_REF =="
BASE_REF="$BASE_REF" python3 "$HERE/extract2.py" 2>/dev/null || BASE_REF="$BASE_REF" python "$HERE/extract2.py"
"$CXX" $FLAGS $DEFS $INCS "$HERE/difftest.cpp" -o "$HERE/difftest.exe"
"$HERE/difftest.exe"

echo
echo "ALL HOST CHECKS PASSED"
