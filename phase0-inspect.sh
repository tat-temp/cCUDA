#!/usr/bin/env bash
# Phase 0 codegen inspection for CUDACyclone.  Run from repo root on the GPU box:
#     bash phase0-inspect.sh
#
# Answers, on the real hardware/toolchain, four questions that re-price the rest of the
# optimization plan:
#   Q1  Does the hot kernel spill / how many registers/thread?      -> occupancy headroom
#   Q2  Does __noinline__ getHash160_33_from_limbs carry a stack     -> the "avoid intermediate
#       frame + LDL/STL (local-memory materialization of its arrays)?    buffer" / noinline lead
#   Q3  Are SHA-256 round constants K[] baked as immediates, or       -> the "K[] as immediates"
#       loaded from the constant bank (LDC) inside the hash fn?          cheap-closeout idea
#   Q4  Is anything spilling to .local independent of the call?       -> confound check before A/B
set -uo pipefail

BIN=CUDACyclone
KERNEL=kernel_point_add_and_check_oneinv
HASH=getHash160_33_from_limbs
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
work="$(mktemp -d)"
echo "GPU compute_cap sm_$ARCH ; work=$work"

echo; echo "===== clean build ====="
make clean >/dev/null 2>&1
if ! make -j"$(nproc)" >"$work/build.log" 2>&1; then
  echo "BUILD FAILED -- tail of log:"; tail -n 30 "$work/build.log"; exit 1
fi
echo "built $BIN"

echo; echo "===== Q1/Q4: ptxas verbose resource report (build-time) ====="
# native-arch rebuild with -Xptxas -v; the report goes to stderr during compilation
make ptxinfo 2>&1 | grep -E "Compiling entry|Function properties|registers|bytes stack frame|bytes spill|Used [0-9]+ registers" || \
  echo "(no ptxas -v lines captured -- see cuobjdump report below, which is authoritative)"

echo; echo "===== Q1/Q4: cuobjdump -res-usage (authoritative, from the built binary) ====="
# REG: registers/thread ; STACK: stack frame bytes ; LOCAL: local-memory bytes (spills or .local arrays)
cuobjdump -res-usage "$BIN" 2>/dev/null > "$work/res.txt" || true
echo "-- hot kernel --"
grep -A2 "$KERNEL" "$work/res.txt" | grep -E "REG:|STACK:|LOCAL:" || echo "(kernel not found in res-usage)"
echo "-- noinline hash fn --"
grep -A2 "$HASH" "$work/res.txt" | grep -E "REG:|STACK:|LOCAL:" || echo "(hash fn not found -- may have been inlined despite __noinline__, itself notable)"

echo; echo "===== Q2/Q3: SASS local-memory + constant-load counts (native arch only) ====="
if ! cuobjdump -sass -arch "sm_$ARCH" "$BIN" > "$work/sass.txt" 2>/dev/null; then
  cuobjdump -sass "$BIN" > "$work/sass.txt" 2>/dev/null || true
fi
echo "total SASS lines: $(wc -l < "$work/sass.txt")"

extract_fn() {  # $1 = substring of the (mangled) function name; prints that function's SASS body
  awk -v pat="$1" '/Function :/{infn=($0 ~ pat)} infn{print}' "$work/sass.txt"
}
report_fn() {   # $1 = label ; $2 = name substring
  local body; body="$(extract_fn "$2")"
  local lines ldl stl ldc
  lines=$(printf '%s\n' "$body" | grep -c . )
  ldl=$(printf '%s\n' "$body" | grep -cE '\bLDL\b')     # local load  (spill or .local array read)
  stl=$(printf '%s\n' "$body" | grep -cE '\bSTL\b')     # local store (spill or .local array write)
  ldc=$(printf '%s\n' "$body" | grep -cE '\bLDC\b')     # constant-bank load
  printf '  %-34s SASS=%-6s LDL=%-4s STL=%-4s LDC=%-4s\n' "$1" "$lines" "$ldl" "$stl" "$ldc"
}
echo "  (LDL/STL = local-memory traffic = spills or .local arrays ; LDC = constant-bank load)"
report_fn "hot kernel [$KERNEL]" "$KERNEL"
report_fn "noinline hash [$HASH]" "$HASH"
echo
echo "  interpretation guide:"
echo "   Q1 occupancy : kernel REG/thread vs the (256,2)=128-reg budget; STACK/LOCAL 0 = no spill (good)"
echo "   Q2 noinline  : hash fn STACK>0 or LDL/STL>0 => its array args DO materialize in .local per call"
echo "                  (would make removing __noinline__ worth an A/B; if 0, VanitySearch's choice stands)"
echo "   Q3 K[] imm.  : LDC in the hash fn ~ K[]/IV[] loads; near-0 => already immediates (skip 1a)"
echo "                  (LDC in the KERNEL is EXPECTED -- c_Gx/c_Gy/c_Jx/c_target_words precomputed points)"
echo "   Q4 confound  : any STACK/LOCAL>0 or LDL/STL>0 present in the CURRENT build must be accounted"
echo "                  for before trusting a noinline-vs-forceinline A/B (it may not be the call's fault)"
echo
echo "full dumps kept in: $work  (res.txt, sass.txt, build.log)"
