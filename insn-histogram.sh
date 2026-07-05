#!/usr/bin/env bash
# Per-function SASS instruction-count histogram for CUDACyclone (native arch only).
# Run from repo root on the GPU box, on whatever branch/binary you want to weigh:
#     bash insn-histogram.sh              # clean-build the current tree, then rank
#     bash insn-histogram.sh CUDACyclone  # weigh an already-built binary (skip build)
#
# Ranks every kernel + device function by its TRUE SASS instruction count so perf work
# targets the fattest functions that still have headroom.  Companion to phase0-inspect.sh
# (which reports occupancy/spills); this one answers "which function is worth trimming".
#
# Counting rule: one instruction == one offset-tagged line '/*hhhh*/'.  The encoding-
# continuation and trailing-encoding tokens print as '/* 0x..' (a space after /*), so a
# hex digit immediately after /* tags instruction offsets only.  This is the exact rule
# that measured getHash160 8576->8504 during the IV-literalization A/B.
set -uo pipefail

BIN="${1:-CUDACyclone}"
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
work="$(mktemp -d)"
echo "GPU compute_cap sm_$ARCH ; binary=$BIN ; work=$work"

# Build only when no binary path was given (i.e. weigh the current working tree).
if [ "$#" -eq 0 ]; then
  echo; echo "===== clean build (current tree) ====="
  make clean >/dev/null 2>&1
  if ! make -j"$(nproc)" >"$work/build.log" 2>&1; then
    echo "BUILD FAILED -- tail of log:"; tail -n 30 "$work/build.log"; exit 1
  fi
  echo "built $BIN"
fi
[ -f "$BIN" ] || { echo "no binary '$BIN' -- build first or pass a path"; exit 1; }

echo; echo "===== native-arch SASS dump ====="
if ! cuobjdump -sass -arch "sm_$ARCH" "$BIN" > "$work/sass.txt" 2>/dev/null; then
  echo "(no per-arch dump for sm_$ARCH -- full fat-binary dump; counts will sum all archs)"
  cuobjdump -sass "$BIN" > "$work/sass.txt" 2>/dev/null || { echo "cuobjdump failed"; exit 1; }
fi
echo "total SASS lines: $(wc -l < "$work/sass.txt")"

# Group offset-tagged instruction lines by their enclosing 'Function :' block.
awk '
  /Function : /                 { fn=$0; sub(/.*Function : /,"",fn); next }
  fn!="" && /\/\*[0-9a-f]+\*\//  { cnt[fn]++ }
  END { for (f in cnt) printf "%d\t%s\n", cnt[f], f }
' "$work/sass.txt" | sort -rn > "$work/histo.txt"

echo; echo "===== per-function instruction histogram (sm_$ARCH), descending ====="
printf '%9s  %s\n' "INSNS" "FUNCTION"
total=0
while IFS="$(printf '\t')" read -r n name; do
  total=$((total + n))
  dem="$(printf '%s' "$name" | c++filt 2>/dev/null || printf '%s' "$name")"
  printf '%9d  %s\n' "$n" "$dem"
done < "$work/histo.txt"
printf '%9d  %s\n' "$total" "== TOTAL over listed functions =="

# Parse-independent cross-check: every offset-tagged line in the whole dump.
xcheck="$(grep -cE '/\*[0-9a-f]+\*/' "$work/sass.txt")"
echo "cross-check: $xcheck offset-tagged lines in dump (>= TOTAL; any extra = code outside Function: blocks)"
echo
echo "dumps kept in: $work  (sass.txt, histo.txt)"
