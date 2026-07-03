#!/usr/bin/env bash
# Per-function SASS opcode histogram for CUDACyclone (native arch only).
# Run from repo root on the GPU box:
#     bash insn-opcodes.sh                    # default: the two hot functions
#     bash insn-opcodes.sh kernel_point_add   # any set of name substrings
#
# For each matched function, tallies base opcodes (mnemonic before the first '.') so we
# can see WHERE its instructions go -- ALU (IADD3/IMAD/LOP3/SHF), local-memory traffic
# (LDL/STL), constant/global loads (LDC/LDG), moves (MOV), control (BRA/etc).  Companion
# to insn-histogram.sh (per-function totals).  Builds the current tree first.
set -uo pipefail

# Function-name substrings to profile (default = the two steady-state hot functions).
PATS=("$@")
[ "${#PATS[@]}" -eq 0 ] && PATS=("kernel_point_add_and_check_oneinv" "getHash160_33_from_limbs")

BIN=CUDACyclone
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')"
work="$(mktemp -d)"
echo "GPU compute_cap sm_$ARCH ; work=$work"

echo; echo "===== clean build (current tree) ====="
make clean >/dev/null 2>&1
if ! make -j"$(nproc)" >"$work/build.log" 2>&1; then
  echo "BUILD FAILED -- tail of log:"; tail -n 30 "$work/build.log"; exit 1
fi
echo "built $BIN"

if ! cuobjdump -sass -arch "sm_$ARCH" "$BIN" > "$work/sass.txt" 2>/dev/null; then
  echo "(no per-arch dump for sm_$ARCH -- full fat-binary dump)"
  cuobjdump -sass "$BIN" > "$work/sass.txt" 2>/dev/null || { echo "cuobjdump failed"; exit 1; }
fi

profile_fn() {  # $1 = name substring
  local pat="$1" total c o pct
  awk -v pat="$pat" '
    /Function : /                 { fn=$0; sub(/.*Function : /,"",fn); want=(fn ~ pat); next }
    want && /\/\*[0-9a-f]+\*\// {
      s=$0
      sub(/^[ \t]*\/\*[0-9a-f]+\*\/[ \t]*/,"",s)   # strip leading ws + /*offset*/
      sub(/^@!?P[0-9]+[ \t]+/,"",s)                # strip optional predicate
      op=s; sub(/[ \t.;].*/,"",op)                 # base opcode (before . ; or ws)
      if (op!="") cnt[op]++
    }
    END { for (o in cnt) printf "%d\t%s\n", cnt[o], o }
  ' "$work/sass.txt" | sort -rn > "$work/op.txt"

  total=$(awk -F'\t' '{s+=$1} END{print s+0}' "$work/op.txt")
  echo; echo "================================================================"
  echo "FUNCTION matching: $pat   (total instructions: $total)"
  echo "================================================================"
  if [ "$total" -eq 0 ]; then echo "  (no instructions matched -- name substring wrong?)"; return; fi
  printf '  %-14s %8s %8s\n' "OPCODE" "COUNT" "SHARE%"
  while IFS="$(printf '\t')" read -r c o; do
    pct=$(awk -v c="$c" -v t="$total" 'BEGIN{printf "%.1f", 100.0*c/t}')
    printf '  %-14s %8d %7s%%\n' "$o" "$c" "$pct"
  done < "$work/op.txt"
}

for p in "${PATS[@]}"; do profile_fn "$p"; done
echo
echo "dumps kept in: $work  (sass.txt, build.log)"
