#!/usr/bin/env bash
# Clock-normalized keys/cycle A/B for CUDACyclone -- v3.1 (second-slot-hardened; +raw Mk/s & power).
# v3.1: also reports RAW Mkeys/s + board power over ALL reps. When a code change moves the clock
# ENDOGENOUSLY (draws more/less power on a power-capped card), keys/cyc divides that real effect
# out -- raw Mk/s is then the honest production metric (see the occ4 / dx-cache findings).
# Compares two branches' steady-state keys/cycle on the SAME box, dividing DVFS out.
# Fixes the classic artifact (a DVFS ramp with unpaired arm clocks fakes a delta):
#   * throttle/co-tenant PREFLIGHT (refuse a degraded box)
#   * untimed HEAT-SOAK so DVFS settles to a flat clock before any sampling
#   * PER-SAMPLE pairing: keys/cyc = mean_i(speed_i / clock_i)  (never mean(speed)/mean(clock))
#   * per-rep clock stability + arm-to-arm clock-match GATES (auto-discard contaminated reps)
#   * headline = clock-matched clean subset; auto-flag inconclusive on full-vs-clean sign flip
# v3 ADDS (the poll-throttle A/B exposed a SECOND-SLOT artifact: the arm that ran 2nd in a
# rep inherited a ~+0.18% warm-up bonus from the 1st arm -- with balanced ABAB alternation it
# cancels in the POINT estimate but bloats variance and can carry a below-bar signal on one
# half of the design):
#   * per-arm untimed PREWARM before EACH timed sample -> both arms enter from an identical
#     just-ran-myself state, so 2nd-slot position no longer confers a bonus (PREWARM=0 = v2)
#   * run_order (pos 1|2) logged per sample; verdict now reports the ORDER SPLIT and the
#     decisive perf-first-only stat (a real code win must survive when the treated arm runs 1st)
# v3.2 ADDS per-arm launch geometry. Until now ONE --grid was applied to BOTH arms, so a
# launch-config experiment (batch size, batches/SM) was structurally unmeasurable here:
# GRID=1024,512 BRANCHES="x x" is an A/A null, not an A/B. Arms are now identified by an
# arm INDEX rather than by branch name, so both arms may sit on the SAME branch and differ
# only in geometry. Set GRID_A/GRID_B; each defaults to $GRID, so every pre-v3.2 invocation
# behaves EXACTLY as before (same labels, same build path, same output columns).
#   * v3.2 also smoke-tests each arm before the soak -- a --grid the binary REJECTS used to
#     surface as a silent NaN 8 minutes later, not as an error. The smoke test additionally
#     cross-checks the two arms' BLOCKS: a short RANGE silently turns a batch-size A/B into
#     an occupancy A/B (total_batches becomes the binding constraint in pick_threads_total).
#   * A/A is now SUPPORTED, not refused -- it is the harness's own noise-floor read, and it
#     used to be silently degenerate (the verdict keyed on branch name, so the two rows
#     aliased and it always printed +0.000%). Rows are keyed on the arm INDEX now.
#   * per-arm SLICES_A/SLICES_B. --slices counts BATCHES per thread per launch, so doubling
#     the batch size doubles the KEYS per launch -- and the worker only tests the stop flag
#     at launch boundaries, so the untimed post-SIGINT tail after every soak/prewarm/sample
#     doubles too. Equalise batch*slices across arms to keep that tail symmetric.
# Usage (from repo root on the GPU box) -- name BOTH arms explicitly; there is NO default
# (a hardcoded branch goes stale the moment it merges/deletes -- that was the perf-insn trap):
#     BRANCHES="main feature" bash bench_ab.sh
#     PREWARM=0 REPS=14 BRANCHES="main feature" bash bench_ab.sh   # reproduce the old v2 behavior
#     # geometry A/B on the CHECKED-OUT branch (touches git not at all; needs no origin/ ref):
#     BRANCHES="rtm1-f1 rtm1-f1" GRID_A=512,512 GRID_B=1024,512 SLICES_B=32 bash bench_ab.sh
#     BRANCHES="rtm1-f1 rtm1-f1" bash bench_ab.sh                  # A/A noise floor
set -uo pipefail

read -ra BRANCHES <<< "${BRANCHES:?name both arms, e.g. BRANCHES='main feature' bash bench_ab.sh}"
GRID="${GRID:-512,512}"
GRID_A="${GRID_A:-$GRID}"   # per-arm launch geometry; default to $GRID => pre-v3.2 behavior
GRID_B="${GRID_B:-$GRID}"
SLICES_A="${SLICES_A:-}"    # optional per-arm --slices (batches per thread per kernel launch);
SLICES_B="${SLICES_B:-}"    # empty = don't pass it, i.e. the binary's default (64)
SMOKE="${SMOKE:-30}"        # per-arm pre-flight seconds: prove each (binary, grid) pair actually runs
RANGE="${RANGE:-100000000000:1FFFFFFFFFFF}"
TARGET="${TARGET:-000000000000000000000000000000000000dead}"
SOAK="${SOAK:-75}"          # untimed heat-soak seconds (settle DVFS to a flat clock)
PREWARM="${PREWARM:-12}"    # v3: per-arm untimed warmup before EACH timed sample (0 = v2 behavior)
WARMUP="${WARMUP:-6}"       # per-run seconds discarded before sampling
WINDOW="${WINDOW:-25}"      # per-run steady sampling seconds
REPS="${REPS:-5}"           # fast screen (5 reps); fine for a total-separation raw-Mk/s effect, too few for a sub-0.5% keys/cyc call
STAB_TOL="${STAB_TOL:-0.01}"   # max in-window clock sd/mean to accept a rep (1%)
CLK_MATCH="${CLK_MATCH:-100}"  # v3.1: relaxed 25->100. An ENDOGENOUS (code-driven) clock gap must NOT
                               # gate reps -- raw Mk/s is judged over ALL reps; 25 only guards keys/cyc noise
REQUIRE_CLEAN="${REQUIRE_CLEAN:-0}"  # 1 = abort if preflight shows throttle/co-tenant

[ "${#BRANCHES[@]}" -eq 2 ] || { echo "need exactly 2 branches, got: ${BRANCHES[*]}"; exit 1; }

# --- arm identity ---------------------------------------------------------------------
# An arm is (branch, grid, slices). Label by branch alone when the geometry is shared, so
# the output of a plain code A/B is byte-identical to v3.1; qualify with @grid only when
# the arms differ in geometry (which is also the case where the branches may match).
# Labels are DISPLAY ONLY -- every recorded row is keyed on the arm index, so an A/A run
# (identical labels) is a valid noise-floor experiment rather than two aliased rows.
for g in "$GRID_A" "$GRID_B"; do
  [[ "$g" =~ ^[0-9]+,[0-9]+$ ]] || { echo "bad grid '$g' -- expected two integers, e.g. 512,512"; exit 1; }
done
for s in "$SLICES_A" "$SLICES_B"; do
  [ -z "$s" ] || [[ "$s" =~ ^[0-9]+$ ]] || { echo "bad slices '$s' -- expected an integer"; exit 1; }
done
[[ "$SMOKE" =~ ^-?[0-9]+$ ]] || { echo "bad SMOKE '$SMOKE' -- expected an integer (0 = skip)"; exit 1; }
ARM_BRANCH=("${BRANCHES[0]}" "${BRANCHES[1]}")
ARM_GRID=("$GRID_A" "$GRID_B")
ARM_EXTRA=("${SLICES_A:+--slices $SLICES_A}" "${SLICES_B:+--slices $SLICES_B}")
if [ "$GRID_A" = "$GRID_B" ]; then
  ARM_LABEL=("${BRANCHES[0]}" "${BRANCHES[1]}")
else
  ARM_LABEL=("${BRANCHES[0]}@$GRID_A" "${BRANCHES[1]}@$GRID_B")
fi
if [ "${ARM_LABEL[0]}" = "${ARM_LABEL[1]}" ] && [ "${ARM_EXTRA[0]}" = "${ARM_EXTRA[1]}" ]; then
  # Not an error: this is the A/A null the sub-0.5% gate needs before any close call is
  # trusted. Index-qualify the labels so the two rows stay distinguishable in the output.
  ARM_LABEL=("arm0:${ARM_LABEL[0]}" "arm1:${ARM_LABEL[1]}")
  AA_NULL=1
else
  AA_NULL=0
fi
LBLW=10
for l in "${ARM_LABEL[@]}"; do [ "${#l}" -gt "$LBLW" ] && LBLW="${#l}"; done

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
REPO="$(git rev-parse --show-toplevel)"; cd "$REPO" || exit 1
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
orig_ref="$(git rev-parse --abbrev-ref HEAD)"

echo "== throttle / co-tenant PREFLIGHT =="
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,power.limit,pstate,utilization.gpu,clocks_throttle_reasons.active --format=csv
echo "-- compute apps (a foreign PID here poisons the A/B) --"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "-- active clock-event/throttle reasons --"
reasons="$(nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -iE 'Thermal Slowdown|Power (Cap|Brake)|Sync Boost|SW Power|HW (Thermal|Power)' | grep -i 'Active' || true)"
echo "${reasons:-  (none parsed -- check the CSV above)}"
if echo "$reasons" | grep -qi ': Active'; then
  echo "!! WARNING: an active throttle reason is present -- sub-0.5% A/B is UNTRUSTWORTHY here."
  [ "$REQUIRE_CLEAN" = "1" ] && { echo "REQUIRE_CLEAN=1 -> aborting."; exit 2; }
fi
echo

# A pure geometry/slices A/B on the branch that is ALREADY checked out needs no git operation
# at all -- and must not perform one. `git reset --hard origin/$b` is destructive (gotcha #4):
# it deletes uncommitted work, and a geometry experiment is exactly the case where the harness
# under test is itself uncommitted. It also simply fails on a local-only branch (no origin/ ref).
SAME_BRANCH=0; [ "${ARM_BRANCH[0]}" = "${ARM_BRANCH[1]}" ] && SAME_BRANCH=1
NO_GIT=0;      [ "$SAME_BRANCH" = "1" ] && [ "${ARM_BRANCH[0]}" = "$orig_ref" ] && NO_GIT=1

if [ "$NO_GIT" = "0" ]; then
  DIRTY="$(git status --porcelain --untracked-files=no)"
  if [ -n "$DIRTY" ]; then
    echo "!! the working tree has uncommitted changes, and this run would 'git checkout' +"
    echo "   'git reset --hard' -- which would DESTROY them:"
    printf '%s\n' "$DIRTY" | sed 's/^/     /'
    echo "   commit or stash first. For a geometry-only A/B, name the CHECKED-OUT branch on"
    echo "   both arms (BRANCHES=\"$orig_ref $orig_ref\" GRID_A=... GRID_B=...) -- that path"
    echo "   builds the working tree as-is and never touches git."
    exit 1
  fi
fi

if [ "$NO_GIT" = "1" ]; then
  echo "== build (geometry-only A/B on the checked-out $orig_ref -- working tree as-is) =="
else
  echo "== build both arms (reset to origin where one exists, clean build) =="
  git fetch --quiet origin >/dev/null 2>&1 || true
fi
for i in 0 1; do
  b="${ARM_BRANCH[$i]}"
  if [ "$i" = "1" ] && [ "$SAME_BRANCH" = "1" ]; then
    # Same branch on both arms (a pure geometry A/B): one build, two arms. Rebuilding would
    # produce a bit-identical binary and only add a chance of the two arms diverging.
    cp -f "$TMP/CUDACyclone.arm0" "$TMP/CUDACyclone.arm1"
    echo "  arm1 ${ARM_LABEL[1]}: same build as arm0 (geometry-only difference)"
    continue
  fi
  if [ "$NO_GIT" = "0" ]; then
    git checkout "$b" >/dev/null 2>&1 || { echo "checkout $b FAILED"; exit 1; }
    if git rev-parse --verify -q "origin/$b" >/dev/null; then
      git reset --hard "origin/$b" >/dev/null 2>&1 || { echo "reset origin/$b FAILED"; exit 1; }
    else
      echo "  (no origin/$b -- benchmarking the LOCAL $b at $(git rev-parse --short HEAD))"
    fi
  fi
  make clean >/dev/null 2>&1
  make >/dev/null 2>&1 || { echo "BUILD FAILED for $b"; exit 1; }
  cp -f CUDACyclone "$TMP/CUDACyclone.arm$i"
  echo "  built arm$i ${ARM_LABEL[$i]} ($(git rev-parse --short HEAD))"
done
[ "$NO_GIT" = "1" ] || git checkout "$orig_ref" >/dev/null 2>&1 || true

if [ "$SMOKE" -le 0 ]; then
  echo "== per-arm smoke test SKIPPED (SMOKE=$SMOKE) =="
  echo "   a --grid the binary rejects will now surface only as kc=NaN after the full rep loop."
else
echo "== per-arm smoke test (${SMOKE}s each) =="
# A --grid the binary REJECTS (non-pow2 batch, batch > MAX_BATCH_SIZE, range not divisible by
# batch) makes it exit immediately. run_once would then just find no "Speed:" line and report
# kc=NaN -- indistinguishable from noise, and only after the whole rep loop had run. Catch it here.
# Abort ONLY on a rejected config, which is exactly "no banner". A banner with no Speed: line yet
# means the config is fine and startup is merely slower than SMOKE, so that only warns -- an abort
# there would break previously-working runs on slow boxes.
ARM_BAT=(); ARM_BLK=(); ARM_SLC=()
for i in 0 1; do
  smoke="$TMP/smoke.arm$i"
  # shellcheck disable=SC2086  # ARM_EXTRA is a pre-validated "--slices N" or empty
  timeout -s INT "$SMOKE" "$TMP/CUDACyclone.arm$i" \
      --range "$RANGE" --target-hash160 "$TARGET" --grid "${ARM_GRID[$i]}" ${ARM_EXTRA[$i]} > "$smoke" 2>&1 || true
  txt="$(tr '\r' '\n' < "$smoke")"
  bat="$(printf '%s\n' "$txt" | sed -nE 's/.*Points batch size[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' | head -n1)"
  blk="$(printf '%s\n' "$txt" | sed -nE 's/.*Blocks[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' | head -n1)"
  slc="$(printf '%s\n' "$txt" | sed -nE 's/.*Batches\/launch[[:space:]]*:[[:space:]]*([0-9]+).*/\1/p' | head -n1)"
  if [ -z "$bat" ]; then
    echo "!! arm$i (${ARM_LABEL[$i]}) never printed its banner with --grid ${ARM_GRID[$i]} ${ARM_EXTRA[$i]}"
    echo "   the binary rejected this configuration -- every rep would have reported NaN."
    printf '%s\n' "$txt" | grep -v '^[[:space:]]*$' | tail -n 6 | sed 's/^/     /'
    exit 3
  fi
  want="${ARM_GRID[$i]%%,*}"
  if [ "$bat" != "$want" ]; then
    echo "!! arm$i (${ARM_LABEL[$i]}): asked for batch $want, binary reports $bat -- refusing to"
    echo "   run an experiment whose treatment did not take effect."; exit 3
  fi
  ARM_BAT[$i]="$bat"; ARM_BLK[$i]="$blk"; ARM_SLC[$i]="${slc:-0}"
  if printf '%s\n' "$txt" | grep -q 'Speed:'; then
    echo "  arm$i ${ARM_LABEL[$i]}: OK (batch=$bat, blocks=$blk, batches/launch=${slc:-?})"
  else
    echo "  arm$i ${ARM_LABEL[$i]}: config accepted (batch=$bat, blocks=$blk) but no Speed: line"
    echo "     within ${SMOKE}s -- startup is slow here; raise SMOKE, and check WARMUP=${WARMUP}s too."
  fi
done

# Both arms print Blocks; compare them. threadsTotal is min(mem, total_batches, SM*bps*256)
# rounded down to a divisor of total_batches = range_len/B, so once range_len/B drops below
# SM*bps*256 the BATCH SIZE starts moving the thread count: a shortened RANGE silently turns
# a batch-size A/B into an occupancy A/B and the verdict would credit the batch size for it.
if [ -n "${ARM_BLK[0]}" ] && [ -n "${ARM_BLK[1]}" ] && [ "${ARM_BLK[0]}" != "${ARM_BLK[1]}" ]; then
  if [ "${GRID_A#*,}" = "${GRID_B#*,}" ]; then
    echo "!! arms differ in BLOCKS (${ARM_BLK[0]} vs ${ARM_BLK[1]}) although batches/SM is the same."
    echo "   RANGE is too short for this batch size, so the launch shape moved with it: this is an"
    echo "   occupancy A/B, not a batch-size A/B. Lengthen RANGE (>= 2^34 at batch 1024) and rerun."
    exit 3
  fi
  echo "  note: arms differ in BLOCKS (${ARM_BLK[0]} vs ${ARM_BLK[1]}) -- expected, batches/SM differs."
fi

# --slices counts BATCHES per thread per launch, so keys/launch = batch * slices. The worker
# tests the stop flag only at launch boundaries, so keys/launch sets the untimed tail that runs
# AFTER every timeout -s INT (soak, prewarm, and each timed sample). Unequal tails mean the
# arms do not enter their timed windows from the same thermal/DVFS state -- what PREWARM exists
# to guarantee. Warn with the value that would equalise it.
kpl0=$(( ARM_BAT[0] * ARM_SLC[0] )); kpl1=$(( ARM_BAT[1] * ARM_SLC[1] ))
if [ "$kpl0" -gt 0 ] && [ "$kpl1" -gt 0 ] && [ "$kpl0" != "$kpl1" ]; then
  echo "!! arms differ in KEYS PER LAUNCH per thread ($kpl0 vs $kpl1): the post-SIGINT tail after"
  echo "   every soak/prewarm/sample is ~$(awk -v a="$kpl1" -v b="$kpl0" 'BEGIN{printf "%.1f", a/b}')x longer on arm1, so PREWARM does not equalise entry state."
  want_slc1=$(( kpl0 / ARM_BAT[1] ))
  if [ "$want_slc1" -gt 0 ] && [ $(( want_slc1 * ARM_BAT[1] )) -eq "$kpl0" ]; then
    echo "   To match them: SLICES_A=${ARM_SLC[0]} SLICES_B=$want_slc1  (keeps batch*slices constant)."
  fi
  echo "   Continuing -- this skews the untimed phases, not the timed window itself."
fi
fi

echo "== heat-soak ${SOAK}s (settle DVFS to a flat clock BEFORE sampling) =="
# shellcheck disable=SC2086
timeout -s INT "$SOAK" "$TMP/CUDACyclone.arm0" \
    --range "$RANGE" --target-hash160 "$TARGET" --grid "${ARM_GRID[0]}" ${ARM_EXTRA[0]} >/dev/null 2>&1 || true
echo "  post-soak clock: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -n1) MHz"

run_once() {  # $1 = binary ; $2 = grid ; $3 = extra args ; echoes "kc clk_mean clk_sd nsamp raw_mean pwr_mean"
  local bin="$1" grid="$2" extra="$3" out="$TMP/out" t=0 pid clk spd pwr
  : > "$out"; : > "$TMP/ratios"; : > "$TMP/clks"; : > "$TMP/speeds"; : > "$TMP/pwrs"
  # shellcheck disable=SC2086
  timeout -s INT $((WARMUP+WINDOW+1)) "$bin" \
      --range "$RANGE" --target-hash160 "$TARGET" --grid "$grid" $extra > "$out" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null && [ "$t" -lt $((WARMUP+WINDOW)) ]; do
    if [ "$t" -ge "$WARMUP" ]; then
      # clock + board power in ONE query so both are read at the same instant
      IFS=',' read -r clk pwr < <(nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n1)
      clk="${clk// /}"; pwr="${pwr// /}"
      spd=$(tr '\r' '\n' < "$out" | sed -nE 's/.*Speed:[[:space:]]*([0-9.]+).*/\1/p' | tail -n1)
      if [ -n "$clk" ] && [ -n "$spd" ]; then
        awk -v s="$spd" -v c="$clk" 'BEGIN{if(c>0)printf "%.6f\n", s/c}' >> "$TMP/ratios"
        echo "$clk" >> "$TMP/clks"
        echo "$spd" >> "$TMP/speeds"
        [[ "$pwr" =~ ^[0-9.]+$ ]] && echo "$pwr" >> "$TMP/pwrs"
      fi
    fi
    sleep 1; t=$((t+1))
  done
  kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  local kc cline sp pw
  kc=$(awk 'NF{s+=$1;n++} END{if(n>0)printf "%.6f",s/n; else printf "NaN"}' "$TMP/ratios")
  cline=$(awk 'NF{s+=$1;ss+=$1*$1;n++} END{if(n>0){m=s/n; sd=(n>1)?sqrt((ss-n*m*m)/(n-1)):0; printf "%.2f %.2f %d",m,sd,n} else printf "NaN NaN 0"}' "$TMP/clks")
  sp=$(awk 'NF{s+=$1;n++} END{if(n>0)printf "%.3f",s/n; else printf "NaN"}' "$TMP/speeds")
  pw=$(awk 'NF{s+=$1;n++} END{if(n>0)printf "%.1f",s/n; else printf "NaN"}' "$TMP/pwrs")
  echo "$kc $cline $sp $pw"
}

: > "$TMP/all.dat"
if [ "$GRID_A" = "$GRID_B" ]; then geom="GRID=$GRID_A"; else geom="GRID_A=$GRID_A GRID_B=$GRID_B"; fi
# Slices belong in the header too: without them a pasted rep block cannot be checked for the
# keys-per-launch symmetry the untimed phases depend on.
[ -z "$SLICES_A$SLICES_B" ] || geom="$geom SLICES_A=${SLICES_A:-default} SLICES_B=${SLICES_B:-default}"
[ "$AA_NULL" = "1" ] && echo "== A/A NULL: both arms are identical -- this measures the HARNESS NOISE FLOOR, not a change =="
echo "== $REPS interleaved reps (SOAK=${SOAK}s PREWARM=${PREWARM}s WARMUP=${WARMUP}s WINDOW=${WINDOW}s $geom) =="
# REPS odd => the ABAB alternation is unbalanced (3 A-first vs 2 B-first at the default 5), so
# the 2nd-slot bonus does NOT cancel in the raw point estimate. The run-order split below
# decomposes it out explicitly; an even REPS also removes it by construction.
for ((r=1; r<=REPS; r++)); do
  if (( r % 2 == 1 )); then order=(0 1); else order=(1 0); fi
  pos=0
  for i in "${order[@]}"; do
    pos=$((pos+1))
    # v3: per-arm untimed prewarm so BOTH arms enter their timed window from an identical
    # just-ran-this-same-binary state -> kills the 2nd-slot warm-up bonus (was ~+0.18%).
    if [ "$PREWARM" -gt 0 ]; then
      # shellcheck disable=SC2086
      timeout -s INT "$PREWARM" "$TMP/CUDACyclone.arm$i" \
          --range "$RANGE" --target-hash160 "$TARGET" --grid "${ARM_GRID[$i]}" ${ARM_EXTRA[$i]} >/dev/null 2>&1 || true
    fi
    read -r kc cm csd ns sp pw < <(run_once "$TMP/CUDACyclone.arm$i" "${ARM_GRID[$i]}" "${ARM_EXTRA[$i]}")
    printf 'rep %2d  %-*s pos=%d keys/cyc=%s  clk=%s MHz (sd %s, n=%s)  raw=%s Mk/s  pwr=%s W\n' "$r" "$LBLW" "${ARM_LABEL[$i]}" "$pos" "$kc" "$cm" "$csd" "$ns" "$sp" "$pw"
    # Field 2 is the arm INDEX, never the label: a label may contain anything (or, in an A/A,
    # be identical for both arms) and either would corrupt the whitespace-split record.
    echo "$r $i $kc $cm $csd $pos $sp $pw" >> "$TMP/all.dat"
  done
done

echo; echo "== verdict =="
python3 - "$TMP/all.dat" "${ARM_LABEL[0]}" "${ARM_LABEL[1]}" "$STAB_TOL" "$CLK_MATCH" <<'PY'
import sys, math, statistics as st
path, A, B, stab, match = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
reps={}
for ln in open(path):
    p=ln.split()
    if len(p)<6: continue
    r=int(p[0]); arm=p[1]          # arm INDEX ('0'/'1'); A and B here are display labels only
    if arm not in ('0','1'): continue
    try: kc=float(p[2]); cm=float(p[3]); csd=float(p[4]); pos=int(p[5])
    except: continue
    sp=float(p[6]) if len(p)>=7 else float('nan')
    pw=float(p[7]) if len(p)>=8 else float('nan')
    reps.setdefault(r,{})[arm]=(kc,cm,csd,pos,sp,pw)
allA=[];allB=[]; rawA=[];rawB=[]; clkA=[];clkB=[]; pwrA=[];pwrB=[]
cleanA=[];cleanB=[]; cleanDiff=[]; diff_Afirst=[]; diff_Bfirst=[]; dropped=[]
for r in sorted(reps):
    d=reps[r]
    if '0' not in d or '1' not in d: continue
    (kcA,cmA,csdA,posA,spA,pwA)=d['0']; (kcB,cmB,csdB,posB,spB,pwB)=d['1']
    allA.append(kcA); allB.append(kcB); rawA.append(spA); rawB.append(spB); clkA.append(cmA); clkB.append(cmB)
    if not math.isnan(pwA) and not math.isnan(pwB): pwrA.append(pwA); pwrB.append(pwB)
    stableA = cmA>0 and csdA/cmA<=stab
    stableB = cmB>0 and csdB/cmB<=stab
    matched = abs(cmA-cmB)<=match
    if stableA and stableB and matched:
        cleanA.append(kcA); cleanB.append(kcB)
        diff=kcB-kcA; cleanDiff.append(diff)
        if posB==1: diff_Bfirst.append(diff)
        elif posA==1: diff_Afirst.append(diff)
    else:
        why=[]
        if not stableA: why.append(f"{A} unstable(sd/mean={csdA/cmA:.3f})")
        if not stableB: why.append(f"{B} unstable(sd/mean={csdB/cmB:.3f})")
        if not matched: why.append(f"clk gap {abs(cmA-cmB):.0f}MHz")
        dropped.append(f"  rep {r}: "+", ".join(why)+f"  (clk {A}={cmA:.0f} {B}={cmB:.0f})")
def cmp(a,b,tag,unit="",prec=3):
    a=[x for x in a if not math.isnan(x)]; b=[x for x in b if not math.isnan(x)]
    if len(a)<2 or len(b)<2:
        print(f"[{tag}] too few reps ({len(a)}/{len(b)})"); return None,False
    ma,mb=st.mean(a),st.mean(b); sa,sb=st.stdev(a),st.stdev(b)
    se=math.sqrt(sa*sa/len(a)+sb*sb/len(b)); t=(mb-ma)/se if se>0 else float('nan')
    sep = min(b)>max(a) or min(a)>max(b)
    print(f"[{tag}] {A}={ma:.{prec}f}{unit} (sd {sa:.{prec}f}, n={len(a)}) | {B}={mb:.{prec}f}{unit} (sd {sb:.{prec}f}, n={len(b)})")
    print(f"[{tag}] delta({B} vs {A})={(mb-ma)/ma*100:+.3f}%  Welch t={t:+.2f}  total-separation={'YES' if sep else 'no'}")
    return (mb-ma), sep
print("== PRODUCTION metrics, ALL reps (gate-independent; the HONEST read when the clock is endogenous) ==")
draw,rsep = cmp(rawA, rawB, "RAW Mkeys/s", " Mk/s", 3)
dclk,csep = cmp(clkA, clkB, "clock",       " MHz", 1)
dpw,_     = cmp(pwrA, pwrB, "power",       " W",   1) if pwrA else (None,False)
if draw is not None and dclk is not None and rsep and csep and (draw>0)==(dclk>0):
    hi = B if draw>0 else A
    pnote = ""
    if dpw is not None:
        if abs(dpw) < 1.0:   # both pinned at the power cap -> the win is efficiency, shown as clock not watts
            pnote = f", at EQUAL power (both ~{st.mean(pwrB):.0f}W cap) -> {hi} is MORE EFFICIENT (more keys/watt)"
        else:
            pnote = f", at {'LOWER' if (dpw<0)==(draw>0) else 'HIGHER'} power"
    print(f">>> ENDOGENOUS clock: {hi} sustains a higher sustained clock{pnote} -> RAW Mk/s is the honest metric.")
    print(f">>> {hi} wins on RAW throughput with TOTAL SEPARATION. Real ONLY IF proof.py is green AND it reproduces.")
elif dclk is not None and not csep:
    print(">>> clock not separated -> not clearly endogenous; the keys/cyc read below is primary.")
print()
print("== keys/cycle, clock-normalized (divides an endogenous clock OUT -> SECONDARY here) ==")
def stat(a,b,tag):
    if len(a)<2 or len(b)<2:
        print(f"[{tag}] too few reps ({len(a)}/{len(b)}) for a stat"); return
    ma,mb=st.mean(a),st.mean(b); sa,sb=st.stdev(a),st.stdev(b)
    se=math.sqrt(sa*sa/len(a)+sb*sb/len(b)); t=(mb-ma)/se if se>0 else float('nan')
    sep = min(b)>max(a) or min(a)>max(b)
    print(f"[{tag}] {A} mean={ma:.5f} sd={sa:.5f} (n={len(a)}) | {B} mean={mb:.5f} sd={sb:.5f} (n={len(b)})")
    print(f"[{tag}] delta({B} vs {A})={ (mb-ma)/ma*100:+.3f}%  Welch t={t:+.2f}  total-separation={'YES' if sep else 'no'}")
    return (mb-ma)
if dropped:
    print("gated-out reps (keys/cyc clean subset only):"); print("\n".join(dropped))
if len(cleanA)<2 and clkA and clkB:
    gap=st.mean([abs(a-b) for a,b in zip(clkA,clkB)])
    if gap>match:
        print(f">>> the clean subset is empty because the arms' clocks differ by ~{gap:.0f} MHz > CLK_MATCH={match:.0f}.")
        print(f"    A GEOMETRY change can move the clock endogenously by more than a code change does;")
        print(f"    the RAW Mk/s block above is gate-independent and stands. To also get a keys/cyc read,")
        print(f"    rerun with CLK_MATCH={gap*1.5:.0f} -- and treat it as secondary, per the endogenous-clock rule.")
print()
dfull=stat(allA,allB,"ALL reps")
dclean=stat(cleanA,cleanB,"CLEAN subset")
print()
base = st.mean(cleanA) if cleanA else (st.mean(allA) if allA else 1.0)
def paired(diffs, tag):
    n=len(diffs)
    if n<2:
        print(f"[{tag}] too few reps ({n})"); return None
    m=st.mean(diffs); sd=st.stdev(diffs) if n>1 else 0.0
    se=sd/math.sqrt(n) if n>0 else 0.0; t=m/se if se>0 else float('nan')
    print(f"[{tag}] mean diff={m:+.6f} ({m/base*100:+.3f}%)  paired t={t:+.2f}  (n={n})")
    return m
print("== run-order split (v3: a REAL code win must survive when the treated arm runs FIRST) ==")
paired(cleanDiff, "CLEAN paired, all")
mAf=paired(diff_Afirst, f"{A}-first ({B} ran 2nd)")
mBf=paired(diff_Bfirst, f"{B}-first ({B} ran 1st) <-- decisive")
if mAf is not None and mBf is not None:
    T=(mAf+mBf)/2.0; P=(mAf-mBf)/2.0
    print(f"decompose: true effect T={T/base*100:+.3f}%  |  2nd-slot bonus P={P/base*100:+.3f}%")
    if T!=0 and abs(P)<0.25*abs(T):
        print(">>> position bonus small vs effect -> order artifact CONTROLLED (prewarm working).")
    else:
        print(">>> position bonus COMPARABLE to effect -> still order-fragile; win not proven to be the code.")
print()
if isinstance(dfull,float) and isinstance(dclean,float) and (dfull>0)!=(dclean>0):
    print(">>> keys/cyc FULL vs CLEAN disagree in SIGN -> keys/cyc inconclusive (EXPECTED if the clock is endogenous; use RAW Mk/s above).")
elif len(cleanA)<12:
    print(f">>> only {len(cleanA)} keys/cyc clean reps/arm (<12); for an endogenous clock the RAW Mk/s call above is what matters.")
else:
    print(">>> keys/cyc secondary; the production call is RAW Mk/s above (a WIN needs separation there + proof.py green + a reproduce run).")
PY
