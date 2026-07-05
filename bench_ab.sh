#!/usr/bin/env bash
# Clock-normalized keys/cycle A/B for CUDACyclone -- v3 (second-slot-hardened).
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
# Usage (from repo root on the GPU box):
#     bash bench_ab.sh                              # default: main vs perf-insn, REPS=40, PREWARM=12
#     BRANCHES="main perf-insn" bash bench_ab.sh
#     PREWARM=0 REPS=14 bash bench_ab.sh            # reproduce the old v2 behavior
set -uo pipefail

read -ra BRANCHES <<< "${BRANCHES:-main perf-insn}"
GRID="${GRID:-512,512}"
RANGE="${RANGE:-100000000000:1FFFFFFFFFFF}"
TARGET="${TARGET:-000000000000000000000000000000000000dead}"
SOAK="${SOAK:-75}"          # untimed heat-soak seconds (settle DVFS to a flat clock)
PREWARM="${PREWARM:-12}"    # v3: per-arm untimed warmup before EACH timed sample (0 = v2 behavior)
WARMUP="${WARMUP:-6}"       # per-run seconds discarded before sampling
WINDOW="${WINDOW:-25}"      # per-run steady sampling seconds
REPS="${REPS:-40}"          # v3: powered for a ~+0.12% true effect vs the strict t~5 bar
STAB_TOL="${STAB_TOL:-0.01}"   # max in-window clock sd/mean to accept a rep (1%)
CLK_MATCH="${CLK_MATCH:-25}"   # max |arm_A_clk - arm_B_clk| MHz to accept a rep pair
REQUIRE_CLEAN="${REQUIRE_CLEAN:-0}"  # 1 = abort if preflight shows throttle/co-tenant

[ "${#BRANCHES[@]}" -eq 2 ] || { echo "need exactly 2 branches, got: ${BRANCHES[*]}"; exit 1; }
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

echo "== build both branches (reset to origin, clean build) =="
git fetch --quiet origin || true
for b in "${BRANCHES[@]}"; do
  git checkout "$b" >/dev/null 2>&1 || { echo "checkout $b FAILED"; exit 1; }
  git reset --hard "origin/$b" >/dev/null 2>&1 || { echo "reset origin/$b FAILED"; exit 1; }
  make clean >/dev/null 2>&1
  make >/dev/null 2>&1 || { echo "BUILD FAILED for $b"; exit 1; }
  cp -f CUDACyclone "$TMP/CUDACyclone.$b"
  echo "  built $b ($(git rev-parse --short HEAD))"
done
git checkout "$orig_ref" >/dev/null 2>&1 || true

echo "== heat-soak ${SOAK}s (settle DVFS to a flat clock BEFORE sampling) =="
timeout -s INT "$SOAK" "$TMP/CUDACyclone.${BRANCHES[0]}" \
    --range "$RANGE" --target-hash160 "$TARGET" --grid "$GRID" >/dev/null 2>&1 || true
echo "  post-soak clock: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits | head -n1) MHz"

run_once() {  # $1 = binary ; echoes "kc_paired clk_mean clk_sd nsamp"
  local bin="$1" out="$TMP/out" t=0 pid clk spd
  : > "$out"; : > "$TMP/ratios"; : > "$TMP/clks"
  timeout -s INT $((WARMUP+WINDOW+1)) "$bin" \
      --range "$RANGE" --target-hash160 "$TARGET" --grid "$GRID" > "$out" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null && [ "$t" -lt $((WARMUP+WINDOW)) ]; do
    if [ "$t" -ge "$WARMUP" ]; then
      clk=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null | head -n1)
      spd=$(tr '\r' '\n' < "$out" | sed -nE 's/.*Speed:[[:space:]]*([0-9.]+).*/\1/p' | tail -n1)
      if [ -n "$clk" ] && [ -n "$spd" ]; then
        awk -v s="$spd" -v c="$clk" 'BEGIN{if(c>0)printf "%.6f\n", s/c}' >> "$TMP/ratios"
        echo "$clk" >> "$TMP/clks"
      fi
    fi
    sleep 1; t=$((t+1))
  done
  kill -INT "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  local kc cline
  kc=$(awk 'NF{s+=$1;n++} END{if(n>0)printf "%.6f",s/n; else printf "NaN"}' "$TMP/ratios")
  cline=$(awk 'NF{s+=$1;ss+=$1*$1;n++} END{if(n>0){m=s/n; sd=(n>1)?sqrt((ss-n*m*m)/(n-1)):0; printf "%.2f %.2f %d",m,sd,n} else printf "NaN NaN 0"}' "$TMP/clks")
  echo "$kc $cline"
}

: > "$TMP/all.dat"
echo "== $REPS interleaved reps (SOAK=${SOAK}s PREWARM=${PREWARM}s WARMUP=${WARMUP}s WINDOW=${WINDOW}s GRID=$GRID) =="
for ((r=1; r<=REPS; r++)); do
  if (( r % 2 == 1 )); then order=("${BRANCHES[0]}" "${BRANCHES[1]}"); else order=("${BRANCHES[1]}" "${BRANCHES[0]}"); fi
  pos=0
  for b in "${order[@]}"; do
    pos=$((pos+1))
    # v3: per-arm untimed prewarm so BOTH arms enter their timed window from an identical
    # just-ran-this-same-binary state -> kills the 2nd-slot warm-up bonus (was ~+0.18%).
    if [ "$PREWARM" -gt 0 ]; then
      timeout -s INT "$PREWARM" "$TMP/CUDACyclone.$b" \
          --range "$RANGE" --target-hash160 "$TARGET" --grid "$GRID" >/dev/null 2>&1 || true
    fi
    read -r kc cm csd ns < <(run_once "$TMP/CUDACyclone.$b")
    printf 'rep %2d  %-10s pos=%d keys/cyc=%s  clk=%s MHz (sd %s, n=%s)\n' "$r" "$b" "$pos" "$kc" "$cm" "$csd" "$ns"
    echo "$r $b $kc $cm $csd $pos" >> "$TMP/all.dat"
  done
done

echo; echo "== verdict (clean = both arms clock-stable AND arm-to-arm matched) =="
python3 - "$TMP/all.dat" "${BRANCHES[0]}" "${BRANCHES[1]}" "$STAB_TOL" "$CLK_MATCH" <<'PY'
import sys, math, statistics as st
path, A, B, stab, match = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
reps={}
for ln in open(path):
    p=ln.split()
    if len(p)<5: continue
    r=int(p[0]); br=p[1]
    try: kc=float(p[2]); cm=float(p[3]); csd=float(p[4])
    except: continue
    pos=int(p[5]) if len(p)>=6 else 0
    reps.setdefault(r,{})[br]=(kc,cm,csd,pos)
cleanA=[]; cleanB=[]; allA=[]; allB=[]; dropped=[]
cleanDiff=[]; diff_Afirst=[]; diff_Bfirst=[]
for r in sorted(reps):
    d=reps[r]
    if A not in d or B not in d: continue
    (kcA,cmA,csdA,posA)=d[A]; (kcB,cmB,csdB,posB)=d[B]
    allA.append(kcA); allB.append(kcB)
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
    print("gated-out reps:"); print("\n".join(dropped))
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
    print(">>> FULL vs CLEAN disagree in SIGN -> INCONCLUSIVE, re-run at a stable clock.")
elif len(cleanA)<12:
    print(f">>> only {len(cleanA)} clean reps/arm (<12) -> underpowered for a sub-0.5% call; add reps or fix the clock.")
else:
    print(">>> clean subset is the headline; call a WIN only at Welch t~5 WITH total-separation AND the B-first split also positive.")
PY
