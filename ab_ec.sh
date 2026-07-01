#!/usr/bin/env bash
# ab_ec.sh — A/B harness: RCKangaroo field arithmetic vs CUDACyclone's own EC field ops.
# Branch: rck-ec-ab.  RUN ON THE RTX 5090 BOX (needs nvcc + the GPU).
#
# Pipeline:
#   0. (optional) sync the local branch to origin  [SYNC=1]  -- avoids the "stale local
#      checkout builds an old binary" trap; git reset --hard is destructive so it is opt-in.
#   1. build: baseline (CUDACyclone) + rckfield/rckinv/rckall + ecbench
#   2. CORRECTNESS GATE (aborts the A/B if anything fails):
#        a. ecbench  -> per-op check vs Python KATs (mul/sqr/inv/sub), both backends
#        b. proof.py -> end-to-end planted-key search vs each RCK variant (must equal baseline)
#   3. THROUGHPUT A/B: cadence-independent steady-state Mkeys/s (Count deltas over a fixed
#      wall-clock window, warmup discarded), variants interleaved to cancel thermal/DVFS
#      drift, Welch's t-test vs baseline.
#
# Usage:   bash ab_ec.sh [REPS] [WINDOW_SEC]
#   REPS       measured windows per variant           (default 6)
#   WINDOW_SEC measured seconds per window            (default 30)
# Env knobs: GRID(512,512) WARMUP(8) RANGE TARGET SYNC(0) SKIP_PROOF(0)
#            PROOF_RANGE PROOF_GRID VARIANTS
set -u

BRANCH="rck-ec-ab"
REPS="${1:-6}"
WINDOW="${2:-30}"
WARMUP="${WARMUP:-8}"
GRID="${GRID:-512,512}"
# Huge power-of-two range with an unfindable target => pure steady-state throughput.
RANGE="${RANGE:-8000000000000000:ffffffffffffffff}"
TARGET="${TARGET:-deadbeefdeadbeefdeadbeefdeadbeefdeadbeef}"
SYNC="${SYNC:-0}"
SKIP_PROOF="${SKIP_PROOF:-0}"
# Small quick range for the planted-key gate (proof.py plants keys inside it).
PROOF_RANGE="${PROOF_RANGE:-8000000000:ffffffffff}"
PROOF_GRID="${PROOF_GRID:-512,512}"
# baseline MUST be first; the rest are compared against it.
VARIANTS="${VARIANTS:-CUDACyclone CUDACyclone-rckfield CUDACyclone-crfield CUDACyclone-rckinv CUDACyclone-rckall}"

LOGDIR="ab_ec_logs"; mkdir -p "$LOGDIR"
say(){ printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# --- 0. sync ---------------------------------------------------------------------------
if [ "$SYNC" = "1" ]; then
  say "sync $BRANCH to origin (git reset --hard)"
  git fetch origin && git checkout "$BRANCH" && git reset --hard "origin/$BRANCH"
fi
say "built from commit: $(git rev-parse --short HEAD)  ($(git rev-parse --abbrev-ref HEAD))"

# --- 1. build --------------------------------------------------------------------------
say "build (make clean; baseline + rck variants + ecbench)"
make clean >/dev/null
make            -j"$(nproc)" >/dev/null || { echo "build baseline FAILED"; exit 1; }
make rckfield   -j"$(nproc)" >/dev/null || { echo "build rckfield FAILED";  exit 1; }
make crfield    -j"$(nproc)" >/dev/null || { echo "build crfield FAILED";   exit 1; }
make rckinv     -j"$(nproc)" >/dev/null || { echo "build rckinv FAILED";    exit 1; }
make rckall     -j"$(nproc)" >/dev/null || { echo "build rckall FAILED";    exit 1; }
make ecbench    -j"$(nproc)" >/dev/null || { echo "build ecbench FAILED";   exit 1; }
echo "OK: $(ls -1 CUDACyclone CUDACyclone-rck* CUDACyclone-crfield CUDACyclone-ecbench 2>/dev/null | tr '\n' ' ')"

# register/spill check on the clean-room build (the make-or-break question — Phase 3 spilled)
echo "-- clean-room kernel resource usage (want REG<=128, LOCAL:0, no spills):"
cuobjdump -res-usage CUDACyclone-crfield 2>/dev/null | grep -A1 kernel_point_add | head -4 || true

# --- 2a. per-op correctness (ecbench) --------------------------------------------------
say "correctness: ecbench (per-op vs KATs, both backends)"
./CUDACyclone-ecbench 5 1 | tee "$LOGDIR/ecbench.txt"
if grep -q "FAIL" "$LOGDIR/ecbench.txt"; then
  echo "!! ecbench reports a correctness FAIL -- aborting A/B."; exit 2
fi

# --- 2b. end-to-end correctness (proof.py planted keys) --------------------------------
if [ "$SKIP_PROOF" != "1" ]; then
  say "correctness: proof.py planted-key gate on each variant (range $PROOF_RANGE)"
  for bin in $VARIANTS; do
    [ -x "./$bin" ] || { echo "missing $bin"; exit 1; }
    echo "-- proof.py -c ./$bin"
    python3 proof.py -c "./$bin" -r "$PROOF_RANGE" --grid "$PROOF_GRID" \
        --start-count 8 --end-count 8 --quartile-count 4 --timeout 120 \
        > "$LOGDIR/proof_$bin.txt" 2>&1
    fails=$(grep -oE "Failures=[0-9]+" "$LOGDIR/proof_$bin.txt" | tail -1 | grep -oE "[0-9]+")
    succ=$(grep -oE  "Successes=[0-9]+" "$LOGDIR/proof_$bin.txt" | tail -1 | grep -oE "[0-9]+")
    echo "   $bin : Successes=${succ:-?} Failures=${fails:-?}"
    if [ "${fails:-1}" != "0" ]; then
      echo "!! $bin FAILED planted-key correctness -- aborting A/B. See $LOGDIR/proof_$bin.txt"; exit 3
    fi
  done
fi

# --- 3. throughput A/B (interleaved) ---------------------------------------------------
say "throughput A/B: $REPS windows x ${WINDOW}s (warmup ${WARMUP}s), grid $GRID, interleaved"
run_window(){ # $1=bin  $2=out
  timeout $((WARMUP + WINDOW + 4)) \
    ./"$1" --range "$RANGE" --target-hash160 "$TARGET" --grid "$GRID" > "$2" 2>&1 || true
}
for r in $(seq 1 "$REPS"); do
  for bin in $VARIANTS; do
    printf '  rep %s/%s  %-24s ...' "$r" "$REPS" "$bin"
    run_window "$bin" "$LOGDIR/${bin}_rep${r}.log"
    printf ' done\n'
  done
done

# --- parse + stats + Welch t (Python) --------------------------------------------------
say "results"
python3 - "$LOGDIR" "$WARMUP" "$REPS" $VARIANTS <<'PY'
import sys, re, glob, math
logdir, warmup, reps = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
variants = sys.argv[4:]
pair_re = re.compile(r'Time:\s*([0-9.]+)\s*s.*?Count:\s*([0-9]+)')
def rate_of(path):
    # parse (t,count) samples (progress line uses \r); throughput = dCount/dt after warmup.
    try: raw = open(path, 'r', errors='ignore').read()
    except FileNotFoundError: return None
    pts=[]
    for line in raw.replace('\r','\n').split('\n'):
        m=pair_re.search(line)
        if m: pts.append((float(m.group(1)), int(m.group(2))))
    pts=[p for p in pts if p[0] is not None]
    if len(pts)<2: return None
    tmax=pts[-1][0]
    warm=[p for p in pts if p[0]>=warmup]
    if len(warm)<2:                      # window too short; fall back to full span
        warm=pts
    (t0,c0),(t1,c1)=warm[0],warm[-1]
    if t1<=t0 or c1<c0: return None
    return (c1-c0)/((t1-t0)*1e6)          # Mkeys/s
def stats(xs):
    n=len(xs)
    if n==0: return 0,0.0,0.0
    m=sum(xs)/n
    v=sum((x-m)**2 for x in xs)/(n-1) if n>1 else 0.0
    return n,m,math.sqrt(v)
data={}
for bin in variants:
    xs=[]
    for r in range(1,reps+1):
        rt=rate_of(f"{logdir}/{bin}_rep{r}.log")
        if rt: xs.append(rt)
    data[bin]=xs
base=variants[0]
bn,bm,bsd=stats(data[base]) if data[base] else (0,0,0)
print(f"{'variant':26s} {'n':>2s} {'mean Mkeys/s':>13s} {'sd':>7s} {'CV%':>6s} {'vs base%':>8s} {'Welch t':>8s}")
def welch(a,b):
    if len(a)<2 or len(b)<2: return 0.0
    na,ma,sa=stats(a); nb,mb,sb=stats(b)
    se=math.sqrt(sa*sa/na + sb*sb/nb) if na>1 and nb>1 else 0.0
    return (ma-mb)/se if se>0 else 0.0
for bin in variants:
    xs=data[bin]
    if not xs: print(f"{bin:26s}  0   (no samples parsed)"); continue
    n,m,sd=stats(xs); cv=100*sd/m if m else 0
    if bin==base:
        print(f"{bin:26s} {n:2d} {m:13.1f} {sd:7.1f} {cv:6.2f} {'--':>8s} {'--':>8s}")
    else:
        d=100*(m-bm)/bm if bm else 0
        t=welch(xs,data[base])
        print(f"{bin:26s} {n:2d} {m:13.1f} {sd:7.1f} {cv:6.2f} {d:+8.2f} {t:+8.2f}")
print("\nrck/cyc < 1.00 (negative vs base%) => RCKangaroo field ops are SLOWER end-to-end.")
print("|Welch t| < ~2 => difference is within run-to-run noise (not significant at this n).")
PY

say "done — logs in $LOGDIR/  (ecbench.txt, proof_*.txt, *_rep*.log)"
