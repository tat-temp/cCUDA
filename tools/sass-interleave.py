#!/usr/bin/env python3
"""Did ptxas actually INTERLEAVE the two hash chains inside getHash160_w2_x2?

Run on the GPU box, from the repo root, after a build:

    make && python3 tools/sass-interleave.py

WHY THIS SCRIPT EXISTS. With -rdc=false, ptxas places intra-module __noinline__ device functions
INSIDE the calling kernel's text section, so `cuobjdump -sass` emits NO `Function :` label for
them -- grepping for "getHash160_w2_x2" returns nothing, which reads like "the function vanished"
rather than "you are looking it up wrong". The bodies have to be found by their CALL.REL.NOINC
target offsets, and those offsets MOVE on every code-size change, so they must be re-derived each
time rather than hardcoded. This project has been bitten by that repeatedly.

WHAT IT MEASURES. The 2-wide hash is a pure SCHEDULING change -- same instructions, different
order -- so instruction counts cannot tell you whether it worked. Producer-to-consumer dependence
distance can.

READ THE MEAN-DISTANCE RATIO, NOT A dist=1 COUNT. The first version of this script tested whether
the dist=1 fraction fell, on the assumption that one hash is a naive serial chain. That assumption
is WRONG and the test is worthless here: measured on real SASS, a single 1-wide hash already has
only 0.1% of its dependences at distance 1 and a MEAN distance of ~26, because ptxas already
schedules each hash with substantial internal slack (SHA's message schedule, RIPEMD's two parallel
lines, independent sub-expressions within a round). That script reported the right answer for the
wrong reason -- a trap worth not re-laying.

The sound test: interleaving two INDEPENDENT chains inserts one chain's instructions between the
other chain's producer/consumer pairs, so the mean distance should roughly DOUBLE. Emitting the
chains back to back leaves each chain's internal schedule untouched, so the mean is unchanged
while the body is ~2x the size. Size ratio says "both chains are here"; distance ratio says
"and they are woven together".
"""
import re
import subprocess
import sys

INSN_RE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9._]*)\s*([^;]*);")
REG_RE = re.compile(r"\bR(\d{1,3})\b")
# opcodes whose first operand is NOT a written register
NO_DEST = ("ST", "STL", "STG", "STS", "RED", "ATOM", "BRA", "BRX", "RET", "EXIT", "BSSY",
           "BSYNC", "CALL", "NOP", "BAR", "MEMBAR", "PMTRIG", "YIELD")


def dump_sass():
    try:
        arch = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
        ).decode().split()[0].replace(".", "")
    except Exception:
        arch = None
    cmd = ["cuobjdump", "-sass"] + (["-arch", "sm_%s" % arch] if arch else []) + ["CUDACyclone"]
    try:
        return subprocess.check_output(cmd).decode("utf-8", "replace")
    except Exception:
        return subprocess.check_output(["cuobjdump", "-sass", "CUDACyclone"]).decode("utf-8", "replace")


def parse(text):
    return [(int(m.group(1), 16), m.group(2), m.group(3)) for m in INSN_RE.finditer(text)]


def body_at(insns, start_off):
    """Instructions from the CALL target offset up to and including its first RET."""
    idx = next((k for k, (o, _, _) in enumerate(insns) if o == start_off), None)
    if idx is None:
        return []
    body = []
    for k in range(idx, len(insns)):
        body.append(insns[k])
        if insns[k][1].startswith("RET"):
            break
    return body


def distance_hist(body):
    """For each instruction, distance back to the nearest producer of any source register."""
    last_write = {}
    dists = []
    for i, (_off, op, ops) in enumerate(body):
        base = op.split(".")[0]
        parts = [p.strip() for p in ops.split(",")]
        srcs, dst = parts, None
        if base not in NO_DEST and parts:
            dst, srcs = parts[0], parts[1:]
        best = None
        for s in srcs:
            for rm in REG_RE.finditer(s):
                d = i - last_write.get(rm.group(1), -10**9)
                if d < 10**8 and (best is None or d < best):
                    best = d
        if best is not None:
            dists.append(best)
        if dst:
            for rm in REG_RE.finditer(dst):
                last_write[rm.group(1)] = i
    return dists


def report(name, body):
    d = distance_hist(body)
    if not d:
        print("  %-14s (no measurable dependences)" % name)
        return None
    n = len(d)
    mean = sum(d) / n
    print("  %-14s %5d insn | dist=1 %5.1f%% | dist=2 %5.1f%% | dist>=3 %5.1f%% | mean %6.2f"
          % (name, len(body), 100 * sum(1 for x in d if x == 1) / n,
             100 * sum(1 for x in d if x == 2) / n,
             100 * sum(1 for x in d if x >= 3) / n, mean))
    return mean


def main():
    text = dump_sass()
    insns = parse(text)
    if not insns:
        print("no SASS parsed -- is CUDACyclone built?")
        return 1

    targets = sorted({int(t, 16) for t in re.findall(r"CALL\.REL\.NOINC\s+0x([0-9a-f]+)", text)})
    if not targets:
        print("No intra-module calls -- the hash may have been inlined. Check CUDAHash.cuh.")
        return 1
    print("CALL.REL.NOINC targets: %s" % ", ".join("0x%x" % t for t in targets))

    bodies = [(t, b) for t, b in ((t, body_at(insns, t)) for t in targets) if b]
    bodies.sort(key=lambda tb: -len(tb[1]))

    print("")
    print("Callee bodies by size (largest should be the 2-wide getHash160_w2_x2):")
    stats = [(len(b), report("0x%x" % t, b)) for t, b in bodies]

    print("")
    print("INTERPRETATION")
    usable = [(n, m) for n, m in stats if m is not None and n > 200]
    if len(usable) < 2:
        print("  Not enough comparable bodies.")
        return 0
    (big_n, big_m), (ctl_n, ctl_m) = usable[0], usable[1]
    size_ratio = big_n / max(ctl_n, 1)
    dist_ratio = big_m / max(ctl_m, 1e-9)
    print("  2-wide body : %5d insn, mean dependence distance %6.2f" % (big_n, big_m))
    print("  1-wide ctrl : %5d insn, mean dependence distance %6.2f" % (ctl_n, ctl_m))
    print("  size ratio  : %.2fx  (expect ~2x: both chains present)" % size_ratio)
    print("  DISTANCE    : %.2fx  (the decisive number -- expect ~2x if woven together)" % dist_ratio)
    print("")
    print("  Do NOT read the dist=1 column as the verdict. ptxas already schedules a single hash")
    print("  with ~%.0f instructions of producer-to-consumer slack, so dist=1 is near zero in BOTH"
          % ctl_m)
    print("  bodies whether or not the chains were interleaved.")
    print("")
    if size_ratio < 1.5:
        print("  => INCONCLUSIVE: the large body is not ~2x, so it may not hold both chains.")
    elif dist_ratio > 1.6:
        print("  => INTERLEAVE FIRED. Distances stretched as expected for 2-way alternation.")
    elif dist_ratio > 1.2:
        print("  => PARTIAL interleaving -- coarser than a clean alternation.")
    else:
        print("  => NOT INTERLEAVED. Both chains are present (~2x size) but each kept its own")
        print("     internal schedule: ptxas emitted them back to back. A warp issues IN ORDER,")
        print("     so chain B cannot fill chain A's stalls. This is a NULL, and no amount of")
        print("     register work changes it.")
        print("")
        print("     Before escalating to hand-interleaved round lists, weigh this: a mean distance")
        print("     of ~%.0f means each hash ALREADY carries substantial internal ILP, which" % ctl_m)
        print("     undercuts the premise that the hash is starved of independent work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
