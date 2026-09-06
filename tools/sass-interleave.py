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

WHAT IT MEASURES. The 2-wide hash is a pure SCHEDULING change: same instructions, different order.
So instruction counts cannot tell you whether it worked. Dependence distance can.

In a single serial dependency chain (one SHA-256), each instruction consumes a value produced by
the instruction immediately before it, so the producer->consumer distance is mostly 1. If ptxas
interleaves two INDEPENDENT chains, consecutive instructions alternate between them and the
typical distance becomes 2 or more.

The 1-wide callee is the built-in control: it is inherently serial, so its distance-1 fraction is
the "not interleaved" baseline. If the 2-wide body shows a markedly LOWER distance-1 fraction than
the 1-wide body, the interleave fired. If the two are about equal, ptxas serialized the chains and
the lever is a null -- escalate to hand-interleaving the round lists.
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
    out = []
    for m in INSN_RE.finditer(text):
        off, op, ops = int(m.group(1), 16), m.group(2), m.group(3)
        out.append((off, op, ops))
    return out


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
    """Producer->consumer distance for each instruction, in instructions."""
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
                r = rm.group(1)
                if r in last_write:
                    d = i - last_write[r]
                    if best is None or d < best:
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
        print("  %-22s (no measurable dependences)" % name)
        return None
    n = len(d)
    at1 = sum(1 for x in d if x == 1) / n
    at2 = sum(1 for x in d if x == 2) / n
    ge3 = sum(1 for x in d if x >= 3) / n
    mean = sum(d) / n
    print("  %-22s %5d insn | dist=1 %5.1f%% | dist=2 %5.1f%% | dist>=3 %5.1f%% | mean %.2f"
          % (name, len(body), 100 * at1, 100 * at2, 100 * ge3, mean))
    return at1


def main():
    text = dump_sass()
    insns = parse(text)
    if not insns:
        print("no SASS parsed -- is CUDACyclone built?")
        return 1

    targets = sorted({int(t, 16) for t in re.findall(r"CALL\.REL\.NOINC\s+0x([0-9a-f]+)", text)})
    print("CALL.REL.NOINC targets found: %s"
          % ", ".join("0x%x" % t for t in targets) if targets else "none")
    if not targets:
        print("No intra-module calls -- the hash may have been inlined. Check CUDAHash.cuh.")
        return 1

    bodies = [(t, body_at(insns, t)) for t in targets]
    bodies = [(t, b) for t, b in bodies if b]
    bodies.sort(key=lambda tb: -len(tb[1]))

    print("\nCallee bodies by size (largest should be the 2-wide getHash160_w2_x2):")
    fracs = []
    for t, b in bodies:
        fracs.append((len(b), report("0x%x" % t, b)))

    print("\nINTERPRETATION")
    if len(bodies) < 2:
        print("  Only one callee body found; cannot compare against the 1-wide control.")
        return 0
    big_n, big_f = fracs[0]
    ctl = [f for f in fracs[1:] if f[1] is not None]
    if big_f is None or not ctl:
        print("  Not enough data.")
        return 0
    ctl_n, ctl_f = ctl[0]
    print("  2-wide body : %d insn, dist-1 %.1f%%" % (big_n, 100 * big_f))
    print("  1-wide ctrl : %d insn, dist-1 %.1f%%" % (ctl_n, 100 * ctl_f))
    print("  size ratio  : %.2fx (expect ~2x if both chains are present)" % (big_n / max(ctl_n, 1)))
    drop = ctl_f - big_f
    if drop > 0.10:
        print("  => INTERLEAVE FIRED. The 2-wide body's dependence distances are markedly longer,")
        print("     which is what alternating between two independent chains looks like.")
    elif drop > 0.03:
        print("  => PARTIAL. Some interleaving, but less than a clean 2-way alternation.")
    else:
        print("  => NOT INTERLEAVED. ptxas emitted the chains back to back, so this is a NULL:")
        print("     no register work will save it. Escalate to hand-interleaving the round lists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
