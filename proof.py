#!/usr/bin/env python3
import argparse
import subprocess
import secrets
import hashlib
import time
import sys
import math
import random
import re
import select
import struct
import threading
from typing import List, Tuple, Optional, Set, Dict
from ecdsa import SECP256k1, SigningKey

BASE58_ALPH = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

def base58_encode(b: bytes) -> str:
    zeros = 0
    for c in b:
        if c == 0:
            zeros += 1
        else:
            break
    num = int.from_bytes(b, "big")
    enc = bytearray()
    while num > 0:
        num, rem = divmod(num, 58)
        enc.append(BASE58_ALPH[rem])
    enc = bytes(reversed(enc))
    return ("1" * zeros) + enc.decode("ascii")

def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()

def _ripemd160_py(message: bytes) -> bytes:
    # Pure-Python RIPEMD-160. Fallback for systems where OpenSSL 3 ships without
    # the legacy provider, so hashlib.new("ripemd160") raises ValueError (common
    # on Ubuntu 22.04+/Debian 12+). Without this every address fails to generate
    # and the whole test run reports 0 successes.
    _r = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
          7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
          3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
          1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
          4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13]
    _rp = [5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
           6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
           15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
           8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
           12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11]
    _s = [11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
          7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
          11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
          11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
          9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6]
    _sp = [8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
           9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
           9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
           15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
           8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11]
    _K  = [0x00000000,0x5A827999,0x6ED9EBA1,0x8F1BBCDC,0xA953FD4E]
    _Kp = [0x50A28BE6,0x5C4DD124,0x6D703EF3,0x7A6D76E9,0x00000000]
    M = 0xffffffff
    def rol(x, n): return ((x << n) | (x >> (32 - n))) & M
    def f(j, x, y, z):
        if j < 16: return x ^ y ^ z
        if j < 32: return (x & y) | ((~x & M) & z)
        if j < 48: return (x | (~y & M)) ^ z
        if j < 64: return (x & z) | (y & (~z & M))
        return x ^ (y | (~z & M))
    ml = (len(message) * 8) & 0xffffffffffffffff
    message = message + b'\x80'
    while len(message) % 64 != 56:
        message += b'\x00'
    message += struct.pack('<Q', ml)
    h0,h1,h2,h3,h4 = 0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0
    for off in range(0, len(message), 64):
        X = struct.unpack('<16L', message[off:off+64])
        A,B,C,D,E = h0,h1,h2,h3,h4
        Ap,Bp,Cp,Dp,Ep = h0,h1,h2,h3,h4
        for j in range(80):
            T = (rol((A + f(j,B,C,D) + X[_r[j]] + _K[j//16]) & M, _s[j]) + E) & M
            A=E; E=D; D=rol(C,10); C=B; B=T
            T = (rol((Ap + f(79-j,Bp,Cp,Dp) + X[_rp[j]] + _Kp[j//16]) & M, _sp[j]) + Ep) & M
            Ap=Ep; Ep=Dp; Dp=rol(Cp,10); Cp=Bp; Bp=T
        T  = (h1 + C + Dp) & M
        h1 = (h2 + D + Ep) & M
        h2 = (h3 + E + Ap) & M
        h3 = (h4 + A + Bp) & M
        h4 = (h0 + B + Cp) & M
        h0 = T
    return struct.pack('<5L', h0,h1,h2,h3,h4)

def _ripemd160(data: bytes) -> bytes:
    try:
        h = hashlib.new("ripemd160")
        h.update(data)
        return h.digest()
    except (ValueError, TypeError):
        return _ripemd160_py(data)

def hash160(b: bytes) -> bytes:
    return _ripemd160(hashlib.sha256(b).digest())

def p2pkh_from_pubkey_compressed(pubkey_comp: bytes) -> str:
    h160 = hash160(pubkey_comp)
    payload = b"\x00" + h160
    checksum = sha256d(payload)[:4]
    return base58_encode(payload + checksum)

def compressed_pubkey_from_priv32(priv32: bytes) -> bytes:
    sk = SigningKey.from_string(priv32, curve=SECP256k1)
    vk = sk.get_verifying_key()
    xy = vk.to_string()
    x = xy[:32]
    y = xy[32:]
    prefix = b"\x03" if (int.from_bytes(y, "big") & 1) else b"\x02"
    return prefix + x

def parse_hex_range(range_str: str) -> Tuple[int, int]:
    if ":" not in range_str:
        raise ValueError("Range must be HEX_START:HEX_END")
    s, e = range_str.split(":", 1)
    s = s.strip()
    e = e.strip()
    if s.startswith(("0x", "0X")): s = s[2:]
    if e.startswith(("0x", "0X")): e = e[2:]
    s = s or "0"
    e = e or "0"
    si = int(s, 16)
    ei = int(e, 16)
    if si > ei:
        raise ValueError("Start > End in range")
    return si, ei

def int_to_priv32_hex(i: int) -> str:
    return i.to_bytes(32, "big").hex()

def parse_batch_from_grid(grid_arg: Optional[str]) -> Optional[int]:
    if not grid_arg:
        return None
    try:
        part = grid_arg.split(",")[0].strip()
        b = int(part)
        if b <= 0 or (b & 1) != 0:
            return None
        return b
    except Exception:
        return None

# Mirrors MAX_BATCH_SIZE in CUDACyclone.cu. Only used for a fast, legible failure when the
# banner probe below could not run; the kernel itself is the real enforcer.
KERNEL_MAX_BATCH = 1024

def probe_batch_size(cyclone_path: str, range_arg: str, grid_arg: Optional[str],
                     timeout_s: float = 90.0) -> Tuple[Optional[int], str, List[str]]:
    """Ask the binary which batch size it will ACTUALLY use, by reading its own banner.

    This is the only authoritative source. Every coverage claim proof.py makes is mod-B
    against the *kernel's* B: the "full mod B residue coverage" block exists to exercise
    every offset inside a kernel batch. If proof.py's B is smaller than the kernel's --
    a stale --grid default, a --batch override, a changed compiled default -- the extra
    residue classes are simply never generated, so they are never tested, and the run
    still reports every test PASS. Nothing about the output looks wrong.

    Returns (batch_or_None, status, last_output_lines). status is one of:
      "ok"            -- the banner line was read
      "launch_failed" -- the binary could not be executed at all
      "child_rejected"-- the binary exited nonzero on its own before the banner (bad args)
      "no_banner"     -- it ran past the banner without the line (binary changed?)
      "timeout"       -- still running after timeout_s without printing it
    The caller must treat launch_failed/child_rejected as fatal: those repeat identically
    on all N test runs, and the binary's own message is already in the tail.
    """
    argv = [cyclone_path, "--range", range_arg, "--target-hash160", "00" * 20]
    if grid_arg:
        argv += ["--grid", grid_arg]
    try:
        p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             bufsize=1, universal_newlines=True)
    except OSError as ex:
        return None, "launch_failed", [f"could not launch {cyclone_path}: {ex}"]

    # Read on a thread, iterating the stream directly. Do NOT poll the raw fd with select()
    # and then readline() the buffered wrapper: CUDACyclone's stdout is block-buffered to a
    # pipe and the whole banner lands in ONE write, so the first readline pulls all of it
    # into Python's buffer, select then reports "nothing readable", and the remaining banner
    # lines sit unread until the child's next flush -- one line per 1 s progress print.
    batch: List[Optional[int]] = [None]
    stopped_early = [False]   # reader returned on purpose (child still running), vs. hit EOF
    tail: List[str] = []

    def reader() -> None:
        try:
            assert p.stdout is not None
            for line in p.stdout:
                tail.append(line.rstrip())
                del tail[:-14]
                m = re.match(r"\s*Points batch size\s*:\s*(\d+)", line)
                if m:
                    batch[0] = int(m.group(1))
                    stopped_early[0] = True
                    return
                if "Phase-1" in line:     # banner is over and the line never came
                    stopped_early[0] = True
                    return
        except Exception:
            pass                          # stream closed under us during teardown

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    th.join(timeout_s)
    timed_out = th.is_alive()

    # EOF on stdout does NOT mean the child has been reaped yet, so an exit status is only
    # trustworthy after an explicit wait(). Reading poll() straight after the reader returns
    # races the child's exit and would misreport a rejected config as a missing banner.
    rc: Optional[int] = None
    if not timed_out and not stopped_early[0]:
        try:
            rc = p.wait(timeout=5)
        except Exception:
            pass
    if p.poll() is None:
        try:
            p.terminate()
            p.wait(timeout=5)
        except Exception:
            try:
                p.kill()
                p.wait(timeout=5)
            except Exception:
                pass
    th.join(timeout=1.0)
    # Close only once the reader has let go. Closing a file object out from under a thread
    # that is blocked reading it deadlocks on the stream lock -- which is exactly the state
    # after a timeout, when the child (or a grandchild holding the pipe) is still alive.
    if not th.is_alive():
        try:
            if p.stdout is not None:
                p.stdout.close()
        except Exception:
            pass

    if batch[0] is not None:
        return batch[0], "ok", tail
    if timed_out:
        return None, "timeout", tail
    if rc not in (0, None):
        return None, "child_rejected", tail
    return None, "no_banner", tail

def run_cyclone_and_watch(
    cyclone_path: str,
    range_arg: str,
    address: str,
    grid_arg: Optional[str],
    match_marker: str = "======== FOUND MATCH! =================================",
    timeout: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    args = [cyclone_path, "--range", range_arg, "--address", address]
    if grid_arg:
        args += ["--grid", grid_arg]
    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    found_priv: Optional[str] = None
    start_time = time.time()
    try:
        assert p.stdout is not None
        for line in p.stdout:
            if match_marker in line:
                for _ in range(20):
                    fd = p.stdout.fileno()
                    rlist, _, _ = select.select([fd], [], [], 0.2)
                    if not rlist:
                        break
                    nxt = p.stdout.readline()
                    if not nxt:
                        break
                    if "Private Key" in nxt:
                        parts = nxt.split(":", 1)
                        if len(parts) > 1:
                            found_priv = parts[1].strip()
                            break
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                return True, found_priv
            if "Private Key" in line and found_priv is None:
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    found_priv = parts[1].strip()
            if timeout is not None and (time.time() - start_time) > timeout:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                return False, None
        p.wait()
        if found_priv is not None:
            return True, found_priv
        return False, None
    finally:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

def gen_series_step(start: int, end: int, first: int, step: int, count: int) -> List[int]:
    out = []
    cur = first
    while len(out) < count and (start <= cur <= end):
        out.append(cur)
        cur += step
    return out

def gen_start_dual_parity(start: int, end: int, count_each: int) -> Tuple[List[int], List[int]]:
    a = gen_series_step(start, end, start, +2, count_each)
    b = gen_series_step(start, end, start + 1, +2, count_each)
    return a, b

def gen_end_dual_parity(start: int, end: int, count_each: int) -> Tuple[List[int], List[int]]:
    a = gen_series_step(start, end, end, -2, count_each)
    b = gen_series_step(start, end, end - 1, -2, count_each)
    return a, b

def full_mod_residue_cover(start: int, end: int, B: int, used: Set[int]) -> List[int]:
    out = []
    for r in range(B):
        v = start + r
        if v < start or v > end:
            continue
        if v in used:
            continue
        out.append(v)
    return out

def quartile_bounds(start: int, end: int) -> List[Tuple[int, int]]:
    size = end - start + 1
    q1_end = start + (size * 25) // 100 - 1
    q2_end = start + (size * 50) // 100 - 1
    q3_end = start + (size * 75) // 100 - 1
    q1_end = max(start, min(q1_end, end))
    q2_end = max(start, min(q2_end, end))
    q3_end = max(start, min(q3_end, end))
    q1 = (start, q1_end)
    q2 = (min(q1_end + 1, end), q2_end)
    q3 = (min(q2_end + 1, end), q3_end)
    q4 = (min(q3_end + 1, end), end)
    return [q1, q2, q3, q4]

def pick_one_with_residue_in_interval(lo: int, hi: int, residue: int, B: int) -> Optional[int]:
    rem = lo % B
    delta = (residue - rem) % B
    n = lo + delta
    if n > hi:
        return None
    cnt = (hi - n) // B + 1
    k = secrets.randbelow(cnt)
    return n + k * B

def quartile_random_mod_coverage(start: int, end: int, B: int, used: Set[int], per_quart: int) -> List[List[int]]:
    qints = quartile_bounds(start, end)
    blocks: List[List[int]] = []
    residues = list(range(B))
    for (lo, hi) in qints:
        if lo > hi or per_quart <= 0:
            blocks.append([])
            continue
        random.shuffle(residues)
        chosen: List[int] = []
        tried = 0
        i = 0
        while len(chosen) < per_quart and tried < 4 * B:
            r = residues[i % B]
            i += 1
            tried += 1
            v = pick_one_with_residue_in_interval(lo, hi, r, B)
            if v is None or v in used:
                continue
            chosen.append(v)
            used.add(v)
        blocks.append(chosen)
    return blocks

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cyclone strong test runner: edges (both parities), full mod-B coverage, quartile random with mod-B diversity."
    )
    parser.add_argument("--range", "-r", dest="range_arg", required=True,
                        help="HEX range START:END (e.g. 200000000:3FFFFFFFF)")
    parser.add_argument("--cyclone-path", "-c", dest="cyclone_path", default="./CUDACyclone",
                        help="Path to CUDACyclone binary")
    parser.add_argument("--grid", dest="grid_arg", default=None,
                        help="Value for --grid passed to CUDACyclone (e.g. 1024,512). "
                             "OMIT to validate whatever the binary ships as its compiled default.")
    parser.add_argument("--batch", dest="batch", type=int, default=None,
                        help="ASSERT the kernel's batch size equals this. Not an override: a "
                             "proof-side batch that disagrees with the kernel voids the mod-B "
                             "coverage claim, so a mismatch is a hard error.")
    parser.add_argument("--timeout", dest="timeout", type=int, default=None,
                        help="Optional timeout in seconds for each CUDACyclone run")
    parser.add_argument("--start-count", type=int, default=128,
                        help="Count per parity at range start (default: 128)")
    parser.add_argument("--end-count", type=int, default=128,
                        help="Count per parity at range end (default: 128)")
    parser.add_argument("--quartile-count", type=int, default=20,
                        help="Random points per quartile (default: 20)")
    args = parser.parse_args()

    try:
        raw_start, raw_end = parse_hex_range(args.range_arg)
    except Exception as ex:
        print("Range parse error:", ex, file=sys.stderr)
        sys.exit(1)

    # The kernel validates the range EXACTLY as given on the command line (CUDACyclone.cu:488,
    # :497-503 and :531), before any of proof.py's clamping below. Mirror all three here so a
    # bad range fails in one second instead of as N consecutive identical FAILs.
    raw_len = raw_end - raw_start + 1
    if raw_len <= 0 or (raw_len & (raw_len - 1)) != 0:
        print(f"Range length (end - start + 1) must be a power of two; got {raw_len}.", file=sys.stderr)
        sys.exit(1)
    if raw_start & (raw_len - 1):
        print(f"Range start {raw_start:#x} must be aligned to the range length {raw_len:#x} "
              f"(CUDACyclone.cu:497-503 rejects this outright).", file=sys.stderr)
        sys.exit(1)

    start_i, end_i = raw_start, raw_end
    if start_i == 0:
        start_i = 1
    order = SECP256k1.order
    if end_i >= order:
        end_i = order - 1
        if start_i > end_i:
            print("Range shrunk below valid curve order.", file=sys.stderr)
            sys.exit(1)

    # --- resolve the batch size the KERNEL will actually use ---------------------------------
    # Authoritative source is the binary's own banner. See probe_batch_size().
    declared = parse_batch_from_grid(args.grid_arg) if args.grid_arg else None
    if args.grid_arg and declared is None:
        print(f"--grid {args.grid_arg!r}: cannot read a positive even batch from its first field.",
              file=sys.stderr)
        sys.exit(1)

    probed, probe_status, probe_tail = probe_batch_size(args.cyclone_path, args.range_arg, args.grid_arg)
    if probed is not None:
        B, b_source, b_verified = probed, "probed from the binary's banner", True
        if declared is not None and declared != probed:
            print(f"MISMATCH: --grid says batch={declared} but the kernel reports {probed}.",
                  file=sys.stderr)
            sys.exit(1)
    else:
        # A binary that cannot be launched, or that REFUSED this configuration, is fatal: it
        # will refuse all N test runs the same way, and the probe is already holding its own
        # error message. Reporting that as a warning is how "848 consecutive FAILs over three
        # hours" used to happen.
        if probe_status in ("launch_failed", "child_rejected"):
            what = ("could not be launched" if probe_status == "launch_failed"
                    else "refused this configuration and exited")
            print(f"CANNOT RUN: {args.cyclone_path} {what}.", file=sys.stderr)
            for ln in probe_tail[-8:]:
                print(f"    | {ln}", file=sys.stderr)
            sys.exit(1)
        # Soft path: the binary is running but never printed the line (timeout / changed banner).
        # --batch is NOT accepted as a source of B here -- it is an assertion, not an override,
        # and taking B from it would silently restore the exact hole this probe exists to close.
        # Only --grid, which was actually handed to the binary, can stand in.
        if declared is None:
            print(f"Cannot determine the kernel's batch size (probe: {probe_status}). Pass "
                  f"--grid A,B so there is at least a value the binary was told to use.",
                  file=sys.stderr)
            for ln in probe_tail[-6:]:
                print(f"    | {ln}", file=sys.stderr)
            sys.exit(1)
        B, b_source, b_verified = declared, f"UNVERIFIED -- probe {probe_status}; assumed from --grid", False
        print(f"WARNING: could not read 'Points batch size' from the binary (probe: {probe_status});",
              file=sys.stderr)
        print(f"         assuming B={B} from --grid. Every coverage claim below is only mod-B",
              file=sys.stderr)
        print("         correct if that is what the kernel actually ran.", file=sys.stderr)
        for ln in probe_tail[-6:]:
            print(f"         | {ln}", file=sys.stderr)

    if args.batch is not None and args.batch != B:
        print(f"MISMATCH: --batch {args.batch} but the kernel's batch is {B}. --batch is an "
              f"assertion, not an override -- a proof-side batch that disagrees with the kernel "
              f"leaves whole residue classes untested while still reporting PASS.", file=sys.stderr)
        sys.exit(1)

    # Mirror the kernel's own batch checks (CUDACyclone.cu:465 and :469).
    if B <= 0 or (B & 1) != 0 or (B & (B - 1)) != 0:
        print(f"Batch size must be even and a power of two; got {B}.", file=sys.stderr)
        sys.exit(1)
    if B > KERNEL_MAX_BATCH:
        print(f"Batch size {B} exceeds the kernel limit {KERNEL_MAX_BATCH}.", file=sys.stderr)
        sys.exit(1)
    if raw_len % B != 0:
        print(f"Range length {raw_len} is not divisible by the batch size {B}.", file=sys.stderr)
        sys.exit(1)

    total_size = end_i - start_i + 1
    if total_size <= 0:
        print("Empty range.", file=sys.stderr)
        sys.exit(1)

    start_A, start_B = gen_start_dual_parity(start_i, end_i, args.start_count)
    end_A, end_B = gen_end_dual_parity(start_i, end_i, args.end_count)

    used: Set[int] = set(start_A) | set(start_B) | set(end_A) | set(end_B)
    residues_block = full_mod_residue_cover(start_i, end_i, B, used)
    used.update(residues_block)
    quart_blocks = quartile_random_mod_coverage(start_i, end_i, B, used, args.quartile_count)

    tests: List[Tuple[str, int]] = []
    for v in start_A: tests.append(("Range start A (start+2k)", v))
    for v in start_B: tests.append(("Range start B (start+1+2k)", v))
    for v in end_A:   tests.append(("Range end A (end-2k)", v))
    for v in end_B:   tests.append(("Range end B (end-1-2k)", v))
    for v in residues_block: tests.append((f"Full mod {B} residue coverage", v))
    qlabels = ["Random Q1 (0–25%)", "Random Q2 (25–50%)", "Random Q3 (50–75%)", "Random Q4 (75–100%)"]
    for label, block in zip(qlabels, quart_blocks):
        for v in block:
            tests.append((label, v))

    # --- coverage invariant ------------------------------------------------------------------
    # Thread i's slice begins at range_start + i*per_thread_cnt and per_thread_cnt is a whole
    # multiple of B (asserted by the kernel itself at CUDACyclone.cu:566-569), so every batch
    # boundary is congruent to range_start mod B and a key's offset inside its batch is exactly
    # (key - range_start) mod B. Use the RAW range start, not the clamped start_i, so the frame
    # is the kernel's. (The `half` shift at CUDACyclone.cu:588 moves where the batch CENTER sits,
    # not where the batch boundaries fall, so it does not enter here.)
    # Assert this rather than trusting that the generators and B stayed in step: an incomplete
    # cover fails no individual test, it just quietly stops testing the offsets it dropped.
    covered = {(v - raw_start) % B for _, v in tests}
    missing = sorted(set(range(B)) - covered)
    if missing:
        print(f"COVERAGE BUG: {len(missing)} of {B} residue classes mod {B} are never tested "
              f"(first: {missing[:8]}). Refusing to report a pass on partial coverage.",
              file=sys.stderr)
        sys.exit(1)

    grid_display = args.grid_arg if args.grid_arg else "(none -- binary compiled default)"
    n_edge = len(start_A) + len(start_B) + len(end_A) + len(end_B)
    n_quart = sum(len(blk) for blk in quart_blocks)
    # The assertion above proves the test set covers every class mod *this* B. That is only a
    # claim about the KERNEL when B came from the kernel, so an unprobed run must not print
    # the word COMPLETE -- that is the false-completeness inference the probe exists to kill.
    cov_claim = (f"COMPLETE ({B}/{B} residue classes)" if b_verified else
                 f"complete mod {B}, but B is UNVERIFIED -- NOT a claim about the kernel")
    print("======== proof configuration =========================")
    print(f"  binary            : {args.cyclone_path}")
    print(f"  grid arg          : {grid_display}")
    print(f"  batch B           : {B}   [{b_source}]")
    print(f"  range             : {args.range_arg}   (length 2^{raw_len.bit_length() - 1})")
    print(f"  mod-{B} coverage   : {cov_claim}")
    print(f"  planned tests     : {len(tests)}  "
          f"= {n_edge} edge + {len(residues_block)} residue + {n_quart} quartile")
    if not residues_block:
        print(f"  note              : the dedicated mod-{B} block is empty -- at this B the edge "
              f"blocks already span all {B} classes")
    print("-------------------------------------------------------")

    stats: Dict[str, Dict[str, int]] = {}
    def bump(label: str, key: str) -> None:
        if label not in stats:
            stats[label] = {"total": 0, "success": 0, "fail": 0}
        stats[label][key] += 1

    out_fname = "cyclone_tests_results.txt"
    with open(out_fname, "w", encoding="utf-8") as ofs:
        ofs.write(
            f"Cyclone strong tests\n"
            f"Range: {args.range_arg}\n"
            f"Cyclone: {args.cyclone_path}\n"
            f"Grid: {grid_display}\n"
            f"Batch(B): {B}  [{b_source}]\n"
            f"Planned tests: {len(tests)}  (mod-{B} residue coverage: {cov_claim})\n"
            f"Date: {time.ctime()}\n\n"
        )
        total_success = 0
        total_fail    = 0

        for idx, (label, priv_int) in enumerate(tests, start=1):
            bump(label, "total")
            priv_hex = int_to_priv32_hex(priv_int)
            try:
                pub_comp = compressed_pubkey_from_priv32(bytes.fromhex(priv_hex))
                addr = p2pkh_from_pubkey_compressed(pub_comp)
            except Exception as ex:
                print(f"=== Test {idx}/{len(tests)} === [{label}]\npriv: {priv_hex}\naddress: (error)\nStatus: FAIL")
                ofs.write(f"{idx}, {label}, {priv_hex}, , ERROR_KEY\n")
                total_fail += 1
                bump(label, "fail")
                continue

            ofs.write(f"{idx}, {label}, {priv_hex}, {addr}, START\n")
            ofs.flush()

            found, found_priv = run_cyclone_and_watch(
                args.cyclone_path, args.range_arg, addr, args.grid_arg, timeout=args.timeout
            )

            if found:
                total_success += 1
                bump(label, "success")
                ofs.write(f"{idx}, {label}, {priv_hex}, {addr}, FOUND, {found_priv}\n")
                print(f"=== Test {idx}/{len(tests)} === [{label}]\npriv: {priv_hex}\naddress: {addr}\nStatus: PASS")
            else:
                total_fail += 1
                bump(label, "fail")
                ofs.write(f"{idx}, {label}, {priv_hex}, {addr}, NO_MATCH\n")
                print(f"=== Test {idx}/{len(tests)} === [{label}]\npriv: {priv_hex}\naddress: {addr}\nStatus: FAIL")

            ofs.flush()
            time.sleep(0.05)

        ofs.write("\nSummary by blocks:\n")
        print("\n================ Summary by blocks ================")
        ordered_labels = [
            "Range start A (start+2k)",
            "Range start B (start+1+2k)",
            "Range end A (end-2k)",
            "Range end B (end-1-2k)",
            f"Full mod {B} residue coverage",
            *qlabels
        ]
        for label in ordered_labels:
            s = stats.get(label, {"total": 0, "success": 0, "fail": 0})
            line = (f"{label:34s} : total={s['total']:4d}  success={s['success']:4d}  fail={s['fail']:4d}")
            ofs.write(line + "\n")
            print(line)

        ofs.write("\nOverall:\n")
        ofs.write(f"Total tests: {len(tests)}\nSuccesses: {total_success}\nFailures: {total_fail}\n")

    print(f"\nDone. Results in {out_fname}. Successes={total_success} Failures={total_fail}")

if __name__ == "__main__":
    main()
