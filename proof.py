#!/usr/bin/env python3
import argparse
import subprocess
import secrets
import hashlib
import time
import sys
import math
import random
import select
import struct
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
    parser.add_argument("--grid", dest="grid_arg", default="512,512",
                        help="Value for --grid passed to CUDACyclone (e.g. 512,512)")
    parser.add_argument("--batch", dest="batch", type=int, default=None,
                        help="Override batch size (first number of --grid). Must be even.")
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
        start_i, end_i = parse_hex_range(args.range_arg)
    except Exception as ex:
        print("Range parse error:", ex, file=sys.stderr)
        sys.exit(1)

    if start_i == 0:
        start_i = 1
    order = SECP256k1.order
    if end_i >= order:
        end_i = order - 1
        if start_i > end_i:
            print("Range shrunk below valid curve order.", file=sys.stderr)
            sys.exit(1)

    B = args.batch if args.batch else parse_batch_from_grid(args.grid_arg)
    if not B:
        print("Cannot determine batch size. Use --grid A,B or --batch.", file=sys.stderr)
        sys.exit(1)
    if (B & 1) != 0:
        print("Batch size must be even.", file=sys.stderr)
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
            f"Grid: {args.grid_arg}\n"
            f"Batch(B): {B}\n"
            f"Date: {time.ctime()}\n\n"
        )
        total_success = 0
        total_fail    = 0

        print(f"Planned tests: {len(tests)}")
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
