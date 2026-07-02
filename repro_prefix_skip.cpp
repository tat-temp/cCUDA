// repro_prefix_skip.cpp
// ---------------------------------------------------------------------------
// Deterministic host-side reproduction of the key-skipping bug in
// CUDACyclone's kernel_point_add_and_check_oneinv (CUDACyclone.cu).
//
// THE BUG (CUDACyclone.cu:135-149, 214-233, 265-283, 326-343):
//   Every candidate is first screened by its 32-bit hash160 prefix word. On
//   ANY prefix match across the warp, the kernel does:
//       if (__any_sync(full_mask, pref)) { ...maybe report...; return; }
//   i.e. the WHOLE warp returns -- BEFORE the end-of-kernel writes of
//   Rx/Ry/counts256/start_scalars and the atomicAdd(d_any_left,1)
//   (CUDACyclone.cu:384-393). A prefix match is true for the real target AND
//   for a ~1-in-2^32 unrelated key ("false prefix collision"). On a FALSE
//   collision no full match is published (found_flag stays FOUND_NONE), yet the
//   warp still bailed without persisting progress. Because the host ping-pongs
//   the point buffers (std::swap(d_Px,d_Rx), CUDACyclone.cu:817) and carries
//   scalars/counts in place, the thread's point desyncs from its scalar on the
//   next launch: it re-reads a stale point, rescans old keys, and a contiguous
//   tail of its assigned sub-range is NEVER scanned -- while the run can still
//   print "KEY NOT FOUND (exhaustive)".
//
// WHY A CPU MODEL IS FAITHFUL:
//   The defect is pure control flow (early return + buffer swap), independent of
//   the secp256k1 / SHA-256 / RIPEMD-160 math. We model:
//     * "point of scalar k" == k. The EC scalar->point map is a bijection, so
//       "which keys get hashed" is exactly "which center scalars get processed".
//       Coverage is what the bug corrupts; the arithmetic is irrelevant to it.
//     * prefixMatch()/fullMatch() are hardcoded fakes: DECOY is an in-range,
//       non-target scalar that prefix-collides with TARGET (same first word)
//       but is not a full match -- precisely the mishandled event.
//   Everything else (batch geometry, +/-G symmetry ordering, per-thread center
//   advance by B, the host relaunch loop, the two point buffers, in-place
//   scalars/counts, d_any_left, the rem==0 fast path) mirrors the real code.
//
//   The FIX toggle (cfg.buggy=false) models the proposed correction: abort the
//   warp only when a full match was actually published, never on a bare prefix.
//
// Build:  g++ -std=c++17 -O2 -o repro_prefix_skip repro_prefix_skip.cpp
// Run:    ./repro_prefix_skip           (exit 0 = bug reproduced as expected)
//
// SCOPE / FIDELITY (independently audited against the source, line by line):
//   * Load-bearing semantics all match: single seeding pass (scalarMulKernelBase
//     runs once at CUDACyclone.cu:668, never re-run in the loop), only d_Px/d_Rx
//     swapped (:817-818) while d_start_scalars/d_counts256 are in place (:765,388),
//     d_any_left zeroed per launch (:761) and incremented only at end-of-kernel
//     (:392), and all four abort sites return before the stores (:148/232/282/343).
//   * The bug has TWO independent manifestations, both reproduced here:
//       (a) multi-launch: after the swap the aborted thread reads a stale point
//           desynced from its in-place scalar and abandons a sub-range tail;
//       (b) single-launch: the aborted warp never bumps d_any_left, so if the rest
//           of the grid has drained, the host declares "exhaustive" with the tail
//           unscanned (Scenario 1 shows exactly this).
//   * This model is CONSERVATIVE. It reads one rem per warp (the kernel reads one
//     per thread) -- harmless under uniform seeding -- and it does NOT model the
//     real kernel's __any_sync/__syncwarp(0xFFFFFFFF) executing after lanes could
//     diverge on rem, which is additional undefined behavior. Real hardware can
//     therefore be worse than this repro, never better.
// ---------------------------------------------------------------------------
#include <cstdint>
#include <cstdio>
#include <vector>
#include <set>
#include <string>
#include <algorithm>

typedef uint64_t u64;
static const u64 SENTINEL  = 0xDEAD000000000000ULL; // stands for uninitialized d_Rx
static const u64 NO_TARGET = 0xFFFFFFFFFFFFFFFFULL; // "no reachable target" (search exhausts)

struct Config {
    u64      range_start;
    u64      range_len;     // power of two
    uint32_t B;             // batch size (even power of two)
    u64      threadsTotal;  // divides range_len; per-thread count is a multiple of B
    uint32_t warp;          // warp-wide abort granularity (real HW: 32)
    uint32_t slices;        // max_batches_per_launch (--slices)
    u64      target;        // scalar whose full hash matches (NO_TARGET => unreachable)
    bool     has_decoy;
    u64      decoy;         // non-target scalar that collides on the prefix word
    bool     buggy;         // true = ships today; false = proposed fix
};

struct RunResult {
    bool     found = false;
    u64      found_scalar = 0;
    u64      launches = 0;
    u64      scan_events = 0;       // total (candidate) hashes performed
    std::set<u64> scanned;         // distinct candidate scalars hashed
    u64      in_range_covered = 0;
    u64      missing_count = 0;
    std::vector<std::pair<u64,u64>> missing; // uncovered intervals within the range
    bool     target_scanned = false;
};

// Candidate ordering inside one batch, matching the kernel exactly:
//   idx 0            -> center                 (CUDACyclone.cu:122-150)
//   idx 2i+1         -> center + (i+1)  [P+iG]  (loop, first sub-block  :186)
//   idx 2i+2         -> center - (i+1)  [P-iG]  (loop, second sub-block :236)
//   idx B-1          -> center - half   [P-half*G, final block          :293)
// Offsets span exactly [-half, +half-1] => B consecutive scalars, no gaps/dups.
static inline int64_t offset_of(uint32_t idx, uint32_t half, uint32_t B) {
    if (idx == 0)     return 0;
    if (idx == B - 1) return -(int64_t)half;
    if (idx & 1u)  { uint32_t i = (idx - 1) / 2; return  (int64_t)(i + 1); }
    else           { uint32_t i = (idx - 2) / 2; return -(int64_t)(i + 1); }
}

static RunResult run_search(const Config& cfg) {
    RunResult R;
    const uint32_t B    = cfg.B;
    const uint32_t half = B / 2;
    const u64 T         = cfg.threadsTotal;
    const u64 ptc       = cfg.range_len / T;              // per_thread_cnt
    const u64 range_end = cfg.range_start + cfg.range_len - 1;

    // Host-persistent state, mirroring CUDACyclone main():
    //   buf0/buf1 = the two device point buffers (d_Px / d_Rx), ping-ponged.
    //   Sarr      = d_start_scalars (updated in place).
    //   remarr    = d_counts256     (updated in place).
    std::vector<u64> buf0(T), buf1(T), Sarr(T), remarr(T);
    for (u64 t = 0; t < T; ++t) {
        u64 pt0   = cfg.range_start + t * ptc + half;     // center scalar (== its point here)
        buf0[t]   = pt0;        // d_Px seeded by scalarMulKernelBase(start_scalars)
        buf1[t]   = SENTINEL;   // d_Rx is not initialized until launch 1 writes it
        Sarr[t]   = pt0;
        remarr[t] = ptc;
    }
    std::vector<u64>* bufs[2] = { &buf0, &buf1 };
    int cur = 0; // bufs[cur] = d_Px (input); bufs[1-cur] = d_Rx (output)

    auto prefixMatch = [&](u64 c) -> bool {
        if (c == cfg.target) return true;
        if (cfg.has_decoy && c == cfg.decoy) return true;
        return false;
    };
    auto fullMatch = [&](u64 c) -> bool { return c == cfg.target; };

    const u64 LAUNCH_CAP = cfg.range_len + 16; // safety net against a modeling mistake

    while (true) {
        std::vector<u64>& in  = *bufs[cur];
        std::vector<u64>& out = *bufs[1 - cur];
        u64 any_left = 0;

        // ---------------- one kernel launch ----------------
        for (u64 w0 = 0; w0 < T; w0 += cfg.warp) {
            u64 w1 = std::min<u64>(w0 + cfg.warp, T);
            std::vector<u64> pt, S, gid;
            for (u64 t = w0; t < w1; ++t) { pt.push_back(in[t]); S.push_back(Sarr[t]); gid.push_back(t); }
            u64 rem = remarr[w0];  // rem is homogeneous within a warp (invariant, held even after desync)

            if (R.found) continue;                         // warp_found_ready -> return (:120)
            if (rem == 0) {                                // rem==0 fast path (:111-115)
                for (size_t l = 0; l < gid.size(); ++l) out[gid[l]] = pt[l];
                continue;
            }

            bool aborted = false;
            uint32_t batches = 0;
            while (batches < cfg.slices && rem >= B && !aborted) {
                for (uint32_t idx = 0; idx < B; ++idx) {
                    bool warpPrefix = false;
                    for (size_t l = 0; l < gid.size(); ++l) {
                        u64 cand = pt[l] + (u64)offset_of(idx, half, B);
                        R.scanned.insert(cand);
                        R.scan_events++;
                        if (prefixMatch(cand)) {
                            if (fullMatch(cand) && !R.found) {           // atomicCAS publish (:137)
                                R.found = true;
                                R.found_scalar = S[l] + (u64)offset_of(idx, half, B);
                            }
                            if (cfg.buggy) warpPrefix = true;            // BUG: bail on any prefix
                        }
                    }
                    // A real full match legitimately stops the search in both variants.
                    if (R.found) warpPrefix = true;
                    if (warpPrefix) { aborted = true; break; }          // __any_sync -> return (:148)
                }
                if (aborted) break;
                for (size_t l = 0; l < gid.size(); ++l) { pt[l] += B; S[l] += B; }
                rem -= B; ++batches;
                if (R.found) break;
            }

            if (aborted) continue;   // faithful bug: returned before the stores => NO write-back
            for (size_t l = 0; l < gid.size(); ++l) {                   // stores at :384-390
                out[gid[l]]   = pt[l];
                Sarr[gid[l]]  = S[l];
                remarr[gid[l]] = rem;
            }
            if (rem > 0) any_left += (u64)gid.size();                   // atomicAdd(d_any_left) :392
        }
        // -------------- end kernel launch --------------

        ++R.launches;
        if (R.found) break;                 // host stops on FOUND_READY, before the swap (:812)
        cur = 1 - cur;                      // std::swap(d_Px, d_Rx) (:817-818)
        if (any_left == 0) break;           // completed_all (:820)
        if (R.launches > LAUNCH_CAP) { std::printf("  [!] launch cap hit -- model error\n"); break; }
    }

    // Coverage over the assigned range.
    R.target_scanned = (cfg.target != NO_TARGET) && (R.scanned.count(cfg.target) != 0);
    bool inGap = false; u64 gapStart = 0;
    for (u64 v = cfg.range_start; v <= range_end; ++v) {
        bool hit = R.scanned.count(v) != 0;
        if (hit) { ++R.in_range_covered; if (inGap) { R.missing.push_back({gapStart, v - 1}); inGap = false; } }
        else     { ++R.missing_count; if (!inGap) { inGap = true; gapStart = v; } }
    }
    if (inGap) R.missing.push_back({gapStart, range_end});
    return R;
}

// ---------------------------------------------------------------------------
static int g_fail = 0;
static void check(const char* name, bool cond) {
    std::printf("    [%s] %s\n", cond ? "PASS" : "FAIL", name);
    if (!cond) g_fail = 1;
}
static void print_run(const char* label, const Config& c, const RunResult& r) {
    std::printf("  %s\n", label);
    std::printf("    launches=%llu  scanned(distinct)=%zu  scan_events=%llu  covered=%llu/%llu  missing=%llu\n",
        (unsigned long long)r.launches, r.scanned.size(),
        (unsigned long long)r.scan_events,
        (unsigned long long)r.in_range_covered, (unsigned long long)c.range_len,
        (unsigned long long)r.missing_count);
    if (!r.missing.empty()) {
        std::printf("    missing intervals:");
        for (auto& iv : r.missing) std::printf(" [%llu..%llu]", (unsigned long long)iv.first, (unsigned long long)iv.second);
        std::printf("\n");
    }
    std::printf("    found=%s", r.found ? "true" : "false");
    if (r.found) std::printf(" (scalar=%llu)", (unsigned long long)r.found_scalar);
    if (c.target != NO_TARGET) std::printf("  target=%llu target_scanned=%s",
        (unsigned long long)c.target, r.target_scanned ? "true" : "false");
    std::printf("\n");
}

int main() {
    // Shared small geometry. B=8 (half=4), 64 keys/thread, slices=2 => several launches
    // per thread so the buffer-swap desync has room to manifest.
    const u64      RS   = 0x1000;   // range_start = 4096
    const uint32_t B    = 8;
    const uint32_t half = B / 2;

    std::printf("================ CUDACyclone prefix-collision key-skip repro ================\n\n");

    // Single-thread configs: thread 3 of 8 (range_len=512), decoy fires in a later
    // launch so the stale-buffer path (not just launch-1) is exercised.
    const u64 T1   = 8, RLEN1 = 512, PTC1 = RLEN1 / T1; // 64 keys/thread
    const u64 pt0_3   = RS + 3 * PTC1 + half;           // thread 3 center (4292)
    const u64 T3_LO   = RS + 3 * PTC1;                  // thread 3 sub-range low  (4288)
    const u64 T3_HI   = RS + 4 * PTC1 - 1;              // thread 3 sub-range high (4351)
    const u64 DECOY   = pt0_3 + 16;                     // false prefix collision at a batch center (4308)
    // With slices=2 the abort desyncs thread 3's point buffer back to its ORIGINAL start point,
    // so it re-hits the same collision center every other relaunch and never advances past it.
    // Everything from the collision onward is abandoned -- [4304..4351] except the point 4308
    // itself -- i.e. 47 keys, until the rest of the grid drains d_any_left and the host stops.
    const u64 GAP_TARGET   = pt0_3 + 48;               // a real key stranded in that dead zone (4340)
    const u64 EARLY_TARGET = pt0_3 + 8;                // a key scanned BEFORE the decoy (4300)

    auto base = [&](void) {
        Config c{}; c.range_start=RS; c.range_len=RLEN1; c.B=B; c.threadsTotal=T1;
        c.warp=1; c.slices=2; c.target=NO_TARGET; c.has_decoy=false; c.decoy=0; c.buggy=true;
        return c;
    };

    std::printf("--- Scenario 0: control, no collision (model sanity == proof.py's 848/848) ---\n");
    { Config c = base(); RunResult r = run_search(c); print_run("buggy build, no decoy, unreachable target", c, r);
      check("full coverage, no gaps", r.missing_count == 0);
      check("every key scanned exactly once", r.scan_events == RLEN1 && r.scanned.size() == RLEN1);
      std::printf("\n");
    }

    std::printf("--- Scenario 1: BUG -- one false prefix collision opens a coverage gap ---\n");
    Config bugCov = base(); bugCov.has_decoy=true; bugCov.decoy=DECOY;               // target unreachable => search exhausts
    RunResult rBugCov = run_search(bugCov); print_run("buggy build, decoy present, exhaustive search", bugCov, rBugCov);
    check("a coverage gap now exists", rBugCov.missing_count > 0);
    { bool all_in_t3 = true;
      for (auto& iv : rBugCov.missing) if (iv.first < T3_LO || iv.second > T3_HI) all_in_t3 = false;
      check("every skipped key belongs to thread 3's sub-range [4288..4351]", all_in_t3); }
    check("thread 3's final key (4351) is abandoned", rBugCov.scanned.count(T3_HI) == 0);
    check("the collision point itself (4308) WAS scanned (it's what wedges the thread)",
          rBugCov.scanned.count(DECOY) != 0);
    check("one false collision strands 47 keys", rBugCov.missing_count == 47);
    check("host still finished 'exhaustively' (no crash/hang)", rBugCov.launches <= RLEN1);
    std::printf("\n");

    std::printf("--- Scenario 2: FIX -- same collision, coverage restored ---\n");
    Config fixCov = bugCov; fixCov.buggy=false;
    RunResult rFixCov = run_search(fixCov); print_run("fixed build, decoy present, exhaustive search", fixCov, rFixCov);
    check("full coverage restored, no gaps", rFixCov.missing_count == 0);
    std::printf("\n");

    std::printf("--- Scenario 3: real impact -- the private key lands in the gap ---\n");
    Config bugFind = base(); bugFind.has_decoy=true; bugFind.decoy=DECOY; bugFind.target=GAP_TARGET;
    RunResult rBugFind = run_search(bugFind); print_run("buggy build, target in the skipped tail", bugFind, rBugFind);
    check("buggy build MISSES the key (never scanned)", !rBugFind.found && !rBugFind.target_scanned);

    Config fixFind = bugFind; fixFind.buggy=false;
    RunResult rFixFind = run_search(fixFind); print_run("fixed build, same target", fixFind, rFixFind);
    check("fixed build FINDS the key", rFixFind.found && rFixFind.found_scalar == GAP_TARGET);
    std::printf("\n");

    std::printf("--- Scenario 4: why it hides -- target scanned before the collision => found ---\n");
    Config mask = base(); mask.has_decoy=true; mask.decoy=DECOY; mask.target=EARLY_TARGET;
    RunResult rMask = run_search(mask); print_run("buggy build, target reached before decoy", mask, rMask);
    check("bug is invisible when the key is found early (as in proof.py)", rMask.found);
    std::printf("\n");

    std::printf("--- Scenario 5: warp amplification -- lane 0's collision skips lane 7's key ---\n");
    const u64 T2 = 64, RLEN2 = 4096, PTC2 = RLEN2 / T2;   // 64 keys/thread, 64 threads
    const u64 pt0_0 = RS + 0 * PTC2 + half;
    const u64 pt0_7 = RS + 7 * PTC2 + half;
    const u64 DECOY0 = pt0_0 + 16;        // false collision in lane 0
    const u64 TGT7   = pt0_7 + 48;        // real key owned by lane 7 (same 32-lane warp)
    auto warpCfg = [&](uint32_t warp) {
        Config c{}; c.range_start=RS; c.range_len=RLEN2; c.B=B; c.threadsTotal=T2;
        c.warp=warp; c.slices=2; c.target=TGT7; c.has_decoy=true; c.decoy=DECOY0; c.buggy=true;
        return c;
    };
    Config w32 = warpCfg(32); RunResult rW32 = run_search(w32);
    print_run("warp=32 (real HW): collision in lane 0, key in lane 7", w32, rW32);
    check("32-wide warp drags lane 7 down -> key MISSED", !rW32.found && !rW32.target_scanned);

    Config w1 = warpCfg(1); RunResult rW1 = run_search(w1);
    print_run("warp=1 (no coupling): same decoy & target", w1, rW1);
    check("with no warp coupling lane 7 is unaffected -> key FOUND", rW1.found && rW1.found_scalar == TGT7);
    std::printf("\n");

    std::printf("============================================================================\n");
    if (g_fail) { std::printf("RESULT: UNEXPECTED -- some checks failed (see FAIL lines above).\n"); return 1; }
    std::printf("RESULT: BUG REPRODUCED. A single 32-bit prefix collision silently drops a\n");
    std::printf("        contiguous block of keys; if the real key is in that block the search\n");
    std::printf("        reports exhaustion without finding it. The fix toggle restores coverage.\n");
    return 0;
}
