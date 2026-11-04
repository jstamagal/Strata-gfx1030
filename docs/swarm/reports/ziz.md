# ziz — lateral report: the roads nobody brought (2026-11-04 ~15:40)

Seat: lateral. Read BRIEF, ledger (QWEN38.md + SUMMARY.md), qsa.cuh, gdn_core.cuh, mtp.cpp:2085-2200,
model.hpp:20-49, MTP microbench. No GPU touched, no build, no config. **V** = verified in source/data this
session; **I** = inference; **G** = gut.

## Framing fact the whole troop can use (V, arithmetic)

With IQ3_S fully resident (post KV_PF_SLOTS=1), the old verdicts invert:

- A 36K prompt at 2700 tok/s = 1517 ms per 4096-token chunk per card. Expert **weights** read once per chunk
  ≈ 24 GB/48 layers × ~24 layers/card ≈ 12.2 GiB → **24 ms at 512 GB/s ≈ 1.6% of the chunk** — that is a LOWER
  bound on expert weight-time (peak-BW one-read; gruff #99 right). Expert GEMM share is **unmeasured**: the mtpq
  microbench prices the DRAFTER's expert kernels, not the main model's prompt path (moe_mmq.cu / moe_ws.cu, group
  shapes ~80 rows/expert). Extrapolating peak-flop gives 21-33%; if the main prompt kernels run at the drafter
  bench's per-row cost instead, experts could be well over half the chunk. **Corrections (gruff #99):** my
  "~2/3 is not experts" was overclaimed from a peak-BW + peak-FLOP model — the honest statement is: expert
  WEIGHT time is small, expert COMPUTE time unknown, and the bucket run decides before anyone builds for either
  side. The 4000 tok/s target needs that one window first either way.
- Agent verify cycle ≈ 6.7 ms wall (149 tok/s; 15.5 ms pre-KV_PF_SLOTS was the old geometry). Expert weight
  READS for 8 verify rows ≥ ~0.2 ms of it (lower bound, peak-BW one-read) — expert weight-read time is small,
  expert compute share unmeasured, same caveat as the prompt line above.

So both headline targets deserve one attribution window each before kernel work. My remaining roads:

## Road 1 (cheapest, most certain): stop writing GDN rollback snapshots nobody reads

- **Claim (V source, I price):** every multi-row GDN window writes an fp32 state snapshot after each row except
  the last: `snap [MAXT-1][nv][HS][HS]` (gdn_core.cuh:11), written unconditionally at gdn_core.cuh:170-176
  (`if (t < T-1) { sp[...] = s[r]; }`) — for T=8: 7 × 3.145 MB × 24 GDN layers ≈ **528 MB written per verify
  window** (16 KB × 48 v-heads × HS×HS is 3.145 MB per layer-snapshot; ×24 layers × 7 rows). At the measured
  46 windows/s that is ~24 GB/s of writes — a WORST-CASE ceiling (assumes every window is T=8; real windows mix
  widths and window 9 is old-UD4 geometry), not an operating estimate (gruff #108). Source proves the writes
  happen; only the attribution window can say what fraction of the critical path they are.
- The waste: window-9 traces measured **46 windows/s** on the pipelined MTP agent arm (2914 windows over ~63 s,
  accounting for the trace's ~4% cost) — 528 MB × 46/s ≈ **24 GB/s of pure write traffic IF every window were
  T=8** (ceiling; see above — gruff #108). Mean acceptance 1.48 drafts/window on agent traffic (ledger window 9),
  so rollback reads touch snap[0]/snap[1] almost always and snap[2..6] ~rarely; on full accept (64.9% of windows,
  97% on edits) **zero** snapshots are read.
- **Fix (three options, cheapest first):** (a) env-gate snapshot stride — snapshot only rows 0..1 plus final S
  (already in S), recompute longer rollbacks by re-running rows (recompute cost ≈ one extra window on the ~35%
  rejected-windows tail); (b) scratch-S + commit-copy: run the window into a scratch state, copy final S
  (3.145 MB/layer ≈ 75 MB/window) on accept — replaces 528 MB of snapshot writes with 150 MB of copy, exact same
  text; (c) skip snapshots entirely and always recompute on reject — simple, only wins when acceptance is high
  (edits). Zero text change in all three (rollback state is bit-identical: same arithmetic re-run or a copy).
- **Build cost:** ~30-80 lines, one kernel arg removed / one buffer swap; no graph topology change needed for (c).
- **Decisive measurement:** frozen-placement bench (adapt-swaps moot on IQ3_S — kahuzi #89.3: adaptive swaps are
  already disabled when all_experts_resident), agent + edits arms, same-md5 gate. **Correctness note (gruff
  #99.4, right):** the acceptable outcomes are same-text-faster or text-preserving-with-replay — any arm that
  changes output is a bug, not a speed result. Option (c) skip+recompute must re-run the row's inputs exactly
  (bit-identical recomputation), not skip state. If cycle time doesn't move ≥3% on edits (the workload with the
  most full-accept windows), my write-bandwidth assumption was wrong — drop it.
- **Failure modes:** snapshots may sit in L2 and cost less than the model (the V620's 4 MB L2 makes 22 MB/layer
  writes mostly spill — which favors the fix); hidden reader besides rollback (searched gl.snap consumers:
  only gdn_core.cuh:38-40 wiring — V); graphs baked against the current arg layout (would need re-capture).

## Road 2 — RETRACTED after gruff's #99 check. Correct picture: the pipelined live path never reaches the serial
draft() gate (mtp.cpp:2085). `chain_launch` (mtp.cpp:2262) takes no min_p, launches ALL n_out step graphs upfront
(mtp.cpp:2299-2302), and the min_p test happens on the host over already-computed probs (generate.cpp `t_rule`,
:17921-17933). No per-step round-trip exists to remove. The real seam in this area is window row-budgeting:
`t_rule` hard-truncates a window at the first chain prob < minp; calibrated survival (kaje) and the untested
STRATA_PL_FRESH_T/_MIN_P, STRATA_PL_GUESS_T/_MIN_P switches (BRIEF:119) are the honest road — kaje/titus ground,
not mine. Lesson recorded: serial-path code is not the pipelined path; check the call site first.

## Road 3 (prompt 2700→4000): re-attribute the chunk before optimizing anything

- **Claim (V arithmetic, above):** expert weights are ~2% of a resident prompt chunk. Every expert-kernel speedup
  (rdna2-mmq, IQ4_NL v_perm, etc.) is now polishing <1/3 of the chunk. The unowned suspects: QSA prompt
  attention (`k_idx_score_tc`, qsa.cuh:162-211 — one f16 GEMM per 64 blocks per 16 rows: at 36K that's
  29490 blocks / 64 × 9216 rows/16 ≈ 268k tile-pairs per QSA layer... I did NOT finish this estimate; G for the
  size, but it's 48×QSA_LAYERS launches per chunk), GDN prompt recurrence (gdn_prompt.cuh, 448 lines, unreads),
  router GEMMs (24 layers × 4096 rows × 2560×512), dense FF layers, and the handoff between cards
  (bakari: ideal link-overlap ≈ 0.7% of 36K — so handoff is NOT it on paper, also G).
- **Decisive measurement (one window, zero code):** `STRATA_PREFILL_TIMING=1` (kahuzi verified the switch
  exists, prefill.cpp:5977-5995 adjacent) + the existing pipeline trace on a 36K arm, bucket the 1517 ms into
  {expert, attn, gdn, router+dense, host, idle}. Whatever bucket is >30% becomes the only prompt road worth
  building for. Budget: one window, ~5 GPU-min.
- **Failure mode:** the timing switch perturbs (profiler-disable risk, kivu #48). Cross-check one bucket against
  kernel-level rocBLAS/hipEvent traces before believing it.

## Things I found wrong / stale

1. **Ledger "prompt expert-streaming-bound ≤0.5%" (QWEN38.md:86) is stale post-residency** — its own caveat says
   so ("That held before full residency"), but it's still quoted in BRIEF §2 ("Earlier finding, now in question")
   without the arithmetic that expert weights are now ~2% of a chunk. Killed idea: any further prompt-expert
   kernel work as a *prompt* lever.
2. **`--mtp-max-t 4` death (-17%, ledger:20) was measured pre-dispatcher-fix** (5fbc5fff6/27d713fbc merged today
   14:47). The wide-MTP fallback was silently running the 8-column kernel for 9-16 columns — corrupted output.
   The -17% number predates the fix; if the wide path was wrong, its cost was measured with garbage. Re-test
   cheap before anyone cites it again. (I: the bug corrupted >8-row windows, and mtp-max-t 4 shouldn't hit it —
   but the *cost* side of that ledger row came from the same binary family. G on magnitude.)
3. **Kahuzi's #89.3 changes the meaning of several adopted wins:** adaptive swaps are disabled on IQ3_S
   (generate.cpp:12278), so the UD4 "+11% adaptive swapping" line (QWEN38.md:33) does NOT transfer to the daily
   driver. Any ledger conclusion that leaned on adaptive placement being active on IQ3_S needs re-reading.

## Verified vs inferred summary

- V: snapshot write site and size (gdn_core.cuh:11,170); the serial draft() gate branch (mtp.cpp:2085-2200) —
  now known NOT to be on the pipelined live path (road 2 retracted); chain_launch launches all steps, no min_p
  (mtp.cpp:2262,2299-2302); t_rule truncation site (generate.cpp:17921-17933); QSA dims top-2048
  (model.hpp:31-49); microbench per-row costs (mtpq microbench2, DRAFTER experts only); window-9 trace rates
  (46 windows/s pipelined MTP agent); dispatcher-fix merge time; kahuzi's residency/adapt facts (generate.cpp:12278).
- I: the 24 GB/s snapshot-write CEILING (assumes all-T=8 windows; the one attribution window replaces it with a
  real number) and the 1.6% expert-weight floor (byte arithmetic at peak bandwidth = lower bounds, not
  operating-point measurements — both carry a one-window kill).
- G: relative sizes of the un-attributed prompt buckets (that's WHY road 3 is measure-first).
