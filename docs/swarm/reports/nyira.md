# nyira (memory lens): what happened the last ten times, and the human cost

Written 2026-11-04 15:45 for the coordinator. Read-only; nothing run on GPUs. V = verified in source/data/web today, I = inference, G = gut.

## Top 3, ranked by expected gain per GPU-minute

### 1. The board's metric is hidden on purpose (steer #74, 15:35): the public LocalMaxxing rules below are a PRECEDENT for what such boards measure, not the target. Optimise all three numbers, as the brief says. Keep this section as the cheapest sanity run: the two public canonical prompts cost ~10 GPU-min and show where decode sits on prose vs code.

*tk-tk-tk.* nyira wrong on "knowable today". Retracted. What stays: every public board of this kind ranks single-stream output tok/s at temperature 0 with spec decoding allowed and counted; prompt speed is a side column. Inference: a private board run by the same community most likely inherits that shape. Assumed, not fact.

**Evidence (V, fetched 15:36).**
- LocalMaxxing Verified rule (github.com/Datura-ai/lium-localmaxxing README, quoting the site): *canonical prompt (reasoning-v1, sha256 9000edaa…3445; a second canonical "code" prompt exists), batch 1 and concurrency 1, >= 256 output tokens, temperature 0, speculative runs allowed but must report draft/accepted counts, engine timings attached, result below the card's theoretical decode ceiling x 1.15.* Boards rank by **output tok/s**. Prompt speed appears only as TTFT and as separate "verified-at-depth" rows (16k/65k/131k, cold prefill).
- Model page title today: "Qwen3.8-Flash-Next-GGUF (180B) — **142.0 tok/s top speed**" (all hardware).
- Closest Strata precedent on the board: @anh, Oct 13, Strata v0.1.43.1 on 5080+5060 Ti (2x16 GB), UD-IQ4_XS, expert streaming: **reasoning 105.7 / code 107.6 tok/s**, TTFT 649/589 ms; at depth 101.4/100.8/94.3 with cold prefill 3.2-3.7k tok/s.
- Identity: the brief (line 10) says "localmaxxxing Discord leaderboard, dual gfx103x category"; localmaxxing.com is the only board of that name, Strata rows are already on it, Qwen3.8-Flash-Next is its 2nd most-tested model (526 runs). **ASSUMED** it is the same board; a Discord-only contest with its own harness would change this. The rule text above is from the Datura README quoting the site (secondary); the primary site pages fetched: get-started (effective-bandwidth rankings "only count single-stream runs without speculative decoding"), api-docs (speculativeDecoding / speculativeMethod / draftModel / speculativeTokens fields), the model page title, the multi-gpu reports page.
- Ledger: IQ3_S frozen essay (prose, 1200 tokens) 98.5 tok/s with proxy; agent code/JSON 149; gen after 36K prompt ~100.

**Mechanism / gain (I).** The agent and edit harnesses are KING's daily driver, not the board. The board's reasoning prompt is prose-like (ledger: prose acceptance ~67%, agent 84% at T=3). So the leaderboard number this box can post today is probably ~100 (reasoning) and ~140+ (code), not 149/372. The 142.0 site-top is within reach on the code prompt already; the reasoning prompt is where the gap is. Every GPU-minute spent on 36K prompt tok/s buys zero leaderboard rank; it buys daily-driver comfort (legitimate, but a different ask — say which one each window serves).

**Decisive measurement.** One window: the two canonical prompts (fetch `configs/` from that repo; the lmx CLI runs `--max-tokens 512 --temperature 0 --warmup 2 --iterations 3`) through the live IQ3_S config, with thinking on and off. Report tok/s out, drafts/accepted, TTFT. That number, not the agent number, is the target to move.

**How it fails.** The dual-gfx103x filter may be empty (no competitor) and the honest goal collapses to "beat 142.0 overall" or "beat the 5080+5060 Ti Strata row". Also the x1.15 decode-ceiling rule with spec decoding: the site's calculator may cap what it will verify; check before submitting (I could not render the filtered board without JS).

### 2. The stall tail is a candidate decode win, and it matches an old pattern: a memory-starved host with a single control thread. Causation is NOT established; one correlated trace decides it (cost: 0 build for the first step)

**Evidence (V).**
- Census (`qwen38-window267/stall-census.txt`): 25 gaps of 0.3-4.2 s in 34 traced arms, quiet and dirty, both binaries, both cards. One 2381 ms gap alone turned an edit case 331 -> 198 tok/s (window L16C "wide_off" row).
- `/proc/vmstat` now: allocstall_movable 397986, allocstall_normal 36566, compact_stall 171872 (compact_fail 111755), thp_fault_fallback 700562. THP `[always]`, defrag `defer+madvise`. `free -g`: 3 GB free, 36 GB cache, **3.3 GB in swap** (/mnt/nvme0/swapfile), vm.min_free_kbytes 524288 (0.5 GB of 92 GB).
- dmesg (uptime 245911 s, ~2.8 days ago): `strata: page allocation stall for 10 secs: order:0, mode:GFP_HIGHUSER_MOVABLE|__GFP_ZERO|__GFP_COMP` — **the engine process itself** blocked 10 s in direct reclaim on an ordinary zeroed anonymous page fault. Three more such stalls from `sh` and `hf-xet` (downloads) today at 11:41 and 13:29.
- `src/core/pinned.cu:457-510`: the engine already carries scar tissue from this exact disease: comment at :472 "MADV_HUGEPAGE only adds direct reclaim and compaction on every fault", :495 "#2126: THP madvise + defrag madvise makes every fault ... reclaim and compact". It happened before, it was patched around the arena, and the control thread's other allocations were left exposed.
- `journalctl -k`: `amdgpu 0000:07:00.0: Runlist is getting oversubscribed due to too many queues. Expect reduced ROCm performance.` on **card 1** at 09:53:33 today and 3x on Oct 30. The Q4_0 configs pin `GPU_MAX_HW_QUEUES=4`; whether the IQ3_S live config does, check. Oversubscribed HWS runlists time-slice queues: a late window on card 1 is exactly what that produces.

**Precedent (V as history, I as applicability).** Redis, MongoDB, PostgreSQL, ScyllaDB, Couchbase all ship the same advice after the same symptom (single-threaded control loop, periodic multi-second stalls, worse under IO/page-cache churn): THP to madvise/never, lock the hot memory, keep the control thread's pages out of reclaim, raise the free-page watermark on big-RAM boxes. The last ten teams who chased this as a GPU or kernel-driver bug found reclaim/compaction in the off-CPU stack.

**Mechanism / gain (I, hypothesis — kivu #70 computed that deleting every listed gap caps the gain at 0.8-7.9% in four ordinary arms, not 149->200).** Host service thread faults a page (fresh anonymous, or one of the 3.3 GB that the kernel swapped out while the PLE/page cache grew) -> direct reclaim/compaction with 0.5 GB min_free on a box running at 2-3 GB free -> the window is launched late or its completion is polled late -> card idle for 0.3-4 s. Removing the tail: per-case 5-20% on the hit cases, mean agent/edits +2-4% (I), #267 gone (I: it is the >20 s tail, same as the ledger's inference), and KING stops losing a loaded engine (~46.8 GiB expert cache reload) every time he downloads a model.

**Zero-GPU first steps (diagnostics, not fixes; no global sysctl change before the trace).** (a) `grep VmSwap /proc/<strata pid>/status` on the next loaded engine — if nonzero, the control thread can hit swap-in stalls, mlock is the fix. (b) the off-CPU stack window the ledger already plans: it decides host-blocked vs GPU-late in one run. (c) if host-blocked: mlockall(MCL_CURRENT) after load (or madvise the control thread's arenas), THP `madvise` (the engine requests MADV_HUGEPAGE itself at pinned.cu:506, so the arena keeps its huge pages), min_free_kbytes up to 1-2 GB (the amdgpu-tune PLAN warns against *lowering* it; raising is the classic answer to "page allocation stall"). Each is a sysctl A/B at zero build cost.

**Decisive measurement.** One off-CPU stack of one >300 ms gap. `__alloc_pages_slowpath`/`try_to_free_pages`/`swap_readpage` on the service thread => host; a clean thread sitting in a HSA poll => GPU-late (then look at queue count / `GPU_MAX_HW_QUEUES`, not memory).

**How it fails.** Stalls are GPU-side (runlist oversubscription, SDMA contention); then the memory fixes are neutral and the queue count is the lever. Or mlock of 60+ GB leaves the box with no page cache for the PLE direct reads and makes prompts slower.

### 3. Raise the harness's resolving power before buying more A/B windows (cost: 0 GPU-min, python over existing traces)

**Evidence (V, ledger).** 36K second runs swing 2468-2738 (10%) "in every arm"; one-case decode swings 10-15% "now and then"; adopted wins are +2.3%, +2.8%, +4.6% on 1-2 runs each; `--pool-workers 8` read -4% from two contaminated cases; route-ahead read -16% once from contamination and 0% clean. The stall census explains these: a single 1-4 s gap in a ~50 s case is 2-8%.

**Precedent.** Every perf effort that A/Bs 1-3% effects with a 3-10% noise floor adopts a mix of real and imaginary wins and later cannot reproduce its own config. The board requires exact engine flags per run; a config built from 40 single-run +1% switches is the one nobody can re-verify.

**Mechanism / gain.** Make the harness report, per case, raw tok/s INCLUDING stalls as the headline (that is the achieved speed and the only number that counts), plus a separate diagnostic column: the >300 ms gap count and the stall-stripped mean. The stripped number is for attributing noise, never for claiming a gain. The diagnostic column separates arm effects from tail noise without a single extra GPU-minute, which multiplies the value of every window that follows. Second: prune the 29 env entries the audit found repeating defaults, and write the ledger window id next to each kept non-default switch in the live config. That is the reproducibility the submission needs and the memory KING will not have in a month.

**Decisive measurement.** Re-score windows 12/13 per case with the gap-count column: if the "CCD placement" and "pool-workers" verdicts hold once gap-carrying cases are flagged, the method is proven; if an arm effect was hidden under one stall, it has paid for itself.

**How it fails.** If stalls correlate with the arm (e.g. a config change causes more stalls), excluding them hides a real loss — which is why the stall count must be reported alongside.

## Things found wrong or stale

- **Brief §1 "optimise all three numbers"**: correct, given the metric is hidden (steer #74). Public precedent ranks output tok/s at c1; prompt speed is a side column there. Weight GPU-minutes toward decode unless the private board says otherwise.
- **Ledger window 5b** ("36K prompt is bound by streaming non-resident experts, cards ~45% busy") was measured before `STRATA_KV_PF_SLOTS=1` full residency; the brief already flags it. Nothing after full residency re-measured card busy %. Any new prompt-kernel verdict needs a fresh busy-% trace first.
- **"Expert hotness counts rejected draft rows"** (brief, not done): adaptive placement was worth ~11% on UD4 (ledger: 74.4 vs 67.1). Correction after kahuzi #89: on live IQ3_S `all_experts_resident == true` (`generate.cpp:12278,12328`) disables adaptive swaps, so this bias only touches the Q4_0/UD4 packs and the single-card case. Cheap fix, but rank it by which pack KING is tuning, not above drafters on IQ3_S.
- **Dead-list verdicts carried across packs (V, ledger lines 10-21):** `STRATA_MTP_STAGE0=1` (-11%) and `--mtp-max-t 4` (-17%) were measured on the UD4 live config this morning (split 24, 2048 chunks, before the proxy, the English vocab, KV_PF_SLOTS and the MMVQ dispatcher fix), never on today's fully-resident IQ3_S geometry. kahuzi #89 showed the same transfer failure for adaptive swaps. Pattern seen every time a ledger spans two packs: the kill is remembered, the geometry is not. One cheap re-run each on IQ3_S before they stay dead (ziz #94 asks the same for `--mtp-max-t 4`).
- **`Runlist is getting oversubscribed`** on card 1 (today 09:53:33, Oct 30 x3): not in any ledger entry. Whoever owns the queue count should check what was running at 09:53 and whether IQ3_S's live config pins `GPU_MAX_HW_QUEUES`.
- **Riser plan (I/G):** with IQ3_S fully resident, card 1's x4 link carries per-window activations and MTP traffic, not experts; the prompt-split gains in window 3 (split 28 +15%) were measured under streaming. Precedent on Gen4 bifurcation risers in consumer boards: frequent link training to Gen3 or x4 — verify `LnkSta` after install. Do not buy for the leaderboard; buy for Q4_0/UD4 daily driving if that is wanted.
- **Single gfx103x (G):** the 5080+5060 Ti row (2x16 GB, streaming, 105.7 reasoning) is the precedent: one V620 streaming over Gen4 x8 is the same regime, expect 60-100 tok/s, not 149.

## Human cost everyone skips

- **The coordinator's reading time is the scarcest resource today**: 19 reports before 16:10. Short wins.
- **Engine exit on #267 costs KING a warm 46.8 GiB expert cache and a serving session, while he is doing the most ordinary thing (downloading a model).** Recovery (time out the window, re-issue, log) beats a perfect root cause; every long-lived server that exits on a timeout ends up in a restart loop that loses its cache.
- **Knob debt**: ~40 STRATA_* switches tested today, several "parked opt-in" branches. KING builds in long stretches and does not remember; each parked switch is a question he will have to answer alone in December. Prune or annotate now.
- **BIOS items wait on KING at the screen**; they are the only items here that cost his body time. Keep that list to what the stall trace proves is needed.

STANCE: metric hidden, so optimise all three but weight decode (public precedent); one correlated off-CPU stack + GPU timeline on a stall before any memory/queue fix; add the gap-count diagnostic column to the harness before another +2% hunt.
CONFIDENCE: low on 1 (precedent only, operator confirmed metric unknown), low-med on 2 (hypothesis; trace decides), high on 3.
