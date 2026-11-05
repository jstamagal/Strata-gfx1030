# 2026-11-04: Qwen3.8-Flash-Next UD-Q4_K_XL, main `strata` engine, 2x V620 layer split 24

Config: the live `strata-unsloth-qwen-UD4_K_XL.json` (build-hip/strata from main 15a55375), 1200-token greedy essay,
thinking off, two essays per server load. Raw output in `qwen38-config-batch1/`.

Greedy text is not repeatable on this engine: the two essays of one load differ, and so do the first essays of two loads
of the same config (adaptive swaps move experts between the cards and the CPU pool, whose arithmetic rounds
differently). Token counts (1037-1180) and draft acceptance move with the text, so single runs carry about +-3% noise.

## Config-only switches (discovery-agent ideas)

| config | run 1 tok/s | run 2 tok/s | verdict |
|---|---|---|---|
| live config, load 1 | 72.6 | 75.4 | baseline |
| live config, load 2 | 72.9 | 76.6 | baseline (mean of the four: 74.4) |
| `STRATA_MTP_STAGE0=1` (drafter on card 0) | 66.5 | 66.3 | loses ~11% |
| `STRATA_KQ_SPLIT=1` | 74.2 | 78.0 | +2%, inside the noise |
| `STRATA_EXPERT_V2K=1` | 74.1 | 72.7 | no change |
| `STRATA_PIPELINE_THETA=0.5` | 74.5 | 75.5 | no change |
| `--mtp-max-t 4` | 62.0 | 61.4 | loses ~17% |
| `--adapt-every 4` | 72.9 | 75.9 | no change |

## Expert placement frozen (`--adapt-swaps 0`): a deterministic bench

With adaptive swaps off, the live config repeats exactly: every essay of every run and every load is byte-identical
(md5 d5dcd30186c2, 996 tokens, 426 of 596 drafts accepted). Raw output in `qwen38-frozen-batch2/`.

| config | run 1 | run 2 | run 3 (tok/s) | text |
|---|---|---|---|---|
| live config, `--adapt-swaps 0`, load 1 | 67.0 | 67.1 | 67.1 | d5dcd30186c2 |
| + `STRATA_KQ_SPLIT=1` | 67.1 | 67.2 | 67.2 | identical |
| + `STRATA_EXPERT_V2K=1` | 67.0 | 67.1 | 67.2 | identical |
| live config, `--adapt-swaps 0`, load 2 | 67.0 | 67.0 | 67.0 | identical |

Both kernel switches are bit-exact and make no measurable difference. Adaptive swapping itself is worth about 11%
(74.4 tok/s mean with it, 67.1 without). Noise on this bench is about 0.2%, so it is the A/B bench for any change
meant to keep the output identical.

## Agent branches, one switch at a time (window 1)

Each branch's own executable on both sides; frozen placement unless noted; 2 essays per load (3 for the head rows).
Raw output in `qwen38-window1/`. Baseline on every branch exe: 67.0-67.2 tok/s, md5 d5dcd30186c2.

| branch | switch | tok/s | text | verdict |
|---|---|---|---|---|
| head (9dee938b) | proxy auto-on, nothing set | 69.0 / 69.1 / 69.1 | identical | **+2.9%** |
| head | + `STRATA_HEAD_Q8_SCREEN=1` | 67.8 / 67.9 / 67.9 | **changed** (1134 tokens) | not a wrong pick: its 492 MiB copy moves the expert layout (window 3 control below) |
| tier (efef3efb) | `STRATA_STALL_READAHEAD=1` | 68.3 / 68.4 | identical | **+1.9%** |
| tier | `STRATA_DRAFT_CONTEXT=1` | 67.7 / 67.8 | identical | +0.9% |
| tier | `STRATA_MTP_KV_SIDE=1` / `STRATA_CPU_EXPERT_WARM=1` | 67.0-67.1 | identical | none |
| tier | `STRATA_DRAFT_CUM=1` | 66.3 / 66.4 | identical (382/495 drafts) | -1% |
| gdn (b81da7a9) | `STRATA_VERIFY_INPUT_FUSE`, `STRATA_MTP_KV_BRANCH`, `STRATA_PLE_SIDE_BRANCH` | 67.1-67.2 | identical | none |
| gdn | `STRATA_MTP_HEAD_PREFETCH=1` | 66.9 / 67.0 | identical | none |
| gdn | `STRATA_QSA_QUERY_FIRST=1` + `GATE_BLOCKS=0` | 66.7 / 66.8 | identical | -0.6% |
| gdn | `STRATA_GDN_LAZY_COMMIT=1` | 66.3 / 66.3 | identical | -1.2% |
| kern (44ae5974) | `STRATA_ROUTE_MASK`, `STRATA_QSA_ROWS4`, `STRATA_TOPK_SUFFIX` | 66.9-67.1 | identical | none at essay length |
| kern | `STRATA_EMBED_HOT=1` | 67.3 / 67.1 | changed (21 MiB moves residency) | none |
| kern | `STRATA_SAMPLER_ONCE=1`, seeded sampling (temp 0.7, top-k 20) | 64.2 / 64.3 vs 64.2 / 64.3 | identical | none |
| kern | 36K prompt + 400 tokens: base / `QSA_ROWS4` / `TOPK_SUFFIX` | 80.9 / 80.9 / 80.9 gen | identical | none at 36K |

rocBLAS table regenerated for the running build 5.8.0.565c4c74 (the repo had only 3d8f334b, so the prompt GEMMs ran
rocBLAS defaults), live config, ~36K prompt, two interleaved loads per side: defaults 1331.3 / 1341.1 and 1331.7 / 1340.1,
new table 1338.7 / 1345.5 and 1338.7 / 1345.4 prompt tok/s: +0.4%. The z GEMM is 1.32x faster alone but overlaps the GDN
recurrence, which stays the tail.

## Integration build = main + head + tier (window 2)

`basalt38/integrate` 11db64ca4 (now main). Raw output in `qwen38-window2/`.

Frozen (`--adapt-swaps 0`, 2 essays): base 69.0 / 69.1, + `STRATA_STALL_READAHEAD=1` 70.3 / 70.4, md5 d5dcd30186c2 both.

Live adaptive config, 4 essays per load (run 1 is warm-up, mean of runs 2-4), two interleaved passes:

| switch | pass 1 | pass 2 | verdict |
|---|---|---|---|
| none | 79.0 | 79.1 | baseline |
| `STRATA_STALL_READAHEAD=1` | 78.4 (one 67.9 outlier excluded) | 78.4 | frozen-only win; stays off |
| `STRATA_ADAPT_EARLY_RANK=1` | 79.4 | 78.4 | noise |
| `STRATA_ADAPT_IDLE_GAPS=1` | 78.6 | 78.8 | none |
| `STRATA_ADAPT_ACROSS=1` | 78.0 | 78.3 | none |
| `STRATA_ADAPT_ONCE=1` | 76.8 | 78.5 | loses |

The live config's greedy text varies run to run (placement adapts), so these compare speed only.

## Window 3

Raw output in `qwen38-window3/`.

- Head reservation fix 8e39ffc31 (reserve the 2-bit proxy before card 1's cache is sized): card 1 drops 8444 -> 8320 slots
  (~350 MiB) for an 85 MB buffer, frozen text changes (1156 tokens, 67.3 / 67.4). Left out of main.
- Q8 verify screen control: forcing the exact head with request `frequency_penalty` 1e-20 + `penalty_last_n` 64 keeps
  d5dcd30186c2 without the screen and 17d89f4ce465 with it, so the screen picks are exact and the text change is the
  layout its 492 MiB copy causes. Not measurable on the frozen bench; parked.
- Layer split (card 0 = PCIe 4.0 x8, card 1 = x4: the board's second slot is x4 in every mode):

| split (card 0 layers) | 36K prompt tok/s | frozen decode tok/s |
|---|---|---|
| 24 (live) | 1295 / 1306 | 69.0 |
| 26 | - | 66.6 / 66.7 |
| 28 | **1496 / 1498 (+15%)** | 64.6 / 64.7 |
| 32 | 1117 / 1118 | 58.5 / 58.6 |

Decode text differs per split (placement), but the decode trend is large and one-way: 24 stays the decode split. A
prompt-only split point near 28 is the open lever.

## DFlash2 vs MTP, serial (window 3b)

build-hip (main before the proxy merge), live adaptive config, essay x4 per load, mean of runs 2-4. DFlash2 pack
q4_0-selq4 (`--draft dflash2 --draft-conf 0.35 --spec 6`), 106,299-id draft vocab unless noted. Raw output in `qwen38-window3/`.

| drafter | pipeline windows | tok/s | drafts accepted |
|---|---|---|---|
| MTP (live) | 2 | 75.1 | ~67% |
| MTP | off | 69.3 | 72% |
| DFlash2 | off | 69.0 | 46% |
| DFlash2, en 40,525-id vocab | off | 68.5 | 45% |

Serial DFlash2 ties serial MTP on prose; pipeline windows are worth +8.4% to MTP, so DFlash2 needs its pipelined
version to compete. Code and tool-call prompts not measured yet.

## Prompt split x chunk, all three packs (windows 4-5)

36K-token prompt (unique first line per run, no prefix reuse), then 300 generated tokens at that context; live
adaptive config; 2 runs per load. Card 0 = PCIe 4.0 x8, card 1 = x4. Raw output in `qwen38-window5/` (and
`qwen38-window4/`).

Why chunk size matters: with `STRATA_SPLIT_OWN=1` the engine replaces `--prefill auto:16384` with 2048-token chunks
at startup (source: no-borrow fallback), and `STRATA_PREFILL_PIPE=283` can only lower that. Routed experts that are
not resident are streamed again for every chunk, so 4096-token chunks halve the streaming passes (18 -> 9 at 36K)
despite ~450 fewer cached experts per card. A cap of 8192 plans 3328-token chunks at 36K (planner hysteresis), so the
8192 rows are not a clean bigger-chunk test.

UD-Q4_K_XL (build-hip = main 11db64ca4 unless noted):

| split (card 0 layers) | chunks | prompt tok/s | gen tok/s |
|---|---|---|---|
| 24 | auto (=2048) | 1331 / 1340 | 78.0 / 80.8 |
| 26 | auto (=2048) | 1691 / 1708 | 80.3 / 85.3 |
| 26 | 2048 | 1699 / 1708 | 74.0 / 85.1 |
| **26** | **4096** | **2012 / 2017** | 74.9 / 84.4 |
| 26 | 8192 (plans 3328) | 1552 / 1557 | 74.0 / 79.6 |
| 25 | 4096 | 1837 / 1846 | 76.5 / 78.1 |
| 27 | auto / 4096 | 1630 / 1645 ; 1959 / 1962 | 83 / 84 ; 80.6 / 80.2 |
| 28 | auto / 4096 / 8192 | 1497 / 1500 ; 1838 / 1846 ; 1532 / 1530 | 78-82 |
| 29 / 30 | auto | 1392 / 1392 ; 1296 / 1297 | 75-84 |

sc117 IQ3_S: split 25 auto (its config) 2695 / 2453; 26 + 4096 2638 / 2650; 27 + 4096 2559 / 2577; 28 + 4096
2458 / 2468 prompt tok/s; gen 73-103 (noisy). Split 24/25 with 4096 not yet measured.

bartowski Swift Q4_0: split 26 auto (its config) 2155 / 2180; **26 + 4096 2382 / 2395 (+10%)**; 27 + 4096
2263 / 2275; 28 + 4096 2110 / 2115 prompt tok/s.

Configs changed: UD-Q4_K_XL split 24 -> 26 and prefill 4096 (2014 vs 1335 this morning, +51%); Swift Q4_0 prefill
4096. IQ3_S waits for the 24/25 x 4096 points.

## Draft-head proxy on a Q6_K head, reservation fix (window 5)

- sc117 IQ3_S, frozen placement, 2 essays: proxy off 95.7 / 95.9, `STRATA_MTP_HEAD_PROXY=1` 98.4 / 98.5 tok/s
  (+2.8%), md5 396d660784f5 both. Auto-enable only covers Q8_0 heads, so IQ3_S and Q4_0 configs now set it.
- Corrected reservation fix (d545c8753, proxy bytes only): card 1 8444 -> 8417 slots (-27), frozen text changes
  (placement), 68.4 / 68.4 tok/s. Merged to main as d8c196e42.

## Exact prompt-kernel switches vs whole-prompt speed (window 5b)

Frozen placement, 36K prompt + 300 tokens, 2 runs. UD-Q4_K_XL split 26 + 4096 chunks; IQ3_S split 25 + 4096 chunks.
Executables from `basalt38/rdna2-mmq` (control = all new switches off).

| switch | kernel alone | prompt tok/s | text |
|---|---|---|---|
| UD control | - | 2006 / 1997 | - |
| `STRATA_PF_Q51_TJ=16` (Q5_1 down tile) | 34-44% faster | 2008 / 2015 (+0.5%) | identical |
| `STRATA_MMQ_Q8K=1` | 3-5% faster | 2005 / 2012 | changed |
| IQ3_S control | - | 2712 / 2722 | - |
| IQ4_NL `v_perm` decoder honoured at TJ20 (was silently TJ32-only) | ~14% faster | 2723 / 2736 | identical |
| scheduler-visible zero-head clamp (IQ2_S) | ~4% faster | 2721 / 2737 | identical |
| `STRATA_PF_IQ_Q8K=1` | 0-3% faster | 2710 / 2722 | changed |

A 36K prompt here is bound by streaming non-resident experts over PCIe (cards ~45% busy), so faster expert kernels
move it by at most ~0.5%. (First-token logit dumps for the non-exact arms were not written: `STRATA_DUMP_FIRST_LOGITS`
only fires in the serial window loop, not with pipeline windows.)

## Window 6

- Next-layer routed-expert prefetch (`STRATA_PF_ROUTE_AHEAD`, `basalt38/tp-prompt`), split 26, frozen, identical
  text in every pair: 2048 chunks 1583 / 1529 -> 1769 / 1776 (+14%); 4096 chunks 2006 / 1927 -> 2068 / 2071 (`0,16`,
  +5%) -> 2086 / 2096 (`0,32`, +6.4%).
- Layer-major prefill with later-stage residuals in VRAM vs host: 1698 / 1704 vs 1696 / 1702 (no difference, and
  below the 4096-chunk control).
- Native expert zero-head chain starts (decode, frozen UD split 24): 67.0 / 67.1 -> 67.1 / 67.1, identical. Neutral.
- IQ3_S `--mtp-q4 head` (Q4_0 draft subset, generic kernel): 96.8 vs Q6_K wide 95.8 vs 2-bit proxy 98.5 tok/s, all
  identical text. Proxy kept.
- Agent workload (six code / tool-JSON prompts, 1,200 tokens max, greedy; UD split 24 live adaptive; DFlash2 build with
  the gfx1030 GEMM table and the proxy): MTP + pipeline windows 118.4 tok/s; DFlash2 serial 99.8 (spec 6, conf .35),
  96.1 (5, .50), 94.0 (4, .50); MTP serial 87.7. On agent work pipeline windows are worth +35% to MTP (+8% on prose).

## Pipelined DFlash2 (window 7)

`STRATA_DF2_PIPELINE=1` surplus-block producer (drafter on card 1, 2.6-4.1 ms per block while the verifier shares
the card). Frozen essays: DFlash2 serial 60.9 / 61.1, pipelined 60.4 / 60.6, pipelined with forced misses 59.9 / 60.1
(same text as pipelined: replay path consistent), MTP pipelined 69.0 / 69.1. Agent workload: DFlash2 pipelined 101.8 vs
MTP pipelined 118.4 tok/s (-14%). Live essays x4: 71.2 vs 78.6 (runs 2-4). 36K prompt (split 24, 2048 chunks):
DFlash2 1395 / 1399 vs MTP 1338 / 1343 prompt tok/s (+4%; MTP's draft-KV prompt pass costs more). MTP stays.

## Window 8: IQ3_S split x chunk, STRATA_IQ_GATHER, 4096 vs 8192 chunks at a 120K prompt

Raw: `qwen38-window8/` (batch26-28, per-arm cfg.json). Same build-hip (11db64ca4). Prompt arms: the 36K or 118,329-token
prompt twice per load with unique first lines; generation is live (adaptive placement), so its numbers are noisy.

IQ3_S, 36K prompt, prompt tok/s (run1 / run2) and 300-token generation tok/s:

| arm | prompt | gen |
|---|---|---|
| split 25, live auto (2048 chunks) | 2697.8 / 2707.4 | 96.3 / 105.8 |
| split 25, --prefill 4096 | 2708.3 / 2719.0 | 101.1 / 98.3 |
| split 24, --prefill 4096 | 2725.1 / 2743.1 | 99.3 / 97.2 |
| split 26, --prefill 4096 | 2403.2 / 2647.6 | 93.5 / 93.7 |

IQ3_S doesn't get UD4's chunk win: 4096 is +0.4% at 36K and split 24 +1.2%, inside the decode noise. Split 26 loses
both. The IQ3_S config keeps split 25 and auto; window 11 checks 2048 vs 4096 at 120K.

STRATA_IQ_GATHER=1, IQ3_S frozen decode: 98.5 / 98.6 vs 98.4 / 98.6 tok/s, same md5, same drafts (460 / 641).
No effect: not adopted.

118,329-token prompt, 100 tokens out:

| arm | prompt tok/s | gen tok/s | expert cache slots card0 / card1 |
|---|---|---|---|
| UD4 26 + 4096 (live) | 1987.3 / 2037.7 | 82.5 / 84.1 | 8443 / 8046 |
| UD4 26 + 8192 | 2219.3 / 2113.3 | 76.5 / 77.8 | 7531 / 7129 |
| IQ3_S 26 + 4096 | 2744.0 / 2630.6 | 98.1 / 85.5 | 12862 / 11264 (card 1 all resident) |
| IQ3_S 26 + 8192 | 2589.0 / 2528.8 | 83.9 / 89.2 | 11399 / 10507 |

UD4 at 120K: 8192 reads +7.6% faster (54.7 vs 58.8 s) but costs ~910 cache slots per card and decodes 8% slower after it;
4096 stays (a full 120K re-read is rare with the prompt cache, decode is every token). IQ3_S: 4096 beats 8192 on both.

## Window 9: DFlash2 at the full block (7 guesses), confidence sweep, whole-block acceptance

Raw: `qwen38-window9/` (cfg.json, xz pipeline traces). Agent workload (6 code / tool-JSON prompts, greedy, 7,200 tokens),
UD4, DF2 exe strata-q38-dflash2 b3097181374b, pipelined, every arm incl. the MTP control with STRATA_PIPELINE_DEBUG=1 and
a pipeline trace (the trace costs ~4%: MTP 113.8 here vs 118.4 untraced). Before this window, the harness's ARG edits
did not reach user_args, so --draft-conf would have stayed at 0.35. Fixed first.

`--spec 8` = 8 verify rows = anchor + 7 guesses = the drafter's trained block (incoai config: block_size 8).
Full accept = a window whose T-1 drafts were all accepted.

| arm | tok/s | windows | full accept | full accept at T=8 | mean accepted drafts |
|---|---|---|---|---|---|
| DF2 spec 7, conf .35 | 93.8 | 2284 | 33.5% | 23% | 2.16 |
| DF2 spec 8, conf .35 | 94.3 | 2083 | 34.8% | 50% | 2.46 |
| DF2 spec 8, conf .20 | 96.0 | 1775 | 29.0% | 42% | 3.07 |
| DF2 spec 8, conf .01 | 79.7 | 1923 | 16.9% | 17% | 2.75 |
| MTP pipelined (control) | 113.8 | 2914 | 64.9% (T=3: 84%) | lookup windows 49% | 1.48 |

Full-width DFlash2 doesn't open up here: best 96.0 vs MTP 113.8 (-16%). DFlash2 + MTP chain: at 3.00 tokens per 29.46 ms
cycle, beating MTP needs whole-block acceptance >= 61-64% with one MTP step, or >= 36-40% with two, even with free
verification. Measured 29-35% overall. Not built.

## Window 10: English draft vocabulary and long lookup windows, IQ3_S

Raw: `qwen38-window10/`. Live IQ3_S config (split 25), build-hip 11db64ca4. Agent = the 6 code / tool-JSON prompts
(7,200 tokens); edit = 3 copy-heavy prompts (rename functions in a file, three changes to a JSON config, add docstrings;
`edit-prompts.json`), greedy.

| arm | agent tok/s | edit tok/s |
|---|---|---|
| base (shared mtp/rt draft vocab: 106,299 ids) | 145.5 | 368.1 |
| --mtp-draft-vocab draft_vocab_en.bin (40,525 ids) | 148.9 | 369.5 |
| STRATA_PL_LOOKUP_MAX_T=12 | 145.5 | 369.0 |
| both | 148.9 | 369.8 |

- **English draft vocabulary: +2.3% on agent work.** The text is identical (same md5 in every case) and so are the
  drafts; all six cases got faster. On the model's own code/JSON output, both subsets miss the same 2.10% of tokens
  (6,965-token corpus), so for English work the 66K CJK rows only made the draft head bigger. Now in all three configs
  as `--mtp-draft-vocab`.
- **Long lookup windows: no effect, because they never switched on.** The engine logs "--lookup-max-t 0: a layer split's
  windows stop at 8 tokens". Copy-heavy edits already accept 97% of drafts at 8 rows (~370 tok/s), so the 8-row cap is
  the ceiling. Lifting it on the split is code work (branch basalt38/split-lookup16).

## Window 11: IQ3_S at a 120K prompt, split 25, 2048 (auto) vs 4096-token chunks

Raw: `qwen38-window11/`. 118,329-token prompt, 100 tokens out, interleaved twice (a / b):

| arm | prompt tok/s (run1 / run2) | gen tok/s |
|---|---|---|
| auto (2048) a | 2744.0 / 2654.4 | 87.1 / 78.8 |
| 4096 a | 2833.1 / 2729.8 | 87.7 / 86.1 |
| auto (2048) b | 2728.6 / 2621.7 | 87.0 / 79.1 |
| 4096 b | 2829.8 / 2671.7 | 87.2 / 86.2 |

At 120K, 4096 is +2.9% on prompt and +4.6% on the generation after it, the same in both repeats (+0.4% at 36K in window 8).
The IQ3_S config now uses --prefill 4096, like UD4 and Q4_0. Overlap watch over windows 9-11: the harness never ran two
engines at once, but free memory fell to 0.8 GB during an engine load.

## Window 12: --pool-workers 4 / 8 / 12 / default (15), IQ3_S

Raw: `qwen38-window12/`. Workers pinned 1-N, host on cpu0 (the engine's default placement).

| workers | 36K prompt tok/s | agent tok/s |
|---|---|---|
| 4 (config) | 2682.0 / 2693.3 | 145.5 |
| 8 | 2696.4 / 2704.6 | 139.2 |
| 12 | 2694.8 / 2704.9 | 145.5 |
| 15 (default) | 2690.5 / 2449.1 | 145.5 |

Worker count doesn't matter on this box: decode hits the expert cache ~99% of the time, so the CPU pool has little to do.
Prompt +0.4% at 8-12 (noise). The 8-worker agent dip is two cases (115 vs 127, 129 vs 148, same text and drafts) while
Astra runs were compiling on the box; the codex runs are now niced to 19. The configs keep 4.

## OS window 1: live OS and driver fixes, IQ3_S (cumulative ladder, auto-revert)

Raw: `qwen38-windowOS/` (batch log with state dumps, step scripts). Keep rule: agent >= best - 0.3% and prompt >= best - 1.5%.

| step | agent tok/s | 36K prompt tok/s | verdict |
|---|---|---|---|
| baseline | 145.5 | 2694.0 / 2705.3 | |
| LACT PF1 memory power state [3] only (was 1-3: it idled at 456/673 MHz between requests) | 145.3 | 2696.9 / 2704.0 | kept (neutral) |
| PCIe MaxReadReq 512 -> 4096, both V620s | 145.4 | 2691.6 / 2701.2 | kept (neutral) |
| GPU interrupts from cpu5/6 to cpu30/31 | 144.4 | 2690.4 / 2706.2 | reverted (-0.7%) |
| preemption full -> lazy | 145.6 | 2690.4 / 2471.2 | reverted (one prompt outlier) |

None of these moves the bottleneck. The LACT change also copied PF1 into LACT's fallback settings, which had card 0 at
150 W. MRRS 4096 lasts until reboot. Found afterwards: card 1's whole PCIe route (root port 00:01.3, card switch
05/06, GPU 07) has ASPM L1 enabled by the BIOS (pcie_aspm=off leaves it alone); card 0's route has it off. OS window 2
tests turning it off.

## Window 13: CPU placement across the 9950X's two CCDs, IQ3_S

Raw: `qwen38-window13/`. --pool-workers 4 takes the first 4 entries of --pool-cores, so every arm ran 4 workers.

| layout | 36K prompt tok/s | agent tok/s | per-case (same text in every arm) |
|---|---|---|---|
| host cpu0, workers 1-4 (CCD0) | 2696.8 / 2709.9 | 142.4 | 127 148 154 **129** 180 128 |
| host cpu0, workers 8-11 (CCD1) | 2688.2 / 2704.7 | 145.5 | 128 148 154 147 180 128 |
| host cpu15 + workers 8-11 (all CCD1) | 2687.6 / 2700.0 | 145.4 | 127 148 154 147 179 128 |
| whole process taskset to CCD0, workers 1-4 | 2686.8 / 2701.4 | 141.5 | 127 148 154 **125** 180 128 |

CCD placement makes no difference. Each slower arm is a single case (json_multi_tool_result_merge, 125-129 vs 147,
same text) while Astra codex runs were active; every other case matches to within 1 tok/s. As with --pool-workers in
window 12, the CPU barely matters for decode at ~99% cache hits. Harness note: a one-case transient of 12-15% shows
up now and then; compare per case before believing an arm-level difference of about 2%.

## Window L16: 16-row lookup windows on the layer split, IQ3_S — FAILED (bug)

Raw: `qwen38-windowL16/`. Binary from basalt38/split-lookup16 at 1eb7ca8d7 (sha256 3e22af24…), all arms the same binary.

| arm | edits tok/s | agent tok/s | text vs base | long windows |
|---|---|---|---|---|
| base (8-row cap) | 366.1 | 145.7 | — | 0 |
| --lookup-max-t 16, long gate on | **245.9** | 143.7 | **differs, all 9 cases** | 62 on edits, 0 on agent |
| --lookup-max-t 16, long gate off | 362.1 | — | same | 0 |

Long windows accepted about 7.5 of 15 drafts (105/210, 196/375, 184/345); the trace puts almost every reject at the
first row of the second half (a=8). Rows 9-16 are not computed correctly somewhere on the stage-0 -> stage-1 path, and
the wrong bonus token at the reject changes the text. The agent arms also change text with zero long windows admitted,
but they are not an equal-geometry pair: the long arm carries S16 and the larger verify reserve, which changes cache
placement, and no agent arm ran with gate off. So that text change may be placement, not a bug. Not adopted; back to
the building agent for a fix.

## OS window 2 and the KV_PF_SLOTS insert, IQ3_S

Raw: `qwen38-windowOS2/` (batchOS2.out, batch36.out).

| change | agent tok/s | 36K prompt tok/s | verdict |
|---|---|---|---|
| baseline | 140.4 | 2709.5 | |
| ASPM off on card 1's route (root port, switch, GPU) | 140.4 | 2715.2 | kept by ruling (neutral); boot unit gpu-pcie-tune.service |
| HSA_ENABLE_INTERRUPT=0 (polling completion) | 140.7 | 2708.4 | neutral, not adopted |
| HSA_ENABLE_MWAITX=0 | 140.5 | 2713.0 | neutral, not adopted |

**STRATA_KV_PF_SLOTS=1** (bounded prompt-KV staging) cuts card 1's owned prompt reserve from 3124 to 2118 MiB. That
lets card 1 cache all 11776 of its experts (11741 before), so both stages get the zero-doorbell verify graph (4 of 4
vs 2 of 4):

| | slots 0 | slots 1 |
|---|---|---|
| agent tok/s (6 cases, every case faster) | 140.4 | **149.0 (+6.1%)** |
| gen after a 120K prompt | 87.3-88.2 | 92.4-101.0 |
| 120K prompt (ABBA, 4 runs each) | 2535-2842, median 2781 | 2634-2718, median 2689 (-3%) |

Agent text changes on 5 of 6 cases (35 experts move from CPU to GPU). A confirmation round (36K prompt, agent in
reverse order, and --kv-resident 61440 as an alternative route to full residency) is queued before adoption.

**KV_PF_SLOTS confirmation** (batch38, reverse order): agent 142.5 (slots 1) vs 138.4 (slots 0), +3.0%. Across both
passes 145.8 vs 139.4, **+4.6%**, with 5 of 6 cases faster in each pass (one-case swings of about 10% both ways).
Gen after a 36K prompt 99.8-103.7 vs 95.8-97.5. 36K prompt 2711/2537 vs 2697/2716 (the 2537 dip also shows up
in other arms). The alternative, --kv-resident 61440 with slots 0, only gets card 1 to 11753 of 11776 experts: no
zero-doorbell graph, no gain. **ADOPTED** in the IQ3_S config (`STRATA_KV_PF_SLOTS=1`); harness copies synced. The
cost is about 3% on a 120K cold prompt. Not applied to Q4_0/UD4: they can't reach full residency, which is where the
gain comes from.

## Window NEXT: integration binary (tp-prompt + rdna2-mmq merges, Q6 proxy auto, MMVQ dispatcher fix), IQ3_S

Raw: `qwen38-windowNEXT/`. basalt38/next at 814720e71 (sha256 d0211251…). The validator passed: all full-text MD5s
match across cur/next/chain arms. KV_PF_SLOTS=1 in every arm.

| arm | agent tok/s | 36K prompt tok/s | edits tok/s |
|---|---|---|---|
| cur (today's build-hip) | 129.3 (contaminated: BF16 download + scans) | 1994 (contaminated) / 2692 | — |
| next | 147.4 | 2720 / 2741 | 369.1 |
| next + STRATA_PF_ROUTE_AHEAD=0,32 | — | 2188 / 2303 (**-16%**) | — |
| next + STRATA_MTP_CHAIN_CAP=1 (5 chain outputs, not 7) | 143.8 (5 of 6 cases within 0.3%; json_multi 126 vs 147) | — | 348.9 (**-5.5%**) |

Route-ahead loses on IQ3_S: with every expert resident there is nothing to prefetch. It stays a UD4/Q4_0 candidate
(its +6.4% came on UD4 26+4096). The chain cap is rejected. The next binary matches cur's text and speed; clean cur
arms are queued (batch39/40).

## Window L16B: 16-row split lookup with the MMVQ dispatcher fix (27d713fbc), IQ3_S

Raw: `qwen38-windowL16B/`. The validator passed: text matches base and the equal-geometry control.

| arm | edits tok/s | agent tok/s | long windows |
|---|---|---|---|
| base (8 rows) | 371.2 | 142.8 | 0 |
| --lookup-max-t 16, gate on | 362.0 (-2.5%) | 135.8 (one-case swings both ways; 4 of 6 cases equal) | 163 on edits, 0 on agent |
| --lookup-max-t 16, gate off | 371.4 | 145.9 | 0 |

Long windows now accept 93-100% of drafts (616/660, 884/885, 880/900) and the text is right, but a 16-row window
costs about as much as two 8-row ones, so there's no gain. Per-stage trace analysis is with the building agent.
**The dispatcher bug was real and is also on main** (single-GPU wide lookup, --batch 9..16, wide-MTP fallback).

**Window NEXT, clean reruns** (batch39/40, no download or scan running). Correction: the -16% route-ahead result
above was contamination (memory pressure from a paused download plus a root file scan), not route-ahead.

| arm | agent tok/s | 36K prompt tok/s (run1 / run2) |
|---|---|---|
| cur | 148.9, second pass 148.8 | 2706 / 2721 |
| next | 148.9 | 2708 / 2540 |
| next + MTP_CHAIN_CAP=1 | 148.9 | — |
| next + ROUTE_AHEAD=0,32 | — | 2697 / 2738, second pass 2707 / 2468 |

Route-ahead is neutral on IQ3_S (every expert resident; run1s within 0.4%). Second runs swing 2468-2738 in every arm:
harness noise, not an arm effect. The next binary equals cur on agent and prompt with identical text. **Merged
into main as ceade1b6b**; build-hip is rebuilt in the next build slot (old binary kept as
build-hip/strata.pre-ceade1b6b).

## Window L16C: lookup16 cost-aware admission policy, IQ3_S

Raw: `qwen38-windowL16C/`. The validator passed: full text matches base and the equal-geometry control.

| arm | edits tok/s | agent tok/s |
|---|---|---|
| base (8 rows) | 372.7 | 146.9 |
| policy, 128-token runway | 374.4 | — |
| policy, 192-token runway (default) | 374.2 (+0.4%) | 146.8 |
| policy, 256-token runway | 374.0 | — |
| 16-row lookup, gate off ("wide_off") | 297.2, json_config alone 331 -> 198 | 146.7 |

The policy admitted 38 of 312 checks at every runway, so the runway threshold doesn't matter. Agent prompts never
reach it (0 checks). The +0.4% is real but small. Parked opt-in on basalt38/split-lookup16; not merged.

The wide_off collapse is not a lookup16 effect. Its text and draft counts equal base. The whole loss is **one 2381 ms
gap** in pipeline.trace: card 1 launched window 109 and finished it 2.4 s later with no other loop event in between
(normal is ~18 ms). See the stall census under window 267.

## Window 267: #267 repro under dirty-page and RAM pressure

Raw: `qwen38-window267/` (script, output, stall census). Binaries: old = next d0211251 (equal to main), fix =
basalt38/verify267 230a21fe1 (c5a091d7). Profiles:
- **auto:** the first failing config. PLE table in RAM by policy (lock given back in old, kept in fix),
  KV_PF_SLOTS=0, kv-resident 61440.
- **direct:** today's live geometry. PLE direct reads, KV_PF_SLOTS=1, kv-resident 65536.

Dirty arms wrote 13-18 GiB at 300 MiB/s; dirty pages peaked around 210 MiB, capped by the box's 256 MiB dirty_bytes.
The ram arm pushed available RAM down to 0.22 GiB.

- **No #267 in any of 9 arms**, quiet or dirty. Inconclusive: dirty pages alone don't trigger it, and nothing is
  established about the fix. **verify267 is not merged:** it has no reproduced failure, and its kept 26.8 GiB PLE
  lock left 0.26 GiB available in a quiet run. The live config uses direct PLE reads, which the fix doesn't touch.
- **The auto profile's text is not reproducible run to run:** two output sets alternate across arms, unrelated to the
  binary, pressure or lock state. The direct profile gave identical text in all 4 arms. The cause is confounded
  between the PLE RAM path and KV_PF_SLOTS=0. The live config is direct.
- Throughput is not comparable to other windows: all arms carried verify/pipeline tracing (126-136 tok/s).

**Stall census** (`qwen38-window267/stall-census.txt`): a gap is >300 ms in pipeline.trace with no loop event, in a
decode request. The census found **25 gaps in 34 traced arms, 0.3-4.2 s each**:
- in quiet arms and dirty arms, in old and fix binaries;
- in windows L16B and L16C as well as 267;
- following both card-1 launches (L1) and card-0 launches/drafts (L0, CD).

These single stalls are the "one-case slowdowns" seen all day (json_multi, json_config, code_async_batcher).
**Inference, not yet shown:** #267 is the >20 s tail of the same stall.

The kernel log shows `page allocation stall for 10 secs` at 13:29:30, during the BF16 download. THP is
enabled=always, defrag=defer+madvise; since boot, compact_stall = 171754 and allocstall_movable = 397974.

Next: an instrumented window that captures strata's off-CPU stacks (>100 ms) and samples vmstat during the stalls.
It splits host-blocked (reclaim, compaction, faults) from GPU-late (the service thread on CPU, polling).

**Natural #267 at 15:29**, audit window 16, arm w16_copy1_b (live config, main build-hip, quiet box). It happened
on the 2nd agent case. The release log says "the GPU finished in 0 ms" on all four verifiers. The GPU was idle,
spinning on a flag; it saw the release write at once, so the flag memory is visible to it. That rules out a stale
GPU cache. (Corrected below: the host never raised SOME window's flag; the reporting verifier's own was up.) The live config
has no DMA host-function path (--pcie-frac 0, copy-kernel mode), so flag A (the plan the CPU pool publishes) is
the likely wait. Window STALL (started 15:39) captures the host side.

Queue bug, fixed: when a window failed, queue-lib's EXIT trap wrote the FAILED marker into the batch log instead of
window<N>.out. Window 17 waited on it from 15:31 to 15:39. The trap now tees the marker into the window log.

## Swarm intake, 15:45 (swarm run 1: titus, nyira, ziz; reports archived out of the swarm's reach)

Taken:
- **Row economics** (titus): window 18 gains arms for `--spec-min-p 0.40` and `STRATA_SPEC_COST=1`, which has
  never run on this box.
- **Prompt re-attribution** (ziz road 3): inserts/49-windowpft.sh runs the 36K prompt with `STRATA_PREFILL_TIMING=1`,
  plus a control, at the next boundary.
- **Verify-tree decider** (titus): asked the mtpq agent for the offline drafter top-2 hit rate on rejected steps.
- **Stall attribution** (nyira, titus): window STALL is running. Seen on the box: swappiness 100, 2 GiB swap in use,
  min_free_kbytes 512 MiB. A 10 s page-allocation stall in strata itself on 11-02.

Checked and corrected:
- **ziz road 2 (host-gated MTP chain) does not apply to the live config.** `draft_first`'s per-step host gate is the
  serial loop's. With `--pipeline-windows 2` the chain goes through `chain_launch`, which launches every step at
  once; the min-p gate applies afterwards in `t_rule`.
- **ziz road 1 (GDN rollback snapshots) is real in source, but sized wrong.** There are 36 GDN layers, and each
  window writes fp32 [nv 48][128][128] = 3.15 MB per layer for every row but the last: ~340 MB for a 4-row window,
  ~790 MB for 8 rows, split across the cards. ziz used 6.7 ms per cycle; that is per token, and a window is ~17 ms.
  Re-estimated cost: ~5-8% of decode (inference). The fix that keeps text exact is to store the per-row GDN inputs
  instead of states and recompute on partial accept. A candidate; it needs a kernel microbench first.
- **titus idea 3 (pipeline depth 3)**: the clamp at 2 is real. From the pipeline traces, card 0 has a window in
  flight 82% of the time on agent and 97% on edits; card 1 65% / 81%, plus the MTP chain on card 1. These are upper
  bounds from host-side intervals. The cards are not "half idle"; depth 3 is low priority.
- **titus**: the stalls are ~2-4% of mean time, a tail problem, not the throughput lever. Agreed.

**Verify-tree decider** (offline, mtpq agent; `strata-q38-mtpq/bench/results/2026-11-04-mtpq/offline/top2-verify-tree.txt`).
Setup: held-out agent B, 4473 prefix-correct drafter steps. The live S2 drafter rejects its top-1 on 782 of them,
and 112 of those rejections are tokens outside the draft vocab.

| when top-1 is rejected | P(drafter top-2 = verifier token) |
|---|---|
| all rejections | 0.42 (0.49 counting only in-vocab targets) |
| top-1 p < 0.4 | 0.23 |
| top-1 p 0.4-0.7 | 0.45 (top-2 holds 0.26 of the probability) |
| top-1 p >= 0.7 | 0.48 |
| edits (thin data, 54 rejections) | 0.28 |

BF16 experts give the same rates, so the tree doesn't depend on the drafter weights. It clears the 0.2 kill line.

Economics (inference): per-step acceptance ~0.83, so a 3-draft chain rejects somewhere in ~44% of windows. A sibling
at every depth adds ~0.18 tokens per window for ~3 extra rows. At this box's row cost (8 rows 18.0 ms, 16 rows
33.2 ms) that is about break-even, before the cost of GDN branch state. The cheap form that could pay is a
**leaf fork**: where the min-p gate would stop the chain (top-1 p 0.4-0.7), verify top-1 and top-2 as two candidates
for the last row. That costs +1 row, raises the chance the last row is accepted from ~0.55 to ~0.75, and needs only a
one-level GDN fork plus a QSA mask change. Ranked below GDN snapshot removal and the stall fix.

## Window STALL: host-side attribution of the single-window stalls (15:39-15:49)

Raw: `qwen38-windowSTALL/`. The run is the live config on main build-hip, 4 agent + 4 edit passes, with pipeline
traces. Alongside it:
- `stall.bt` logged every off-CPU interval >=100 ms of every strata thread, with kernel stacks, and block I/O >50 ms;
- a 100 ms sampler logged compact_stall, allocstall, pgscan_direct, pgmajfault and PSI.

The text is identical across all passes.

- **5 gaps >300 ms** (433, 762, 1125, 300, 617 ms); 3 of them are decode-time gaps in agent requests 4-5 and edit
  request 3. **0 slow block I/O.**
- **The host was not blocked.** In the run with the two biggest gaps (ws_agent_2, request 4), no strata thread except
  idle ones (pool workers, which sleep through resident decode) was off-CPU >=200 ms during decode. In that run,
  compaction, allocstall and direct reclaim are nonzero only at t=11 s (model load). All reclaim activity across
  the 8 runs clusters at load time, t=6-26 s after each start.
- **The memory-reclaim hypothesis is falsified for these stalls** (nyira and cantsbee proposed it; swappiness 100 and
  failing compaction are real on the box, but absent at the moments of the stalls). With the natural #267
  ("GPU finished in 0 ms" on release), the picture is: the host thread was running and did not see what it waits
  for, and the GPU was spinning on a flag the host had not raised.
- Kernel stacks are unsymbolized: kptr_restrict=2 hides kallsyms even from root. The next trace must set it to 1
  for the run and restore it afterwards.

**Lead: the #649 handshake switches**, opt-in since upstream 0.1.38 for "gfx1030 verify timeouts" and never run
here:
- `STRATA_VERIFY_COHERENT=1`: the handshake words and rows go in hipHostMallocCoherent (fine-grained) memory. Today
  they are default mapped host memory.
- `STRATA_DOORBELL_STORE=1`: the GPU stores each step's ring instead of a read-modify-write over PCIe.

Mechanism (inference): a GPU progress write to non-coherent host memory, or a PCIe RMW, that the host doesn't see
until something flushes it. This gives random 0.3-4 s delays, and a 20 s deadlock (#267) when nothing else flushes.
**inserts/50-windowcoh.sh** runs 8 passes per switch with a gap census; text must match window STALL.

cantsbee (swarm run 1, archived) also proposes:
- prompt as a chunk-pipeline problem; window PFT, queued, measures where the time goes;
- cursor-chase lookup for edits: edits-only, small, kept on the list.

**Swarm chat intake, 16:15** (troop run 20261104-152916: the bus log and gruff's writeup; the full table is in
docs/swarm/BRIEF.md under "Already proposed and judged").

Found only in the chat:
- **The pipelined `pl_launch` lacks the early PLE flag guard.** The serial path has it against a documented HIP lock
  cycle: the host inside hipGraphLaunch waits on a runtime lock while the GPU spins on the PLE flag. It matches the
  natural #267 and window STALL. Being implemented on basalt38/pl-early-ple, with a slow-launch logger.
- **Guess-outcome cycle table** (tuck): right 61% at 12.8 ms; wrong 32% at 22.2 ms; misses take 41% of decode time.
- **Stage wall time:** ~8 ms fixed plus ~0.9 ms per row.
- **HC_PERSIST** needs QFUSE/NODE_FUSE off.
- **The resident-complement mode** frees the 46.84 GiB host expert copy.
- **SH_STREAM=0** under full residency.
- **Two prefill fusions:** PREFILL_SWIGLU_QUANT and PREFILL_COMBINE_WRITE.
- **The IQ kernel's 4 -> 5 row cliff.**
- **Survival calibration.**

The zero-build items are queued as inserts/51-windowzb.sh (with a per-kernel profile of both cards).

## Window MTPQ-AB1: MTP drafter weights rebuilt from BF16 (16:25)

Raw: `qwen38-mtpqAB1/`. Binary: strata-q38-mtpq build-hip (563dbb2f). Live IQ3_S config with KV_PF_SLOTS=1. Six
drafters, agent and edits, two passes in reverse order. The gates passed (11776 slots, 4 zero-doorbell graphs, 100%
resident), and the text is identical everywhere. GPTQ was calibrated on agent cases 0/2/4; 4 agent cases and all of
edits are held out.

| drafter | MiB | agent tok/s (both passes) | held-out agent | edits |
|---|---|---|---|---|
| live S2 | 839 | 148.99 (149.0 / 149.0) | 147.98 | 340.5 / 368.6 (pass 1 slow) |
| q2w (weighted refit) | 839 | 150.13 (+0.8%) | +0.9% | 368.5 |
| **s2gptq** (all S2, GPTQ, same bytes) | 839 | **151.90 (+2.0%)**, 152.0 / 151.8 | **+1.4%** | 366.6 |
| mixr15 (+97 Q4_0-GPTQ experts) | 967 | 150.85 (+1.2%) | +1.1% | 368.6 |
| mixr40 (+134 Q4_0-GPTQ experts) | 1016 | 151.42 (+1.6%) | +1.6% | 369.5 |
| qo4 (q/o projections Q4_0) | 817 | 146.98 (-1.3%) | -0.4% | 368.5 |

Edits are flat for every variant against live's clean second pass (368.6). The summary's +4% edit figures compare
against live's slow first pass, a stall-type outlier.

Profile arms (one agent pass, with DRAFT_TIMING / MTP_PROFILE on):

| drafter | tok/s | ms per round |
|---|---|---|
| live | 148.0 | — |
| q4n, all native Q4_0 RTN | **151.5** | — |
| mixr40 | 150.5 | — |
| n2live, live bytes in the native Q2_0 layout | **91.4** | 25.2 |

The q4n smoke run earlier read -4.5%. n2live has the same bytes and acceptance as live but is 38% slower, so the
native grouped path has a large fixed cost in some layouts. The mtpq agent is reading the priority-0 profile arms.
Adoption is held until that analysis lands; s2gptq is the low-risk candidate (same VRAM, +2.0% / held-out +1.4%).

**MTPQ-AB1, mtpq agent's analysis** (`strata-q38-mtpq/bench/results/2026-11-04-mtpq/ab1-stallfree.txt`, a445b20f8):
- **n2live's -38% was window stalls:** 7 gaps over 300 ms (17.1, 2.1, 1.4, 2.9, 3.0, 2.2, 0.8 s) add up to its whole
  deficit. Without the stalls it sits ~1% behind live, in line with the native Q2_0 kernel's microbench. The 17.1 s
  gap came close to #267's 20 s limit.
- **q4n's -4.5% in 41z was also stalls:** three cases ran 1.0-1.2 s slower at identical draft counts. Run clean,
  q4n gives 151.5 and 151.0 (priority on and off) against live's 148.0, i.e. **+2.0-2.3%**. RTN uses no
  calibration, so every case is held out. Cost: +675 MiB on card 1; it keeps 11776 slots with 1038 MiB free.
- **STRATA_HIP_MTP_PRIORITY=0 is neutral for both drafters.** The chain (3.7-3.9 ms) runs hidden under card 1's
  ~10 ms stage-1 window. DRAFT_TIMING / MTP_PROFILE print nothing in pipelined decode (synchronous path only).

Next:
- window ZB checks s2gptq on the main binary;
- mtpq_ab2 (live / s2gptq-a / q4n / q4_0-gptq-a, 2 passes, traced) and a corpus capture are staged by the agent;
- adoption follows ZB.

**MTP drafter roadmap** (mtpq agent's interim summary, 16:40; branch basalt38/mtp-quant at 390eb3dae).

Offline:
- Live S2 gets 98.7% of BF16's gated hits per step. Q4_0 and Q4_0/Q8_0 experts match BF16.
- Upgrading only the down matrices to Q4_0 removes 49% of the S2-GPTQ error at 1/3 of the bytes.

Not built:
- **S2 grouped kernel with Q4_0 down rows:** keeps the S2 path, adds about 236 MiB, should get most of q4n's gain.
- **Training tiers** (main model frozen throughout):

| tier | what | engineering | compute | expected |
|---|---|---|---|---|
| A | rank-r head adapter + logit bias with a gate-aware loss | ~0.5 day runtime | CPU, minutes | +1-3%, low confidence (a 3.6K-step probe gave ungated E[L] +2.1%, gated -1.2%) |
| B | quantization-aware tuning of the S2 experts (fp16 scales + mixer) to the BF16 drafter's distribution (KL on replayed captures) | ~1 day | ~1 GPU-hour on one V620 | targets Q4-quality drafts at S2 bytes and speed; best value per cost |
| C | self-distil the whole MTP block to the deployed abliterated IQ3_S verifier (7-step chain unrolled) | 3-4 days | data 2-4 h on the GPU pair, training 4-10 h on one V620 | largest upside: 82-84% per-depth acceptance, and top-2 holds 42% of misses |

Training data comes from STRATA_MTP_CAPTURE (~360 records per second of generation, 72 KB each). Full-block
training needs a round-input hook (~1 h); distillation needs a verify top-k dump (~0.5 day).

**Correction to the natural-#267 reading above** (from the early-flag agent's source read, verified in
verify.cpp):
- "the window never finished" comes from done(). In zero-doorbell mode that is reached only after service() has
  raised that verifier's own PLE flag.
- The four "finished in 0 ms" lines print after done()'s own 5 s release has already drained the GPU.
- So the reporting verifier's flag WAS up. Its graph was held by something else: most likely an earlier graph on
  the same stage stream, spinning on its own unraised flag while the host polled the later window. Inference.
- "The host never raised the flag" is withdrawn for that verifier. "Host waits on GPU, GPU waits on a host flag"
  still stands as a cross-window ordering cycle.
- Two older #267 logs (w_kvs2_kv61_agent, mtpq_cap_agent) say the release did NOT free the GPU within 5 s, so not
  every #267 is a host-flag spin.

**Early-flag build** (basalt38/pl-early-ple, 72ca0237f, binary fae158d9; 21/21 CPU tests, none of which cover the
pipelined path):
- `STRATA_PL_EARLY_PLE_FLAG=1` raises every pipelined zero-doorbell window's PLE flag before cudaGraphLaunch. In
  the live config that removes every host-flag spin from the verify graphs.
- `STRATA_PL_LAUNCH_SLOW_MS` logs slow pl_launch / commit calls, together with the state of every in-flight
  window.
- Window EPF (4+4 agent/edit per mode, traced) runs at 16:4x, ahead of window COH.
- If e1 still stalls, the host-flag theory is out for verify.

## Window PFT: where a fully resident 36K IQ3_S prompt spends its time (16:35)

Raw: `qwen38-windowPFT/`. Main build-hip d4d10152, live config.

| arm | run 1 | run 2 |
|---|---|---|
| control | 2546 tok/s | 1985 (the recurring second-run outlier) |
| STRATA_PREFILL_TIMING=1 | 2463 | 2580 (small perturbation) |

Per-chunk GPU timeline (4096 tokens, steady chunks, ~1250 ms; wall 1266-1297 ms):

| phase | ms | share |
|---|---|---|
| gemm gate/up | 260 | 21% |
| qsa attn | 159 | 13% |
| gdn recurrence | 125 | 10% |
| gdn (rest) | 110 | 9% |
| qsa proj | ~105 | 8.5% |
| gemm down | 104 | 8% |
| gdn out proj | 67 | 5% |
| hc read down+silu | 66 | 5% |
| hc read up | 57 | 4.5% |
| dequant | 45 | 3.5% |
| hc write | 35 | 3% |
| router+shared | 24 | 2% |
| embed | 24 | 2% |
| combine | 23 | 2% |
| host grouping | 21 | 2% |
| gather | 13 | 1% |
| host waits per chunk | ~20 + 20-40 | |

- **Experts in total** (gate/up + down + dequant + gather + combine) are ~36%. **GDN** ~25%, **QSA** ~21%, **HC** ~13%.
  "Expert-streaming-bound" is dead post-residency; the prompt is spread across phases.
- **The first chunk after server start costs 2208 ms** (gate/up 1216 ms against ~260 steady), a one-time warm-up of
  ~1 s. It is in every harness run 1.
- **Arithmetic** (inference): 9 chunks at ~1.27 s plus fill/drain is close to the observed 14.3 s. A 4000 tok/s 36K
  prompt needs the slower stage's chunk near 0.85 s, i.e. ~-33% on a spread profile. No single fusion gets there.
- **Window ZB** tests SWIGLU_QUANT (the gate/up path) and COMBINE_WRITE (combine / hc write).
- **Candidates by size:** the gate/up GEMM (21%), GDN prompt recurrence + gdn (19%), QSA attention (13%), the HC
  reads (10%).

**Correction (16:45): the GDN rollback-snapshot candidate is DEAD for the live engine.** ziz's snapshot writes
(`gdn_core.cuh`, "~5-8%" in the swarm intake above) are in `src/split/`, the opt-in strata-split engine
(`STRATA_BUILD_SPLIT`, OFF, not in build-hip). The coordinator endorsed it without checking which binary the file
belongs to. The live verify path already replays kept rows from stored per-row inputs (`gdn_commit_layers`).

The live-path remainder is PlSnap: card 0's GDN state is copied D2D once per speculative window on a side stream,
and restored on doomed windows. Removing it with ping-pong parity is a medium build, low rank (tuck, run 2).

## Window EPF: early PLE flag against the stalls (16:55)

Raw: `strata-ab/windowepf.out` (to be copied as `qwen38-windowEPF/`). Earlyple build fae158d9 (main 9d0a5e5b0 +
`STRATA_PL_EARLY_PLE_FLAG` / `STRATA_PL_LAUNCH_SLOW_MS`), live IQ3_S config, 4 agent + 4 edit passes per mode,
interleaved, `STRATA_PL_LAUNCH_SLOW_MS=50` on every arm. Text matches window STALL in all 16 arms.

| set | agent tok/s pooled | gaps >300 ms (sum) | edit tok/s pooled | gaps (sum) | #267 | slow lines |
|---|---|---|---|---|---|---|
| e0 (flag after launch, today) | 141.7 | 6 (7185 ms) | 334.5 | 4 (3228 ms) | 0 | 0 |
| e1 (flag before launch) | 143.7 (+1.4%) | 4 (4116 ms) | 303.0 (-9.4%) | 4 (5277 ms) | 0 | 0 |

- **The early flag does not remove the stalls.** e1 has a gap in every agent arm (351-1368 ms) and a 2830 ms edit gap.
  The tok/s deltas are the gaps moving around (edit -9.4% is one 2.8 s gap in 3600 tokens).
- **The slow-launch logger saw nothing:** no `cudaGraphLaunch`, event record, early-flag row wait or commit wait+launch
  took >50 ms in 16 gaps of 0.35-2.8 s. With window STALL (host not blocked off-CPU, no reclaim) the host is
  **polling a window the GPU has not finished**: the GPU is late. The HIP-lock-cycle reading is dead for these stalls.
- Next suspect, queued (window HWQ, insert 51 head): **hardware-queue sharing.** The live config pins
  `GPU_MAX_HW_QUEUES=4`, never measured on this config. A card runs well over 4 streams (two verifiers per stage with
  `cs_`/`sh_cs_`/`copy_`, `pl_stream`, the PlSnap side stream, the drafter), so HIP multiplexes them onto 4 in-order
  AQL queues. The zero-doorbell graphs carry device spin-waits (`wait_flag_ge`, `wait_copy_i32`, `doorbell_wait`):
  one spinning at a queue's head holds every later packet of every stream sharing that queue until the host raises
  its flag. HWQ: `GPU_MAX_HW_QUEUES` 4 vs 16, 4+4 passes each, gap census, kernel log checked for runlist
  oversubscription (only 2 lines today, both 09:53, none during the afternoon stall runs). Run 1 nyira's branch
  ("a clean HSA poll => look at the queue count"), tuck run 2.
- Not merged: the early flag stays on its branch (basalt38/pl-early-ple) as a logger.

## Swarm intake, run 2 late chat (17:00)

The run's final synthesis lead (gruff, gpt-6-astra): adopt s2gptq after ZB; the top fresh code candidate is the
full-A rollback GDN skip; HANDOFF_PIPE is the cheapest new prompt probe. Taken and queued:

- **Full-A rollback GDN skip** (bakari; source audit gruff, pablo, poppy): a wrong guess after A was taken whole
  (undo_keep == A.T) restores stage 0's GDN snapshot and replays the same A.T commit the stage already made before B
  launched; B's window writes no GDN state or conv history (always published: no self commit; not a batch window).
  Eligible: 779/3215 verdicts (24.2%) in wl16c, 722/3176 in ws_agent_2 (F1-keyed counts, two parsers agree).
  Built as `STRATA_PL_UNDO_KEEP_GDN` (branch basalt38/pl-undo-keep-gdn bdb6954d0, default off): skips the restore and
  runs the commit graph with GDN count 0 (the GDN/conv commit kernels return at once); QSA indexer and PLE history
  still replay. Window UKG (after HWQ): off/on, agent x3 + edit x2, text must match, trace read of stage-0 L0->F0 of
  the fresh window after a kept vs partial rollback. Coordinator's caveat (inference): the undo runs on stage 0 while
  card 1 drafts the fresh window's chain (launched after the verdict), so it may already hide under the chain; the
  trace read says which.
- **`STRATA_HANDOFF_PIPE`** (twig): the opt-in pieced stage hand-off + pinned worker for prompts (exact), never run
  on the ledger. Window ZB prompt arms hp4/hp8 added.
- **Profiler observer effect** (gruff, bikari, sabyinyo): `STRATA_VERIFY_PROFILE` disables conv/GDN fusion,
  IDX_PAIR, gateQ8, the shared stream and combine fusion. zb_prof is a per-kernel-class ranking only, never live
  shares. Low-perturbation alternatives in tree: `STRATA_VERIFY_NODES`, the `trace_m_` breadcrumbs. Installed
  rocprofv3 has `--hip-graph-trace` and lists gfx1030; graph-node attribution needs segmented scheduling,
  `AMD_DIRECT_DISPATCH=1`, a sole launcher thread and no GraphExecUpdate.
- **No-guess windows are slower than wrong guesses** (24.7 vs 22.2 ms; cantsbee): fresh/no-guess windows get
  `fresh_chain_n`, launched after the verdict, so they pay the chain serially. Same family as chain-late (window CHN).
- **Drafter per-step bytes** (mtpq agent, from rt/dense.txt and the live log): dense ~111 MiB (bf16 HC mixers 37.5,
  q_proj Q8_0 31.9, o_proj 15.9, fc 13.3, shared expert 5, k/v/indexer/router ~7), head proxy ~31 MiB (2-bit fitted
  over 40525 rows), routed experts ~13 MiB. `--mtp-q4 head` would turn the proxy off and read more (dropped from ZB).
  Levers: a 16-24K draft vocab (proxy scales with rows; offline study approved, no code: `--mtp-draft-vocab`) and Q8_0
  HC mixers (~-19 MiB/step, kernel change in the fused gr_read path, not started).
- **Node cost on this card** (tuck): AMD_HIP.md has an on-box V620 captured-graph number: empty kernel 2.6 us, a
  verify window ~750 nodes/card. Scale reference only. The historical three-switch bundle (HEAD_WIDE, ROUTE_PLAN,
  HEAD_SAMPLE) is already in the live config; ROUTE_PLAN=2 was ~0.2% there (noise).
- **IQ 4->5-row cliff is per expert group** (kivu, mikeno): the bench gives every group T entries, so 85.6->125.3 us
  is a maximal-sharing synthetic. Unpriced until a live route-group multiplicity histogram.
- **Arithmetic checks** (ndume): stage slope is 0.91 ms/row T1->T3 but 1.42 ms/row T3->T8; the full verify tree
  loses at either slope; the cycle table lacks tokens per window for the no-B class (it cannot be reconciled to
  149 tok/s without it).
- **#267 recovery** (nyira): two older #267 logs show the release did not free the GPU in 5 s, so in-process recovery
  is impossible there; respawn + replay in serve/server.py is the recovery road (MCP side effects make replay unsafe
  for tool calls; bikari).
- **Prompt warm-up** (gruff, sabyinyo): report cold first-chunk and warm steady-state separately; prewarming moves
  cost before READY, not speed.
- Longshots, not queued: exact small-alphabet IQ2_S decoder (pip; CPU-exhaustive parity on 32768 cases, no GPU
  number), progressive 3->4->exact head screens (kaje; 3-bit screens keep 42-113 of ~124 rows on CPU samples, misses
  the byte break-even), wider GDN final-state banks.
- **`STRATA_SPEC_COST` does not reach the pipelined drafter** (bakari, run 2 final; coordinator-checked): the cost rule
  is `mtp.set_chain_rule`, called only in the serial `MtpDrafter::draft`; `chain_launch`/`chain_poll` never call it.
  Under `--pipeline-windows 2` the flag only sets the robust lookup policy and turns `STRATA_DRAFT_CUM` off. Audit
  window 18's `w18_mtp3_cost` arm is annotated: a null there rejects nothing. A real pipelined cost rule would pick
  the width from probabilities already on hand, with no per-step host wait.
- `STRATA_VERIFY_PROFILE=0` still turns the profiler on (`getenv != nullptr`); control arms must unset it. ZB's speed
  arms never set it.
- Main-model head on card 1, isolated (kahuzi; mtpq bench `basalt-head/q6-final.log`): Q6_K 248320x2560 (521 MB),
  HEAD_WIDE T=1 1.007 ms (518 GB/s), T=8 1.31 ms. Scale only; it cannot be subtracted from overlapped stage intervals.

## Window CHN: shorter forced chains for the guessed window (17:07)

Raw: `strata-ab/windowchn.out`, arms `wch_*`. Main build-hip, live IQ3_S config, `STRATA_PL_GUESS_T` (clamps the
guessed window's forced chain), agent x2 + edit x1, pipeline traces + DECODE_TIMING. Text matches live in every arm.

| arm | agent p1 / p2 tok/s | stage-0 idle | chain wait | speculative window | edit tok/s |
|---|---|---|---|---|---|
| live | 148.8 / 147.2 | 18.4-19.2% | 10.2-10.3% | 11.5-12.5 ms, 2.1-2.75 tok | 368.1 |
| GUESS_T=2 | 140.9 / 137.6 (-5 to -7%) | 14.3-15.4% | 8.4-9.6% | 10.4-12.8 ms, 1.7-3.1 tok | 357.8 |
| GUESS_T=1 | 115.2 / 109.6 (-23 to -26%) | 9.3-9.7% | 7.6-7.9% | 8.9 ms, 1.0 tok | 367.2 |

- **Dead as a knob.** Shorter guessed windows cut stage-0 idle but lose far more tokens per ms (a 1-row window still
  costs ~8.9 ms: the ~8 ms fixed stage cost).
- **The chain wait is mostly fixed latency, not drafter steps.** With a 1-step forced chain, 99% of speculative
  windows are still "chain late" and chain wait is still 7.6-7.9% of agent decode (vs 10.3% live). About 3/4 of the
  wait does not scale with chain length. Per-step drafter savings (draft vocab, Q8_0 mixers, S2Q4D) can reach only
  the ~1/4 that does: a few tenths of a percent each, not the 10%.
- Inference, to check in source next: the fixed part is the path from the verdict to the guessed chain's outputs
  reaching stage 0 (host poll of the verdict, chain launch on card 1 behind stage 1's verify kernels on the same
  card, host poll of the outputs, stage-0 launch). The pipeline log reports 0.005 ms host time per chain launch, so it
  is GPU-side waiting or poll cadence, not launch cost. Candidate road: start the guessed chain before the verdict
  (from the lookup or the previous chain's tail), or give the drafter its own CU partition so it does not queue
  behind stage 1.
- **Draft vocabulary below 40K: dead** (mtpq agent offline study, commit 75555417d in its tree; exact Q6_K replay
  matches the GPU's proxy+rescore top-1 on 99.8% of agent B steps, 100% of edit). Gated E[L] vs 40,525 rows at
  p >= 0.4/0.5/0.7, held-out agent B: 16K -7.8/-8.0/-7.5%, 24K -6.3/-6.1/-5.9%, 32K -1.8/-1.7/-1.8%; edit: -23%,
  -16%, -10% at 16/24/32K. Edits copy identifiers from the input (out-of-vocabulary 1.9% at 40K, 7.0% at 24K), so a
  fixed cut loses there; draft context rows would cover them but switch the 2-bit proxy off. With CHN pricing per-step
  savings at tenths of a percent, closed.
- Offline gate curve for window 18 (live drafter, full vocab): gated E[L] agent B 3.570 / 3.421 / 2.956 at
  p >= 0.4 / 0.5 / 0.7; edit 5.295 / 5.202 / 5.029. Longer accepted runs at a lower gate cost more verified rows
  (0.9-1.4 ms/row); window 18's engine arms decide.

## Swarm run 2 final (gruff, 17:13; saved as docs/swarm/reports/gruff.md from KING's paste)

Its top three map onto the queue as follows:
1. **Full-A rollback GDN skip** — built (`STRATA_PL_UNDO_KEEP_GDN`, bdb6954d0), window UKG queued. It skips both
   the restore and the GDN replay (count word 0), as the final requires. Parity: with undo_keep == A.T the kept state
   is the output of the same commit kernel on the same rows from the same starting state as the replay, so it is
   identical by construction wherever B wrote nothing; text match over ~50K tokens of agent+edit greedy decoding
   (hundreds of kept rollbacks) tests that end to end. Forced-miss/EOS/cancel cases are covered by the same identity.
2. **HANDOFF_PIPE** — ZB prompt arms now 2, 4 and 8. A 118K follow-up only if one wins.
3. **Cost-aware width selection reaching the pipelined MTP** — code road, not queued. CHN adds a constraint: a
   guessed window's fixed ~8 ms makes wider windows cheaper per token (1-row guessed windows: 8.9 ms for 1.0 tok;
   live 3-row: ~12 ms for 2.1-2.75 tok), so any cost rule must use V620 per-class costs, not a per-row model.
Dissents noted: COHERENT stays cut (CLR source; DOORBELL_STORE runs alone). Replay after a #267 respawn needs output
and tool idempotency first (server-side MCP actions).

## Window COH, DOORBELL_STORE arm (17:17)

Raw: `strata-ab/windowcoh.out`, arms `wc_dbs_*`. Main build-hip, live config + `STRATA_DOORBELL_STORE=1`, 4 agent + 4
edit passes, traces. The COHERENT arm was cut (CLR maps the memory coherent already). Text matches window STALL.

| set | agent tok/s pooled | gaps >300 ms | edit tok/s pooled | gaps |
|---|---|---|---|---|
| dbs | 145.6 | 3 in 4 arms (999; 1818, 904) | 368.8 | 0 in 4 arms |
| EPF e0 control (same config, earlyple binary, 16:55) | 141.7 | 6 in 4 arms | 334.5 | 4 in 4 arms |

- Fewer gaps (3 in 8 arms vs a census of ~0.74 per arm, p ~0.13 by Poisson): suggestive, not significant. Not
  adopted on this alone; it rides along if HWQ (running now) leaves the stalls.

## Window HWQ: GPU_MAX_HW_QUEUES 4 vs 16 (17:17-17:31, stopped)

Raw: `strata-ab/windowhwq.out`, arms `hq_*`. q4 (live) arms ran normally (hq_q4_edit_1 368.8 tok/s). **With 16 queues
the engine ran at 1.8 tok/s** (warm-up 55-token prompt 9.6 tok/s, 16 tokens in 10 s; the first agent case at
1159/1200 tokens after 637 s), and the kernel log printed "Runlist is getting oversubscribed due to too many queues"
46 times in two minutes, all on card 0 (03:00.0). Stopped by the coordinator after the first 16-queue arm.

- Reading (inference): past the HWS runlist's slots the scheduler time-slices queues, and a queue whose head kernel
  spin-waits on a host flag (the zero-doorbell `wait_flag_ge` family) burns its slices while the work it waits for is
  descheduled. That is the stall mechanism in its extreme form, and it shows that queue mapping and spin-waits interact
  on this card. The live 4 does not oversubscribe (no kernel lines in any 4-queue run today), so the stalls at 4 would
  be the in-order sharing form, not time-slicing.
- Window HWQ2 (head of insert 53): probes 8, 6, 5 with a watchdog that kills a probe on the first oversubscription
  line, then a 4+4-pass A/B of 4 against the largest count that fits.

## Window UKG: full-A rollback GDN skip (17:36-17:48)

Raw: `strata-ab/windowukg.out`, arms `ukg_*`. Branch basalt38/pl-undo-keep-gdn bdb6954d0 (binary 575a3a51), live
IQ3_S config, off/on via `STRATA_PL_UNDO_KEEP_GDN`, agent x3 + edit x2, traces + DECODE_TIMING.

| set | agent p1/p2/p3 tok/s | kept / rollbacks per agent pass | edit p1/p2 tok/s |
|---|---|---|---|
| off | 149.0 / 149.0 / 148.9 | 0 / 1007 | 368.6 / 369.3 |
| on | 149.1 / 149.2 / 149.0 | 719 / 1007 | 368.8 / 306.6 (one 2041 ms gap) |

- **Exact:** text matches off in all 10 arms, through 2157 kept rollbacks on agent and 62 on edit.
- **Worth ~nothing:** stage-0 L0->F0 of the fresh window after a kept rollback drops 7.91->7.88 ms (T1), 8.75->8.73
  (T2), 9.71->9.67 (T3); means 0.02-0.08 ms. After a partial rollback (control): unchanged. The restore and replay
  were already hidden: they run on card 0 while card 1 drafts the fresh window's chain (coordinator's caveat). Agent
  +0.1%, inside noise. Not merged; the branch stays as a record.
- **Side finding, the big one:** 1 stall gap in these 10 arms (the census is ~0.74 per arm). See the stall-rate
  section below.

## Stall rate tracks host load and request starts (17:50-17:58, from existing traces + a live kernel trace)

- **By time of day** (all of today's traced arms, the 16-queue arm excluded): 0.24 gaps per arm 11:30-14:30 (quiet),
  ~1.0 in 15:00-15:30 (swarm run 1 + the #267 pressure tests) and 16:30-17:13 (swarm run 2), 0.07 after the swarm
  ended (17:14-17:48: UKG 1 in 10, COH tail and HWQ q4 0). The swarm ran ~23 agent processes on this box.
- **By request position** (435 requests, 61 gaps): 26 gaps (43%) start within the first second of decode (windows
  1-14). Early gaps by request number: 9 / 4 / 5 / 8 / 0 / 0 for requests 1-6 (agent suites have 6, edit 3). Requests
  1-4 follow a conversation-cache park that allocates new memory (the cache holds 4; from the 5th park on it evicts
  one and reuses its memory); requests 5-6 never had an early gap. Late gaps: 7/5/4/5/3/11.
- One request shows the whole episode: we_e0_agent_3 parked request 2 in 488 ms (others 24-30 ms), read request
  3's 297-token prompt in 878 ms (others 420-490 ms), then stalled 2424 ms 147 ms into decode.
- **Live kernel trace** (strata-ab/evict.bt from 17:49): kcompactd0 calls `amdgpu_amdkfd_evict_userptr` (every queue
  of the engine quiesced) in bursts of 7-8 within 0.4-0.7 s, restores 0-43 ms, while each new engine loads; none
  seen during decode on the quiet box so far, and the one ZB gap (zb_nofuse_1, 1441 ms, request 4, 157 ms into
  decode) had no userptr eviction within 0.5 s. The box: 2 GB available, strata RSS 82.9 GiB, Mlocked 44 MB, THP
  always, compaction_proactiveness 20; kcompactd moved 2.5M pages in one 30 s tick during a load.
- Working hypothesis (inference): the park's host allocations (ConversationBuffer segments, pinned staging, and a
  VRAM stage that can be freed and re-allocated: cudaFree syncs the device) under RAM/VRAM pressure push TTM or the
  MMU notifier into evicting the process; a full KFD restore re-validates every BO including the 46.84 GiB userptr
  arena. Second probe `evict2.bt` (eviction fence + `amdgpu_amdkfd_gpuvm_restore_process_bos` duration) running from
  17:56. Window EVP (forced compaction during decode) is first in insert 53.

## Window ZB, drafter arm: s2gptq-a ADOPTED (18:12)

zb_s2g_1/2 vs zb_live_1/2 (main build-hip d4d10152, live IQ3_S config, only `--mtp` differs): 151.9 / 151.7 vs
148.8 / 148.9 tok/s (+2.1% / +1.9%); held-out 4 cases +1.5% (150.0 vs 147.8), calibration cases +4.3% / +1.7%. Text
identical in all 6 cases both passes; accepted drafts up in every case; gates hold (CUDA1 11776 slots, 4 zero-doorbell
graphs, 100% resident); drafter 839 MiB and CUDA1 free 1712 MiB in both (no VRAM change). No stalls (each case within
0.1% across passes). Matches AB1 on the mtpq binary (+2.0%, held-out +1.4%). Read by the mtpq agent, checked by the
coordinator.

- **Adopted:** `strata-sc117-IQ3_S-qwen-abliterated.json` (gitignored live config) `--mtp` now
  `~/Strata-data/mtp/q38-s2-gptq-a` (was `~/Strata-data/mtp/rt`). Revert = that one line.
- The harness copy `strata-ab/qwen-iq3s.json` stays on rt for the rest of today's queue, so windows stay comparable
  with today's ledger; refresh it at the next natural break.
- Other configs still on rt (swift Q4_0 x5, instruct IQ3_XXS) are other packs: untested with this drafter, unchanged.

## Window ZB, the rest (17:17-18:17; read 18:22)

Main build-hip d4d10152, harness IQ3_S config (rt drafter), agent x2 per arm, 36K prompt x2 runs per load.

| arm | what | agent p1 / p2 tok/s | text vs live | verdict |
|---|---|---|---|---|
| live | control | 148.8 / 148.9 | 6/6 | - |
| sh0 | STRATA_SH_STREAM=0 | 130.3 / 130.2 | 6/6 | the live knob is worth +14%: keep |
| nofuse | STRATA_QFUSE=0 + STRATA_NODE_FUSE=0 | 140.3 / 144.4 | 6/6 | the two live fusions are worth +3-6% (p1 had one 1441 ms gap): keep |
| hcp | nofuse + STRATA_HC_PERSIST=1 | 147.3 / 147.3 | 0/6 | +2-5% over nofuse but text differs in every case, and below live: dead |
| cmpl | `--resident-experts` | 148.8 / 148.2 | 6/6 | speed-neutral, and NO 46.84 GiB host arena (below) |
| s2g | s2gptq-a drafter | 151.9 / 151.7 | 6/6 | adopted (section above) |

36K prompt (run1 / run2 tok/s): ctl 2715.9 / 2734.9; swq 2721.9 / 2539.4 (one dip); cmw 2715.3 / 2738.3; cmpl
2714.0 / 2730.8; HANDOFF_PIPE=2/4/8 2713.2 / 2736.0, 2716.8 / 2738.0, 2714.8 / 2738.3. Every prompt knob is flat:
PREFILL_SWIGLU_QUANT, PREFILL_COMBINE_WRITE and HANDOFF_PIPE are dead on this config.

- **`--resident-experts` drops the expert arena.** zb_live_1's log has `expert arena: cudaHostRegister PORTABLE ok`
  (46.84 GiB registered userptr, THP always, not mlocked); zb_cmpl_1's has none of it, only `locked resident cache
  complement ready: resident 0.00 GiB, pinned 0.00 GiB` plus 0.47 GiB of pinned exchange buffers. With all 24576
  expert pairs in VRAM (12800 + 11776 slots) the arena is a RAM copy nothing reads at steady state; the prompt path's
  streamed ring is unaffected (cmpl prompt = ctl). Side effect: `exchange rotation unavailable` (no-op at 100%
  resident). This is the strongest stall-fix candidate: ~47 GB less RAM on a box running at ~2 GB available, and the
  arena is exactly what KFD userptr eviction and restore walk. Window RXP (queued) checks memory, reclaim and gaps.
- zb_prof (STRATA profile on, 115.0 tok/s, profiling overhead): steady windows 19.1-20.6 ms, avg T 2.29-2.80, 2.03-2.67
  tokens/window; stage 0 GPU: VRAM hits 1.96-2.98 ms, q8+qkv gemv ~1.05, out-proj 0.84, z 0.68; stage 1 ~10% lighter.

## Window EVP: forced compaction makes stalls, but mostly not through userptr eviction (done 18:34)

Main build-hip d4d10152, harness IQ3_S config, traces on. Sets: q quiet, p = `echo 1 > /proc/sys/vm/compact_memory`
every 10 s for the whole arm, c = `--conversation-cache-mib 0`. Agent x3 + edit x2 per set, interleaved.

| set | agent tok/s | agent gaps > 300 ms | edit tok/s | edit gaps |
|---|---|---|---|---|
| q | 148.9 | 0 | 345.1 | 1 (1346 ms, ev_q_edit_1 request 3) |
| p | 146.4 (-1.7%) | 5 (400, 508, 425 / 386 / 486 ms; sum 2205) | 368.5 | 0 |
| c | 148.9 | 0 | 368.2 | 0 |

- Forced compaction causes stalls: 5 gaps in 3 agent arms vs 0 quiet. They are shorter (386-508 ms) than the
  natural 1-4 s gaps.
- Only 1 of the 5 sits next to a KFD userptr eviction (18:20:06.424, in the `tee` writing compact_memory, 2 ms after
  the gap started). The other 4 have no eviction or restore within -0.5/+0.1 s. The second probe (evict2.bt: the
  eviction fence, process suspend, full BO restore) logged NOTHING in 40 min, loads and teardowns included: that path
  never fires here. So compaction stalls the pipeline mostly some other way (inference: page migration of memory the
  host threads or the GPU touch: migration entries block the faulting thread; the decode host threads were seen in
  direct compaction/reclaim, compact.bt).
- The conversation cache off (c) changes nothing on a quiet box (both 0 gaps); its edit runs match p's speed.
- The quiet edit gap (1346 ms, request 3 of ev_q_edit_1) had no eviction nearby either.
- Quiet agent arms had 0 gaps, so a quiet-only fix test proves nothing: window RXP was changed to run live and
  `--resident-experts` both quiet and under compact_memory every 10 s (16 arms).

## Window HWQ2: 4 hardware queues is the ceiling (18:34-18:36)

GPU_MAX_HW_QUEUES 8, 6 and 5 each oversubscribed card 0's HWS runlist on the first probe (watchdog killed each run).
4 (live) is the largest count that fits. Window HWQ3 (3 and 2 vs 4, agent x2, only because HWQ2 found nothing above 4)
is appended after RXP. Inference to check: the vision process (strata-vision on GPU) also holds card-0 queues.
- KFD sysfs during an AB2 arm (18:37): the engine holds 5 compute + 1 SDMA queues on EACH card at
  GPU_MAX_HW_QUEUES=4, and the picture reader (strata-vision, started by the config's "vision" section, idle in every
  benchmark) holds 3 compute + 1 SDMA on card 0. So card 0 runs 8 compute + 2 SDMA, and one more engine queue
  oversubscribes. HWQ3 part B runs without the reader (config copy minus the "vision" section; the engine keeps
  --vision, so the engine run is identical): 4 as control, then probes 6 and 5. If more queues help, the trade is the
  reader on CPU / on demand / on a spare card (`"vision": {"cuda_device": N}` moves it).

## Window AB2 (mtpq drafters, 18:37-18:58)

mtpq branch binary 563dbb2f, harness IQ3_S config, only `--mtp` differs; agent x2 + edit x2 per drafter, interleaved.

| drafter | MiB | agent tok/s (pooled, vs live) | held-out agent | edit tok/s (vs live) | text | gates |
|---|---|---|---|---|---|---|
| live rt | 839 | 148.94 | 147.94 | 368.40 | - | OK |
| s2gptq-a (adopted) | 839 | 151.89 (+2.0%) | 150.00 (+1.4%) | 366.15 (-0.6%) | 6/6, 3/3 | OK |
| q4n | 1514 | 152.68 (+2.5%) | 151.82 (+2.6%) | 367.19 (-0.3%) | 6/6, 3/3 | OK |
| q4g | 1514 | 150.35 (+0.9%; p2 148.4) | 150.75 (+1.9%) | 356.04 (-3.4%; p2 345.2) | 6/6, 3/3 | OK |
| king (KING's quant) | - | 143.48 (-3.7%) | 142.79 | 361.21 (-2.0%) | 1/6, 3/3 | FAIL: CUDA1 11705 slots, 2 zero-doorbell graphs |

- **s2gptq-a edit -0.6% is ONE draw, not a measured loss** (correction, 19:05). 366.2 / 366.1 vs 368.6 / 368.2 comes from
  7 accepted tokens: 3097 vs 3104 of 3600, 503 vs 496 rounds. Greedy decoding with the same drafts makes the two passes
  identical replays (they differ only in timing), so 3 cases are one sample, and a 7-token swing is the size of the
  draft drift any weight change causes. Offline edit can't settle it either: same 3 cases (959 steps, s2gptq-a +2/-1
  vs live), and the replay is teacher-forced on live's chain states, so deep-chain drift (edit runs 7.2 tok/round) is
  invisible. Settling it: a 24-case edit set from llama.cpp and forgecode files (mtpq agent), capture plus an engine
  A/B of rt vs s2gptq-a, appended after the corpus capture. The adoption stays until that reads.
- q4n is the best drafter here (+2.5% agent, -0.3% edit) but takes +675 MiB on card 1: blocked by the no-VRAM-growth
  rule. q4g's pass-2 dips look like stalls (single-pass drops of 2.6% and 6.1%).
- KING's quant pushes the expert cache below full residency (11705 of 11776 slots on CUDA1, so only 2 zero-doorbell
  graphs) and changes the text: every mtpq drafter beats it.
- The IQ2_XXS drafter A/B (mtpqi_*) started 18:58.

## Window mtpqi: IQ2_XXS gate/up drafter (18:58-19:11)

mtpq branch binary 563dbb2f (all arms), harness IQ3_S config, only `--mtp` differs; agent x2 + edit x2.

| drafter | loaded MiB / CUDA1 free | agent tok/s (vs live) | held-out agent | edit tok/s (3-case, one draw) |
|---|---|---|---|---|
| live rt | 839 / 1712 | 148.90 | 147.88 | 368.18 |
| s2gptq-a | 839 / 1712 | 151.82 (+2.0%) | 150.01 (+1.4%) | 366.08 (-0.6%) |
| iq2g (q38-native-iq2xxs-gptq-a) | 802 / 1750 | 150.50 (+1.1%) | 149.08 (+0.8%) | 370.48 (+0.6%) |

Text identical everywhere, gates OK, no gaps. On agent, iq2g sits between live and s2gptq-a; its offline +2.8% did not carry
over in full. It frees 38 MiB on card 1. Edit differences on 3 cases are single draws (see the AB2 correction). The
24-case edit A/B (live / s2gptq on main, livebr / iq2g on the branch, per-case paired accepted tokens with a sign test)
and its capture are appended to insert 54.
- Paired accepted tokens per case, pass 1 (mtpq agent's reader): agent s2gptq vs live 6 won / 0 lost (sign test p 0.031,
  +1.66%); iq2g vs live 5 / 0 / 1 tied (p 0.062, +1.24%); iq2g vs s2gptq 2 / 4 (p 0.69): the two don't separate on
  agent. Edit (3 cases): iq2g vs s2gptq +9 tokens, 2 / 0 / 1; undecided until edit24. The offline replay ranked iq2g
  well above s2gptq on agent; the engine has them level: the replay picks direction, not small margins.
- s2gptq's agent acceptance moved between passes on 2 cases with identical text (696 / 688, 748 / 746); live and iq2g
  repeated exactly. Inference: in the pipelined engine, acceptance can depend on timing (a chain that lands late, or
  falls below the gate, changes which drafts get verified), so "acceptance is deterministic" holds only to ~0.1%.

## Window NTV: mtp/native-experts engine parity HOLDS (19:16-19:23)

Branch mtp/native-experts 31398223c built in the slot (binary 0bf25b59; 11/11 ctests, mtp_experts_parity included).
(a) On live rt the branch equals main's binary case for case, agent 6/6 and edit 3/3 (text md5, draft_n, accepted).
(b) q38-native-iq2xxs-gptq-a loads as `native, 0 S2 Q2_0 + 512 native (ggml types gate/up 16, down 42), 637.5 MiB`,
text equals live 9/9, and draft counts equal the mtpq worktree binary's mtpqi_iq2g runs 9/9. Gates OK on all six
runs; one 1343 ms gap (ntv_br_live_edit). The branch is merge-ready, but it only matters once a native-format drafter
is adopted. Not merged yet: AB3 runs on the branch binary and decides that.

## #267-class hang in decode (mtpq edit24 capture, 19:30; first misread as a prompt read)

ab/mtpq_cap_edit24 (mtpq branch binary 563dbb2f with the capture hook, harness IQ3_S config, rt): request 4
(wedit_json_config_2, 1690-token prompt) hung reading its prompt: "no progress for 60 s ... waiting for the GPU
(attention, router) at layer 3 of the prompt chunk from token 1024", and the engine stopped itself (exit -6); the
server restarted it, and the other 23 cases completed. Stall report: stage 1 odd verifier "IN FLIGHT T=3 pos 1910 ...
GPU rang 0, flags served 1 A 0 B 0; window event PENDING", host idle 93 s; on release "the GPU finished in 6 ms". So a
zero-doorbell verify graph sat on card 1 waiting for host flags (plan, copies = 0) while the host waited for prefill
work stuck behind it. The previous request's trace has every L1 matched by an F1. Main's binary ran the same prompts in
the same order (mtpqe_live_edit24_p1) without hanging, so it is not deterministic. Same "GPU finished in 0 ms"
signature as the natural #267 deaths.
- Working hypothesis (inference, under test): more streams than the 4 HW queues per card means ROCclr maps several
  streams onto one in-order queue. A parked zero-doorbell graph blocks other streams' work behind it on that queue:
  a deadlock if the host waits on that work (#267), a stall if the flag comes later anyway (longer under compaction).
- An Opus xhigh agent (background subagent in this session) is reading the code: graph enqueue vs flag timing, which
  streams exist per card, ROCclr's stream-to-queue mapping. It delivers a fix branch (basalt38/parked-graph), an
  in-slot builder, and a staged repro plus A/B window. The capture's 23 good cases are usable offline.
- **Correction (from the capture): the hang was in DECODE, not the prompt read.** The capture holds 214 emitted tokens of
  wedit_json_config_2 at cells 1697-1910 (prompt ends ~1696) and 334 drafter records at cells 1696-1913, so the
  1690-token prefill finished and the engine hung 214 tokens into decode. The dump's "pos 1910" is that request's last
  emitted token, and the watchdog's phase label ("reading the prompt (batched) ... chunk from token 1024") was stale, a
  second, smaller bug. The shape is classic #267: the stage 1 odd window in flight with its plan and copies flags never
  raised while the host sat idle. The edit24 capture was repaired (the restarted engine's seq/request ids renumbered;
  mtpq/cap/edit24, raw copy in edit24-raw): 23 full cases plus 214 tokens of the 24th.

## edit24 A/B: s2gptq-a does NOT cost edits (19:37-19:54)

24 edit cases (mtpq/prompts/edit24.json, llama.cpp + forgecode files), 2 passes. live / s2gptq on main's build-hip,
livebr / iq2g on the mtpq branch binary. Per-case paired accepted tokens (mean over passes; |delta| <= 2 = tie):

| pair | won / lost / tied | sign test p | accepted |
|---|---|---|---|
| s2gptq vs live | 4 / 3 / 17 | 1.000 | 22971.5 vs 22973.5 (-0.01%) |
| iq2g vs live | 6 / 3 / 15 | 0.508 | 22934.0 vs 22973.5 (-0.17%) |
| iq2g vs livebr | 5 / 3 / 16 | 0.727 | +0.12% |
| livebr vs live (same drafter, two binaries) | 1 / 3 / 20 | 0.625 | -0.30% |

- **The AB2 edit -0.6% was a single draw: no edit cost.** s2gptq-a stays adopted with no caveat (+2.0% agent).
- Text identical to live in all 24 cases for every arm.
- Throughput across arms (live 300.0, s2gptq 311.0, livebr 304.5, iq2g 309.9 tok/s) is NOT drafter signal. Pass
  order and stalls dominate it: s2gptq's +3.6% is mostly its pass 2 (312.2 vs 300.3), and live's pass 2 had a #267
  restart.
- Acceptance drifts between passes much more than 0.1% on some JSON cases (livebr json_config_2 912 / 802, iq2g
  json_config_1 865 / 952) with identical text. Timing-dependent drafting (the suffix/lookup drafter runs beside the
  MTP; late chains) makes per-case acceptance noisy, so pairs need means over passes and ties, as the reader does.
- **#267 on main's binary:** mtpqe_live_edit24_p2, wedit_py_rename_4, at decode start after a 953-token prompt:
  "verify: the window never finished; its GPU waits were released but the GPU did not finish within 5 s", releases
  finished in 3/0/1/0 ms, engine exit 1, restarted. That's 2 #267-class failures in ~216 edit24 requests within 30 min:
  edit24 is a usable repro set (handed to the parked-graph agent).

## Audit window 18: draft confidence x chain length (19:54-20:05)

Main build-hip, harness IQ3_S config (rt drafter), one pass each, agent unless noted (config-audit/runs/w18_*).

| arm | tok/s | drafts accepted / drafted |
|---|---|---|
| mtp-max-t 3, min-p 0.70 (live) | 148.1 | 4030 / 4760 |
| mtp-max-t 3, min-p 0.50 | 145.0 (-2.1%) | 4230 / 5437 |
| mtp-max-t 3, min-p 0.40 | 142.1 (-4.1%) | 4258 / 5749 |
| mtp-max-t 2, min-p 0.70 | 139.1 (-6.1%) | 3349 / 3959 |
| mtp-max-t 2, min-p 0.50 | 135.1 | 3496 / 4344 |
| mtp-max-t 3, min-p 0.70, STRATA_SPEC_COST=1 | 141.1 (-4.7%) | 4030 / 4798 |
| edits, --spec 8 (live) | 368.6 | 3104 / 3265 |
| edits, --spec 5 | 303.3 | 2798 / 2997 |

The live settings are the best tested. A lower min-p accepts more drafts but pays for more drafted rows. SPEC_COST
loses 4.7% with the same accepted count: single pass, no trace, so a stall can't be excluded; the pipelined drafter
never calls the `chain_rule_` it changes, so its live effect is indirect at best.

**Correction (21:18, swarm run 3, nyira; checked against the per-case outputs):** the pooled numbers above are not
verdicts for min-p 0.50 or SPEC_COST. Per case, SPEC_COST's loss is one json_patch outlier (+21.8% time with the same
acceptance; median case +1.3%) and min-p 0.50's is one outlier too (median +0.3% while accepting more drafts). Only
min-p 0.40 is slower across cases (median +3.4%). One pass, pre-RXP (eviction stalls possible). Both re-tested post-RXP
with per-case medians and KFD eviction tags in window SW3 (insert 565).

## Window RXP: `--resident-experts` removes the stalls; ADOPTED (20:05-20:27)

Main build-hip, harness IQ3_S config (rt drafter), traces on, interleaved. l = live, r = + `--resident-experts`
(no 46.84 GiB hipHostRegister'd expert arena; all 24576 expert pairs VRAM-resident either way). pl / pr = the same
under `compact_memory` every 10 s. Quiet sets: agent x2 + edit x1; provoked: agent x3 + edit x2.

| set | live tok/s, gaps > 300 ms | resident-experts tok/s, gaps |
|---|---|---|
| agent quiet | 144.3, 3 (2680 ms) | 147.8 (+2.5%), 0 |
| agent compaction | 139.0, 8 (8416 ms) | 148.0 (+6.5%), 0 |
| edit quiet | 368.5, 0 | 355.4 (-3.6%), 1 (365 ms: the whole difference in a 1-arm set) |
| edit compaction | 320.1, 2 (2954 ms) | 367.8 (+14.9%), 0 |

- **Gaps: live 13 (14.05 s) vs resident-experts 1 (0.37 s) over 9 arms each.** Compaction no longer hurts at all.
- Memory per arm: live MemAvailable floor 1.0-1.8 GiB, strata RSS 81 GiB; resident 44.5-71.1 GiB available, RSS
  31-58 GiB. thp_fault_fallback ~13-17k per live arm vs 59-524. KFD userptr EVICT lines inside each arm's span: live
  13901 / 68 / 119421 / 121537 / 57933, resident 66-196 (the ~66 every teardown fires).
- Cost: engine load 52-67 s vs 34-45 s (+15-25 s, one-time per start). The log notes "exchange rotation unavailable"
  (a no-op at full residency).
- Text identical; gates OK; no #267 in either set (none expected at this request count).
- **ADOPTED 20:28:** `--resident-experts` added to `strata-sc117-IQ3_S-qwen-abliterated.json` (gitignored live
  config; revert = delete that one line). Stock Strata parses the same flag, so the queued stock-vs-modded window
  (it snapshots the live config) stays same-flags for both binaries. The harness copy qwen-iq3s.json stays as it is
  for today's comparability.
- Mechanism (open): the registered arena is tied to the stalls even on a quiet box, although few evictions sat inside
  decode spans in EVP. What touches the arena during fully resident decode (adaptive swap/exchange, SDMA) is with the
  parked-graph agent, plus whether #267 itself goes away (edit24 repro arm with the flag).

## ROOT CAUSE of the stalls and #267: KFD queue evictions of GPU-registered host memory (parked-graph agent, 20:40)

Opus xhigh subagent; full report saved at ~/inbox/agents/claude/parked-graph/REPORT.md.
- gfx1030 has no XNACK. When a page of GPU-mapped host memory changes on the CPU side (compaction, reclaim/swap, THP
  split or collapse, a free), KFD evicts EVERY queue of the process on every GPU node (`svm_range_evict` ->
  `kgd2kfd_quiesce_mm`); the restore worker retries every ~2.5 ms and fails while pages keep moving. Nothing runs on
  the GPU until one succeeds. A 20+ s eviction is #267 (flags raised, no wave running to read them); shorter ones are
  the 0.3-4 s window stalls.
- libhsakmt registers every `hipHostRegister` range as an unpinned KFD **SVM** range (HSA_USE_SVM on by default), not a
  userptr. That is why evict.bt (it probed `amdgpu_amdkfd_evict_userptr`) and the EVP gap map saw few evictions: they
  were looking at the wrong path. ROCclr also pins pageable copies over 64 KiB in place (32 MiB chunks), which stay
  GPU-mapped; a 59.2 MiB one is unmapped at every request end, which fits the gaps that cluster at request starts.
- KFD's own event stream (kfd_events.py --all, root, 70 s; coordinator check of the log): 189 evictions with
  trigger=SVM, 12 with trigger=USERPTR, 6288 restore events (mostly failed retries). The serving engine was frozen 7.7%
  of 36 s (longest 793 ms); a loading engine was frozen 10.1 of 31 s.
- The 46.84 GiB arena was the biggest registered range AND kept the box at 1-2 GiB available, so kcompactd, kswapd
  (swappiness 100) and khugepaged churned over every registered page. That is why RXP's `--resident-experts`
  (adopted) took the gaps from 13 to 1. The remaining registered ranges: the KV host copy (`kv_host_alloc`, ~3.09
  GiB), ROCclr's pinned-in-place copy buffers, a few small ones.
- **Correction of my correction:** case 1 (mtpq_cap_edit24) DID hang in the prompt read; the watchdog's label was
  right. Its first snapshot shows no pipelined loop yet; the 214 captured tokens were decoded during the report's 2 s
  sleep, after the queues came back (major faults 655 -> 11097), and the watchdog then killed an engine that had
  recovered. Case 2 (main binary) hung in decode (a 25-30 s eviction). The shared-HW-queue hypothesis is not the
  cause: an eviction stops every queue regardless.
- Branch basalt38/parked-graph (47d396d88): STRATA_KFD_WATCH (own-process KFD event log, no root), STRATA_KFD_EVICT_WAIT
  (the #267 deadlines and the watchdog count only time the GPU could run; the engine waits out an eviction instead of
  dying), a watchdog recheck after its report, the failing request's trace written, decode stage labels,
  STRATA_MLOCK_REGISTERED (from KING's cl-mlock/1020). Builds in insert 55's slot; window PGR (insert 57, ~2.5-3 h):
  edit24 under pressure on main / main + resident / branch, a quiet speed A/B, and diagnostics
  (HSA_SVM_HOST_UNREGISTER_DEBUG, mlock, hugetlb arena).

## Window HWQ3: keep GPU_MAX_HW_QUEUES=4 (20:27-20:44)

Main build-hip, harness IQ3_S config (arena, rt drafter), agent x2 per count, traces.

| set | pass 1 | pass 2 |
|---|---|---|
| A: with the picture reader, q4 (live) | 148.9 | 148.9 |
| A: q3 | 145.9 | 145.1 |
| A: q2 | 146.3 | 146.2 |
| B: no reader, q4 | 146.8 (text 5/6, gap 548 ms) | 141.3 (gaps 328, 1907 ms) |
| B: no reader, q6 (fits: no oversubscription) | 140.7 | 140.0 |

- Fewer queues cost 1.7-2.3%; 6 queues (possible only without the reader) cost 4-6% against A's q4. 4 stays.
- The B runs overlap the start of swarm run 3 (~20:40) and the arena was still in the harness config: their gaps
  and the B-q4 shortfall are not clean signal. B-q4 pass 1 changed the text in 1 of 6 cases against A-q4 (only the
  reader's config section differs; the engine args are identical). Unexplained, single occurrence: watch for it.
- The queue-sharing theory of the stalls is dead anyway (root cause above), so there is no reason to move the reader.

## Window STOCK: upstream Strata vs this fork, same config, same env (21:24-21:37)

Stock = ~/Strata-clean at ffa80d73 (newest upstream), built with main's cmake options against its pinned llama.cpp;
modded = main build-hip (f42ede4d2 tree, binary from 14:59). Both on a copy of the live config taken at window start
(resident experts, s2gptq-a drafter, vision reader, every env var including fork-only knobs, which stock ignores).
Interleaved, 2 passes each. Both loads: 100% of experts resident, MTP drafter on card 1; stock captures 2 zero-doorbell
graphs, the fork 4 (even/odd verifier per stage).

| workload | stock | fork | fork / stock |
|---|---|---|---|
| agent (6 cases x 1200 tok) | 96.9 / 97.0 tok/s | 149.7 / 151.5 tok/s | **1.55x** |
| edits (3 cases x 1200 tok) | 152.2 / 152.1 tok/s | 365.9 / 365.8 tok/s | **2.41x** |
| 36K prompt | 1779 / 1774 / 1774 / 1772 tok/s | 2696 / 2721 / 2706 / 2710 tok/s | **1.52x** |
| decode after the 36K prompt (256 tok) | 74.5-75.4 tok/s | 101.0-108.2 tok/s | 1.38x |

- Text: edits identical in all 6 runs; agent text differs from stock in every case (stock is self-consistent across
  passes, as is the fork). Inference: the fork's numerics-changing kernels (fused HC/node paths, 2-bit head proxy etc.)
  move greedy argmax on long agent outputs; edits are copy-dominated and survive. Not a correctness verdict either way.
- Accepted drafts are similar (agent 4251 vs 4108 of ~4800): the speed is the verify/compute path, not acceptance.
- Output: ~/inbox/agents/claude/strata-ab/windowstock.out, arms ab/st_*.

## MTP drafter quant: corpus-calibrated builds scored; offline replay over-predicts (mtpq agent, 21:41)

Branch basalt38/mtp-quant: table bench/results/2026-11-04-mtpq/offline/corpuscal-table.txt (78c7ffbe6), scorer depth
cap fix a985414d6 (52 edit24 rows had reached depth 8-14 although no chain drafts more than --spec 8; capped, live
reads 4.911 vs the GPU's own 4.913 on the same steps). Gated E[L] at p >= 0.7 vs live, agent B / edit / edit24:

| drafter | agent B | edit | edit24 | MiB |
|---|---|---|---|---|
| bf16 (ceiling) | +3.6% | +2.0% | +3.7% | - |
| s2-gptq-a (live) | +0.7% | +0.7% | +0.9% | 675 |
| s2-gptq-c1 (QAT base) | +2.3% | +2.1% | +1.8% | 675 |
| s2-gptq-c | +1.0% | +1.1% | +1.8% | 675 |
| iq2xxs-gptq-a / -c / rtngu-c | +2.8 / +2.4 / +2.8% | +2.4 / +3.8 / +2.6% | +3.1 / +3.1 / +3.2% | 637.5 |
| q2q4-c | +3.3% | +2.3% | +2.5% | 900 (over the VRAM rule) |
| q4_0-gptq-c | +3.8% | +3.1% | +4.1% | 1350 (over the VRAM rule) |

- **Offline over-predicts the engine.** Same 24 edit cases: offline s2gptq-a +0.9%, iq2g-a +3.1%; engine -0.01% and
  -0.17% accepted (inside the same-drafter floor). On agent the order flips at gate 0.7 (offline iq2g >> s2gptq,
  engine s2gptq +2.0% vs iq2g +1.6% tok/round); gate 0.4 orders them right. The gate itself matches the GPU (78.9% vs
  78.6% of steps >= 0.7; mean conf 0.861 vs 0.859): the gap is method (teacher-forced per-step vs timing, late chains).
- **So: offline differences under ~3% don't pick a drafter; the engine's paired counts do.** BF16 is only +3.6% / +3.7%
  offline and the engine sees roughly half: the whole drafter-quant road at today's VRAM is worth ~2% at most.
- Calibration source isn't the lever (corpus vs agent-A builds about level, lower KL). Per-route S2 quality is: QAT
  (insert 56, running) then AB3 on agent + edit24.

## Pinned-host lane (GPT-6 Astra xhigh via codex, 21:26-22:00): which host memory can freeze the queues

Report: ~/inbox/agents/claude/astra-pinned/REPORT.md (ROCm 10.3.0a20261023 rocm-systems 39fd0ea5 and CachyOS
7.3-rc6 source, fetched). Branch basalt38/pinned-host (02a1049847d3, on basalt38/parked-graph).
- **Correction to ROOT CAUSE:** `kv_host_alloc` on Linux defaults to a host allocation (hipHostMalloc path), not
  hipHostRegister; registration is a fallback or `STRATA_KV_HOST_REGISTER=1`. On this ROCm, ordinary hipHostMalloc /
  paged hsaKmt host memory is a KFD **USERPTR** BO (HMM notifier: page moves evict queues), and hipHostRegister is an SVM
  range. Both freeze the queues on a page move. KFD **GTT** BOs (hsaKmt NonPaged, HIP VMM host) have no user-page
  notifier, but TTM can still evict them under memory pressure (trigger=TTM). mlock and hugetlb are mitigations only:
  notifiers stay; partially mapped mlocked THPs can still be split.
- The 59.2 MiB range unmapped at every request end matches stage 0's **GDN checkpoint** exactly (19 GDN layers x 3.117
  MiB; stage 1's is 53.0 MiB), pinned in place by ROCclr in 32 MiB chunks at `checkpoint_save`. The 6 MiB range is the
  mrope table. The 24 MiB and 0.4 MiB ranges are unattributed. Host KV images: 6.19 GiB trunk + 0.52 GiB per drafter.
- `STRATA_HOST_IMMUNE=1` (default off): HSA_USERPTR_FOR_PAGED_MEM=0 + HSA_SVM_FOR_PAGED_MEM=0 before HIP init (paged
  host allocations become GTT), GPU_PINNED_MIN_XFER_SIZE raised so ROCclr stages instead of pinning in place, KV host
  storage GTT, persistent per-device GDN checkpoint buffers from session start. Ten TUs compile; nothing run.
- Queued: one build slot with gdnlife / iqsigned / pinnedhost (insert 566), then window PIN: main vs branch
  HOST_IMMUNE 0/1, edit24 under compact_memory + page-cache pressure, 6 passes, KFD SVM/USERPTR/TTM counts, text
  identity. PGR (insert 57) trimmed to its quiet phase B: its arena arms test the pre-RXP config.

## Window SW3: run 3's zero-build checks (21:51-23:03; three arms contaminated, re-run queued as SW3B)

Main build-hip, a copy of the live config (resident experts, s2gptq-a, vision), KFD events logged as root on every arm.
**Contamination:** an idle llama-cli (LFM3-1.7B, KING's shell) held GPU queues 22:16:57-22:35:50. Arms
sw3_s_p50_edit_1, sw3_s_d2_edit_1 and sw3_s_d2_agent_2 overlapped it: card 1 ran ~330-440 ms per window instead of
~18 (13.5-15.5 tok/s requests), two **#267s with no KFD eviction** (longest 18 ms). Inference: a second process's
queues oversubscribe the HWS runlist (same collapse as GPU_MAX_HW_QUEUES > 4): a second #267 mechanism, and Strata
needs the GPUs to itself even when the other process is idle. gpu-guard.sh now kills foreign GPU processes during
queue windows. Per-case medians over 3 passes below survive one bad pass; pooled numbers for those sets don't.

- **T: THP shrink_underused 1 vs 0 (pablo) is DEAD.** edit24 under compact_memory + page-cache pressure, 3 arms each:
  307.3 vs 306.4 tok/s, gaps 3 / 3, KFD episodes 918 vs 1085, frozen 5.1 s vs 11.3 s (longest 279 vs 1245 ms),
  splits went UP with 0 (split_page 70 vs 1196). The box setting stays.
- **S, agent (per-case median vs live, 3 passes):** min-p 0.5 +0.35%, min-p 0.4 -0.01%, SPEC_COST +0.25%: all
  neutral. Window 18's pooled losses were eviction noise, but there is no gain either: live 0.7 stays.
- **S, edit24:** min-p 0.5 -0.34% (16 of 24 cases slower). DOWN2 +0.27% (18 of 24 faster, worst -18.3% in the
  contaminated pass); agent DOWN2 -0.51%. The fused_gr_bench -17-22% on that kernel at 5-8 rows is ~0.3% of an edit
  window: below adoption value; SW3B re-measures its contaminated passes.
- **F: the cold first prompt chunk is gone** (nyira's item): chunk 1 is 1310-1328 ms vs 1280-1340 steady, GEMM warm-up
  on GPU 0 127-147 ms (6892 ms pre-RXP). The old 1 s was the pre-RXP load's eviction recovery. 36K with timing on:
  2572-2623 tok/s.
- **Dispatch census written** (kaje): ab/sw3_s_b_agent_1/dispatch.json and ab/sw3_s_b_edit_1/dispatch.json.
- Quiet arms still see short KFD evictions post-RXP (12-30 episodes per agent arm, mostly < 10 ms; a few 400-700 ms,
  mixed SVM and USERPTR): what window PIN's HOST_IMMUNE arm targets.
- Text identical to live in every clean arm. Output: strata-ab/windowsw3.out.

## Windows GDNLIFE and IQSIGNED: two exact kernel lanes, both neutral at engine level (23:08-23:33)

Built in one slot with main's options (gdn-life quad2c 186 VGPRs; iq-signed iq2s<18> 97 / fused<18> 102 VGPRs).
- **GDNLIFE** (basalt38/gdn-life 928b78907: in-place v_fma state update in gdn_rec_quad2c_kernel, 220 -> 186 VGPRs):
  bitwise (rec_head 32 PASS; stage-bench y/state hashes identical). Recurrence + norm -3.4% / -4.2% / -4.9% at
  T 4096 / 16384 / 32768 (control kernel within 0.5%). 36K prompt on the engine: main 2698.5, branch 2673.2 tok/s
  (-0.94%, one slow load; main spread 1.54%): neutral. The kernel is ~3.4 ms of a ~1300 ms 4096-token chunk, so the
  saving is ~0.01% of the prompt; "GDN recurrence ~10% of prompt" in the brief was all of GDN. Parked, not merged.
- **IQSIGNED** (basalt38/iq-signed 6edc7415f: IQ3_XXS gate/up on a 16 KiB signed codebook in LDS, STRATA_IQ_SIGNED_CB):
  bitwise (lab parity 594 comparisons, 0 differ, synthetic + real rows; decode_expert_lab hash same off/on).
  Microbench on layer 0's real experts, the dispatch's kernel: T=1 gate/up 33.3 -> 34.5 us (+3.7%: the table fill),
  T=2 -1.6%, T 3-8 -4.4% to -6.0%; whole expert step -2% at T>=3. Engine (flag 0 vs 1, 3 agent / 2 edit24 passes):
  agent per-case median +0.10%, edit24 +0.10%: neutral. Parked.
- **The finding that matters: the decode expert kernels are at the memory roofline.** The same microbench reads
  IQ3_XXS gate/up at 377 GB/s at T=1, 456 at T=3, 477-483 GB/s at T 4-8 (V620 peak 512 GB/s); with the signed table
  501-514. Instruction-count work on these kernels can't buy engine speed. Decode speed now comes from fewer expert
  bytes per accepted token (smaller expert quants, longer accepted windows, fewer distinct experts per window) or from
  the non-expert part of the window.

## Window PIN: STRATA_HOST_IMMUNE=1 removes every KFD queue eviction under pressure (11-04 23:2x - 11-05 00:23)

basalt38/pinned-host 02a104984 (on parked-graph). edit24 under compact_memory every 10 s + the page-cache streamer,
6 passes, interleaved, a copy of the live config (resident experts), KFD events as root per arm:

| arm | edit24 tok/s | KFD queue evictions (node events: SVM / USERPTR / TTM) |
|---|---|---|
| main | 311.5 | 5604 / 369 / 0 |
| branch, HOST_IMMUNE=0 | 309.6 | 4464 / 396 / 0 |
| **branch, HOST_IMMUNE=1** | **312.3** | **0 / 0 / 0** |

- Per engine process: main and branch-off 200-310 episodes per pass (longest up to 1299 ms); immune 0 in every pass.
- Text identical in every case that completed. Main's pass 5 hit **#267** (two edit24 cases lost, engine restarted);
  its own KFD log shows only short evictions before it (longest 6 ms in that engine life): this #267 is not explained
  by a long eviction, so a third #267 trigger, or a lost-event gap in the logger, remains open.
- Next: window IMMUNE (insert 5657), the quiet adoption check (agent, edit24, 36K prompt) before HOST_IMMUNE goes
  live; staging copies instead of pin-in-place could cost prompt H2D bandwidth.
- Output: strata-ab/windowpin.out, arms ab/pin_*.
