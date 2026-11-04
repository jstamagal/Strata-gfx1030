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

**Correction (17:00): the GDN rollback-snapshot candidate is DEAD for the live engine.** ziz's snapshot writes
(`gdn_core.cuh`, "~5-8%" in the swarm intake above) are in `src/split/`, the opt-in strata-split engine
(`STRATA_BUILD_SPLIT`, OFF, not in build-hip). The coordinator endorsed it without checking which binary the file
belongs to. The live verify path already replays kept rows from stored per-row inputs (`gdn_commit_layers`).

The live-path remainder is PlSnap: card 0's GDN state is copied D2D once per speculative window on a side stream,
and restored on doomed windows. Removing it with ping-pong parity is a medium build, low rank (tuck, run 2).

## Window EPF: early PLE flag against the stalls (16:52)

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

## Swarm intake, run 2 late chat (17:05)

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

## Window CHN: shorter forced chains for the guessed window (17:25)

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

## Swarm run 2 final (gruff, 17:45; saved as docs/swarm/reports/gruff.md from KING's paste)

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
