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
