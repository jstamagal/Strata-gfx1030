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
