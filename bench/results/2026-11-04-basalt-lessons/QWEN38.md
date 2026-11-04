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
