# 2026-11-04: lessons from Basalt, Ornith side (strata-q36, 2x V620 TP)

Basalt (jesdga95/basalt, a CUDA/Blackwell fork of Strata for Qwen3.8-Flash-Next) was read for speed ideas. This
folder records the Ornith 1.5 122B part; the Qwen3.8 part is separate work.

## Ornith UD-Q3_K_XL decode, 1200-token greedy essay (`--mtp 3`, thinking off, page cache warm)

Every row produced byte-identical text (md5 71611428fd…) and identical drafts (566 of 1305).

| engine / setting | tok/s | verify ms/step | MTP ms/step |
|---|---|---|---|
| morning: build-hip 137b2d4c, `STRATA_Q36_TP_ATTN_GRAPH_ROWS=1` | 88.3 | 18.09 | 2.25 |
| same engine, `STRATA_Q36_TP_ATTN_GRAPH_ROWS=0` | 93.2 | 16.91 | 2.27 |
| + router mask grouping (basalt/router-mask) | 94.3 | 16.67 | 2.28 |
| + ordered head-group attention (basalt/attn-rows) | 105.1 | 15.04 | 1.86 |
| live after merge (main ae265494, rebuilt build-hip, live config) | 106.6 | | |

UD-Q4_K_XL hybrid TP, warm run: `GRAPH_ROWS=0` 64.6 -> 65.7 tok/s, identical text (attention/router not re-run).

## What changed

- `STRATA_Q36_TP_ATTN_GRAPH_ROWS=0` in both live Ornith configs: the flag sent contexts up to 1024 cells to the
  per-head `k_attn_part` (uncoalesced int8 K reads, ~225 us per call even at tiny contexts).
- `src/qwen35/q36_attn_rows.cuh`: int8 GQA-16 attention with packed K loads shared across head groups, V loads issued
  ahead of the ordered accumulation, same split-K chunks and reduction trees. Default on gfx1030 int8 HD256;
  `STRATA_Q36_ATTN_ROWS=0` turns it off. Test `q36_attn_rows`: 594 cases bitwise equal to both old kernels.
  Device-0 kernel time at nt=4: ctx 1000 173.8 -> 28.9 us, ctx 30000 1582 -> 217 us, ctx 65536 3200 -> 473 us.
- `src/qwen35/q36_router_mask.cuh`: router expert grouping by token masks and popcount scans (Basalt 10dd14e3),
  unrolled projection loads. `STRATA_Q36_ROUTER_FAST=0` turns it off. Test `hip_q36_router_mask`: 24,384 cases
  bitwise equal. Kernel 26.2 -> 13.1 us (T=1), 26.9 -> 17.1 us (T=4).

## Tried and parked

- CPU pool phase fusion (branch basalt/pool-fuse, Basalt 2c4a0eb6): bitwise identical, speed-neutral on Q4 hybrid
  (CPU pool 3.21 -> 3.13 ms/step, decode 65.5 -> 65.9 tok/s; load-to-load spread is larger).
- Reply-end commit bound (branch basalt/stop-bound, Basalt cd6a9c60): no effect with `--draft-vocab 32768`, which
  cannot propose the end tokens; 12 two-turn chats reused identically with and without it.

## Files

- `ornith-Q3-morning/`: `tools/strata_bench.py --runs 3 --chat-tokens 400 --long` on the morning engine and
  setting; the matching run of the new engine was stopped before it started.
