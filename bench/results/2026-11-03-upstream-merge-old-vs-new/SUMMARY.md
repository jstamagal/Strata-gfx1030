# 2026-11-03: upstream 0.2.10 merge + tpk tensor-parallel port, serving speed on 2x V620

`tools/strata_bench.py` (3 runs, medians, greedy, 400-token chat), the real HTTP API, one model at a time, page cache
dropped before each run. Every run below uses the old tree's bench script, so the agent test sends the same text.

## Qwen3.8-Flash-Next unsloth UD-Q4_K_XL (`strata`, layer split)

| build | chat decode | 4K prompt | 16K prompt | agent decode | agent cache hit |
|---|---|---|---|---|---|
| old: ornith-122b 1f76d1e2 | 75.9 | 1244 | 2151 | 83.3 | 70.0% |
| live: f8307f65, build-hip | 74.6 | 1230 | 2123 | 83.8 | 69.9% |

## Ornith 1.5 122B UD-Q3_K_XL (`strata-q36`)

| build | chat decode | 4K prompt | 16K prompt | agent decode |
|---|---|---|---|---|
| old: ornith-122b 1f76d1e2 (layer split) | 63.5 | 477 | 395 | 49.9 |
| new: + upstream 0.2.10, bcc174a6 (layer split, same text) | 63.5 | 477 | 395 | 51.5 |
| tpk: /home/admin/work/engines/strata-q36-tpk-int8 (tensor parallel) | 106.3 | 665 | 562 | 88.3 |
| live: f8307f65 = new + tpk, build-hip (tensor parallel) | 103.8 | 666 | 564 | 86.2 |

Run-to-run drift of identical builds is about +-2-3%.

Notes:
- A fresh CMake configure leaves STRATA_PREFILL_MMQ off (setup.py turns it on); without it strata-q36 reads prompt
  experts on the native grouped kernels and Ornith prompt is 15-30% slower.
- The agent test's cache hit varies 40-70% between Ornith builds: a turn whose re-sent reply does not tokenize
  identically cannot resume the GDN state and re-reads the prompt. Text-dependent, present in every build, tpk too.
- `*-oldfiles` / `new` / `newflags` folders: the merge's server and engine with the merged tree's docs (different
  agent text) and a configure without prefill MMQ; kept for the record, not comparable rows.
