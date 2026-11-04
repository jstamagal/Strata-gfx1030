# 2026-11-03: the stranded Ornith lanes (UnbiasedStrata /home/admin/work/lanes, 10-21/22), judged on 2x V620

Ornith 1.5 122B, `tools/strata_bench.py` (3 runs, medians, greedy, 400-token chat), the real HTTP API, page cache dropped
before each run, every run with this tree's bench script. Baseline is the live engine (ornith-122b f8307f65 + snapshot,
tensor parallel with TP_BRANCHES): 103.8 chat decode, 666 / 564 prompt at 4K / 16K, 86.2 agent decode
(`../2026-11-03-upstream-merge-old-vs-new/`). Run-to-run drift of identical builds is about +-2-3%.

## mega: persistent TP decode stages (branch port/mega, STRATA_Q36_TP_PERSIST=1)

Correctness gate (`tools/qwen35/q36_tp_persist_gate.py`, 28 greedy requests): PASS, tokens and MTP acceptance identical.

| run | chat decode | 4K prompt | 16K prompt | agent decode |
|---|---|---|---|---|
| live (TP_BRANCHES) | 103.8 | 666 | 564 | 86.2 |
| mega-persist (persist needs TP_BRANCHES off) | 91.0 | 666 | 563 | 78.2 |
| mega-persist-eh (+ MTP_SPLIT_EH) | 92.0 | 665 | 560 | 79.8 |

Gate timing, same binary, both without TP_BRANCHES: 96.6 tok/s plain vs 96.4 persistent over the same 5,448 tokens.
Verdict: no gain. HIP graphs already hide the launches persistence removes, and persistence cannot coexist with
TP_BRANCHES (it needs a linear capture), which is worth 10-14% (live vs mega-persist above; persist itself is speed-neutral, so that row is in effect TP without branches). Parked as branch port/mega, not merged.

## kv8: hybrid TP experts, Q4 decode, expert pack, host-streamed KV, YaRN (branch port/kv8)

K1, no regression on the live Q3_K_XL config (kv8 build, same env and args, every kv8 feature off):

| run | chat decode | 4K prompt | 16K prompt | agent decode | agent cache hit |
|---|---|---|---|---|---|
| live | 103.8 | 666 | 564 | 86.2 | 57.0% |
| kv8 build | 105.8 | 665 | 560 | 85.4 | 69.6% |

Same answers as live TP in the two-turn shelf test.

## Live build-hip reconfigured (2026-11-04 00:00): STRATA_Q6K_EXPERTS=ON, STRATA_MMQ_WIDE_LDS=ON

The live build's CMake cache still had STRATA_Q6K_EXPERTS=OFF from an old configure (setup.py and CMake default it
ON), so Ornith UD-Q4_K_XL refused to load ("blk.46.experts are Q5_K/Q6_K, which the native expert kernels do not
take"). Wide LDS is upstream's default and what the tpk / mega / kv8 builds use.

| run | chat decode | 4K prompt | 16K prompt | agent decode |
|---|---|---|---|---|
| live Q3_K_XL before | 103.8 | 666 | 564 | 86.2 |
| live Q3_K_XL after (`ornith-q3-live-q6k-wide`) | 107.6 | 666 | 564 | 89.8 |
| live Q4_K_XL layer split, after (`live-q4-ls`; 172 of 256 experts per layer in VRAM, the rest on the CPU) | 47.4 | 430 | 379 | 36.7 |

Q4_K_XL answers the shelf test right (item 37 -> 12) and resumes from the snapshot on the edited second turn.

K2, Ornith UD-Q4_K_XL (does not fit both cards: experts spill to the CPU):

| run | chat decode | 4K prompt | 16K prompt | agent decode | short-chat TTFT |
|---|---|---|---|---|---|
| live layer split (`live-q4-ls`, 172/256 experts per layer in VRAM) | 47.4 | 430 | 379 | 36.7 | 1.78 s |
| kv8 hybrid TP (`kv8-q4-hyb`, STRATA_Q36_TP_HYBRID=1, 191/256 half-experts per layer per rank, 93.6% of expert entries on the GPUs) | 63.0 | 421 | 391 | 50.8 | 1.52 s |

Accuracy (40 long-context shelf lookups + 40 two-digit products, greedy): live layer split 38/40, 40/40; hybrid TP 39/40, 40/40.
The layer-split row is the old Q4 config as it was (draft vocab 65536, f16 KV); the hybrid row uses the Q3 config's
args (int8 KV, draft vocab 32768, spec-min-p 0.45), which the new Q4 config now has.

Two bugs in the ported hybrid TP, both fixed on port/kv8 before these numbers:
- warmup deadlock (GPU 0 spinning, GPU 1 idle): the fused mixer-norm join waits on the device for the peer while the
  hybrid layer waited on the host before the peer's half was queued; tp_layer_pair now runs a third phase for the
  host half (06d2f19c);
- the first prompt's rebalance hit an unallocated H.vmap_d, and CK's exit() then hung on stdin's lock for 90 s
  (7c60f121).

K3, host-streamed KV (`--kv-resident 65536 --max-context 262144`, Q3_K_XL TP: a 57344-cell VRAM ring + 8192-cell
staging window per card, every cell in mapped host memory, 0.36 GiB of VRAM rings per card):

| prompt | read | decode after it | passphrase at the start found |
|---|---|---|---|
| 68,352 tokens (11K cells past the ring) | 240 tok/s average, 284 s | 23.0 tok/s | yes |
| 117,930 tokens (60K past the ring) | 104,448 read in 2,072 s, not finished in 3,000 s | - | - |

Correct, but cold cells cross PCIe for every token and every 8-token prompt window past the ring, so speed falls
with depth. Opt-in only; the live configs keep 65535 cells in VRAM.

K4, `--batch 2` with two concurrent two-turn shelf conversations (snapshot resume on the edited second turn):
before the fix the conversation the server moved from the solo path into a batch slot answered "Item is filed under
shelf 11." for item 87 (alone: "Item 87 is filed under shelf 11. ..."), with STRATA_Q36_SNAPSHOT on and off alike.
Cause: a stop during the sampled speculative step (taken whenever a presence penalty is set, temperature 0 included)
kept only x but emitted the token sampled after the dropped drafts (fixed in 5f2631ad). After: both conversations'
first turns match the one-at-a-time answers token for token in every round; the batch-decoded second turn of one
conversation words its second sentence differently (same facts, same every round).

## Shipped (ornith-122b: stop fix 5f2631ad, kv8 merge 25c0ce94, vis bf87019f; build-hip rebuilt from it)

| config | chat decode | 4K prompt | 16K prompt | agent decode | agent cache hit |
|---|---|---|---|---|---|
| strata-ornith-1.5-122b-UD-Q3_K_XL.json (TP, unchanged) (`final-q3`) | 104.5 | 665 | 562 | 84.4 | 69.9% |
| strata-ornith-1.5-122b-UD-Q4_K_XL.json, now hybrid TP (`final-q4`) | 63.2 | 421 | 391 | 50.0 | 69.7% |

The Q4_K_XL config moved from build-hip/strata (layer split, draft vocab 65536, f16 KV) to build-hip/strata-q36 with the
Q3 config's TP env plus STRATA_Q36_TP_HYBRID=1, int8 KV, draft vocab 32768, spec-min-p 0.45, engine_ready_s 1800.

## vis: Ornith vision on TP (branch port/vis, fast-forwarded into ornith-122b)

Parked at first for "no mmproj on disk"; unsloth/Ornith-1.5-122B-GGUF ships mmproj-F16.gguf (909 MB, now in the HF
cache), and the lane had its own conversion at /home/admin/work/models/mmproj-Ornith-1.5-122B-f16.gguf all along.

- Picture reader parity (tools/vision/check_embeddings.py, CPU, against an independent llama.cpp mtmd): built on the
  vendored third_party/llama.cpp, square is bit-identical but wide / tall differ (relative L2 0.0015 / 0.0007, the
  same numbers the lane recorded); built on upstream ggml-org llama.cpp bee23322f (~/Repos/llama.cpp-ggml-org) all
  three are bit-identical. build-vision/bin/strata-vision is built on that checkout (statically linked); the engine
  keeps its pinned ggml.
- TP vision on the GPUs (tools/vision/check_tp.py --run: eager and graph, MTP 0 and 3, growing / equal-size / text
  between images): PASS, every case token-identical, TP_VISION_GRAPH graphs captured for main and MTP.
- HTTP (OpenAI and Anthropic endpoints, the red square / blue circle / "ORNITH 4217" fixtures at the server's
  1024-token minimum): both shapes, colours and positions right, the text read as "ORNITH 4217" on every image.
- Cost on text, Q3_K_XL TP with --vision and the reader on the CPU (`final-q3-vision`): 107.1 / 665 / 562 / 86.9
  against 104.5 / 665 / 562 / 84.4 without. strata-ornith-1.5-122b-UD-Q3_K_XL.json now has --vision,
  STRATA_Q36_TP_VISION_GRAPH=1 and a "vision" block.
- Q4_K_XL hybrid TP with --vision (`final-q4-vision`): the three pictures read right, 191/256 half-experts per layer
  as without vision, 63.2 / 422 / 391 / 51.6 against 63.2 / 421 / 391 / 50.0. strata-ornith-1.5-122b-UD-Q4_K_XL.json
  has vision too (its reader reads that config's own GGUF).

## Not merged

- mega (port/mega): speed-neutral, conflicts with TP_BRANCHES.
- hyfast (stranded/hyfast): kv8 plus one WIP commit (2810d7a8, a `--main-qwen35-check` preflight toward running
  qwen35moe on the main strata engine; it includes a header the commit does not add) - groundwork for another project.
- Host-streamed KV is merged but off: correct at 68K, slow past the VRAM ring.
- Divergence earlier than the last prompt still re-reads the conversation: one more snapshot is ~149 MiB of recurrent
  state per session (36 GDN layers x 64 heads x 128x128 fp32 + conv, half on each card; VRAM the Q4_K_XL config
  spends on experts), and clients that clear old tool results diverge
  before any recent snapshot anyway.
