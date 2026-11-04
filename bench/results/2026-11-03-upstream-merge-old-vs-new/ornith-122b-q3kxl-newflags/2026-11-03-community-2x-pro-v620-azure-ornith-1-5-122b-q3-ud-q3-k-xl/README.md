# Community benchmark on AMD RADEON PRO V620 Azure + AMD RADEON PRO V620 Azure

Measured on 2026-11-03 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD RADEON PRO V620 Azure + AMD RADEON PRO V620 Azure, 64 GiB VRAM
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 92.7 GiB; storage: sda     ST26000DM000-3Y8103 sata; sdb     CT1000P310SSD8      usb; sdc     ASM236X NVME        usb; sdd     ASM236X NVME        usb; nvme0n1 SHPP41-2000GM       nvme; nvme1n1 SHPP41-2000GM       nvme; nvme2n1 SPCC M.2 PCIe SSD   nvme
- OS: Linux 7.3.0-rc6-2-cachyos-rc; GPU driver: 7.3.0-rc6-2-cachyos-rc; CUDA (driver supports up to): not read
- Strata: project 0.1.56, commit bcc174a6; engine not read
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: ornith-1.5-122b-q3-UD-Q3_K_XL
- Context: 65535 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata-q36 --kv-cache-type int8 --native Ornith-1.5-122B-UD-Q3_K_XL-00001-of-00003.gguf --max-context 65535 --prefill auto --mtp 3 --draft-vocab 32768 --spec-min-p 0.45
```

Environment of the config: `STRATA_Q36_DROP_PACKED=1`, `STRATA_Q36_STAGE1=1`, `STRATA_EXP_MODE=8`, `STRATA_EXPERT_DOWN_L=1`, `STRATA_TSUM=1`, `STRATA_MMQ_KQ=1`, `STRATA_Q36_TP=1`, `STRATA_Q36_MTP_ADAPT=1`, `STRATA_Q36_TP_FUSE2_CONV_AB=1`, `STRATA_Q36_TP_FUSE2_GDN_Q8=1`, `STRATA_Q36_TP_FUSE2_JOIN_NORM=1`, `STRATA_Q36_TP_FUSE2_MOE_NORM=1`, `STRATA_Q36_TP_FUSE2_ROUTER=1`, `STRATA_Q36_TPX_DOWN16=1`, `STRATA_Q36_TP_ROUTER_ALIAS=1`, `STRATA_Q36_TP_ATTN_GRAPH_ROWS=1`, `STRATA_Q36_TP_JOIN_ROW=1`, `STRATA_Q36_TP_JOIN_MOE_PUBLISH=1`, `STRATA_Q36_MTP_SHORT_HEAD=1`, `STRATA_Q36_MTP_CONF=1`, `STRATA_Q36_TP_PARTIAL_SPLIT=1`, `STRATA_Q36_TP_ROUTER_PAIR=1`, `STRATA_Q36_MTP_SPLIT_HEAD=1`, `STRATA_Q36_TP_JOIN_WIDE=1`, `STRATA_Q36_TP_JOIN_Q8_ROUTER=1`, `STRATA_Q36_TP_BRANCHES=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 400 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 273 (206-400) | 9 | not measured | 63.4 (58.6-75.2) | 0.34 (0.31-0.39) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 331.8 (331.7-332.0) | not measured | 12.12 (12.11-12.12) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 289.2 (289.0-289.3) | not measured | 55.54 (55.51-55.57) |
| agent session, follow-up turns | 13635 (13630-13635) at the last turn | see below | up to 64 | 3 | see below | 50.3 (46.7-57.7) | 9.82 (4.74-35.01) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 57.0 % (36.5-69.9); earlier-turn tokens read again per session: 5457 (2-14153) (earlier turns that the cache did not hold; the 12760 (12758-12760) new tokens of the files are read once); first turn (nothing cached) first token: 1.90 s (1.90-1.93).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 315.9 | 1.932 | 55.8 |
| 1 | 3856 | 618 | 3238 | 329.9 | 9.82 | 49.0 |
| 2 | 5389 | 3920 | 1469 | 309.4 | 4.752 | 50.3 |
| 3 | 8633 | 5452 | 3181 | 297.7 | 10.691 | 54.2 |
| 4 | 10861 | 8697 | 2164 | 276.1 | 7.844 | 48.1 |
| 5 | 13635 | 10925 | 2710 | 264.9 | 10.237 | 52.0 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 473 s.
