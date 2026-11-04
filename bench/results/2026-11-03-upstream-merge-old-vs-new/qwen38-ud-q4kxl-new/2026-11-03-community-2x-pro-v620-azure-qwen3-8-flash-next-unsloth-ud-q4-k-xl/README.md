# Community benchmark on AMD RADEON PRO V620 Azure + AMD RADEON PRO V620 Azure

Measured on 2026-11-03 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD RADEON PRO V620 Azure + AMD RADEON PRO V620 Azure, 64 GiB VRAM
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 92.7 GiB; storage: sda     ST26000DM000-3Y8103 sata; sdb     CT1000P310SSD8      usb; sdc     ASM236X NVME        usb; sdd     ASM236X NVME        usb; nvme0n1 SHPP41-2000GM       nvme; nvme1n1 SHPP41-2000GM       nvme; nvme2n1 SPCC M.2 PCIe SSD   nvme
- OS: Linux 7.3.0-rc6-2-cachyos-rc; GPU driver: 7.3.0-rc6-2-cachyos-rc; CUDA (driver supports up to): not read
- Strata: project 0.1.56, commit bcc174a6; engine 0.1.56
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-unsloth-ud-q4_k_xl
- Context: 262144 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged (hip backend); thinking off for the suite; temperature 0

```text
strata --pack unsloth-ud-q4_k_xl --native Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --ple-gguf Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf --expert-profile expert-profile.bin --expert-cache auto --prefill auto:16384 --spec 8 --spec-min-p 0.7 --mtp rt --max-context 262144 --kv int8 --kv-resident 65535 --pipeline-windows 2 --adapt-every 1 --adapt-swaps 160 --adapt-decay 0.97 --pool-workers 4 --vision --pcie-frac 0.00 --trim-stage-weights --mtp-max-t 3
```

Environment of the config: `STRATA_DENSE_MMQ=0`, `STRATA_KQ256=0`, `STRATA_GDN_CHUNKED=0`, `STRATA_SPEC_COUPLED=0`, `STRATA_HIP_PROMPT_F16=1`, `STRATA_ROCBLAS_TUNING=hip`, `STRATA_SELECT_SIMT=1`, `STRATA_IO_THREADS=64`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE=auto`, `STRATA_HIP_ADAPT_KERNEL_COPY=1`, `STRATA_GDN_CONVL2=1`, `STRATA_STAGER_SLEEP=0`, `STRATA_SPLIT_OWN=1`, `STRATA_QFUSE=1`, `STRATA_SH_STREAM=1`, `STRATA_SPEC_GUMBEL=1`, `STRATA_PREFILL_HELP=1`, `STRATA_PREFILL_CPU_SHARE_MAX=1024`, `STRATA_HIP_MTP_PRIORITY=1`, `STRATA_ADAPT_LAG=2`, `STRATA_EXCHANGE_ROTATE=1`, `STRATA_PL_PLE_PREFETCH=1`, `STRATA_PL_PLE_LATE=1`, `STRATA_PL_EARLY_CHAIN=1`, `STRATA_ATTN_MERGE_V2=1`, `STRATA_PREFILL_PIPE=283`, `STRATA_PF_Q4_WIDE=1`, `STRATA_PA_DOT4=1`, `STRATA_GDN_HEAD=1`, `STRATA_GDN_PP=2`, `STRATA_GDN_NOY=1`, `STRATA_HC_WIDE=3`, `STRATA_GR_NORM_FAST=1`, `STRATA_TSUM=1`, `STRATA_DQ8=1`, `STRATA_HC_GATED16=1`, `STRATA_HC_DOWNINJ=1`, `STRATA_PF_STREAM_ROUTED=1`, `STRATA_PF_DIRECT=1`, `STRATA_PF_NL=1`, `STRATA_PF_Q51=1`, `STRATA_MMQ_RESIDENT_SORT_NE=1`, `STRATA_PL_LOOKUP_STREAK=12`, `STRATA_HEAD_WIDE=1`, `STRATA_ROUTE_PLAN=1`, `STRATA_HEAD_SAMPLE=1`, `STRATA_EXP_MODE=8`, `STRATA_EXPERT_DOWN_L=1`, `STRATA_KV_PREFETCH=1`, `STRATA_SPLIT_MTP_BATCH=1`, `STRATA_TOPK_STREAM=1`, `STRATA_TOPK_AUTO=1`, `STRATA_KV_PF_RESIDENT=1`, `STRATA_TUNING_CACHE=strata`, `STRATA_NODE_FUSE=1`, `STRATA_HC_WIDE8=1`, `STRATA_HC_UP_TSUM=1`, `STRATA_MTP_FUSE=1`, `STRATA_EXP_IQ2S=1`, `STRATA_PL_LOOKUP_RESYNC=1`, `STRATA_PF_SIDE_STREAM=1`, `STRATA_PF_NOWIDEN=1`, `STRATA_PF_IQ_TJ=20`, `STRATA_PF_Q2=1`, `STRATA_PF_NL_TJ=20`, `STRATA_PF_NL_PERM=1`, `STRATA_HC_DOWNSK=8`, `STRATA_HC_UPMIX16=1`, `STRATA_IDX_PAIR=1`, `STRATA_QSA_GATE_Q8=1`, `STRATA_PF_IQ2S=1`, `STRATA_PF_IQ3=1`

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (started by the benchmark from the install configuration); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 400 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 261 (200-400) | 9 | not measured | 77.7 (71.7-111.0) | 0.24 (0.23-0.25) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1250.0 (1211.2-1258.3) | not measured | 3.24 (3.22-3.34) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2147.5 (2147.3-2148.3) | not measured | 7.50 (7.50-7.51) |
| agent session, follow-up turns | 13635 (13618-13638) at the last turn | see below | up to 64 | 3 | see below | 84.0 (69.8-99.9) | 4.75 (2.87-5.09) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.9 % (69.9-69.9); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12760 (12758-12760) new tokens of the files are read once); first turn (nothing cached) first token: 1.30 s (1.24-1.33).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 502.2 | 1.237 | 57.2 |
| 1 | 3856 | 618 | 3238 | 639.4 | 5.091 | 69.8 |
| 2 | 5389 | 3919 | 1470 | 480.4 | 3.085 | 76.3 |
| 3 | 8633 | 5452 | 3181 | 628.4 | 5.09 | 84.0 |
| 4 | 10861 | 8696 | 2165 | 500.0 | 4.356 | 79.3 |
| 5 | 13635 | 10924 | 2711 | 552.3 | 4.935 | 98.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 152 s.
