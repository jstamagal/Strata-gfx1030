# Community benchmark on AMD RADEON PRO V620 Azure + AMD RADEON PRO V620 Azure

Measured on 2026-11-03 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: AMD RADEON PRO V620 Azure + AMD RADEON PRO V620 Azure, 64 GiB VRAM
- CPU: AMD Ryzen 9 9950X 16-Core Processor (16 cores, 32 threads); installed RAM: 92.7 GiB; storage: sda     ST26000DM000-3Y8103 sata; sdb     CT1000P310SSD8      usb; sdc     ASM236X NVME        usb; sdd     ASM236X NVME        usb; nvme0n1 SHPP41-2000GM       nvme; nvme1n1 SHPP41-2000GM       nvme; nvme2n1 SPCC M.2 PCIe SSD   nvme
- OS: Linux 7.3.0-rc6-2-cachyos-rc; GPU driver: 7.3.0-rc6-2-cachyos-rc; CUDA (driver supports up to): not read
- Strata: project 0.1.56, commit c37f8321; engine not read
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: ornith-1.5-122b-q3
- Context: 65535 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 400 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 302 (229-400) | 9 | not measured | 107.8 (94.4-122.2) | 0.21 (0.21-0.23) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 665.2 (664.3-665.9) | not measured | 6.04 (6.04-6.05) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 563.7 (556.5-575.9) | not measured | 28.50 (27.89-28.87) |
| agent session, follow-up turns | 13632 (13629-13635) at the last turn | see below | up to 64 | 3 | see below | 90.5 (75.5-105.4) | 5.81 (2.71-6.28) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.7 % (69.7-69.9); earlier-turn tokens read again per session: 67 (4-67) (earlier turns that the cache did not hold; the 12759 (12758-12760) new tokens of the files are read once); first turn (nothing cached) first token: 1.17 s (1.17-1.18).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 520.5 | 1.174 | 99.7 |
| 1 | 3856 | 618 | 3238 | 558.8 | 5.801 | 87.6 |
| 2 | 5389 | 3856 | 1533 | 549.9 | 2.793 | 105.4 |
| 3 | 8632 | 5452 | 3180 | 509.0 | 6.254 | 75.5 |
| 4 | 10855 | 8691 | 2164 | 478.3 | 4.531 | 90.5 |
| 5 | 13629 | 10919 | 2710 | 462.5 | 5.866 | 90.9 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 228 s.
