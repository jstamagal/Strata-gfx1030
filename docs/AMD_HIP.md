# AMD Radeon: the HIP backend

Strata runs on AMD Radeon cards through its HIP backend, the same engine as on NVIDIA compiled for AMD. This page is for
people with an RX 6000, 7000 or 9000 card or a Radeon AI PRO R9700, and for anyone building the engine from source.
The Ryzen AI Max "Strix Halo" APU has its own page: [STRIX_HALO.md](STRIX_HALO.md). Integrated Radeon chips in mini PCs are
covered under [Integrated GPUs](#integrated-gpus-mini-pcs).

## Quick start

Use the matching [desktop app or Terminal](INSTALL.md). The engine package carries the HIP runtime, so only the AMD
graphics driver must be installed (Windows: [AMD Software: Adrenalin Edition](https://www.amd.com/en/support/download/drivers.html);
Linux: the kernel's amdgpu driver, no ROCm install). Terminal:

```sh
strata launch --plan       # shows what would happen, changes nothing
strata launch              # sets up and serves
```

Source checkout on Linux ([SOURCE_INSTALL.md](SOURCE_INSTALL.md#amd-cards)):

```sh
./setup.sh --backend hip
```

Setup picks `--backend hip` by itself on a PC with no NVIDIA card Strata can use. Other AMD architectures and mixed AMD/NVIDIA
runs are not supported. Use the [package table](#ready-made-packages) to match your card.

## Supported cards

| Card | Architecture | Status |
| --- | --- | --- |
| RX 7900 XT / XTX | gfx1100 | Maintainer-tested |
| RX 9070 / 9070 XT, Radeon AI PRO R9700 | gfx1201 | Maintainer-tested ([RDNA4](#rdna4-gfx1201)) |
| RX 9060 XT | gfx1200 | Validated by its owner |
| RX 7800 XT / 7700 XT | gfx1101 | Validated by its owner |
| RX 7600 XT | gfx1102 | Reported working, not validated |
| RX 6800 / 6900 series, RX 6700 XT | gfx1030, gfx1031 | Has a package, not tested on a real card ([RDNA2](#rdna2-gfx1030)) |
| Ryzen AI Max (Strix Halo) | gfx1151 | [STRIX_HALO.md](STRIX_HALO.md) |
| Radeon 780M / 760M | gfx1103 | Linux, opt-in build: [RADEON_780M.md](RADEON_780M.md) |
| Instinct MI50 / MI60, Radeon VII | gfx906 | Source build, community-tested ([gfx906](#gfx906-instinct-mi50--mi60-radeon-vii-wave64-built-from-source)) |

Strix Point (890M / 880M, gfx1150) and Krackan (860M / 840M, gfx1152) are named and not supported by setup. RDNA1
(gfx1012) builds by hand: [OLDER_GPUS.md](OLDER_GPUS.md#amd-building-gfx906-and-gfx1012).

DeepSeek V4.1-Flash runs on the Radeon AI PRO R9700 on Linux, in the app and in the Terminal. The RX 9070 (gfx1200) engine package
carries it but it is untested. Details: [MODELS.md](MODELS.md#deepseek-v41-flash).

<a id="ready-made-packages-023"></a>
## Ready-made packages

The Terminal one-liner and `strata update` pick the package whose GPU code covers the card. The app installer
(`Strata-0.2.7-windows-x64-setup.exe`, `Strata-0.2.7-linux-x86_64.AppImage`) has no engine inside: on its first start the app
downloads the matching package for the card. You can also press **Update engine** in Settings (or run `strata update`).

| Package (variant) | OS | GPU code inside | ROCm |
| --- | --- | --- | --- |
| `rx9070` | Windows | gfx1200, gfx1201 | 10.2.0a20260930 (TheRock) |
| `rdna3` | Windows | gfx1100, gfx1101, gfx1102, gfx1151 | 10.2.0a20260930 |
| `rdna2` | Windows | gfx1030, gfx1031 | 10.2.0a20260930 |
| `rx9070` | Linux | gfx1200, gfx1201 | 7.10.0a20251120 (`gfx120X-all` wheels) |
| `rx7900` | Linux | gfx1100, gfx1101, gfx1102 | 7.10.0a20251120 (`gfx110X-dgpu` wheels) |
| `halo` | Linux | gfx1151 | 7.14.1 (TheRock) |
| `rx6000` | Linux | gfx1030, gfx1031 | 7.13.0a20260515 (`gfx103X-all` wheels) |

Each package carries the engine, the CPU and GPU picture readers for its architectures, and the rocBLAS and hipBLASLt kernel
data for them only. A package whose GPU code does not cover the card is refused with one line that names the card family.
The RX 6000 packages are offered only to a PC whose card reports gfx1030 or gfx1031. They run the plain kernels: RDNA2 has no
matrix cores, and hipBLASLt ships no gfx1030 kernels, so there is no tuning table.

**Automatic card.** With no card named in the model's settings, the engine start asks the package's
`strata-device --list-devices` and skips cards the package has no code for and integrated chips, while a card of its own can
run. `HIP_VISIBLE_DEVICES`, a `"gpu"` in the settings file or `hip_ordinal` is never overridden.

**Tuning tables.** Each package carries the hipBLASLt tuning tables of `tools/hip` that match the hipBLASLt it bundles, as
`tuning/<arch>-hipblaslt-<version>.txt`:

| Package | Table |
| --- | --- |
| Windows `rx9070` | gfx1201 (hipBLASLt 1.5.0) |
| Windows `rdna3` | gfx1100, gfx1151 (1.5.0) |
| Linux `rx7900` | gfx1100 (1.2.0) |
| Linux `halo` | gfx1151 (1.4.1) |
| Linux `rx9070`, `rx6000`, Windows `rdna2` | none |

A table changes which GEMM algorithm runs the prompt, so rounding differs and answers are not bit-identical to plain
hipBLAS. Packaged engines therefore use the table by default only on Strix Halo (gfx1151). On every other card it is off
until you ask:

| Setting | Effect |
| --- | --- |
| `STRATA_HIPBLASLT_TABLE=on` (or `"hipblaslt_table": true` in the run config) | Use the packaged table when it matches |
| `STRATA_HIPBLASLT_TABLE=off` | Never use it, also on Strix Halo |
| `STRATA_HIPBLASLT_TUNING=<file>` | Use this table file; always kept |

The start log says which table was taken or why none was (`hipBLASLt tuning table: ...`). Please send that line and a
prompt-speed run with `on` and `off` from an RX 7000 or Strix Halo card.

## Install with setup (source checkout, Linux)

```sh
./setup.sh --backend hip
```

| Step | What setup does |
| --- | --- |
| Detection | Finds the card through the kernel's KFD topology. Integrated Radeon GPUs are listed as not supported (Strix Halo has its own page). |
| ROCm | Uses a system ROCm 7 in `/opt/rocm` (or `$ROCM_PATH`) with hipcc and hipBLAS when present. Otherwise, or when it is older than 7.0, installs ROCm into `.venv` from AMD's TheRock wheels (about 10 GB, no sudo). |
| Engine | Compiles it for the card's architecture (10-20 minutes, once; again when you pick another architecture). Needs a C++ compiler and git (`sudo apt install build-essential git`). |
| Tuning table | Uses `tools/hip/<arch>-hipblaslt-<version>.txt` only when it matches both the architecture and the installed hipBLASLt version. Otherwise it says so and uses plain hipBLAS. On gfx103x the same role falls to a rocBLAS table. |
| Pictures | On by default. CPU reader under 24 GB of VRAM, GPU reader from 24 GB; `--vision gpu` / `--vision cpu` switch it. If the GPU reader cannot start, the CPU reader is used. |

TheRock wheel index per family: `gfx110X-dgpu` (gfx1100, gfx1101), `gfx120X-all` (gfx1200, gfx1201), `gfx103X-all`
(gfx1030), `gfx1151` (Strix Halo). Default versions: 7.10.0a20251120, 7.13.0a20260515 for `gfx103X-all`, 7.14.0a20260608 for
`gfx1151`. When the index no longer offers the version, setup warns and takes the newest of the same line.
`STRATA_ROCM_VERSION` and `STRATA_ROCM_INDEX` override them. The gfx1030 wheels are unvalidated; a system ROCm 7 is the
tested path there.

**Several cards.** Setup takes one card (the one with the most VRAM, or `--gpu N`) unless you name more.

| Option | Effect |
| --- | --- |
| `--gpus 1,0` | Split the layers across these cards (numbers as setup lists them). The faster card goes last and is the Main card. |
| `--gpus all` | Every supported card, most VRAM first |
| `--gpu-roles 1=main,0=helper` | Say the roles yourself ([MULTI_GPU.md](MULTI_GPU.md)) |

Every chosen card must be one of the supported architectures. Cards of two families (for example gfx1100 and gfx1201) need
a system ROCm 7, because AMD's wheels hold one family. A split pays only when no single card holds the model's experts.

Setup does not offer tuning (calibration) on AMD. A tuning run by hand (`./setup.sh --calibrate`) is saved for that AMD card and
reused. The Monitor shows the card's load, VRAM, temperature and power from Linux sysfs.

## Windows

Install the [app](INSTALL.md) (`Strata-0.2.7-windows-x64-setup.exe` from the
[release page](https://github.com/Niko1221/Strata/releases/latest); it downloads the engine for the card), or
install the [Terminal](INSTALL.md#the-terminal) and run `strata launch`. You need Windows 10 or 11 (64-bit), a supported card
and a current AMD driver. No ROCm or HIP SDK, no compiler, no admin rights. `START-HERE.bat --backend hip` is a
source-checkout option for developers, not a way to start the released app.

- **Detection:** setup reads the display adapters Windows lists. An integrated Radeon is listed as not supported, except Strix
  Halo ([STRIX_HALO.md](STRIX_HALO.md)). Phoenix / Hawk Point (780M / 760M) is not supported on Windows; on Linux it has an
  opt-in build ([RADEON_780M.md](RADEON_780M.md)).
- **HIP runtime:** the engine folder holds its own `amdhip64_7.dll` and `amd_comgr.dll`, because Windows looks in the
  program's folder before System32, where some AMD drivers install an `amdhip64_7.dll` that crashes the bundled libraries. The
  engine log names the runtime it loaded (`strata generate: HIP runtime ...`).
- **Before the model download,** setup runs `engine\strata-device.exe --list-devices`. If the HIP runtime does not see the
  card, setup stops and points to the driver. With an integrated Radeon, a discrete card is HIP device 1, not 0; `--gpu N` and
  the config's `"gpu"` are HIP numbers.
- **Several cards:** use the app's card selector or `strata launch --gpu 0,1`. [MULTI_GPU.md](MULTI_GPU.md) lists the
  combinations measured. A tested Linux pair does not validate the same pair on Windows.
- **VRAM budget:** `hipMemGetInfo` on Windows does not subtract what the desktop and other programs hold. The engine lowers
  the free figure by what Windows' video memory budget for the process withholds (logged once). `STRATA_WDDM_BUDGET=0` turns
  that off.
- **WMMA prompt attention** is opt-in on Windows (`STRATA_HIP_WMMA=1`); it is the default on gfx12 on Linux.

**Reporting a Windows AMD run.** Give your card, the driver version (AMD Software > System), the output of the two commands
below (from the Strata folder, after `set PATH=%CD%\engine\rocm\bin;%PATH%`), the end of `strata-<model>.log`, and the speed
lines the server window prints for a first answer.

```bat
engine\strata-device.exe --list-devices
engine\strata-device.exe --selftest
```

**Building it yourself.** `tools\hip\build_windows.bat` (Visual Studio 2022 Build Tools with the C++ workload, Python, git; no
admin, no AMD GPU) installs ROCm from AMD's TheRock wheels into `.rocm-win`, builds, and packages
`dist\strata-windows-x64-hip.zip`. `START-HERE.bat --backend hip --prebuilt dist\` installs that one.
`tools\hip\build_windows.bat tests` also builds the HIP tests. `STRATA_HIP_ARCHS` picks other architectures; for the RX 6000
package use `STRATA_HIP_ARCHS="gfx1030;gfx1031"` and `package_engine.py --variant rdna2`.

## Build

Requirements: a working ROCm driver and runtime, HIP development headers and compiler, hipBLAS and (for tuned dense prefill)
hipBLASLt development files, CMake 3.24+, a C++20 host compiler, and Git. The model still needs enough system RAM and a fast
SSD for its PLE table.

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

`CMAKE_HIP_ARCHITECTURES` is `gfx1100`, `gfx1101`, `gfx1200`, `gfx1201`, or a list such as `"gfx1100;gfx1201"` (one binary
for both). gfx1102, gfx1030 and gfx1031 build with a warning. At startup the engine and `strata-device` compare each GPU
they use with the architectures the binary was compiled for and require wave32. A binary carried to another card stops with
the card's name, its architecture and the build's list.

- If CMake cannot find the HIP compiler, add `-DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++`. Use the compiler and
  libraries from the same ROCm installation.
- Do not enable both GPU backends in one build.
- CMake fetches the pinned llama.cpp revision. For an offline build, point `STRATA_GGML_DIR` at a checkout of that exact
  revision.
- Linux engine packages: `VARIANT=rx6000 packaging/build_hip_linux.sh`.
- **A hand-built engine has no tuning table.** The packaged engines carry `tuning/<arch>-hipblaslt-<version>.txt` beside the binary
  ([Ready-made packages](#ready-made-packages) lists which); a build made with plain `cmake` and started
  through `serve/server.py` has none, and the prompt GEMMs fall back to plain hipBLAS. On Strix Halo (gfx1151) that read prompts
  20-25% slower in one report (1,043 against 826 tok/s at 8K, hipBLASLt 1.4.1, a single run per arm, not a paired A/B). Point
  `STRATA_HIPBLASLT_TUNING` at the file in `tools/hip/` (`gfx1151-hipblaslt-100401.txt` for hipBLASLt 1.4.1), or copy it into a
  `tuning/` folder next to the `strata` binary. The start log then has the line `prefill gemm: hipBLASLt tuning enabled (N rows,
  gfx1151, version 100401)`; without it, no table was taken.

Run the registered checks with:

```sh
cmake --build build-hip -j2
ctest --test-dir build-hip --output-on-failure --timeout 60 -E '^(ple_parity|platform_memory_test)$'
```

`ple_parity` needs an external model fixture. `platform_memory_test` needs a memlock limit above the shell's default.

## Model and serving configuration

Prepare a supported model pack and its MTP runtime with the existing tools. For OrcaRouter IQ3_XXS, follow the
[compatibility conversion](ORCA.md). Add `--experts-bin` to the `tools/iq_pack.py` command: mmap requires the pack's
`experts.bin`, which the default native packing does not emit and which uses more disk. Keep each model's pack and tokenizer
together.

Start args for the original GSQ-RCO IQ3_XXS model (replace the paths):

```sh
build-hip/strata --serve \
  --pack packs/iq3xxs \
  --native /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto \
  --prefill 512 --spec 4 --spec-min-p 0.5 --mtp mtp/rt \
  --max-context 4096 --kv int8 --pool-workers 15 \
  --adapt-every 0 --pcie-frac 0 --vram-reserve-mib 1024
```

`--serve` is the engine's internal token protocol. For a browser or OpenAI API, put these args in the `args` array of the
[server JSON example](ORCA.md#local-server), set `exe` to `build-hip/strata`, and use the matching pack's `tokenizer`
directory:

```sh
.venv/bin/python -m serve.server --engine strata --config strata-hip.json --port 8080
```

- Select the AMD device with `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` if needed.
- Start with a modest context and prefill chunk. The 4K context above is a smoke-test start, not a model limit.
- The worker count (15) was used on a 16-core CPU; measure it for yours.
- Keep the PLE file on an SSD.
- Large pinned host allocations can fail on ROCm even when RAM is free. `--mmap-experts` avoids allocating the full pinned
  arena; it depends on OS file-cache residency and does not make SSD access equal to RAM.
- **The card also drives a Linux desktop:** keep more VRAM free than the default, for example
  `./setup.sh --vram-reserve-mib 3072`. With the cache filling the card, the desktop's next VRAM need makes amdgpu move
  memory to system RAM, and the OOM killer can end the compositor or apps.

**Long prompts on a big card: the lend-coverage line.** With `--resident-experts` the prompt path borrows cache slots, and
the experts of those slots must also sit in RAM. After the first request of a start the engine prints
`FileExpertSource: N of M of the prompt path's lendable slots keep their experts in RAM too`. When N is not M, the missing
experts are read from the pack during the prompt (also a `WARNING` line). The resident set is sized from `MemAvailable` at
start minus `STRATA_RESIDENT_HEADROOM_GIB` (default 4). If N is short of M, lower the headroom a little and restart; if the
machine then swaps or stalls, raise it. Re-check after a ROCm update.

### Switches for shared-expert and draft streams

| Switch | Default | Effect |
| --- | --- | --- |
| `STRATA_SH_STREAM=1` | Off on HIP, except the gfx1151 table | Shared-expert fork on its own stream. Output is identical; speed goes both ways by PC. `./setup.sh --calibrate` measures it and writes it to the config's `"env"` only when it wins by more than 3%. |
| `STRATA_MTP_SIDE_STREAM` | Off on HIP, on for gfx103x | `=0` turns it off on gfx103x, `=1` creates it on another HIP card |
| `STRATA_DENSE_MMQ=1` | Off | Dense GGUF prompt projections through int8 MMQ kernels instead of dequantize + hipBLAS. Greedy text can diverge after a few dozen tokens (q8_1 rounding). |
| `STRATA_HIP_ADAPT_KERNEL_COPY=1` | Off | Adaptive-tier swaps copied by a kernel instead of the SDMA engine; a workaround for a gfx1030 hang with MMQ prompts and adaptive swaps (untested by the reporter) |

**Why `STRATA_SH_STREAM` goes both ways.** The fork has to hide something. With experts on the CPU in the window, the shared expert forks
off only after the doorbell is published (`STRATA_SH_FORK_LATE`, `src/core/verify.cpp`), so it runs while the GPU would otherwise wait for the
CPU pool. On a 16 GB card whose cache holds only part of the experts that wait is about half of a verify window (one RX 6900 XT under Linux,
4,859 slots, `STRATA_DECODE_TIMING=1`: GPU-reach wait 20.7 of 39.1 ms per window), which is where #1272's gain comes from. When every routed
expert is on the GPU there is no wait to fill, and the fork is two cross-stream event dependencies per layer and nothing else: 7fd67dd measured
it on an R9700 with `--resident-experts` and the KV stream path running (48,067-token prompt), 43.4 tok/s with the fork against 62.9 without;
a 256-token prompt barely moved (65.4 and 62.7). Windows adds its own cost to every cross-stream dependency (#816).

**Windows with KV streaming: leave it off.** On one RX 6900 XT under Windows 11 (the 0.2.3 `rdna2` package, engine 0.1.48, Qwen3.8-Flash-Next
IQ3_S, 64K context, `--kv int8 --kv-resident 32768 --vram-reserve-mib 1536`) the fork together with KV streaming stopped the engine on its first
request ("verify: unspecified launch failure", in other starts first reported by the MTP draft step; exit code 0xC0000409). With the fork off, or
without `--kv-resident`, the same setup answered normally. Removing the other switches one at a time (`STRATA_HIP_PROMPT_F16`, `STRATA_HC_PERSIST`,
`STRATA_HIP_ADAPT_KERNEL_COPY`, `STRATA_PREFILL_CPU_SHARE`, `STRATA_PIPED_CONCURRENT`, `STRATA_PREFILL_STREAM_MIN`, `STRATA_IO_THREADS`, a rocBLAS
table) did not clear it; removing `STRATA_SH_STREAM` did. Two RX 6900 XT under Linux (layer split, the same model, the fork and `--kv-resident
32768`) have served without the fault since 0.1.40. It is one PC and one card, and the faulting kernel was not isolated. So `--calibrate` does
not measure the fork on Windows when KV streaming is on, `strata tune` never picks it there, and a config that sets it by hand gets a warning
line at the start; nothing is refused. If you have an RX 6000 card on Windows and want to help, run it once with `STRATA_SH_STREAM=1` and
`--kv-resident` and tell us in [#3895](https://github.com/Niko1221/Strata/pull/3895) whether the first request answered, with the card and the driver.


## Speed switches

Switches for the prompt path. A switch you set yourself, to anything, is never overridden by an architecture default. Put
them in the server config's `env`, or in the shell.

| Switch | Cards | Default | Effect |
| --- | --- | --- | --- |
| `STRATA_HIP_WMMA` | gfx12; gfx11 on request | gfx12 Linux: on with int8 KV. Windows and gfx11: `=1` | QSA prompt attention on the matrix cores. `=0` off; `=auto` takes it only for batches of `STRATA_HIP_WMMA_MIN_Q` (64) or more queries. Not bitwise: greedy text can differ from about token 50 on. |
| `STRATA_PF_FUSED` | gfx12 native IQ packs | gfx12 Linux: on for IQ3_S, IQ3_XXS and Q2_0 gate/up. IQ2 formats and Windows: `=1` | Prompt experts on int8 matrix-core kernels instead of MMQ. `=0` keeps MMQ; `STRATA_PF_FUSED_NATIVE=0` keeps MMQ with the flag set. |
| `STRATA_SCORES_THREAD` | gfx12, gfx1100 | gfx12: on; gfx1100: `=1` | Exact long-prompt QSA scorer (below). `=0` off. |
| `STRATA_SELECT_WMMA=1` | gfx12 | Off | Matrix-core block scorer for the QSA top-k; selects slightly differently |
| `STRATA_HC_Q8` | gfx12, gfx1151 | On | 8-bit hyper-connection weights; `=0` restores the bf16 read ([below](#strix-halo-8-bit-hyper-connection-weights-default-on-gfx1151)) |
| `STRATA_GDN_CHUNKED=1` | gfx11 | Off | Chunked prompt GDN (next section) |
| `STRATA_PF_GEMM`, `STRATA_PF_PAD` | gfx11 only | Off | RDNA4 runs neither; the engine does not pad where no kernel reads the stride |
| `STRATA_HIP_ROUTER_OLD=1` | all HIP | Off | Run the portable MoE router kernel instead of the fast one |
| `STRATA_PREFILL_WARM=0` | all | On | Turn off the warm-up BF16 and FP16 products run while the expert arena loads |
| `STRATA_HIP_QUEUE_PRIME=0` | gfx1200, gfx1201, Linux | On | Turn off the ordered opening of two GPU queues at start (it avoids slow starts) |
| `STRATA_PLE_AUTO_RAM=0` | all | On where the RAM fits | Keep the n-gram table in RAM when `--ple-io` was not given (RDNA4: 8K prompt +22.5%, see [RDNA4](#rdna4-gfx1201)); `=0` keeps the direct reads |
| `STRATA_TOPK_OLD=1`, `STRATA_TOPK_STREAM=0` | HIP | Off | Run the original top-k kernel, or the register kernel in the 90K-135K band |

### Opt-in chunked prompt GDN on gfx11

`STRATA_GDN_CHUNKED=1` runs the prompt's gated DeltaNet recurrence in 16-token chunks on the gfx11 matrix cores, with FP16
operands and FP32 accumulation, and fuses the output norm. It is off by default because rounding differs from the
token-by-token FP32 recurrence. The CUDA switch of the same name uses a separate FP32 implementation. `=2` is the same as 1
here.

The HIP path accepts gfx1100, gfx1101, gfx1102, gfx1150 and gfx1151 and prompt chunks of at least 64 tokens. Other cards,
shorter chunks, an output row stride not divisible by eight, and `STRATA_GDN_REC_HEADS` use the existing recurrence. Speed
and numerical checks were run on gfx1151 only.

> **Measured on:** Ryzen AI Max+ 395 (gfx1151), ROCm 7.14.1, UD-IQ4_XS, engine version not recorded, whole-prompt `prompt_ms`, 11 alternating
> pairs per row, geometric mean of off / on. "Section 5" is the nine variables of STRIX_HALO.md section 5; "defaults" is the
> engine without them.

| Tokens | Arm | Off -> on | Off / on |
| ---: | --- | ---: | ---: |
| 4,096 | section 5 | 3.703 -> 3.641 s | 1.018 |
| 4,096 | defaults | 5.914 -> 5.829 s | 1.014 |
| 32,768 | section 5 | 26.157 -> 25.589 s | 1.021 |
| 32,768 | defaults | 44.082 -> 43.479 s | 1.015 |
| 128,000 | section 5 | 107.149 -> 104.866 s | 1.021 |
| 128,000 | defaults | 193.439 -> 191.055 s | 1.014 |

That is about 2% on top of section 5 and 1.5% on the defaults. First-token KL against the run without the switch (12 prompts of
1.5K-24K tokens) is 0.004 to 0.007 mean, with the same top token in all of them. Long generation and perplexity were not
measured. To check it on your build:

```sh
cmake --build build-hip --target hip_gdn_chunked hip_gdn_rec_head
ctest --test-dir build-hip -R 'hip_gdn_(chunked|rec_head)' --output-on-failure
for i in 1 2 3; do ./build-hip/hip_gdn_chunked --bench-default; ./build-hip/hip_gdn_chunked --bench; done
```

### Exact long-prompt QSA scorer

`STRATA_SCORES_THREAD=1` enables a HIP port of the exact score kernel on gfx1100 for batches of at least 64 queries with at
least 16,385 live index blocks. On gfx12 it is on by default for those batches and `STRATA_SCORES_THREAD=0` turns it off. It
is bitwise equal to the warp scorer. The default, short-prompt and decode paths keep the existing scorer.
`STRATA_SCORES_THREAD_TRACE=1` prints an engagement line.

> **Measured on:** Radeon AI PRO R9700 (gfx1201), 256-query batches, fresh prompts, engine 0.1.45.

| Prompt | Scorer stage | Prompt read |
| --- | --- | --- |
| 131K tokens | 3.6x faster | 8.7% faster |
| 230K tokens | 4.0x faster (at 240K) | 16.4% faster |

On one RX 7900 XTX the gain was 10-13% at about 131K tokens and 22% at 240K. Cached follow-ups and a two-task coding pilot
showed no repeatable gain, so this is opt-in on gfx1100, not a coding-speed recommendation. Data and limits:
[bench/results/2026-10-11-gfx1100-qsa-scores](../bench/results/2026-10-11-gfx1100-qsa-scores/README.md).

### Deep contexts: the prompt's top-k selection

Past about 135K cells (and from about 90K cells on a prompt read) the top-k selection runs the wide kernel, as on CUDA, with
the same ids as the reference radix. On an R9700 a fresh 500K-token prompt reads 34% faster than with the reference kernel
(716 -> 962 tok/s; 1,284 tok/s with the exact scorer above).

## Linux: verify timeouts while the kernel reclaims host memory

A `verify: timed out at layer N` or "no progress for 60 s" message does not by itself mean a kernel bug. Two community
reports found the same mechanism: the GPU's queues are suspended while the kernel reclaims host pages the GPU has pinned
through a KFD userptr. Each is one machine and one workaround; neither is a default, and neither is known to matter on
Windows or other ROCm versions.

| Case | Workaround in the server JSON `env` |
| --- | --- |
| Paged host allocations (two R9700, ROCm 7.2, #750) | `"HSA_USERPTR_FOR_PAGED_MEM": "0"` |
| `--mmap-experts` (RX 6800, ROCm 7.2.4, 31 GiB RAM, #920) | `"GPU_PINNED_MIN_XFER_SIZE": "1048576"` |

Merge the field into the existing config (this is not a complete run config), change one setting at a time, restart the
engine, and remove it to go back. If the engine runs in a systemd unit, limit it with `MemoryMax`, never `MemoryHigh`:
`MemoryHigh` counts the page cache that `--mmap-experts` reads from.

## RDNA4 (gfx1201)

The RX 9070 / 9070 XT and the Radeon AI PRO R9700 run the same kernels as gfx1100: wave32, 64 KiB of LDS per workgroup, the
signed dot4 instruction and a 100 MHz wall clock. Switch defaults for gfx12 are in [Speed switches](#speed-switches).

- **Build and tests:** `-DCMAKE_HIP_ARCHITECTURES=gfx1201 -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON`. On an R9700,
  `ple_parity` (needs the Q2_0 fixture) and `expert_multi_test` (needs AVX-512) fail for reasons outside the GPU.
- **Architecture check:** a gfx1100-only build stops on a gfx1201 card at startup with a plain message.
- **hipBLASLt:** the gfx120X wheel's hipBLASLt is 1.2.0 and no table ships for it, so Linux `rx9070` uses plain hipBLAS. A
  table calibrated for it measured within noise end to end. Tables for 1.2.2 (system ROCm 7.2.4), 1.4.1 and 1.5.0 exist in
  `tools/hip` ([Tuning table](#tuning-table)). With ROCm 7.2.4's hipBLASLt 1.2.2 the plain path is far off, and its table
  roughly doubled prompt speed.
- **Two cards in one run (layer split):** `"backend": "hip", "gpu": [1, 0]` in the config (setup writes it with `--gpus 1,0`).
  With every expert on the GPUs the split gives the same tokens as the R9700 alone. On an R9700 plus a 9070 XT the split
  does not pay, because the R9700 alone holds all 12,288 expert pairs. A split is worth it when no single card holds the
  model's experts. On Linux a split pins the whole expert arena (the 8 GiB cap is for Windows and WSL2).
  `"split_skip_if_fits": true` (opt-in) runs such a pair on the first card alone when it holds every profiled expert.
  The conversation cache refuses a split at start (exit 2). See [MULTI_GPU.md](MULTI_GPU.md).
- **First prompt after a start** pays two one-time hipBLAS library loads. The engine runs a tiny BF16 and FP16 product while the
  arena loads so the first request does not carry them (`STRATA_COLD_TRACE=1` times them).
- **Pictures:** the card's reader (`strata-vision-hip`) keeps about 1.4 GiB of VRAM. It has not run on an RDNA3 card
  (gfx1100-gfx1102) or on Windows with an AMD card.
- **PLE table residency (0.2.7):** when `--ple-io` is not given and the machine has room for the n-gram table plus
  8 GiB, the engine keeps the table in RAM (`--ple-io ram`: map, pre-touch, mlock attempt; `STRATA_PLE_AUTO_RAM=0`
  turns the policy off, an explicit `--ple-io` always wins). The rows are random reads in a 26.8 GiB table, so this
  is the one prompt lever that paid on this card. Measured on p620 (R9700 32 GB, IQ3_S, engine 0.1.51, 24576 expert
  pairs of which 13207 resident): 8K prompt 2045 tok/s with direct reads, 2276 with `mmap`, **2506 with the table in
  RAM (+22.5%)**; 32K prompt 2522 against 2754 (+9.2%). Decode unchanged, answer md5 identical; the project's
  `ple_reader_selftest` proves the mapped rows are the same bytes as the direct ones. The table is 26.8 GiB
  (320,001,536 rows of 90 B), so a start reads it once: about 3 s when it is in the file cache, up to a minute
  from a cold disk, and it needs the RAM for it while the engine runs.
- **Decode: the miss split starts at 0.15 on gfx12 (0.2.7).** `--pcie-frac` decides how many of the missing
  expert rows are fetched over PCIe and how many are computed by the CPU pool. The x16 share (0.55) was measured on
  a machine where the link was the cheap side; on an R9700 with idle CPU cores it is the pool: IQ3_S, 3 rotated
  pairs, PLE direct, **8K decode 86.1 -> 94.6 tok/s at 0.15 (+9.9%), 32K 67.2 -> 84.4 (+25.6%)**, 64K 69.1 -> 72.3
  (+4.7%), prompt speed unchanged. Under `serve/server.py` a short chat decodes 84.4 -> 93.2 (+10.4%). The CPU lane
  rounds a missing expert's FP32 rows differently from the GPU lane, so this changes bits: the first-token KL of 0.0
  against 0.55 is **0.0013 mean** (median 0.0011, max 0.0047, same top-1 in 12 of 12 prompts), below the engine's own
  implementation-noise scale. gfx12 native packs therefore **start** at 0.15 (`STRATA_PCIE_FRAC_DEFAULT=old` restores
  0.55; an explicit `--pcie-frac` always wins); the CUDA and gfx11 parts keep their own measured values. This is the same
  class of finding as the pcie-frac default being wrong on the 3060 and P100 in the CUDA plan.
- **Not validated:** answer-quality benchmarks and contexts beyond 16K on the 9070 XT.

> **Measured on:** Radeon AI PRO R9700 32 GB and RX 9070 XT 16 GB (gfx1201), Ryzen 9 3900X, 47 GB RAM, Ubuntu 24.04
> (KVM/VFIO guest), system ROCm 7.14, no hipBLASLt table, engine 0.1.30, Coder IQ1_M, `--expert-cache auto --prefill auto
> --spec 4 --spec-min-p 0.5`, MTP, `--kv int8`, 32K context, 128 greedy tokens.

| Card | Expert cache | 4K prompt | 4K decode | 16K prompt | 16K decode |
| --- | --- | ---: | ---: | ---: | ---: |
| R9700 32 GB | 12,288 slots | 982 tok/s | 45.5 tok/s | 1,402 tok/s | 48.3 tok/s |
| RX 9070 XT 16 GB | 4,931 slots | 782 tok/s | 30.8 tok/s | 1,235 tok/s | 35.4 tok/s |

Later engines are faster; see [History](#history--older-measurements) for the R9700 gains from WMMA attention, fused
experts and the router kernel.

## Integrated GPUs (mini PCs)

On a Radeon that shares the computer's memory (Radeon 780M / 760M gfx1103, 890M / 880M gfx1150, Strix Halo gfx1151) the
matrix-core prompt set below is on by default on the Radeon 780M (gfx1103, Linux) and opt-in on the others, because it
changes bits and only the 780M was measured. The engine picks the set from the device's own "integrated" flag and the
architecture. A discrete card never gets it, and neither does a computer that also holds a discrete card beside the iGPU.
The engine prints one line at start-up that names what it set.

| Switch | What it does |
| --- | --- |
| `STRATA_HIP_WMMA=1` | Prompt attention on the matrix cores (needs `--kv int8`) |
| `STRATA_PF_GEMM=1` | The prompt's FP16 projections on the 128 x 256 WMMA GEMM |
| `STRATA_PA_FAST=1` | Prompt attention with single FP16 q and p |
| `STRATA_SELECT_WMMA=1` | The block scorer on the matrix cores |
| `STRATA_HC_UPMIX=1` | The hyper-connection up projection with its epilogue |

`STRATA_IGPU_PROMPT_SET=0` turns the whole set off, `=1` turns it on where it is not the default. Each switch keeps `=0` as its
off switch. `STRATA_IGPU_DEFAULTS=0` turns the set off whatever else is set. The same code applies on Windows but was not
tested there. Strix Halo has separate speed switches ([STRIX_HALO.md](STRIX_HALO.md)).

> **Measured on:** Radeon 780M (Ryzen 7 255), Q2_0, `--kv int8`, 4K prompt, engine version not recorded, one run each.

| Setting | Prompt | Decode |
| --- | ---: | ---: |
| None of the switches | 109.5 tok/s | 16.6 tok/s |
| First four | 214.7 tok/s (+96%) | 17-19 tok/s (noise) |
| All five | 233 tok/s (+111%) | same |

A repeat with 6 interleaved pairs gave prompt x1.60 at 1K and x2.03 at 4K, decode x0.999, identical greedy ids. The
switches round differently from the old path; on an R9700 with the same kernels the teacher-forced KL was 0.0003 to 0.0014.

**The 780M resets its GPU on some firmware.** On the current Ubuntu firmware (gc 11.0.1 MES) a GPU reset can follow when the
kernel evicts the process's GPU queues while a kernel with barriers runs ([RADEON_780M.md](RADEON_780M.md)). The prompt set does
not cause it. `STRATA_ARENA_PIN_GIB=none` shortened the eviction time but did not stop the resets and cost 1-4% speed, so it
is not the default. A newer `linux-firmware` is the likely cure and is a host change.

## Community-validated cards

Run by their owners, not on the maintainers' machines. Setup accepts them like gfx1100 and gfx1201. A hipBLASLt table ships
for gfx1200; gfx1101 has none (make one with [Tuning table](#tuning-table) and compare the prompt speed).

> **Measured on:** the machine in each row, Coder IQ1_M, MTP, owner-calibrated tables.

| Card | Machine | Result |
| --- | --- | --- |
| gfx1101, RX 7800 XT 16 GB (#254) | Ryzen 9 5950X, 121 GiB RAM, hipBLASLt 1.4.1, engine 0.1.29, 64K context | Prompts 4K-9K at 898-953 tok/s, decode 38-44 tok/s |
| gfx1200, RX 9060 XT 16 GB (#256) | Ryzen 9 7950X, 64 GiB RAM, ROCm 7.2, hipBLASLt 1.2.2, engine 0.1.29 | 2,374-token prompt 540 tok/s, decode 27.1 tok/s; 65K prompt 748-753 tok/s; 130K prompt 725 tok/s; llama.cpp HIP on the same card: 20 tok/s decode, 450 prompt |
| gfx1102, RX 7600 XT 16 GB (#942) | i7-13700K, 64 GB, Swift IQ3_XXS, 146 requests up to 51K context | About 43 tok/s decode, 250-300 tok/s prompt; setup accepts it, unvalidated |

The shipped gfx1200 table (`gfx1200-hipblaslt-100202.txt`) against plain hipBLAS, engine 0.1.31, two runs each:

| Prompt | With table | Plain hipBLAS | Gain |
| --- | --- | --- | --- |
| 2,374 tokens | 597-599 tok/s | 396-398 tok/s | 1.50x |
| 65,045 tokens | 754-757 tok/s | 380 tok/s | 1.99x |
| 130,091 tokens | 719 tok/s | 363-370 tok/s | 1.98x |

## RDNA2 (gfx1030)

The RX 6800 / 6800 XT / 6900 XT / 6950 XT run the same kernels: wave32 and 64 KiB of LDS per workgroup. RDNA2 has no
`v_dot4_i32_iu8`, so `dp4a` uses the plain signed `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`). There is no WMMA; the QSA scorer
takes the ordered FP32 fallback. CMake lists gfx1030 as unvalidated. The `rx6000` and `rdna2` packages are built and not tested on
a real RX 6000 card; the numbers here come from community machines with engines built from source. Please report a run on an
RX 6000 card with the end of `strata-<model>.log`.

hipBLASLt ships no gfx1030 kernels, so there is no hipBLASLt table and the plain hipBLAS path runs. rocBLAS on gfx1030 has tuned
kernels for FP16-in / FP16-out GEMMs only; the prompt path's FP16-in / FP32-out and BF16 products run generic kernels about
6.6x slower. The switches below address that. All are opt-in and none is bit-identical to the default.

| Switch | Effect |
| --- | --- |
| `STRATA_HIP_PROMPT_F16=1` | Runs the 16-bit prompt GEMMs FP16 in and out (the engine prints a tip on gfx103x). `STRATA_F16_RANGE=1` prints, per prompt, the largest value that entered or left FP16 (FP16 ends at 65504); `STRATA_DBG_NAN=1` counts non-finite FP16 outputs. Checks: [2026-10-04-rdna2-fp16-prompt](../bench/results/2026-10-04-rdna2-fp16-prompt/README.md). |
| `STRATA_RDNA2_SGEMM=1` | Widens both GEMM inputs to FP32 (exact) and runs SGEMM, no 65504 limit. `=all` forces it on any AMD card. With `STRATA_HIP_PROMPT_F16=1` the FP16 route runs first and SGEMM takes the products it leaves. Slower than the FP16 route. |
| `STRATA_ROCBLAS_TUNING=<file or dir>` | A rocBLAS solution table for the FP16 route; setup points the engine at the directory when a table for the architecture exists. The log says `rocBLAS tuning enabled (... rows ...)` or why not. See [rocBLAS table](#rocblas-table-gfx103x). |
| `STRATA_SELECT_SIMT=1` | QSA block scores on the FP32 tiled scorer instead of the warp kernel; helps long prompts (128K: +7.4%). Not bitwise the warp kernel. [Data](../bench/results/2026-10-07-rdna2-qsa-select-simt/README.md). |
| `STRATA_HC_PERSIST=1` | The verifier's hyper-connection read as one persistent kernel (HIP, wave32). Measured on 2x RX 6900 XT: -2.8% to -4.2% ms per verify window. Not offered on gfx11 and gfx12. Opt-in for measurement on gfx103x. [Data](../bench/results/2026-10-12-hc-persist-gfx1030/README.md). |
| `STRATA_FILE_CACHE_OLD_RULE` | gfx906 keeps the old file-tier cache rule; `=1` forces it on any Linux build, `=0` the current one |

  The exact calls on the card (opA = T, opB = N, FP32 out): N 10240 x T 8192 x K 2560 86.7 -> 27.7 ms including
  the widening; N 320 x T 8192 x K 10240 (BF16) 10.9 -> 7.5 ms. Against a float64 reference on random inputs in
  [-1, 1], the SGEMM route is as close as or closer than the native call on all 10 FP16 / BF16 shapes checked; the
  two differ by at most 4.3e-6 of the largest output. A 9-task code-repair benchmark (the model patches seeded bugs
  in a browser game until 38 browser checks pass) solved 9/9 on the first try with both, model time 458 -> 374 s.
  With the switch on and `STRATA_HIP_PROMPT_F16` unset the SGEMM route runs; with `STRATA_HIP_PROMPT_F16=1`
  the FP16 route above runs first and the SGEMM route takes the products it leaves (`beta != 0`). Same RX 6800, one
  session, fresh 12K-token prompts: native 310, SGEMM 439, FP16 (#835) 494 tok/s; `STRATA_F16_RANGE=1` on the Coder
  IQ1_M peaked at 105.3 (activations) and 424.8 (FP16 outputs) with nothing beyond 65504, and the repair benchmark
  solved 9/9 on the first try with both (model time 374 s SGEMM, 307 s FP16). The SGEMM route is the slower of the
  two; no value passes through FP16, so it has no 65504 limit to stay under. It is not bit-identical to the native
  call (at most 4.3e-6 of the largest output apart), which can flip a near-tied greedy token like any such change.
- **rocBLAS solution table** (for the `STRATA_HIP_PROMPT_F16=1` route above; with it off, nothing runs through the
  table): rocBLAS picks the kernel for each of those FP16 GEMMs from its own table by shape, and
  on gfx1030 that pick is far from the best kernel the library holds for some shapes: the GDN input projection
  (N = 10240, K = 2560) below about 1,150 tokens runs at 5 TFLOPS where the best of rocBLAS's own 237 solutions for the
  same shape runs at 32-38 (7x; the kernel choice flips at T = 1152). `tools/hip/tune_rocblas` enumerates the library's
  solutions per shape and token bucket, checks each against the default kernel's output and writes the winners as
  `tools/hip/<arch>-rocblas-<version>.txt`; the engine reads the table through `STRATA_ROCBLAS_TUNING` (a file, or
  the directory, from which it takes the file of its own rocBLAS build) and runs those solutions in place of rocBLAS's
  choice, the default kernel for every shape without a row. Solution indices are valid for one architecture and one
  rocBLAS build (the full version string, `5.6.0.8d1ae90e` for ROCm 10.0.0's), and the engine refuses any other
  table. Shipped: `gfx1030-rocblas-5.6.0.8d1ae90e.txt` (ROCm 10.0.0, the `/opt/rocm` setup installs on Ubuntu), measured
  on an RX 6900 XT. Other gfx1030 cards (RX 6800 / 6800 XT / 6950 XT, Radeon PRO W6800 / V620) run it as well - the
  indices are valid on every gfx1030 - but have fewer CUs or other clocks, so their own table may pick differently;
  `tune_rocblas` makes one in about 5 minutes ([Tuning table](#tuning-table)), and a report of what it changed is welcome. Setup
  points the engine at the directory when a table for the card's architecture exists, and the engine's log says
  `rocBLAS tuning enabled (... rows ...)` or why not. Measured on the 2x RX 6900 XT machine above (one card, #835 +
  this, `STRATA_PREFILL_TIMING=1`, temperature 0, the same answers with and without): a 799-token prompt's GPU time
  3,329 -> 2,985 ms (the GDN projections 443 -> 177 ms, `hc read` 270 -> 77), 495 tokens 2,477 -> 2,139 ms (365 ->
  146), 8,874 tokens 11,978 -> 11,459 ms (the 682-token tail chunk's projections, `hc read` 1,200 -> 898); a
  1,108-token prompt did not change (3,391 -> 3,395 ms): its GDN time fell 598 -> 201 ms and the whole saving
  reappeared as `wait copy`, the experts' PCIe transfer, which bounds that prompt. Per-run lines and the tuner's
  summary: [bench/results/2026-10-05-rdna2-rocblas-table](../bench/results/2026-10-05-rdna2-rocblas-table/README.md). Decode does not use these GEMMs and is unchanged. See [Tuning
  table](#tuning-table) for making a table for another rocBLAS build.

- **Decode IQ expert gate/up (`STRATA_EXP_IQ2S=1`, opt-in):** use with `STRATA_EXP_MODE=8`. It covers
  IQ2_S, IQ3_XXS and IQ3_S gate/up. IQ2_S's shared codebook holds two-bit magnitude selectors; a byte
  permutation forms the signed weights. IQ3 uses a small sign-mask table. Activations are read from global
  memory. Unset or `0` keeps the original kernel. Set it before starting the engine. On one PRO V620 (gfx1030),
  Linux, the 24-layer lab at 4 tokens x 10 experts gave gate/up per layer (IQ4_NL-down pair): IQ2_S
  84.4 -> 73.0 us, IQ3_XXS 87.6 -> 77.1, IQ3_S 89.7 -> 84.0. Small IQ3 calls now keep the original
  kernel: IQ3_XXS with up to 10 entries and IQ3_S with up to 20 entries. This pack uses ten entries per token.
  That restores IQ3_S at 1 token from 42.3 to 40.1 us, and at 2 tokens from 60.5 to 60.4 us.
  All 240 cases at 1–8 tokens match both the original q8_1 intermediate and final output bit for bit.
  [Before/after timings, two-GPU probe rows and limits](../bench/results/2026-10-20-gfx1030-iq2s-decode/README.md).
  Other HIP cards, Windows, CUDA and SYCL were not tested.

- **Decode Q2_0 expert down (`STRATA_EXP_DOWN=1`, opt-in, HIP):** for the 2560-by-640 expert shape.
  A small shared lookup table expands packed weights into signed bytes. The kernel reads activations directly
  from global memory and processes eight rows per warp. Unset or `0` keeps the existing down path.
  On one PRO V620 (gfx1030), Linux, the 24-layer lab behind IQ2_S gate/up gave down per layer:
  T=1 20.1 -> 12.8 us, T=4 44.8 -> 40.7 us, T=8 74.8 -> 66.3 us. All Q2_0 rows at T=1–8 improved.
  The measured base uses `STRATA_EXP_MODE=8`, `STRATA_EXP_IQ2S=1`, `STRATA_EXPERT_DOWN_L=1`, and `STRATA_TSUM=1`.
  All 240 hash cases match the original q8 intermediate and final output bit for bit.
  [Part 2 timings, probe rows and limits](../bench/results/2026-10-20-gfx1030-iq2s-decode-part2/README.md).
  Other cards, shapes and model packs were not tested.

- **IQ prompt expert tiles (`STRATA_PF_IQ2S=1`, `STRATA_PF_IQ3=1`, opt-in, gfx103x):** 256 weight rows by
  16 token rows, scalar activation loads and packed signs. The switches cover IQ2_S and IQ3_XXS / IQ3_S,
  respectively; unset keeps MMQ. With MMQ's q8_1 activations every tested output matched bit for bit
  (`hip_prefill_iq{22,18,21}_*` CTests). On one PRO V620, Linux, amdclang 24 from `/opt/rocm`, best of five after
  warmup, 16 experts x 40 rows, W[1280 x 2560]: IQ2_S 9.9 -> 11.2 T MAC/s, IQ3_XXS 10.6 -> 12.3,
  IQ3_S 10.5 -> 11.4; at 320 rows: 13.4 -> 16.0, 15.7 -> 17.6, 15.6 -> 16.9. On 2x PRO V620 with
  `sc117-abl-iq3_s`, `strata_bench.py --runs 1`, two baselines and one IQ2_S run: 4K 1,897-1,902 -> 1,937 tok/s,
  16K 2,511-2,520 -> 2,546. Adding IQ3 in a separate A/B: 4K 1,930-1,932 -> 1,958, 16K 2,527-2,556 -> 2,570;
  that extra gain is close to run variation. Both prompt answers started with “The”. The pack has no IQ2_S / IQ3
  down matrix: K=640 cannot hold 256-value blocks, so down parity uses K=768. Other GPUs, Windows, CUDA, SYCL,
  longer prompts and end-to-end residual identity were not tested.
  Tried on the same kernels and dropped (2026-10-20, one PRO V620, `hip_prefill_expert_bench`, every variant the
  same bits as the 16-row tile, gate+up 16 experts x 40 / 80 / 320 rows): 8, 12, 24 or 32 token rows a tile instead
  of 16 (IQ3_S 11.4 -> 9.8 / 10.6 / 10.7 / 9.4 T MAC/s at 40 rows; 24 rows gains 5% on IQ2_S at 320 rows only); two
  weight rows a thread on 128 threads (9-12% slower); the kernels compiled for 64-lane waves in a translation unit
  of their own (`-mwavefrontsize64`, 5-12% slower). End to end the two switches are still worth having on: on 2x
  PRO V620 with `STRATA_HC_DOWNSK=8 STRATA_HC_UPMIX16=1 STRATA_PF_NOWIDEN=1` and the strata-sync base settings
  (which leave them off), alternated, `--runs 1`: 4K 1,976.3 / 1,976.0 -> 2,045.6 / 2,047.9 / 2,032.3 / 2,045.4
  tok/s, 16K 2,622.0 / 2,619.4 -> 2,700.6 / 2,702.7 / 2,698.9 / 2,697.0 (+3.4%, +3.0%). `STRATA_PA_DOT4_CFG=16` on top
  of that loses: 1,967.1 / 1,967.4 and 2,587.7 / 2,589.2 (-3.7%, -4.1%). With the IQ tiles on, the routed experts' gate / up are
  still the largest GPU phase of the first stage on a 16K prompt (1,065 of 5,434 ms).

- **Q4_0 / Q4_1 prompt experts on 256 x 32 tiles (`STRATA_PF_Q4_WIDE=1`, opt-in, gfx103x):** a plain Q4_0 GGUF's
  prompt experts do not go through rocBLAS or llama.cpp's MMQ on RDNA2 but through Strata's own int8 kernel
  (`src/prefill/moe_q4.cu`, on by default there, `STRATA_PF_Q4=0` turns it off). Its 64 x 32 tile waits on shared
  memory: every 4 dots need a 16-byte read of activations from LDS. With the switch the Q4_0 products read a tile's
  token rows through scalar loads instead (no shared memory, no barrier), fold the zero point into the weight bytes,
  and give each thread one weight row against 32 token rows; the Q4_1 products keep shared memory and put two weight
  rows on a thread. The output is the same, bit for bit (`hip_prefill_q4_parity` compares every output of both
  shapes; a 3.2K-token prompt's final residuals, `STRATA_PREFILL_DUMP_R` with the CPU share off, are the same file
  with and without). Measured on 2x Radeon PRO V620 (gfx1030, 32 GB each, Ryzen 9 9950X, 92 GB, Linux, ROCm in
  `/opt/rocm`, rocBLAS 5.8.0), this branch: the kernel alone, 16 experts x 320 token rows, gate+up 6.4 -> 22.3 T MAC/s,
  down 6.2 -> 23.6 (Q4_0) and 7.6 -> 15.9 (Q4_1) - llama.cpp's MMQ on the same card and shapes runs its i-quant
  kernels at 13-18 (`hip_prefill_expert_bench`). End to end, bartowski's Swift Q4_0 on both cards (`--layer-split 24`,
  the `q4_0-stage13-q.json` settings, `tools/strata_bench.py --runs 1`, the same day, two runs each): a 4K prompt
  1,083-1,090 -> 1,294-1,298 tok/s, a 16K prompt 1,393-1,398 -> 1,595-1,597 tok/s; decode unchanged. With
  `STRATA_PREFILL_TIMING=1` the 16K prompt's `gemm gate/up` fell 2,556 -> 888 ms and `gemm down` 1,227 -> 386 ms of a
  10.4 s timeline; about a quarter of that saving came back as waiting for the streamed experts' copies
  (`embed+steps` and `wait copy`, +600 ms), which is what bounds that prompt now. An i-quant pack does not use this
  kernel and is unchanged (IQ3_S: 1,400 / 1,830 tok/s without, 1,406 / 1,833 with). Not measured: one card, a
  gfx1031 / gfx1032 card, Windows.

- **The dense projections around the rocBLAS GEMMs (`STRATA_DQ8=1`, `STRATA_HC_GATED16=1`, `STRATA_HC_DOWNINJ=1`,
  opt-in):** `STRATA_DENSE_TIMING=1` (debug) prints, per prompt, the GPU time of every dense projection by step and
  shape. On 2x Radeon PRO V620 with the IQ3_S pack at 2048-token chunks the large products already run at rocBLAS's
  best (N 10240 K 2560 at 35 TFLOP/s, N 2560 K 6144 at 31, N 6144 K 2560 at 34, N 10240 K 320 at 32), so a kernel of
  our own for them had nothing to win; the time was around them. (1) `STRATA_DQ8=1`: the dequantization of Q4_0, Q5_0,
  Q4_K, Q5_K, Q6_K and IQ4_XS weights with four threads per 32-value group and one 16-byte store each, as Q8_0
  already ran. The same bits (`hip_dq8_parity`: every type into FP16, BF16, FP32 and padded FP16). A 10240 x 2560
  Q6_K weight: 1.42 -> 0.08 ms, which was 2.0 ms in front of a 3.0 ms GEMM in every GDN layer of every chunk; the
  other types 0.11 -> 0.06-0.07 ms. (2) `STRATA_HC_GATED16=1`: the hyper-connection up projection's T x 10240 FP16
  output is not widened in place; the mix widens each value as it reads it. The same bits. (3)
  `STRATA_HC_DOWNINJ=1`: the down (N 320) and inject (N 4) projections, which read the same T x 10240 image, as one
  N 324 product; it takes what N 320 takes, and the N 4 product cost 0.19 ms on its own. Another rocBLAS kernel may
  run, so the sums can round differently: against an FP64 reference of the same FP16 operands both sequences are at
  rel L2 3.1e-4 for silu(down) and 2.0e-4 for inject (`hip_hc_read_parity`, which also checks (2) bit for bit). The
  table gained rows for N 324 K 10240, and for the GDN alpha / beta projections as the engine calls them (N 48 into
  rows of 96: `ldy` 96 had no row, so they ran the default kernel, 0.123 -> 0.056 ms at 2048 tokens). End to end, a
  3K-word prompt's final residuals (`STRATA_PREFILL_DUMP_R`, the CPU share off) are the same file with (1) + (2);
  with (3) and the new rows they differ by rel L2 3.5e-2, where the new rows alone give 4.2e-2 and running without
  the table 3.3e-2 (any change of a GEMM's summation order moves them that far), and the answer's text is the same.
  Measured on 2x Radeon PRO V620 (gfx1030, 32 GB each, Ryzen 9 9950X, 92 GB, Linux, ROCm in `/opt/rocm`, rocBLAS
  5.8.0), this branch, `--layer-split 24`, the `iq3_s-stage13-q.json` settings plus `STRATA_PA_DOT4=1
  STRATA_GDN_HEAD=1 STRATA_GDN_PP=2 STRATA_GDN_NOY=1` in both arms, `tools/strata_bench.py --temperature 1.0 --top-p
  0.95 --top-k 20 --runs 1`, the same night, two runs each: a 4K prompt 1,497-1,499 -> 1,600-1,602 tok/s, a 16K
  prompt 1,963-1,964 -> 2,094-2,099 tok/s; with (1) + (2) only, one run: 1,588 / 2,082. Of the first stage's
  8.62 s of marked GPU time on the 16K prompt (8.04 s with all three), dequantization fell 386 -> 53 ms, the hyper-connection GEMMs + widening
  + mix 812 -> 660 ms. bartowski's Swift Q4_0 (its dense weights are Q4_0 and Q8_0, which were not slow), one run
  each: 1,136 -> 1,157 and 1,468 -> 1,489 tok/s. Decode does not use these paths. Not measured: one card, another
  gfx103x card, Windows, K-quant expert packs; tried and dropped as measured no gain: one GEMM for the GDN qkv + gate
  projections (both already at 34-35 TFLOP/s), a retune of N 640 / N 512 at 2048 tokens (best 0.283 against the
  default's 0.285 ms; 0.201 against 0.218).
- **The hyper-connection products on v_dot2 (`STRATA_HC_DOWNSK=8`, `STRATA_HC_UPMIX16=1`, opt-in, gfx103x, the
  FP16 prompt path):** two hand-written FP16 GEMMs on `v_dot2_f32_f16` that read the BF16 weights as they are
  (each value converted in the kernel as the FP16 path converts it, so no conversion pass and no FP16 copy).
  (1) `STRATA_HC_DOWNSK=<S>`: the [down; inject] product of `STRATA_HC_DOWNINJ` (N 324, K 10240) with K split in
  S parts (S divides 10240 into 64-value steps: 1, 2, 4, 5, 8, 10, 16 or 20; the other values up to
  31 run `STRATA_HC_DOWNINJ`'s product). rocBLAS's best kernel for that shape runs at 17 TFLOP/s here, because at 2048 tokens its tiles make
  about 48 workgroups for 72 compute units; rocBLAS's own split (strided batched, FP16 in, FP32 out) has no tuned
  kernel on gfx1030 and ran 1.8-3.2 ms. Here each workgroup sums a 128 x 112 tile of one part in FP32 into `gated`,
  and the SiLU kernel adds the S parts in order and rounds each output to FP16 as the FP16-out GEMM does. S = 8:
  down + inject + SiLU at 2048 tokens 0.778 -> 0.563 ms (`hip_hc_downsk_parity --bench`; 1024 tokens 0.412 ->
  0.329, 4096 tokens 1.231 -> 1.056); in the engine (`STRATA_DENSE_TIMING=1`, first stage, 16K prompt) the
  product went from 0.832 ms plus 0.051 ms of weight conversion and 0.022 ms of widening to 0.587 ms. (2)
  `STRATA_HC_UPMIX16=1`: the up projection (N 10240, K 320) with the mix as its epilogue, as `STRATA_HC_UPMIX` does
  on gfx11 and CUDA: a thread sums 8 tokens x 8 rows (2 outputs with their 4 streams), rounds each sum to FP16 and
  runs `gr_mix_r`'s arithmetic in its order, so the T x 10240 `gated` image is not written nor read back. Up +
  mix at 2048 tokens 0.686 -> 0.606 ms, 4096 tokens 1.507 -> 1.175 (`hip_hc_upmix16_parity --bench`; at 1024
  tokens 0.287 -> 0.293, no gain). Both change only the order of the FP32 sums: against an FP64 reference of the
  same FP16 operands the outputs are as far as the default's (silu(down) rel RMS 3.0-3.2e-4 in both, inject
  1.7-2.1e-4, `mixed` 5.7-6.1e-5), finite, and the same bits run to run; 0.4-0.6% of the FP16 SiLU values and 0.3%
  of `mixed` differ from the default's. A 3K-word prompt at temperature 0 (the CPU share off): the final residuals
  are the same bits run to run without the switches, 4.0e-2 (rel L2) from them with (1), 3.5e-2 with (1) + (2) -
  what any change of a GEMM's summation order does (see the paragraph above); the answer's text is the same with
  (1) + (2), and with (1) alone it takes another word from its 21st on (both read well). Measured end to end on 2x Radeon PRO V620 (gfx1030,
  ROCm in `/opt/rocm`, rocBLAS 5.8.0), the IQ3_S abliterated pack, `--layer-split 24`, the coordinator's base
  settings (with `STRATA_HC_DOWNINJ=1`, `STRATA_HC_GATED16=1` and the rocBLAS table) in every arm,
  `tools/strata_bench.py --temperature 1.0 --top-p 0.95 --top-k 20 --runs 1`, the arms alternated on one binary:
  without the switches a 4K prompt 1,888.5 / 1,892.5 / 1,886.8 / 1,890.6 tok/s and a 16K prompt 2,505.0 / 2,506.3 /
  2,499.8 / 2,504.3; with (1) 1,912.6 / 1,920.1 / 1,913.2 and 2,540.7 / 2,552.6 / 2,532.4 (+1.4%, +1.5%); with (1)
  + (2) 1,849.3 / 1,933.5 / 1,934.7 / 1,939.1 and 2,572.4 / 2,566.2 / 2,561.7 / 2,567.3 (+2.4% on the 4K prompt
  without its one low run, the first after a rebuild and not explained, and +2.5% on the 16K). With `STRATA_DENSE_TIMING=1` the first
  stage's marked GPU time on the 16K prompt went 6,779 -> 6,644 (1) -> 6,549 ms (1) + (2); the up projection, its
  weight conversion and the mix 0.79-0.83 ms -> 0.65 ms a half.
  Kernel shapes tried for (1) at 2048 tokens: 16 x 7 sums a thread on 8-byte LDS reads 0.53 ms; 8 x 7 sums
  0.56-0.58; 16-byte reads 0.58; 64 x 112, 256 x 112, 128 x 168 and 128 x 84 tiles 0.60-0.70; padding the
  20,480-byte row strides of X and W changed nothing; double-buffered LDS spilled registers; the same kernel built
  for 64-lane waves (`-mwavefrontsize64`) ran 0.79 ms against 0.54 (2 waves a SIMD at 254 VGPRs instead of 4). With the loads and
  barriers taken out the inner loop alone runs at about 29 TFLOP/s, so what is left is the LDS-to-register traffic
  and the register budget (the kernel uses 254 of 256 VGPRs). Not measured: one card, another gfx103x card,
  Windows, other packs; CUDA and SYCL builds not built (the new code is HIP-only, with no-op stubs elsewhere).
- **The FP16 GEMM outputs read without the in-place widening (`STRATA_PF_NOWIDEN=1`, opt-in, HIP, the FP16
  prompt path, the same bits):** every dense product here is rocBLAS FP16 in / FP16 out, and `widen_rows_f16` then
  turns the halves into floats in place: about 105 ms of the first stage's GPU time on a 16K prompt
  (`STRATA_DENSE_TIMING=1`; GDN qkv 26 ms, gate 14, QSA q 12, the shared expert's gate + up 9, the out
  projections 9, the shared expert's down 9, the rest under 6 each). With the switch the GDN qkv / gate, the QSA q,
  both attention out projections and the shared expert's gate / up stay as the GEMM wrote them
  (`Gemm::keep_f16_next`, now also honoured by `native()` and `f16()` when the weight fits the scratch in one piece)
  and their readers widen each value as they load it: `gdn_conv`'s fused conv + L2 kernel (`STRATA_GDN_CONVL2`),
  the recurrence's grid-stride out norm (`STRATA_GDN_HEAD`), `split_q`, `gate_attn`, `gr_write_norm_rs` and
  `swiglu_pair`. Any other variant of those readers gets the rows widened first. `__half2float` is what the
  widening stores, so the readers see the same floats: `hip_pf_nowiden_parity` checks each reader on the halves
  against itself on the widened rows (T 1 .. 2048, the same bits), and one 3K-word prompt at temperature 0 (the CPU
  share off) gives the same final residuals and the same answer with and without the switch, also with
  `STRATA_HC_DOWNSK=8 STRATA_HC_UPMIX16=1` on in both arms. Widening left on the
  first stage: 33 ms; its marked GPU time 6,779 -> 6,606 ms (the readers also read half the bytes). On 2x Radeon
  PRO V620, the IQ3_S abliterated pack, the settings of the paragraph above, the arms alternated on one binary:
  without 1,890.6 / 1,895.3 tok/s (4K prompt) and 2,500.9 / 2,503.8 (16K), with 1,924.6 / 1,925.8 / 1,923.9 and
  2,549.7 / 2,551.5 / 2,546.4 (+1.8%, +1.9%). On top of `STRATA_HC_DOWNSK=8 STRATA_HC_UPMIX16=1` (1,937.6 /
  2,566.7 the same hour): 1,970.7 / 1,966.8 / 1,971.2 and 2,613.2 / 1,924.5 / 2,613.5 (+1.7% and
  +1.7% without the one low 16K run, not explained (other agents share the box)); against
  no switch at all +4.1% / +4.4%. Not done: the shared expert's down output
  (read by the MoE combine's four variants, 9 ms), the router / QSA k / v / indexer products (N 512, 6 ms), GDN
  alpha / beta (4 ms). Not measured: one card, other gfx103x cards, Windows, other packs; CUDA and SYCL not built
  (the readers' templates default to their FP32 form there).
- **Streamed prompt experts by routing (`STRATA_PF_STREAM_ROUTED=1`, opt-in):** a pack that does not fit VRAM (Q4_0
  here: 19.6K of 24.6K expert pairs resident; UD-Q4_K_XL: 17.5-18.8K) streams the rest per prompt chunk. From 1,024
  tokens a chunk on, the engine copies EVERY non-resident expert of every layer, in a fixed order, so that the copies
  run ahead of the routing. `STRATA_PF_STREAM_STATS=1` prints what that costs, per chunk and stage: on Q4_0 a
  2,048-token chunk copies 2,450-2,484 experts (6.5 GiB) of which the routing picked 480-1,530; on UD-Q4_K_XL an
  8,192-token chunk copies 3,970-4,448 (11.8-13.3 GiB) for 1,740-1,935 picked. The experts that are not in VRAM are
  the rarely used ones, so most of those bytes are never read. On this PC the first card's slot is PCIe 4.0 x8 and the
  second card's x4 (`current_link_width` of the root port above each card in sysfs), and the
  second stage (layers 24-47) was waiting for its copies: 6.5 GiB a chunk is about a second on that link, as long as
  the whole chunk took. With the switch a layer's copies are queued when its routing is known, only for the experts it
  picked, and the layer walks its resident experts first, the streamed ones last (the row tables are rebuilt for that
  order as `STRATA_MMQ_RESIDENT_SORT_NE` rebuilds them). The same bytes reach the same products: a 1,832- and a
  5,504-token prompt's final residuals (`STRATA_PREFILL_DUMP_R`) are the same file with and without, on Q4_0 and
  UD-Q4_K_XL, and the same as the build this branch started from. `=2` keeps the routed list in id order (the A/B
  arm; measured on the commit before the reorder, with `STRATA_PF_DIRECT=1`: Q4_0 1,713 / 2,162 tok/s in id order,
  1,765 / 2,218 with the streamed experts last; UD `stage13-q` 1,357 / 1,703 and 1,401 / 1,770). It applies only
  where every streamed expert is copied from page-locked memory and no helper card takes part; any other chunk
  keeps the fixed list - a ONE-chunk prompt on two cards does (the idle stage helps and streams a share over its
  own link: a 1,825-token prompt still copied 1,498 + 1,519 experts for 462 + 396 picked).

- **Prompt experts read where they lie (`STRATA_PF_DIRECT=1`, opt-in, gfx103x):** an MMQ launch takes a group of 16
  experts from one buffer, so each expert was first copied out of its cache slot (1.5-4 MB, about 9% of the GPU time of
  a 2,048-token chunk: `dequant` in `STRATA_PREFILL_TIMING`). With the switch the gate/up product of every MMQ type
  takes a pointer per expert (`third_party/llama.cpp-patches/0003`: `mmq_args::x_each`), and so do both products of
  the Q4_0 / Q4_1 kernel. llama.cpp's own down products keep the copy: they read K in steps of 256 values, 128 past
  each row of a 640-column matrix and past the matrix for its last row, and only the group buffer's zeroed tail makes
  that harmless. `hip_prefill_expert_bench` runs every expert type both ways with the experts apart and FP16 NaNs
  between them: the same output hash, the same speed.

- **IQ4_NL down products on Strata's 256 x 32 tile (`STRATA_PF_NL=1`, opt-in, gfx103x):** 39 of the IQ3_S pack's 48
  layers keep the down projection in IQ4_NL, which is Q4_0's block with a table behind the code. The Q4_0 tile for
  that format, reading llama.cpp's own q8_1 activations and adding each 32 values as its MMQ kernel does: the bench's
  output hash is MMQ's (one known exception: MMQ also adds four zero terms past a 640-column row, which can turn an
  output of exactly -0 into +0). The kernel itself is about MMQ's speed where an expert has 40 rows (16 experts, down:
  0.100 -> 0.095 ms) and faster where it has many (x 320 rows 15.2 -> 17.2 T MAC/s; a skewed 5,120 rows 0.610 ->
  0.493 ms); what it brings is that it never reads past a matrix, so with `STRATA_PF_DIRECT=1` those layers lose the
  down copy too. Gate/up in IQ4_NL (no pack here) stays with MMQ, which is faster at K 2560 (0.166 against 0.193 ms).

- **Q5_1 down products on the same tile (`STRATA_PF_Q51=1`, opt-in, gfx103x):** UD-Q4_K_XL keeps the down projection
  of 43 of its 48 layers in Q5_1. The tile calls llama.cpp's own `vec_dot_q8_1_q8_1_impl` on registers, so the output
  hash is MMQ's (the same -0 note). It is NOT faster than MMQ as a kernel (16 experts, down: x 40 rows 0.105 -> 0.111
  ms, x 320 rows 14.1 -> 13.4 T MAC/s, a skewed 5,120 rows 0.653 -> 0.624 ms); it is there so that those layers'
  down matrices are read in place with `STRATA_PF_DIRECT=1`. End to end that is small: see the table.

  Why there is no such kernel for the IQ3_S pack's gate/up types: a two-card prompt reads 2,048-token chunks, which
  route about 40 rows to an expert, and there MMQ is close to the Q4_0 tile. 16 experts x 40 rows of gate+up on a
  V620: Q4_0 tile 0.157 ms (13.4 T MAC/s; 22.4 at 320 rows), MMQ IQ4_XS 0.165, IQ3_XXS 0.195, IQ3_S 0.199, IQ2_S
  0.211, Q4_K 0.192, Q5_K 0.193 (0.133 of IQ3_S's 0.199 ms is there at ONE row an expert: the decode into shared
  memory, which a register tile would pay again in table lookups). `hip_prefill_expert_bench --rows 40` prints them.

  Measured on 2x Radeon PRO V620 (gfx1030, 32 GB each, PCIe 4.0 x8 and x4, Ryzen 9 9950X, 92 GB, Linux, ROCm in
  `/opt/rocm`), `--layer-split 24`, `tools/strata_bench.py --temperature 1.0 --top-p 0.95 --top-k 20 --runs 1`, the
  same night, `STRATA_PF_Q4_WIDE=1 STRATA_PA_DOT4=1 STRATA_GDN_HEAD=1 STRATA_GDN_PP=2 STRATA_GDN_NOY=1` in every
  arm, prompt tok/s at 4K / 16K (the range where an arm ran more than once):

  | pack, settings | without | with | switches |
  |---|---|---|---|
  | Swift Q4_0, `q4_0-stage13-q` | 1,320-1,323 / 1,632-1,638 | 1,753-1,765 / 2,207-2,218 | ROUTED, DIRECT |
  | UD-Q4_K_XL, `stage12-u` (8,192-token chunks) | 969-974 / 1,485-1,499 | 1,122 / 1,613-1,618 | ROUTED, DIRECT |
  | the same | | 1,141 / 1,630 | ROUTED, DIRECT, Q51 |
  | UD-Q4_K_XL, `stage13-q` (2,048-token chunks) | 841 / 990 | 1,401-1,421 / 1,769-1,781 | ROUTED, DIRECT |
  | the same | | 1,431-1,441 / 1,763-1,787 | ROUTED, DIRECT, Q51 |
  | IQ3_S (all experts resident), `iq3_s-stage13-q` | 1,495 / 1,959-1,965 | 1,517 / 1,984-1,985 | DIRECT |
  | the same | | 1,573-1,576 / 2,053-2,055 | DIRECT, NL |
  | the same | | 1,577-1,578 / 2,049-2,051 | `STRATA_MMQ_RESIDENT_SORT_NE=1` alone |
  | the same | | 1,635-1,638 / 2,119-2,122 | DIRECT, NL, `STRATA_MMQ_RESIDENT_SORT_NE=1` |

  With the routed stream the UD pack is faster on 2,048-token chunks and its own prompt buffers than on the
  8,192-token chunks that were its best setting before (a small chunk no longer pays a full pass over the streamed
  experts). Decode is not touched by any of them (chat decode medians of 56-89 tok/s over the arms, no pattern).
  The residual dumps of all arms above match per pack. Not measured: one card, a card whose experts are not
  page-locked (the routed stream then does not engage), `--prefill-pipe` values other than 1, prompts past 16K,
  Windows, CUDA (DIRECT, NL and Q51 are gfx103x-only; the routed stream is not backend-specific but has only run here).

- **K-quant MMQ wide-LDS activation windows (`-DSTRATA_MMQ_WIDE_LDS=ON`, llama.cpp patch 0002) on a real model:**
  the same V620 pair, a second build with only that option changed. Kernel, 16 experts of gate+up, stock -> wide:
  Q4_K x 40 rows 0.192 -> 0.182 ms, x 160 rows 11.3 -> 13.1 T MAC/s, x 320 rows 14.0 -> 16.3; Q5_K 0.194 -> 0.192 ms,
  11.1 -> 13.0, 13.7 -> 16.1; the same output hashes, `hip_prefill_kquant_parity` passes. UD-Q4_K_XL end to end with
  the routed stream and direct experts: `stage13-q` 1,401-1,421 / 1,769-1,781 -> 1,415-1,433 / 1,766-1,800 tok/s,
  `stage12-u` 1,122 / 1,613-1,618 -> 1,142 / 1,640; the residual dumps are the same file. So it is correct and worth
  about 1% of a UD prompt on gfx1030, at the edge of what these runs resolve: small, because at 40 rows an expert the
  tile is J = 40 and most of its time is the weight decode, not the activation window.

- **QSA block scores on the FP32 tiled scorer (`STRATA_SELECT_SIMT=1`, opt-in, as on CUDA below sm_80):** without
  matrix cores the prompt path's QSA selection scores every (query, KV block) pair on the warp kernel, the one prompt
  term that grows with the context. The switch runs the FP32 tiled scorer instead (FP32 in another summation order:
  not bitwise the warp kernel, so off by default), tiled 32/128/16/8 on gfx103x. `qsa_select_bench`, 256 queries:
  2.69 -> 0.587 ms at 128K (the original 32/64/32/4 tiling: 0.695), 0.67 -> 0.169 ms at 32K; FP64 gate passed,
  selections identical 256/256. Every tiling gives the same bytes (each score is one fmaf chain over k in order), and
  the fastest tiling was the same with 60 and 72 of the card's 80 CUs (`HSA_CU_MASK`). On the 2x RX 6900 XT split
  (0.1.40.3, IQ3_S, `--kv int8 --kv-resident 32768`, two cold starts per arm, ABBA) a 128K-token prompt reads 1,735 ->
  1,861 and 1,727 -> 1,858 tok/s (+7.4%), 32K +0.8%; decode is not affected (the text changes with the scores' order,
  so its speed is not compared). Measurements:
  [bench/results/2026-10-07-rdna2-qsa-select-simt](../bench/results/2026-10-07-rdna2-qsa-select-simt/README.md).
- **The prompt's small kernels on a second stream (`STRATA_PF_SIDE_STREAM=1`, opt-in), and prompt chunks for the
  two-card pipeline (`STRATA_PREFILL_PIPE=283`).** Where a 16,051-token prompt's time goes on 2x Radeon PRO V620
  (IQ3_S abliterated pack, layer split 24, every expert in VRAM, `STRATA_PF_IQ2S=1 STRATA_PF_IQ3=1`), GPU ms of stage 0
  by `STRATA_PREFILL_TIMING=1` (stage 1 within 2% on every line): expert gate/up 1067, GDN input projections 820,
  hyper-connection read 669, QSA attention 584, expert down 547, QSA projections 435, GDN out projection 317,
  router + shared expert 264, GDN recurrence 256, of 5,697. A kernel trace (rocprofv3) of the same prompt: 5.43 s of
  kernels in a 6.07 s span on stage 0, of which the FP16 GEMMs of rocBLAS are 2.0 s at 31-35 TFLOP/s (the card's
  FP16 peak is 40.5), the expert tiles 1.9 s and the attention 0.58 s; the card stood idle 0.22 s in ~6 us gaps between
  dependent kernels and 0.14 s in gaps of 0.05-2 ms, most of them while the host grouped a layer's routed ids after
  the per-layer sync. The two stages run 2,048-token chunks in step (740 ms a chunk each, 6 and 12 ms of it the
  hand-off copy), so only shorter kernels or fewer gaps make the 16K prompt faster.
  `STRATA_PF_SIDE_STREAM=1` gives the prompt path a second stream with its own rocBLAS handle and dequantization
  scratch (62 MiB per card, made at the first chunk of 1,024 tokens or more; without the room it says so and
  everything stays on one stream) and runs there what does not depend on the big product beside it: the MoE half's
  shared expert beside the router, the host's grouping and the routed experts; the GDN alpha / beta beside qkv and the
  gate z beside the conv and the recurrence (whose 96 blocks leave most of the card free; only the norm after it reads
  z, and it waits for z by an event, `gdn_set_z_ready`); the QSA k / v / indexer projections, K's norm and RoPE, the
  indexer's appends and the selection beside the q projection, the K/V append and q's norm and RoPE. The same kernels
  on the same inputs, joined by events before any reader: the final residuals of a 3,000- and a 9,000-word prompt
  (`STRATA_PREFILL_DUMP_R`) hash the same as with the switch off, and so do the greedy answers.
  `STRATA_PF_DEFER_SHARED=1` is the small step without a second stream: the shared expert is queued right after the
  routing sync (as the routed-only walk already did) and fills part of the grouping gap; the same bits.
  The chunks: with `--prefill-pipe` (`STRATA_PREFILL_PIPE`) 1 a 4K prompt reads as two 2,048-token chunks, and the
  second card waits for the first chunk and the first card for the last one: a chunk costs about 87 ms plus 0.31 ms a
  token on each card here (from the timed 1,021- and 2,048-token chunks), b/a = 283 tokens, and with that ratio the
  planner takes four 1,024-token chunks at 4K, 1,024-1,280-token chunks for 2.7-3.5K, 768 for 1.5K and keeps 2,048
  at 16K (the best count there). The chunk geometry changes the rounding (as every `--prefill-pipe` value does):
  a 3,190-token prompt read in 1,280-token chunks ends in other residuals and a greedy answer that begins the same and
  then differs; no quality measurement was made.
  Prompt tok/s, `tools/strata_bench.py --runs 1 --temperature 1.0 --top-p 0.95 --top-k 20`, the agent row the median
  first-token time of its five follow-up turns (1.5-3.5K new tokens each):

  | arm | 4K | 16K | agent turn |
  |---|---|---|---|
  | without (7 runs) | 1949 (1939-1957) | 2568 (2564-2571) | 1.91-2.37 s |
  | `STRATA_PF_DEFER_SHARED=1` (2) | 1958, 1956 | 2578, 2578 | 1.92, 1.97 s |
  | `STRATA_PF_SIDE_STREAM=1`, the shared expert only (2) | 1996, 2000 | 2632, 2637 | 1.88, 1.91 s |
  | `STRATA_PF_SIDE_STREAM=1`, all of it (2) | 2022, 2023 | 2675, 2673 | 1.86, 2.55 s |
  | `STRATA_PREFILL_PIPE=283` (2) | 2183, 2175 | 2564, 2563 | 1.68, 1.69 s |
  | both (2) | 2232, 2238 | 2676, 2675 | 1.60, 1.62 s |

  That is +14.6% at 4K and +4.2% at 16K with both. The agent turns are noisy at one run: a run with a turn of
  3-3.5 s appeared with the switches off (2.37) and on (2.55). Tried in the same series and not faster: `ROC_ACTIVE_WAIT_TIMEOUT=50000`
  (the host spins on the per-layer sync instead of sleeping) and `HIP_FORCE_DEV_KERNARG=1` (1947 / 2563).
  Not measured: one card, other packs, other cards than gfx1030 (the switches are not gfx-specific; `STRATA_PREFILL_PIPE=283`
  is this rig's ratio, not a default for others), prompts past 16K, decode (the switches are prompt-only), CUDA and
  SYCL builds (not built).
- **Expert tiles at 2,048-token chunks: the tiles' heights and a Q2_0 down tile (`STRATA_PF_IQ_TJ=20`,
  `STRATA_PF_Q2=1`, `STRATA_PF_NL_TJ=20`, opt-in, MMQ's / the default's bits).** At the two-card split's 2,048-token
  chunks an expert gets about 40 token rows. The IQ tiles took 16 a block (three decodes of the weights for 40 rows),
  the IQ4_NL down tile 32, and the IQ3_S pack's nine Q2_0 down layers still ran on llama.cpp's MMQ (197 ms of a 16K
  prompt's stage 0, about twice the IQ4_NL tile's time a layer, plus the copy of those matrices into the group buffer).
  `STRATA_PF_IQ_TJ` (8 / 12 / 16 / 20 / 24 / 32) and `STRATA_PF_NL_TJ` (16 / 20 / 24 / 40) set the heights; every
  height gives the same bits, as each output's sum runs over the same values in the same order. `STRATA_PF_Q2=1` runs
  the Q2_0 down products on the IQ4_NL tile's pattern from MMQ's D4 activations with MMQ's arithmetic: 0 different
  bits against MMQ on every output of `hip_prefill_expert_bench --type 42` (with `STRATA_PF_DIRECT=1` too, which then
  reads those matrices in place). Microbench, 16 experts x 40 rows / a skewed 5,120 rows, ms: IQ3_XXS gate/up
  0.950 -> 0.908 skewed at 20 rows (even at 40), IQ4_NL down 0.101 / 0.499 at 32, 0.086 / 0.461 at 16, 0.096 / 0.434
  at 20; Q2_0 down MMQ 0.100 / 0.625, the tile at 24 rows 0.089 / 0.483. End to end (2x PRO V620, IQ3_S,
  `STRATA_PF_SIDE_STREAM=1` in every arm, prompt tok/s at 4K / 16K, one bench run each unless two are given):
  `STRATA_PF_IQ_TJ` 16: 2023 / 2671, 2025 / 2675; 20: 2033 / 2694, 2037 / 2694, 2033 / 2691, 2023 / 2687; + `STRATA_PF_Q2=1`:
  2055 / 2707; + `STRATA_PF_NL_TJ=16`: 2058 / 2717, 2051 / 2711; + `STRATA_PF_NL_TJ=20` instead: 2058 / 2727, 2057 / 2718: with all three +1.5% / +1.2% over `STRATA_PF_IQ_TJ=20` alone and +1.7% / +1.9% over the side stream alone.
  The final residuals of the 3,000- and 9,000-word prompts are unchanged with all three. Before these, a kernel trace
  with `STRATA_PF_SIDE_STREAM=1` showed that the gap while the host groups a layer's routed ids (0.7 ms a layer without
  the switch) is already filled by the shared expert on the second stream (0.05-2 ms gaps on stage 0: 143 -> 43 ms), so
  grouping on the GPU was not built.
- **Decode: the hyper-connection read on the whole card (`STRATA_HC_WIDE=3`, opt-in).** A verify window reads the
  hyper-connection weights twice per layer (four BF16 matrices of 6.5 MB); on a two-card split of 2x Radeon PRO V620
  that was about a quarter of a window's GPU time (`STRATA_VERIFY_PROFILE=1`: 4.3 of 16.7 ms per window on the first
  card, IQ3_S, windows of about 4 tokens). The default down projection is 41 blocks of one warp per row (328 wavefronts), and the up projection does
  each row's pending write and gate on single lanes. Bit 1 runs the down projection as one block per row (2,592
  wavefronts, the row's eight tiles side by side, their 40 chunk dots per lane added in the default order); bit 2 moves
  the up projection's per-element epilogue onto threads that work on 32 neighbouring columns. Every output is bitwise
  the default read's (`fused_gr_bench 300 1 8 24 3`: 32 of 32 cases of 1-8 tokens equal, the q8_1 image included).
  One read with another layer's weights every call, in microseconds: 88.2 -> 62.8 at 4 tokens, 65.0 -> 50.2 at 2,
  111.7 -> 82.0 at 6. Decode on 2x V620 (layer split 24, `--spec 5`, the draft layer, `--pipeline-windows 2`, every
  expert in VRAM, greedy, three fixed questions, one run each, with `STRATA_GR_NORM_FAST=1`, the split norm kernel of
  gfx12): IQ3_S 133.5 / 81.0 / 78.7 -> 140.7 / 85.2 / 82.6 tok/s (the answers identical, 400 / 260 / 202 tokens);
  `STRATA_GR_NORM_FAST=1` alone 134.2 / 81.6 / 79.2. Unsloth UD-Q4_K_XL (71% of the experts in VRAM, `--adapt-every 0
  STRATA_IQ_MT_MIN=1` so two runs give the same text): 98.2 / 50.2 / 65.0 -> 101.0 / 51.8 / 67.4, the answers
  identical, also after 3.2K- and 12.7K-token prompts. Prompt speed is unchanged (the prompt path does not use these
  kernels). Measured on gfx1030 only; other cards keep the default until someone measures them.
- **Decode: the gfx1151 table's decode switches on gfx1030** (the same 2x V620, IQ3_S, the same three questions, one
  run each, against 133.5 / 81.0 / 78.7 tok/s; none is on by default here): `STRATA_TSUM=1` 135.3 / 82.2 / 79.7, the
  answers identical, and 142.9 / 86.6 / 83.8 together with `STRATA_HC_WIDE=3 STRATA_GR_NORM_FAST=1`.
  `STRATA_EXPERT_V2=1`, `STRATA_LFUSE=1`, `STRATA_GDN_SPLIT=1` and `STRATA_PLE_BATCH=1`: the same text, speed within
  0.3%. `STRATA_MMVF_ROWS=1` 132.6 / 80.4 / 78.1 and `STRATA_ATTN_LANECELL=1` 132.7 / 80.6 / 78.2: the same text,
  slower. `STRATA_Q8_PACKED=1 STRATA_Q6_PACKED=1`: 122.9 / 70.7 / 76.8, slower, and the answers differ from the
  default's on this card (they did not on gfx1151).
- **Decode: the output head, the routing and the sampler as fewer, wider launches (`STRATA_HEAD_WIDE=1`,
  `STRATA_ROUTE_PLAN=1` or `2`, `STRATA_HEAD_SAMPLE=1`; opt-in, measured on gfx1030 only).** Measured in a captured
  graph on a Radeon PRO V620, a kernel costs about 4 us before it does anything (`router_chain_bench`: an empty kernel
  2.6 us, a one-row BF16 projection 3.9 us), and a verify window is about 750 of them per card.
  - `STRATA_HEAD_WIDE=1`: the head is a Q6_K matrix of 248,320 x 2,560 (497 MiB). The default launch reads it at
    538 GB/s for one column (0.97 ms), but takes 1.17 / 1.38 / 1.63 ms for 3 / 4 / 5 columns, then 0.04-0.05 ms for the
    greedy pick and 0.01 ms for the non-finite check. The switch runs persistent workgroups whose threads keep their
    activation words in registers, ask for the next row's weights while the current row is computed, and keep each
    workgroup's first logit in the pick's order; one small launch merges those. Logits, picks and flags are bitwise
    the default's (`hip_head_parity`: 608 cases of 1-8 columns with tied maxima and non-finite scales, 0 mismatches,
    with and without `STRATA_TSUM=1`). Head + pick + check, 1 / 2 / 3 / 4 / 5 columns, in microseconds
    (`hip_head_lab`, synthetic weights of the real shape, the card idle, medians of 9 batches of 20):
    1034 / 1090 / 1220 / 1432 / 1702 -> 1011 / 1016 / 1022 / 1032 / 1107.
  - `STRATA_ROUTE_PLAN=1`: the routing top-10 and the all-resident plan in one launch instead of two (only in the
    zero-doorbell graph, without `STRATA_ROUTE_PRIOR` / `STRATA_ROUTE_RESIDENT`). `=2` also runs the router's BF16
    projection in that launch: 512 workgroups compute a row each and a 513th waits for them, with the stores fenced
    and counted by an atomic add and the logits read back with device-scope atomic loads (what rocPRIM's look-back
    scan does). Ids, weights and the plan's bytes are the separate kernels' (`hip_route_plan_parity`: 600 cases, and
    for `=2` 150,000 launches on changing activations and weights in one reused buffer, 0 mismatches). The router
    chain of one layer in a 48-layer graph, 1 / 3 / 5 tokens, in microseconds: 15.3 / 16.5 / 18.5 -> 13.3 / 14.6 /
    16.6 (`=1`) -> 11.5 / 12.6 / 16.4 (`=2`).
  - `STRATA_HEAD_SAMPLE=1`: a sampled request's draws as nodes of the window's graph. The graph is captured once, so
    the sampler ran after it as two launches of its own and a second sync; here its parameters (the chain, the seed,
    the window's first position) are staged in mapped memory with the window's inputs and read at run time. Requests
    with penalties or a logit bias, and `STRATA_SPEC_PROB`'s judged windows, sample after the graph as before. The
    token is `sample_tokens`' for the same parameters (`hip_head_parity`: 400 parameter sets, 0 mismatches).
  - Decode on 2x V620 (IQ3_S, layer split 24, `--spec 3 --spec-min-p 0.7`, the draft layer, `--pipeline-windows 2`,
    every expert in VRAM, with `STRATA_HC_WIDE=3 STRATA_GR_NORM_FAST=1 STRATA_TSUM=1 STRATA_DQ8=1
    STRATA_HC_GATED16=1` and the prompt switches above on in both arms; three fixed questions of 400 new tokens,
    greedy and sampled with temperature 1.0, top-p 0.95, top-k 20, seed 7, `STRATA_SPEC_GUMBEL=1`; one run each, the
    off arm run twice, pooled tok/s): greedy 116.2 -> 117.3-117.4, sampled 107.2 -> 108.4 with all three (`ROUTE_PLAN=2`,
    two runs), 117.1 / 108.2 with `ROUTE_PLAN=1`; the texts identical in every arm. Alone: `STRATA_HEAD_WIDE=1` (its
    `STRATA_HEAD_WIDE_V=2` kernel) 116.4 / 107.4, `STRATA_ROUTE_PLAN=1` 116.7 / 107.7. An answer that copies its
    prompt (windows of 5 tokens): 175.8 -> 180.8-181.1 (three runs; 177.5 in a fourth with the earlier
    `STRATA_HEAD_WIDE_V=2` kernel; its drafts depend on timing, so it varies more).
    Not measured: other models, one card, windows wider than 5 tokens, a card where some experts are not in VRAM (the
    routing switches then do nothing), CUDA (`STRATA_HEAD_WIDE` and `STRATA_ROUTE_PLAN=2` are HIP kernels; the other
    two compile there but were not built).
- **Decode: the dense projections one wavefront per row, and fewer launches per window (gfx1030, all opt-in).**
  A verify window's dense projections (GDN qkv / gate / out, QSA q / k / v / out, the head) run on `native_mmvq`'s
  kernels: a block of four wavefronts per two rows, partial sums through shared memory, a barrier. With
  `STRATA_MMVQ_ROW=1` one wavefront does a row: each lane plays the default layout's four threads in turn and adds
  their sums in the default order, so every output is bitwise the default kernel's (`hip_mmvq_row_parity`: 10 GGUF
  types, 1-8 columns, 12 layouts, with and without `STRATA_TSUM`, 3,840 runs each way equal). One call on a V620, a
  different one of 1.2 GB of weight copies every call so that the card's 128 MB third-level cache does not hold them
  (six copies in rotation are read from that cache and flatter every kernel), microseconds, default -> row:

  | matrix | 1 column | 2 | 3 | 4 | 6 |
  |---|---|---|---|---|---|
  | Q6_K 2560 -> 10240 | 48.8 -> 49.5 | 50.8 -> 50.2 | 53.0 -> 51.1 | 58.7 -> 55.1 | 88.7 -> 66.0 |
  | Q5_K 2560 -> 10240 | 43.8 -> 42.1 | 48.8 -> 42.7 | 65.5 -> 46.0 | 86.1 -> 51.9 | 109.5 -> 66.1 |
  | Q4_K 2560 -> 6144 | 26.8 -> 25.1 | 30.6 -> 25.7 | 37.8 -> 28.0 | 48.6 -> 30.4 | 63.6 -> 39.5 |
  | IQ4_XS 2560 -> 6144 | 28.2 -> 27.0 | 29.9 -> 28.3 | 32.8 -> 29.4 | 35.3 -> 30.0 | 42.0 -> 33.1 |
  | Q4_0 2560 -> 10240 | 37.5 -> 35.6 | 37.4 -> 35.9 | | 40.3 -> 36.8 | |
  | Q6_K 2560 -> 248320 (head) | 978 -> 978 | 988 -> 988 | 1041 -> 1032 | 1206 -> 1092 | 2011 -> 1423 |

  Not faster and left on the default kernels by the switch's table: Q8_0 6144 -> 2560 (40.2 -> 40.4 at 4 columns; it
  reads at about 415 GB/s, the head at 430-530), Q4_0 6144 -> 2560, and matrices of a few hundred rows (Q6_K
  2560 -> 512: 6.6 -> 7.4). At 1-3 columns the Q6_K matrices are at the card's memory speed either way; the gain is
  where the default kernel is short of arithmetic (Q4_K / Q5_K, and everything from 4 columns up).
  `STRATA_MMVQ_ROW=2` takes the kernel for every type and shape (not timed: Q3_K, Q5_0, IQ4_NL, Q2_0).

  A dependent kernel costs about 2.4 us in a window graph whatever it does, so the other switches remove launches
  from the chain, each bitwise: `STRATA_MMVQ_SET=1` runs the projections that read the same input as one launch (a GDN
  layer's qkv and gate, a QSA layer's k, v and q; `hip_mmvq_row_parity` checks sets of 2 and 3 of six types);
  `STRATA_NORM_ROPE=2` the fused RMSNorm + RoPE launch with the rotation written as the
  unfused kernel writes it (`hip_norm_rope_parity`: bitwise on gfx1030; `=1`, the CUDA form, differs in the rotated
  high half and changes the text). Existing switches measured in the same series: `STRATA_FUSE_CONVAB=1`,
  `STRATA_DF_PLE=1`, `STRATA_HEAD_MIX_MULTI=1`, and `STRATA_ROUTE_PLAN=1` (above).

  The series below was run on a tree without the head, routing and sampling switches above and before gfx103x had
  its switches on by default; its `STRATA_ROUTE_PLAN=1` arm ran an earlier one-launch kernel of the same ids, weights
  and plan words, not the one in this tree. Not measured again on this tree, and not with `STRATA_HEAD_WIDE=1`, which
  takes the head away from the kernels `STRATA_MMVQ_ROW` replaces.

  Decode on 2x Radeon PRO V620 (layer split 24, IQ3_S, every expert in VRAM, `iq3_s-stage13-q` with `--spec 3
  --spec-min-p 0.7` and `STRATA_HC_WIDE=3 STRATA_GR_NORM_FAST=1 STRATA_TSUM=1` on in every arm; three fixed questions
  of 400 new tokens greedy, the same three sampled with seed 7, one copy-heavy answer; pooled tok/s, one run an arm,
  two runs of the first arm gave 116.2 and 116.3; the greedy and sampled texts are the same hashes in every arm):

  | arm (each adds to the one above) | greedy | sampled | copy-heavy |
  |---|---|---|---|
  | without | 116.3 | 107.2 | 175.8 |
  | `STRATA_MMVQ_ROW=1` | 117.6 | 108.4 | 185.2 |
  | `STRATA_ROUTE_PLAN=1` | 118.1 | 108.8 | 186.0 |
  | `STRATA_MMVQ_SET=1` | 119.5 | 110.1 | 187.4 |
  | `STRATA_FUSE_CONVAB=1` | 120.1 | 110.6 | 188.1 |
  | `STRATA_NORM_ROPE=2` | 121.0 | 111.2 | 189.2 |
  | `STRATA_DF_PLE=1 STRATA_HEAD_MIX_MULTI=1` | 121.9 | 112.2 | 191.3 |

  In place of the SET and CONVAB arms, `STRATA_DF_BRANCH=1` (the same projections on side streams) gave 120.0 /
  110.3 / 189.0, and 120.3 with `STRATA_NORM_ROPE=2`: the two are alternatives, not a sum. `STRATA_LFUSE=1` on top of
  the second arm: 118.2 (its launches are on the shared expert's own stream here). Tried and removed: the router's
  BF16 projection one wavefront per row, bitwise equal but 8.6 -> 24.6 us a call and 118.1 -> 111.5 tok/s (a 2.6 MB
  matrix wants more wavefronts in flight, not fewer). Swift Q4_0 (`q4_0-stage13-q`, the same
  changes; its text and speed move from run to run with the adaptive tier, about 2 tok/s): one run each,
  106.5 / 106.4 / 155.5 without, 104.6 / 106.0 / 159.3 with the first five switches, 109.5 / 109.5 / 159.3 with
  `STRATA_FUSE_WAIT=1 STRATA_FUSE_HITS=1` as well: nothing that these runs resolve. With `--adapt-every 0 STRATA_IQ_MT_MIN=1`, so that two runs give
  the same text: 96.9 / 93.4 / 145.1 -> 99.7 / 95.9 / 148.5 with the five switches, the greedy and sampled hashes
  equal. Not measured: one card,
  `--spec 5`, UD-Q4_K_XL and IQ3_XXS, long contexts, any card other than gfx1030 (the row kernel and the sets are
  compiled for every wave32 AMD card but nothing here says they are faster elsewhere; `STRATA_NORM_ROPE=2` must pass
  `hip_norm_rope_parity` on a card before it is used there).
- **Decode: fewer nodes in the window graph (`STRATA_NODE_FUSE=1`, opt-in).** With the switches above a 4-token
  verify window still held 663 nodes on the first card and 651 on the second (IQ3_S, layer split 24, counted with
  `STRATA_VERIFY_NODES=1`). `STRATA_NODE_FUSE=1` takes them to 514 and 508 (1-token windows: 652 -> 540, 648 -> 532)
  with launches that compute every value bitwise as the launches they replace:
  - the MoE combine of a layer made inside the next layer's first hyper-connection norm (one launch less a layer);
  - the GDN conv + L2 norm and the a/b gates made by the recurrence kernel's own blocks before the recurrence starts
    (two less a GDN layer; h, gate and beta are still stored for the commit);
  - a QSA layer's three per-head RMSNorm + RoPE launches (k, q, indexer q) as one, and the indexer tail's window
    snapshot made by the indexer append (three less a QSA layer);
  - the shared expert reads the q8_1 image of its input that the hyper-connection read already wrote, and runs gate
    and up through `STRATA_MMVQ_SET`'s kernel (two less a layer, on the shared expert's own stream);
  - two one-launch forms that existed but were off on HIP: the n-gram block's window steps in one pass
    (`STRATA_DF_PLE=1`'s path, about 25 nodes on the first card) and the head's hyper-connection read as one launch
    set (`STRATA_HEAD_MIX_MULTI=1`'s path), which now also writes the head projection's q8_1 input.

  Any other value of the variable is a bit mask of the single fusions (listed in
  `include/strata/kernels/node_fuse.hpp`). `hip_node_fuse_parity` checks the four new kernels against the launches
  they replace for 1-8 tokens; `verify_batch_parity`, `gr_parity` and `hip_head_parity` pass with the engine's switches
  for the two existing forms.

  Decode on 2x Radeon PRO V620 (IQ3_S abliterated, every expert in VRAM, `--spec 8 --spec-min-p 0.7 --mtp-max-t 3
  --pipeline-windows 2` with this card's defaults and `STRATA_IDX_PAIR=1 STRATA_QSA_GATE_Q8=1`;
  `tools/decode_probe.py --warm`: three questions of 400 new tokens greedy, the same three sampled with seed 7, one
  copy-heavy answer; pooled tok/s, the arms alternated):

  | run | greedy | sampled | copy-heavy |
  |---|---|---|---|
  | without, 1 | 127.0 | 116.0 | 348.4 |
  | `STRATA_NODE_FUSE=1`, 1 | 129.2 | 118.0 | 353.5 |
  | without, 2 | 126.9 | 116.0 | 347.9 |
  | `STRATA_NODE_FUSE=1`, 2 | 129.2 | 117.9 | 353.2 |
  | without, 3 | 127.0 | 116.0 | 348.2 |
  | `STRATA_NODE_FUSE=1`, 3 | 129.2 | 117.9 | 353.2 |
  | all but the combine (`=508`), 2 runs | 128.7, 128.7 | 117.5, 117.5 | 351.8, 352.1 |

  +1.8% greedy, +1.7% sampled, +1.5% copy-heavy; the greedy and sampled texts are the same hashes in every run. Single
  fusions alone gave much less than their sum in the whole (one run each, greedy: the GDN one 127.7, the combine
  126.7, the n-gram and head forms 127.0, the QSA and shared-expert ones 127.2): the two cards run as a pipeline, and
  shortening one card's stage (the n-gram block is only on the first, the head only on the second) shows little until
  the other's is shorter too.
  UD-Q4_K_XL (experts partly in RAM, so the doorbell graph; `--adapt-every 0 STRATA_IQ_MT_MIN=1`, 200 new tokens):
  the two runs without gave different greedy texts, so that config does not repeat here; both runs with the switch
  gave the second run's text, the sampled text was the same in all four, and the speeds (72.6-73.6 greedy) are inside
  that config's run-to-run spread. A 12.7K-token and a 6.4K-token prompt (one run each, greedy, 200 new tokens): the
  same texts, decode 93.1 -> 94.6 and 94.3 -> 95.8 tok/s. Not measured: CUDA and SYCL builds
  (the new kernels are compiled for every backend; the SYCL port keeps its own copies and is untouched), one card,
  Q4_0, a stage split into two token groups (the GDN fusion stays off there), batch windows, contexts past 13K.
- **Decode: 7-8 token windows keep the wide hyper-connection down (`STRATA_HC_WIDE8=1`, opt-in).** With
  `STRATA_NODE_FUSE=1` on, the stage profile (`STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1`) of a copy-heavy answer
  (7.4 tokens a window) showed the hyper-connection down at 0.99 ms and the up at 0.94 ms per window over a card's 18
  GDN layers, twice their time at 2 tokens. The cause: `STRATA_HC_WIDE`'s one-block-per-row down ran only when the
  whole window fit one launch of the staged kernel's token chunk, 6 tokens on a 64 KB LDS card, so a 7- or 8-token
  window fell back to the staged kernel in two launches. The row kernel keeps no tokens in dynamic shared memory;
  `STRATA_HC_WIDE8=1` gives it all 8 in one launch (its token loop). Every token's sums are the row kernel's:
  `hip_node_fuse_parity` compares the read with `STRATA_HC_WIDE=3 STRATA_HC_WIDE8=1` against the default (staged,
  chunked) read, 1-8 tokens, with and without the pending write, every output bitwise. Same setup as the table above,
  `STRATA_NODE_FUSE=1` in both arms, alternated:

  | run | greedy | sampled | copy-heavy |
  |---|---|---|---|
  | without, 1 / 2 / 3 | 129.2 / 129.1 / 129.2 | 117.9 / 117.8 / 117.8 | 353.2 / 353.2 / 353.0 |
  | `STRATA_HC_WIDE8=1`, 1 / 2 / 3 | 129.2 / 129.2 / 129.1 | 117.6 / 117.6 / 117.5 | 372.0 / 372.0 / 372.1 |

  Copy-heavy +5.3%, greedy unchanged, sampled -0.3%: the first sampled question, 164.3 -> 163.5 in all three pairs,
  offers 224 draft tokens instead of 219 and keeps the same 196: faster stages change how many drafts the pipelined
  windows carry, not the text (which part of the pipeline sizes them by timing was not traced). Same hashes in every
  run. Not measured: windows of 7-8 tokens on cards whose staged chunk is 8 already (they take the row kernel without
  the switch).
- **Decode: the wide up kernel's sums as one transposed butterfly (`STRATA_HC_UP_TSUM=1`, opt-in).** The wide
  hyper-connection up kernel summed each row's dot once per token with a 5-step xor butterfly, 20-40 cross-lane moves
  a row at 4-8 tokens; its time grew by about 5 us a token. `STRATA_HC_UP_TSUM=1` sums them as one transposed
  butterfly (`s26_tsum.cuh`, which the Q8 read already used: every level adds the pairs the butterfly adds, so each
  sum is the same bits), from 4 tokens on (at 2-3 the exact-T kernels stay faster). `fused_gr_bench`, cold weights,
  up incl. a stamp, 4 / 5 / 6 / 7 / 8 tokens: 30.9 / 35.4 / 42.8 / 48.7 / 53.8 -> 26.2 / 30.9 / 35.9 / 38.8 / 42.6 us.
  `hip_node_fuse_parity` compares the read with it against the wide read without it, 1-8 tokens, every output
  bitwise. Same setup, `STRATA_NODE_FUSE=1 STRATA_HC_WIDE8=1` in both arms, alternated:

  | run | greedy | sampled | copy-heavy |
  |---|---|---|---|
  | without, 1-4 | 129.3 / 128.4 / 129.3 / 129.3 | 117.7 / 117.2 / 117.7 / 117.7 | 371.9 / 372.0 / 371.7 / 372.0 |
  | `STRATA_HC_UP_TSUM=1`, 1-4 | 129.4 / 129.4 / 129.4 / 129.4 | 117.4 / 117.3 / 117.3 / 117.3 | 381.5 / 184.8 / 381.8 / 381.5 |

  Copy-heavy +2.6%, greedy +0.1%, sampled -0.3% (the second sampled question offers 186 drafts instead of 181 and
  keeps the same 116: the window sizing above again). Pair 2 was disturbed (its copy-heavy answer ran at half speed
  with the same drafts, and that pair's run without was slow too); a fourth pair replaced it. Same hashes in every run.
- **File-tier cache rule on gfx906 (`STRATA_FILE_CACHE_OLD_RULE`):** on an MI50 ROCm queues are suspended while the kernel reclaims around the process's HMM-tracked arena (#1752, KevinX8: stalls, prompt 46 -> 360 tok/s once the file tier stops churning the page cache). The gfx906 backend therefore keeps the old rule (the file cache is used only when the room left is at least the bytes read from the files); other HIP cards keep the current rule. `STRATA_FILE_CACHE_OLD_RULE=1` forces the old rule on any Linux build, `=0` the current one.
- **Windows** (the report above): setup's ready-made Windows engine 0.1.39 runs the RX 6800 beside a 780M iGPU
  (HIP numbers the RX 6800 1; setup points the engine at it), and `tools/hip/build_windows.bat` with
  `STRATA_HIP_ARCHS=gfx1030` builds a working engine from the `gfx103X-all` 10.2.0a20260930 wheels.
- **gfx1031** (RX 6700 XT, #524): setup knows it (the `gfx103X-all` wheels, unvalidated); its reporter runs it daily
  on one card. More reports: an RX 6700 XT 12 GB run as gfx1030 on ROCm 7.2.4 (#1027: IQ2_XS, decode 30-32 tok/s, prompt about 300 tok/s, 6 of 6
  needles), and an RX 6800M 12 GB on Windows with a self-built engine (#915, #1078: Q2_0, decode 9-27 tok/s, prompt 50-114 tok/s; the
  experts stream from disk with 31 GB of RAM). The ready-made Windows zip has no gfx1031 code unless it is built with it: `STRATA_HIP_ARCHS=gfx1031`
  in `tools\hip\build_windows.bat` (the default list has it from 0.1.40.2 on).
- **gfx1150** (Radeon 890M, Ryzen AI 9 HX PRO 370, Strix Point, #1217): builds and runs from 0.1.40.2 as an unvalidated target (CMake warns; the
  device code already covers it: the WMMA guards and `gfx_arch_is_gfx11_wmma()` include it). One community machine: MINISFORUM N5 PRO, 96 GB DDR5-5600, GTT
  raised to 64 GiB (`ttm.pages_limit=16777216 ttm.page_pool_size=16777216`), TheRock ROCm 7.14.1 for gfx1150, an unprivileged LXC container, IQ3_XXS: chat and
  tool calls work, decode 15-19 tok/s, the HIP ctest passes apart from tests that need files a public checkout lacks.
  Prompts read at about 125-130 tok/s on plain hipBLAS and 1.5-1.7x faster with `tools/hip/gfx1150-hipblaslt-100401.txt` (ROCm 7.14.1's
  hipBLASLt; 3.6K tokens 124 -> 216 tok/s, 7K 130 -> 226 tok/s, decode unchanged):
  [bench/results/2026-10-07-community-gfx1150](../bench/results/2026-10-07-community-gfx1150/README.md).
  Setup does not install for it yet (an integrated Radeon other than Strix Halo is named and not supported): build by hand with `-DCMAKE_HIP_ARCHITECTURES=gfx1150`
  and point `STRATA_HIPBLASLT_TUNING` at the table yourself.
- **gfx1152** (Radeon 860M / 840M, Ryzen AI 300 "Krackan", #1625): in CMake's unvalidated list like gfx1150, so
  `-DCMAKE_HIP_ARCHITECTURES=gfx1152` builds with a warning. It runs the portable kernels (the matrix-core guards name
  gfx1150 and gfx1151 only), no card has reported on it, and setup does not install for it (an integrated Radeon other
  than Strix Halo is named and not supported). A report from a real 860M is welcome.
- **gfx1032** (RX 6600/6600 XT): unvalidated; built with `STRATA_HIP_ARCHS=gfx1030;gfx1032` and ran under Windows as
  the *helper* expert GPU of an RX 6800 ([SECOND_GPU.md](SECOND_GPU.md)), 3.4-3.8k expert slots in its 8 GB and
  +8-18% decode over the 6800 alone. Its 14 WGP make a helper row about 2-3x the 6800's cost, so it pays as a
  cache tier for the CPU pool, not as a compute peer. No report of it as a primary card.
- **Not validated:** setup's own build path and the
  `gfx103X-all` wheels on gfx1030, images, answer-quality benchmarks. RDNA1 (gfx1012, RX 5500 XT) builds by hand:
  [OLDER_GPUS.md](OLDER_GPUS.md#amd-building-gfx906-and-gfx1012).

On a 16 GB card keep the defaults for `--pcie-frac` and `--adapt-every`; the 7900 XTX values in
[AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md) cost 8.6 and 13.6 tok/s on an RX 6900 XT.

> **Measured on:** RX 6900 XT 16 GB (gfx1030), i7-13700KF, 63 GB RAM, NixOS, ROCm 7.2.3, engine 0.1.30, Swift 1.5 IQ3_XXS,
> `--context 131072 --kv-resident 32768 --adapt-every 1 --vram-reserve-mib 1024`, 15 CPU pool workers (one per physical
> core except the host thread).

| Workload | Result |
| --- | --- |
| Decode, 200 greedy tokens | 38-42 tok/s (8 workers: 36; 24 workers: 27) |
| Prefill, 2,000-token prompt | 246 tok/s |
| Prefill, auto 8,192-token chunk (8K and 16K prompts) | 330-339 tok/s |
| Decode after a 16K prefill | 45.5 tok/s |

The 16 GB card holds 6,310 expert slots at 16K context and 5,247 at 131,072.

Other gfx103x and RDNA2-adjacent cards (community reports):

| Card | Report |
| --- | --- |
| gfx1031, RX 6700 XT 12 GB (#524, #1027) | Setup knows it (`gfx103X-all` wheels, unvalidated). IQ2_XS on ROCm 7.2.4: decode 30-32 tok/s, prompt about 300 tok/s, 6 of 6 needles. The default list of `build_windows.bat` includes it. |
| RX 6800M 12 GB, Windows, self-built (#915, #1078) | Q2_0: decode 9-27 tok/s, prompt 50-114 tok/s; experts stream from disk with 31 GB RAM |
| gfx1032, RX 6600 / 6600 XT | Unvalidated. Ran under Windows as the helper expert GPU of an RX 6800 ([SECOND_GPU.md](SECOND_GPU.md)): +8-18% decode over the 6800 alone. No report as a primary card. |
| gfx1150, Radeon 890M (#1217) | Builds and runs as an unvalidated target. MINISFORUM N5 PRO, 96 GB, TheRock ROCm 7.14.1, IQ3_XXS: decode 15-19 tok/s, prompts 125-130 tok/s on plain hipBLAS and 1.5-1.7x faster with `gfx1150-hipblaslt-100401.txt`. Setup does not install for it: build with `-DCMAKE_HIP_ARCHITECTURES=gfx1150` and point `STRATA_HIPBLASLT_TUNING` at the table. [Report](../bench/results/2026-10-07-community-gfx1150/README.md). |
| gfx1152, Radeon 860M / 840M (#1625) | Builds with a warning; portable kernels; no report; setup does not install for it |

On Windows, setup's ready-made engine runs an RX 6800 beside a 780M iGPU (HIP numbers the RX 6800 as 1).

## gfx906 (Instinct MI50 / MI60, Radeon VII): wave64, built from source

gfx906 is wave64 and has no WMMA and no packed byte arithmetic, so the wave32 backend refuses it. A separate opt-in build
compiles the CUDA sources as HIP through a small compat layer (`include/strata/platform/hip_compat/`), with a CUDA warp mapped
to a logical half of the 64-lane wavefront. Setup does not build it: build by hand and run `serve/server.py` with a config as
on any other card.

The gfx906 build is community-tested: the maintainers have no gfx906 card, and the engine here is checked to compile for
gfx906 and nothing more. The attention switches `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` and `STRATA_GFX906_ATTN_REDUCE12=1` are
off by default and compiled only into this build. A wave32 card seen by this build is refused at start. The Qwen3.6 family
is not part of this build; `strata --families` lists what the binary runs. Please report how it runs on your card.

**ROCm.** AMD's current ROCm releases no longer ship gfx906 libraries. The build used HIP 7.14 from the community image
[`mixa3607/rocm-gfx906:7.14-complete`](https://hub.docker.com/r/mixa3607/rocm-gfx906) (rocBLAS/hipBLAS with gfx906 kernels)
on the kernel's amdgpu driver. hipBLASLt is not used.

```sh
# inside the image, with this checkout at /src and llama.cpp at the pinned commit in third_party/llama.cpp
apt-get update && apt-get install -y cmake ninja-build git python3 build-essential
cmake -S . -B build-906 -G Ninja -DSTRATA_HIP_GFX906=ON -DCMAKE_HIP_ARCHITECTURES=gfx906 \
  -DSTRATA_GGML_DIR=/src/third_party/llama.cpp \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
ninja -C build-906 strata
```

`STRATA_HIP_GFX906` turns on the CUDA targets and refuses `STRATA_ENABLE_HIP` (one HIP build at a time). Run the container with
`--device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1`; on some boards
`HSA_OVERRIDE_GFX_VERSION=9.0.6` is needed for the runtime to accept the card.

**What differs from the CUDA build** (each A/B switch restores the CUDA layout, and was checked bitwise against it):

| | gfx906 | Switch |
| --- | --- | --- |
| Grouped native experts | Mode 7: signs as `dp4a(g ^ m, u) - dp4a(m, u)`, grids and activations in LDS. IQ formats only (K-quant / Q5_1 / Q8_0 experts take the CUDA layout). | `STRATA_EXP_MODE=2` |
| MMVQ | One wavefront per row, 64-lane butterfly, no LDS | `STRATA_MMVQ_WAVE=0` |
| Router top-10 | One wavefront, register argmax | none |
| Verify-window routing | The window's tokens in one multi-column BF16 projection and one top-k | `STRATA_ROUTE_PER_TOKEN=1` |
| Hyper-connection read | Norm loads in flight, 8 lanes per row for `up` | `STRATA_GR_FAST=0` |
| `gr_down_multi`, opt-in | Split along K, about 3% per window; sums in another order | `STRATA_GR_SPLIT=1` |
| IQ4 table lookup | llama.cpp's `v_perm_b32` sequence | none |
| QSA prompt attention | The FP32 kernel | none |

**Two cards: try `--pipeline-windows 2` (off by default on gfx906).** The first card starts the next verify window while the
second still verifies this one ([MULTI_GPU.md](MULTI_GPU.md)). Add `--trim-stage-weights` with the explicit `--layer-split`.
In the config: `"args": [ ..., "--pipeline-windows", "2", "--trim-stage-weights" ]`. `--pipeline-windows 0` is the plain loop.
`--batch` did not help a single interactive user. Nobody has compared the tokens against the serial loop on gfx906.

> **Measured on:** 2x Instinct MI50 16 GB, Ryzen 7 5700X, 64 GB DDR4-3600, ROCm 7.14 (community image), engine 0.1.43.1,
> original IQ3_XXS, 131,072 context, `--layer-split 27 --spec 3 --kv int8 --kv-resident 32768`, median of 7 runs, 1.6K-token
> prompt, 128 greedy tokens (#3173). One machine, one model.

| Setting | Decode | Prompt |
| --- | ---: | ---: |
| Without pipelining | 49.2 tok/s | 353 tok/s |
| `--pipeline-windows 2` | 66.9 tok/s (+36%) | 353 tok/s |
| plus `--trim-stage-weights` | +2% | 371 tok/s |

Not done on gfx906: setup detection and automatic build, Windows, images (`--vision`), MI60 and Radeon VII (same ISA, not run),
a single-card run. Other measurements are under [History](#history--older-measurements).

## Tuning table

A hipBLASLt table holds solution ids that are valid only for one GPU architecture and one hipBLASLt version, so it is calibrated
on the card with the ROCm the engine runs with. To calibrate the shipped gfx1100 table's rows for another card or version (a
few minutes):

```sh
cmake --build build-hip --target tune_hipblaslt
CASES=$(awk 'NR>2 {printf " --case %s,%s,%s,%s,%s", $1, $5, $2, $3, $4}' tools/hip/gfx1100-hipblaslt-100200.txt)
./build-hip/tune_hipblaslt $CASES --tuning-out table.txt
```

The file's second line names the architecture and version (`STRATA_HIPBLASLT_TUNING_V1 gfx1201 100401`). Save it as
`tools/hip/<arch>-hipblaslt-<version>.txt` for setup, or point `STRATA_HIPBLASLT_TUNING` at it. Compare the prompt speed with and
without it before keeping it. For an engine package, `packaging/stage_tuning.py` (Linux) and `tools/hip/package_windows.py`
copy the tables that match the bundled hipBLASLt into the package's `tuning/` folder. `STRATA_HIPBLASLT_VERBOSE=1` reports
fallbacks; look for `fallbacks=0`. The test `hip_prefill_hipblaslt_gemm` is a smoke test (it checks 2 rows), and it does not
prove that every solution id is valid.

Tables in `tools/hip`:

| File | Card, calibrated with | Notes |
| --- | --- | --- |
| `gfx1100-hipblaslt-100100.txt`, `-100200.txt`, `-100202.txt` | RX 7900 XTX | 100202 is for system ROCm 7.2.4 |
| `gfx1100-hipblaslt-100401.txt` | RX 7900 XTX, ROCm 10.0.0 | 43 rows; setup uses it only with hipBLASLt 1.4.1 |
| `gfx1100-hipblaslt-100500.txt` | RX 7900 XTX, ROCm 10.2.0a20261003 nightly | 32 rows |
| `gfx1201-hipblaslt-100202.txt` | R9700, for hipBLASLt 1.2.2 | 24 of the gfx1100 table's 26 shapes |
| `gfx1201-hipblaslt-100500.txt` | R9700, ROCm 10.2.0a20260914 (`gfx120X-all` nightly) | 32 rows. The engine can check only the version number, so recalibrate for a different 1.5.0 build. |
| `gfx1200-hipblaslt-100202.txt` | RX 9060 XT | See [Community-validated cards](#community-validated-cards) |
| `gfx1151-hipblaslt-100400.txt`, `-100401.txt`, `-100500.txt` | Radeon 8060S (Strix Halo) | 100400 is for the 7.14.0a wheels setup installs; 90 rows. Used by default on Strix Halo. |
| `gfx1150-hipblaslt-100401.txt`, `-100500.txt` | Radeon 890M | Setup does not use it; point `STRATA_HIPBLASLT_TUNING` at it |
| `gfx1103-hipblaslt-100401.txt` | Radeon 780M | 143 rows. Setup installs for gfx1103 only with `STRATA_EXPERIMENTAL_GFX1103=1`. [Report](../bench/results/2026-10-12-community-gfx1103/README.md). |

> **Measured on:** the card in each row, fresh prompts, decode unchanged unless stated; plain hipBLAS against the table.

| Table | Card, workload | Prompt speed |
| --- | --- | --- |
| `gfx1100-hipblaslt-100401.txt` | RX 7900 XTX, Swift 1.5 IQ3_XXS, 262144 ctx, hipBLASLt 1.4.1, medians of 3 | 1.7K tokens 682 -> 1,038 tok/s; 6.5K tokens 857 -> 1,524 tok/s |
| 43-row revision | RX 7900 XTX, Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, 3,151-token prompt | 711 -> 938 tok/s (+31.9%) |
| `gfx1100-hipblaslt-100500.txt` | RX 7900 XTX, engine 0.1.38 lineage | 131,071-token prompt 926 -> 1,687 tok/s |
| `gfx1201-hipblaslt-100202.txt` | R9700, full IQ3_XXS, `--kv int8 --kv-resident 32768` | 32K prompt 638 -> 1,177 tok/s; 7K prompt 656 -> 1,164 tok/s |
| `gfx1201-hipblaslt-100500.txt` | One R9700, 4,210-token prompts | 1,531 -> 1,590 tok/s (+3.9%) |
| `gfx1151-hipblaslt-100400.txt` | Strix Halo | 3.1x to 14.1x faster per GEMM (median 6.2x); whole-model prompt speed not measured |
| `gfx1103-hipblaslt-100401.txt` | Radeon 780M, Q2_0, 4K prompt, 32K context, MTP | 90 -> 187 tok/s; 277 with the WMMA opt-ins |

Tuning on the R9700 with hipBLASLt 1.2.0 or 1.4.1 gave 0 to 3% end to end (within noise), so no table ships for those and
Linux `rx9070` uses plain hipBLAS. A 1.2.0 table is not bit-identical: first-token KL 0.0157 mean against plain over 24
prompts, top-1 the same 24 of 24.

When you calibrate, time a heavy case first: the first case of a fresh `tune_hipblaslt` process times its baseline while the
clocks are still ramping. The first fresh prompt after a model loads is also not a measurement.

### rocBLAS table (gfx103x)

On RDNA2 the prompt GEMMs run through rocBLAS in FP16, and hipBLASLt has no gfx1030 kernels. The rocBLAS table does for
rocBLAS what the table above does for hipBLASLt. A table is valid for one architecture and one rocBLAS build (the full
version string, tweak hash included), so calibrate it on the card with the rocBLAS the engine loads (`/opt/rocm/lib` is
rocBLAS 5.6.0.8d1ae90e on ROCm 10.0.0):

```sh
cmake --build build-hip --target tune_rocblas
LD_LIBRARY_PATH=/opt/rocm/lib ./build-hip/tune_rocblas --tuning-out table.txt   # 14 shapes x 8 token buckets, about 5 min
```

The tool times rocBLAS's default kernel and every solution the library lists for each shape at T = 256 ... 8192 (the
engine's dense GEMM shapes; `--case T,N,K,ldy` adds one, `--tokens` changes the buckets), checks the fastest ones
against the default kernel's output (within FP16 rounding, no write outside the N x T result) and that they also run at
smaller, odd token counts, and writes one row per shape and bucket: the winning solution index where it beats the
default by `--min-gain` (1.05), `default` otherwise - a `default` row is what keeps a neighbouring bucket's solution from
reaching token counts where rocBLAS's own choice measured best. The file's header names the architecture, the version
and the width of the card it was measured on (`STRATA_ROCBLAS_TUNING_V1 gfx1030 5.6.0.8d1ae90e mp=40`: HIP's
`multiProcessorCount`, which on RDNA counts workgroup processors of two CUs each, so 40 on an RX 6900 XT; older tables
without `mp=` still load). One architecture covers cards of several widths - gfx1030 is the RX 6800 and Radeon PRO
W6800 (60 CUs, mp 30), the RX 6800 XT and Radeon PRO V620 (72, mp 36; the W6800 and V620 are the RDNA2 cards on
ROCm's support list) and the RX 6900 XT / 6950 XT (80, mp 40), and smaller RDNA2 cards run as gfx1030 through
`HSA_OVERRIDE_GFX_VERSION` - with the same memory system, so where kernels are close the fastest one can differ; a
table from another width still loads (its solutions are valid there) and the engine logs the mismatch.
`tune_rocblas --to-cache` writes the card's own table to a per-device cache directory (`STRATA_TUNING_CACHE`, default
`$XDG_CACHE_HOME/strata`, else `%LOCALAPPDATA%\Strata\cache` on Windows, else `~/.cache/strata`) as
`<arch>-mp<N>-rocblas-<version>.txt`, and with `STRATA_ROCBLAS_TUNING` pointing at a directory the engine takes that
file first, then `<arch>-mp<N>-rocblas-<version>.txt` in the directory, then `<arch>-rocblas-<version>.txt`; its
`rocBLAS tuning enabled` line names the file. Save a table as `tools/hip/<arch>-rocblas-<version>.txt` (the version
reaching token counts where rocBLAS's own choice measured best. The file's header names the architecture and version
(`STRATA_ROCBLAS_TUNING_V1 gfx1030 5.6.0.8d1ae90e`); save it as `tools/hip/<arch>-rocblas-<version>.txt` (the version
with trailing dots dropped) for setup, or point `STRATA_ROCBLAS_TUNING` at the file. `STRATA_ROCBLAS_VERBOSE=1` prints
the solution picked for each shape and a `tuned_launches=... fallbacks=...` summary at exit; a solution rocBLAS refuses
at run time (status 11 once while it sets up its workspace is normal) falls back to the default kernel for that call, and
after three refusals for good. `hip_prefill_rocblas_gemm` (with `STRATA_ROCBLAS_TUNING` set; skips otherwise) is the
smoke test: it refuses a table for another architecture or build and checks Gemm's FP16 route against hipBLASEx on the
table's first two shapes.

Shipped: `gfx1030-rocblas-5.6.0.8d1ae90e.txt` (ROCm 10.0.0, the `/opt/rocm` setup installs on Ubuntu), measured on an RX 6900 XT.
Other gfx1030 cards run it too, but a table calibrated for them may pick differently; `tune_rocblas` makes one in about 5
minutes. Data: [2026-10-05-rdna2-rocblas-table](../bench/results/2026-10-05-rdna2-rocblas-table/README.md).

## Strix Halo: 8-bit hyper-connection weights (default on gfx1151)

On Strix Halo (gfx1151) and gfx12 the hyper-connection down and up projections are read as Q8_0 by default (quantized at load
from the pack, or the GGUF's own Q8_0), which halves the 13.3 MB each read moves. Decode is about 9% faster on Strix Halo
(+7.9% to +10.7% across json, code, tool call, chat and agent prompts). `STRATA_HC_Q8=0` restores the bf16 read. It is not
bit-identical to the bf16 read: a greedy answer can pick another token at a near tie and go its own way.

> **Measured on:** RTX A4000, IQ3_XXS, quant-quality yardstick (teacher forced, 5 texts of 1,000 tokens, reference = bf16
> read), 2026-10-22. The Q8_0 weights are the same bytes on every card.

| Text | Top-1 same | KL |
| --- | ---: | ---: |
| Code | 97.9% | 0.0077 |
| Documents | 93.5% | 0.0229 |
| English | 94.7% | 0.0107 |
| Other languages | 98.0% | 0.0060 |
| Agent transcript | 93.0% | 0.0779 |
| All 4,995 positions | 95.4% | 0.0251 |

For scale, bf16 on both sides with only the experts computed on the CPU instead of the GPU (a rounding-only difference the
engine takes whenever an expert is not resident) gives KL 0.0228 over all positions with top-1 95.4%. The 8-bit read is at
that level.

## History / older measurements

Dated results from earlier engines. They show how the numbers moved and are not instructions. Narrative from the original
write-ups (method notes, bug hunts, per-run logs) was dropped; the numbers and their labels are kept.

**RDNA4 (gfx1201), R9700 (Q2_0 32 GB unless stated), Linux, ROCm 7.14, interleaved starts.**

| Change | Engine | Result |
| --- | --- | --- |
| Packed byte ops (`__byte_perm` as one `v_perm_b32`, #262) | 0.1.31 | R9700 decode +15% (46.0 -> 53.0 tok/s, 4K prompt); 9070 XT +5-7% |
| Router kernel without serial FP64 sum | 0.1.32 | R9700 decode 62.4 -> 70.0 tok/s; a user's A/B with TheRock 7.10 and an i5-12600K: +1-4% inside 12-20% spread (#432) |
| WMMA prompt attention (Coder IQ1_M) | 0.1.32 | R9700 prompts 4K 1,784 -> 2,427, 16K 1,797 -> 2,700 tok/s; a user on TheRock 7.10: +29-34% (#432). Kernel 7.2-7.5x the portable one |
| WMMA prompt attention (IQ3_S, 431072 ctx, gfx1201 hipBLASLt 1.4.1 table) | 0.1.40 | 8,830-token prompt 1,571 -> 2,064 tok/s (+31.4%); 4,210 tokens 1,699 -> 1,990 (+17.1%); decode 89.8 -> 86.9 (unchanged) |
| WMMA gfx12 default (Q2_0) | not recorded | 20K prompt x1.46 (`=1`) / x1.54 (`auto`); 4K x1.33 / x1.34; quality: KL 0.0003 (6,024 positions), HumanEval 157/164 (default 156) |
| Fused int8 prompt experts (`STRATA_PF_FUSED=1`, Coder IQ1_M, ROCm 7.14) | 0.1.42 | 4K prompt 1,612 -> 1,353 ms (+19.1%), 16K +8.4%, 32K +8.5%. First-token KL vs MMQ: IQ3_S 0.00284 vs 0.00332, IQ3_XXS 0.00388 vs 0.00340 (pooled 24 prompts) |
| Fused experts for Q2_0 gate/up | not recorded | One layer, 2,048 tokens: MMQ 24.5 ms, fused 4.4 ms. Prompt 4K +16.7%, 20K +19.0%; KL vs MMQ 0.0004 |
| Both defaults together (Q2_0 32 GB) | not recorded | 4K 1,151 -> 2,286 tok/s, 20K 1,384 -> 2,751; IQ3_XXS 20K 1,576 -> 1,997 |
| Queue priming at start (Linux) | 0.2.1 | About half of fresh Linux starts read prompts 10-23% slower (4K 1,980 vs 2,290 tok/s, p620); queue priming at start fixed it, answers identical |
| Layer split R9700 + 9070 XT (4K / 16K prompt, decode) | 0.1.30 | R9700 alone 1,794 / 1,804 tok/s, 51-52 decode; split 1,384 / 1,906, 42-43; 9070 XT alone 1,016 / 1,519, 38-40 |
| Layer split with experts streamed (R9700 cache 1,500, 16K prompt) | 0.1.31 | Split 1,803 tok/s against 1,287 |
| `split_skip_if_fits` (R9700 first) | 0.1.31 | 4K prompts 1,776 tok/s (split: 1,244), decode about 60 tok/s (split: about 51) |
| Pictures on the R9700 | not recorded | `strata-vision-hip`: 57 ms for a 260-token picture, 0.40 s for 972 tokens, against 3.13 s and 14.1 s on an EPYC CPU reader; cosine 0.99985. A 780M with a gfx1103 reader: 0.40 s against 2.42 s |
| Windows RX 9070 XT (#325), engine 0.1.34 | 0.1.34 | Coder IQ1_M at 32K: 29.2 tok/s decode, about 181 tok/s prefill, correct answers; ctest 42 of 46 |
| Top-k selection, R9700 | 0.1.45 | 256-query selection 0.29 -> 0.21 ms at 98K, 1.16 -> 0.49 ms at 262K, 22.0 -> 0.93 ms at 524K |

**Packages and tables.**

| Result | Where |
| --- | --- |
| R9700 and Linux `rx9070`: a 1.2.0 table read 4K prompts 2,302 vs 2,119 tok/s and 20K 2,671 vs 2,507, decode 91.2 vs 93.2 (-2.1%), mixed by start mode (0.2.1, package 0.1.45.3, Q2_0) | Not shipped |
| Strix Halo: prompts 251 -> 529 tok/s (2.11x) with the table (community A/B) | Default on gfx1151 |
| R9700, own table with hipBLASLt 1.4.1, IQ3_S 431072 ctx: 8,830-token prompts 1,606 -> 1,613 tok/s (+0.5%), decode 87.1 -> 85.1 (0.1.40) | Kept local, not shipped |
| RX 6800 (Windows, #380 #377): `--expert-cache auto` filled the card past what Windows keeps, decode fell 41 -> 30 tok/s; fixed by the video-memory budget (0.1.37) | Fixed |

**Shared-expert stream, measured by platform.** 2x RX 6900 XT Linux (#1272): turning it off costs 5-13% decode (one card 47.5 /
46.9 / 48.3 -> 50.5 / 48.5 / 51.3 tok/s with it on). RX 6800 Windows (#816): on was 12-22% slower. 4x R9700: fork off was 13-45%
faster.

**Draft side stream, RX 6800 Windows (`STRATA_HIP_PROMPT_F16=1`, IQ3_XXS, two runs each).** A 7.9K-token prompt read 798-808
tok/s with the stream and 584-596 without; 30.4K tokens 855-863 against 553-600. Decode unchanged.

**Lend coverage, RX 7900 XTX (gfx1100), v0.1.40.2, 50 GB RAM, 74K prompt (#1389).** 6201 of 6201 slots covered: `wait copy` 0.2%,
2,583 tok/s. 5093 of 6202: 36.8%, 1,810 tok/s. Same reporter's other levers: `STRATA_PF_FUSED=1 STRATA_PF_GEMM=1
STRATA_PF_SWITCH_MIN_T=4096` 925 -> 1,092 tok/s; `--resident-experts` 1,092 -> 2,503 tok/s; `--prefill auto:32768` +23% at 74K.

**RDNA2, other measurements.**

| Workload | Result |
| --- | --- |
| `STRATA_HIP_PROMPT_F16=1`, 2x RX 6900 XT, GSQ-RCO IQ3_S, engine 0.1.39 | 9.4K / 34.7K / 105.8K prompts 744 / 915 / 926 tok/s instead of 439 / 466 / 461; decode unchanged |
| `STRATA_RDNA2_SGEMM=1`, RX 6800 over OCuLink, Ryzen 7 8845HS, Windows 11, Coder IQ1_M, 12K prompts, engine 0.1.39 | Prompt 312 -> 452 tok/s; GPU timeline 37.2 -> 25.5 s; decode 29-34 vs 29-35 tok/s; same session: native 310, SGEMM 439, FP16 494 tok/s |
| rocBLAS table, 2x RX 6900 XT, 799-token prompt | GPU time 3,329 -> 2,985 ms; 8,874 tokens 11,978 -> 11,459 ms; 1,108 tokens unchanged (bounded by expert PCIe transfer) |
| `STRATA_SELECT_SIMT=1`, 2x RX 6900 XT, IQ3_S, 0.1.40.3 | 128K prompt 1,735 -> 1,861 tok/s (+7.4%); 32K +0.8%. `qsa_select_bench`, 256 queries: 2.69 -> 0.587 ms at 128K |
| `STRATA_HC_PERSIST=1`, 2x RX 6900 XT, ROCm 10.0 | Read 1.35-2.10x; -4.2% / -2.8% ms per verify window at 4K / 32K |
| Draft side stream, RX 6800 Windows | See above |
| `STRATA_PF_PAD`/`STRATA_PF_GEMM` on RX 9070 (gfx1201), ROCm 7.2.4, Coder IQ1_M | 4K prompt 1,247 vs 1,243 tok/s, 8K 1,330 vs 1,321 (noise); an earlier crash with padded GEMMs on RDNA4 was fixed in 0.1.46 |

**Package checks (0.2.1 to 0.2.3).** Initial package checks (build, link, `package_engine.py --verify`, GPU code listing,
`strata --version` in a clean environment, feed selection by architecture) passed without an RX 7000 or Strix Halo card.
Later evidence: [release 0.2.3](release-notes-0.2.3.md) records Windows Strix Halo inference with engine 0.1.48. An RX 7900 XTX
on Windows 11 ran the 0.1.40 Windows engine ([report #1173](../bench/results/2026-10-06-community-rx-7900-xtx/README.md));
other RX 7900 owners ran Linux builds ([COMMUNITY.md](../bench/results/COMMUNITY.md)).

**Original backend validation (PR #94, engine 0.1.20, RX 7900 XTX, Ryzen 9 7950X3D, 64 GiB, ROCm 7.1).** 25 registered tests
passed on 2026-09-29. A real GSQ-RCO IQ3_XXS server smoke with mmap, expert profile, automatic cache and MTP passed four
requests; the cache held 11,142 experts (18.08 GiB). Short checks, not a benchmark. Current evidence:
[AMD_HIP_PERFORMANCE.md](AMD_HIP_PERFORMANCE.md).

**MI50 (gfx906), port this build was cut from (builds on 0.1.30 / 0.1.33).** 2x MI50 16 GB (85 W each), Xeon E5-2666 v3, 32 GB,
Coder IQ1_M, 131,072 context, `--kv int8 --kv-resident 32768 --layer-split 27 --prefill 4096`, MTP `--spec 4 --spec-min-p 0.5`:
all 12,288 experts in VRAM; a 17,043-token agent prompt read in 43.5 s (392 tok/s), decode 57.7-59.3 tok/s; 407 requests, 202,940
tokens in a load test without stalls. llama.cpp (3cf0325, ROCm) on the same machine: 26.9 tok/s `tg128`.

**Community gfx1150 (890M).** Prompts 3.6K tokens 124 -> 216 tok/s and 7K 130 -> 226 tok/s with the table, decode unchanged.

**gfx1151 table (100400).** `hip_prefill_hipblaslt_gemm` passes (`launches=4 fallbacks=0`); HIP ctest 61 of 62.
