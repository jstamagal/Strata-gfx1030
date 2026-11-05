# Host GTT experiment

`STRATA_HOST_IMMUNE=1` is an opt-in Linux HIP allocation policy. It selects
KFD GTT for ROCr host allocations and persistent ROCclr staging for linear
pageable copies. GTT has no userptr HMM notifier. It can still be evicted by
TTM; the name does not promise that queues can never be evicted.

The policy was read against ROCm 10.3.0a20261023, rocm-systems
`39fd0ea559a5970e42afd87c30f275744e00b8d1`, and CachyOS Linux
`4a01eeb1d794483311055ae3b7bb6ab379903383` (7.3-rc6-1 source tag).
It sets `HSA_USERPTR_FOR_PAGED_MEM=0`, `HSA_SVM_FOR_PAGED_MEM=0`, and
`GPU_PINNED_MIN_XFER_SIZE` before HIP initializes. Both HSA switches are
needed: disabling userptr alone selects SVM in this runtime.

One GDN checkpoint host buffer per device is reserved at session creation
and retained until process exit. Explicit host registration is refused in
this arm, so optional callers take their staged paths. A failed host KV or
embedding allocation cannot silently switch those buffers to VRAM.

The policy is off by default. No kernels, model arithmetic, or device buffer
sizes are changed. Text identity, memory peaks, and speed require the queued
hardware window; they have not been measured for this change.

These scripts are copies of the coordinator's staging files. They are
machine-specific and have not been started or queued:

- `pinnedhost-in-slot.sh`: waits for `buildslot.open` and expect name
  `pinnedhost`, then configures and builds only in that slot.
- `66-windowpin.sh`: snapshots the live config and edit24, runs main and the
  branch with the feature off/on, and applies compaction plus file-cache
  pressure. Root KFD logs are filtered to each engine's PID and time span.
- `66-windowpin-read.py`: checks each case's text hash, reports raw per-node
  SVM/USERPTR/TTM eviction counts and episodes, gaps over 300 ms, and tok/s.
- `66-windowpin-pressure.py`: owned pressure workers, stopped by recorded
  PID. No persistent system settings are changed.

Default `PASSES=6`; use a fresh `RUN` for a repeated window. The live config
must still have resident experts, vision, two pipeline windows, and 524288
context cells. Image encoder allocations are in a separate process and are
outside this main-engine policy. Rectangular/image host copies in ROCclr
have other pinning paths; the live main-engine host copies audited here are
linear.
