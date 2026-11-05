#!/bin/bash
# Expect name: pinnedhost. Prepared only; the coordinator starts this builder.
set -uo pipefail
H=/home/ronald_rump/inbox/agents/claude/strata-ab
W=/home/ronald_rump/inbox/agents/claude/strata-pinned-host
B=$W/build-hip
L=$H/staging/pinnedhost-build.log
export TMPDIR=/home/ronald_rump/inbox/agents/claude/astra-pinned/scratch
export CCACHE_TEMPDIR=$TMPDIR
export PATH=${PATH//\/home\/ronald_rump\/.venv\/bin:/}
until [ -f "$H/buildslot.open" ] && grep -qx pinnedhost "$H/buildslot.expect" 2>/dev/null; do sleep 20; done
# Always release the slot, with an explicit failure result in the log.
trap 'touch "$H/buildslot.done.pinnedhost"' EXIT
{
    echo "pinnedhost build start $(date -Is) $(git -C "$W" rev-parse HEAD)"
    nice -n 10 cmake -S /home/ronald_rump/inbox/agents/claude/strata-pinned-host -B /home/ronald_rump/inbox/agents/claude/strata-pinned-host/build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang -DCMAKE_HIP_ARCHITECTURES=gfx1030 -DCMAKE_HIP_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm '-DGPU_TARGETS=gfx1030;gfx1030;gfx1036' -DSTRATA_AMD_COMMUNITY_TUNING=ON -DSTRATA_BUILD_CONVERSATION_TESTS=OFF -DSTRATA_BUILD_DSV41=OFF -DSTRATA_BUILD_SPLIT=OFF -DSTRATA_BUILD_TESTS=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_SYCL=OFF -DSTRATA_GFX1012_PORTABLE_DOT=OFF -DSTRATA_GGML_DIR=/home/ronald_rump/Strata_donttouch/third_party/llama.cpp -DSTRATA_HIP_GFX906=OFF -DSTRATA_ISA_FLOOR= -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_RDNA2_DOT_Z=OFF -DSTRATA_MMQ_RDNA2_Q8K=ON -DSTRATA_MMQ_RDNA2_TILES=ON -DSTRATA_MMQ_TAB=OFF -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_MMQ_VEC_COPY=OFF -DSTRATA_MMQ_WIDE_LDS=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_ORACLE_BUILD_DIR= -DSTRATA_ORACLE_SOURCE_DIR= -DSTRATA_ORCA_Q4KS_MMQ=ON -DSTRATA_PARITY_GFX1100_QSA_SCORES=OFF -DSTRATA_PLE_FIXTURE_DIR=/home/ronald_rump/inbox/agents/claude/strata-pinned-host/bench/micro -DSTRATA_PORTABLE=OFF -DSTRATA_PREFILL_MMQ=ON -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_RDNA2_DOT_Z_ATTN=OFF -DSTRATA_RDNA2_DOT_Z_DENSE=OFF -DSTRATA_RDNA2_DOT_Z_EXPERTS=OFF -DSTRATA_RDNA2_DOT_Z_HEAD=OFF -DSTRATA_RDNA2_DOT_Z_MODE=1 -DSTRATA_RDNA2_DOT_Z_PROMPT=OFF -DSTRATA_WERROR=OFF -DSTRATA_WINDOW_MAX_T=auto
    rc=$?
    echo "configure exit $rc"
    if [ "$rc" != 0 ]; then echo "build exit $rc"; exit "$rc"; fi
    # Compare engine options; the fixture path intentionally names this worktree.
    diff <(grep -E '^STRATA_[A-Z0-9_]+:(BOOL|STRING)' /home/ronald_rump/Strata_donttouch/build-hip/CMakeCache.txt | sort) \
         <(grep -E '^STRATA_[A-Z0-9_]+:(BOOL|STRING)' "$B/CMakeCache.txt" | sort)
    rc=$?
    if [ "$rc" != 0 ]; then echo 'build exit 90 (main build options changed; coordinator must refresh)'; exit 90; fi
    nice -n 10 cmake --build "$B" -j16 --target strata
    rc=$?
    echo "build exit $rc"
    if [ "$rc" != 0 ]; then exit "$rc"; fi
    sha256sum "$B/strata" | tee "$H/staging/pinnedhost-build.sha256"
    git -C "$W" rev-parse HEAD > "$H/staging/pinnedhost-build.commit"
    echo "pinnedhost build end $(date -Is)"
} > "$L" 2>&1
