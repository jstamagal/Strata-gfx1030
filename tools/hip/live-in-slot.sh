#!/bin/bash
# Expect name: live. Coordinator launches this; HIP configure/build and GPU parity are SLOT ONLY.
set -euo pipefail
H=/home/ronald_rump/inbox/agents/claude/strata-ab
W=/home/ronald_rump/inbox/agents/claude/strata-q38-live
export TMPDIR=$W/scratch CCACHE_TEMPDIR=$W/scratch
export PATH=${PATH//\/home\/ronald_rump\/.venv\/bin:/}
mkdir -p "$TMPDIR"
until [ -f "$H/buildslot.open" ] && grep -qw live "$H/buildslot.expect" 2>/dev/null; do sleep 20; done
trap 'rc=$?; touch "$H/buildslot.done.live" || rc=$?; echo "build exit $rc"; exit "$rc"' EXIT
if [ -e "$H/staging/live-build.log" ]; then
    mkdir -p /home/ronald_rump/inbox/agents/claude/trash/2026-11-05
    mv "$H/staging/live-build.log" "/home/ronald_rump/inbox/agents/claude/trash/2026-11-05/live-build-$(date +%s%N).log"
fi
exec > "$H/staging/live-build.log" 2>&1
echo "live slot build start $(date -Is) $(git -C "$W" rev-parse HEAD)"
python3 -I "$W/tools/hip/live_build.py" build
echo "live slot build end $(date -Is)"
