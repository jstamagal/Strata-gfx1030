#!/bin/bash
# Queue-only Window LIVE, BOTH V620s. Runtime estimate ~26-32 min, not measured:
# 12 decode loads (6 agent ~68s + 6 edit24 ~128s), plus 9 separate fresh 36K loads (~35-55s each).
# m/L0/L agent+edit24, two FULLY mirrored passes; then three separate first-request 36K loads per arm.
# No stalls dropped. Reader prints state-aware comparisons and the explicit LIVE_SWITCH GO/HOLD gate.
set -euo pipefail
H=/home/ronald_rump/inbox/agents/claude/strata-ab
W=/home/ronald_rump/inbox/agents/claude/strata-q38-live
export TMPDIR=$W/scratch CCACHE_TEMPDIR=$W/scratch
shopt -s nullglob
existing=("$H"/ab/live_*)
[ "${#existing[@]}" -eq 0 ] || { echo 'LIVE_SWITCH HOLD | ab/live_* exists; refusing to start'; exit 1; }
mkdir -p "$TMPDIR"
set +e
(
    echo "window LIVE start $(date -Is)"
    python3 -I "$W/tools/hip/live_window.py"
    rc=$?
    echo "window LIVE end $(date -Is) exit $rc"
    exit "$rc"
) > "$H/windowlive.out" 2>&1
rc=$?
echo "WINDOWLIVE DONE exit $rc; readout $H/windowlive.out"
exit "$rc"
