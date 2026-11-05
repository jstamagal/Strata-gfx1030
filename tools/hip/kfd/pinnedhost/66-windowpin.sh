#!/bin/bash
# Window PIN: main vs one branch binary, HOST_IMMUNE=0 vs 1. Do not autoqueue.
# compact_memory every 10s plus the page-cache streamer; no sysctl changes.
# Live config is copied once at window start; every arm uses that exact snapshot.
set -euo pipefail
H=/home/ronald_rump/inbox/agents/claude/strata-ab
W=/home/ronald_rump/inbox/agents/claude/strata-pinned-host
MAIN=/home/ronald_rump/Strata_donttouch/build-hip/strata
BR=$W/build-hip/strata
K=$W/tools/hip/kfd
EDIT24=/home/ronald_rump/inbox/agents/claude/mtpq/prompts/edit24.json
LIVE=/home/ronald_rump/Strata_donttouch/strata-sc117-IQ3_S-qwen-abliterated.json
IOF=(/home/ronald_rump/Strata-data/models/bartowski-swift-Q4_0/ukisai_Swift-Qwen3.8-Flash-Next-Q4_0-0000{1,2,3}-of-00003.gguf)
PASSES=${PASSES:-6}
RUN=${RUN:-pin}
export TMPDIR=/home/ronald_rump/inbox/agents/claude/astra-pinned/scratch
[[ $RUN =~ ^[a-zA-Z0-9_-]+$ ]] && [[ $PASSES =~ ^[1-9][0-9]*$ ]] || exit 2
cd "$H"
exec > "$H/window${RUN}.out" 2>&1
SNAP=$H/staging/${RUN}-inputs
[ ! -e "$SNAP" ] || { echo 'GATE FAIL: snapshot exists; use a fresh RUN'; exit 1; }
for f in "$MAIN" "$BR" "$EDIT24" "$LIVE" "$K/kfd_events.py" "$K/kfd_episodes.py" "${IOF[@]}"; do
    [ -f "$f" ] || { echo "GATE FAIL: missing $f"; exit 1; }
done
sudo -n true
grep -qx 'build exit 0' "$H/staging/pinnedhost-build.log"
sha256sum -c "$H/staging/pinnedhost-build.sha256"
for ((i=1;i<=PASSES;i++)); do for arm in m b0 b1; do
    [ ! -e "$H/ab/${RUN}_${arm}_e$i" ] || { echo 'GATE FAIL: output already exists'; exit 1; }
done; done
mkdir "$SNAP"
cp "$LIVE" "$SNAP/live.json"
cp "$EDIT24" "$SNAP/edit24.json"
sha256sum "$MAIN" "$BR" "$SNAP/live.json" "$SNAP/edit24.json" > "$SNAP/hashes"
git -C "$W" rev-parse HEAD > "$SNAP/branch-head"
git -C /home/ronald_rump/Strata_donttouch rev-parse HEAD > "$SNAP/main-head"
python3 -I - "$SNAP/live.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); a=j['args']
for flag in ('--resident-experts','--vision'): assert flag in a, flag
for flag,value in (('--pipeline-windows','2'),('--max-context','524288')): assert a[a.index(flag)+1] == value
# Reject a live snapshot that already overrides the variables that distinguish this experiment.
for k in ('HSA_USERPTR_FOR_PAGED_MEM','HSA_SVM_FOR_PAGED_MEM','GPU_PINNED_MIN_XFER_SIZE'):
    assert k not in j.get('env',{}), k
PY
for f in /proc/sys/vm/compact_unevictable_allowed /proc/sys/vm/swappiness \
         /sys/kernel/mm/transparent_hugepage/enabled /sys/kernel/mm/transparent_hugepage/shrink_underused \
         /sys/module/ttm/parameters/pages_limit; do printf '%s ' "$f"; cat "$f"; done > "$SNAP/settings"
boot_ns() { python3 -I -c 'import time; print(time.clock_gettime_ns(time.CLOCK_BOOTTIME))'; }
kp= cp_pid= ip= sp= ksuper= csuper=
cleanup() {
    # Only recorded child PIDs from this arm, never discover-and-kill by name.
    # Also cover a startup error before both pid files have appeared.
    if [ -z "$kp" ] && [ -n "${D:-}" ] && [ -s "$D/kfd.pid" ]; then kp=$(cat "$D/kfd.pid"); fi
    if [ -z "$cp_pid" ] && [ -n "${D:-}" ] && [ -s "$D/compact.pid" ]; then cp_pid=$(cat "$D/compact.pid"); fi
    pgrep -af '[6]6-windowpin-pressure.py|[k]fd_events.py|[6]6-windowpin.sh' || true
    if [ -n "$ip" ]; then kill "$ip" 2>/dev/null || true; wait "$ip" 2>/dev/null || true; ip=; fi
    if [ -n "$cp_pid" ]; then sudo -n kill "$cp_pid" 2>/dev/null || true; cp_pid=; fi
    if [ -n "$kp" ]; then sudo -n kill "$kp" 2>/dev/null || true; kp=; fi
    if [ -n "$sp" ]; then kill "$sp" 2>/dev/null || true; wait "$sp" 2>/dev/null || true; sp=; fi
    if [ -n "${D:-}" ]; then rm -f "$D/kfd.pid" "$D/compact.pid"; fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM
fails=0
echo "WINDOWPIN start $(date -Is) passes=$PASSES run=$RUN"
for ((i=1;i<=PASSES;i++)); do
    order=(m b0 b1); ((i % 2 == 0)) && order=(b1 b0 m)
    for arm in "${order[@]}"; do
        n=${RUN}_${arm}_e$i; D=$H/ab/$n; mkdir "$D"
        exe=$BR; immune=0
        [ "$arm" != m ] || exe=$MAIN
        [ "$arm" != b1 ] || immune=1
        # PID monitor includes only engines whose executable matches this arm.
        ( while true; do
            pids=
            for q in /sys/class/kfd/kfd/proc/*; do
                q=${q##*/}
                actual=$(readlink "/proc/$q/exe" 2>/dev/null || true)
                [ "$actual" != "$exe" ] || pids="$pids,$q"
            done
            printf '%s %s %s' "$(date +%s%3N)" "$(awk '/^MemAvailable/{print $2}' /proc/meminfo)" "${pids#,}"
            for c in /sys/class/drm/card[0-9]*/device; do
                [ ! -r "$c/mem_info_gtt_used" ] || printf ' %s:%s:%s' "${c%/device}" "$(cat "$c/mem_info_gtt_used")" "$(cat "$c/mem_info_vram_used")"
            done
            printf '\n'; sleep 1
          done ) > "$D/mem.log" & sp=$!
        sudo -n sh -c 'umask 022; echo $$ > "$1"; exec python3 -I "$2" "$3" --all --secs 7200' \
            sh "$D/kfd.pid" "$K/kfd_events.py" "$D/kfd.log" & ksuper=$!
        sudo -n sh -c 'umask 022; echo $$ > "$1"; exec python3 -I "$2" compact' \
            sh "$D/compact.pid" "$H/staging/66-windowpin-pressure.py" & csuper=$!
        for ((r=0;r<50;r++)); do
            if [ -s "$D/kfd.pid" ] && [ -s "$D/compact.pid" ] && [ -s "$D/kfd.log" ]; then break; fi
            sleep .1
        done
        kp=$(cat "$D/kfd.pid"); cp_pid=$(cat "$D/compact.pid")
        [ -s "$D/kfd.log" ] || { echo 'GATE FAIL: KFD logger did not start'; exit 1; }
        ionice -c3 nice -n 19 python3 -I "$H/staging/66-windowpin-pressure.py" io "${IOF[@]}" & ip=$!
        cp /proc/vmstat "$D/vmstat.before"
        boot_ns > "$D/span"
        echo "ARM $n $(date -Is) immune=$immune"
        if env -u HSA_USERPTR_FOR_PAGED_MEM -u HSA_SVM_FOR_PAGED_MEM -u GPU_PINNED_MIN_XFER_SIZE CFG="$SNAP/live.json" MODEL=qwen3.8-abl-iq3_s PROMPTS="$SNAP/edit24.json" \
            "$H/agab2.sh" "$n" "EXE=$exe" "STRATA_HOST_IMMUNE=$immune" \
            STRATA_KFD_WATCH=1 STRATA_MLOCK_REGISTERED=0 STRATA_PIPELINE_DEBUG=1 \
            STRATA_PIPELINE_TRACE=pipeline.trace > "$D/agab.out" 2>&1; then rc=0; else rc=$?; fi
        echo "$rc" > "$D/harness.status"; ((fails+=rc!=0)) || true
        boot_ns >> "$D/span"
        cp /proc/vmstat "$D/vmstat.after"
        sleep 1
        cleanup
        wait "$ksuper" 2>/dev/null || true; wait "$csuper" 2>/dev/null || true
    done
done
python3 -I "$H/staging/66-windowpin-read.py" "$H/ab" "$K/kfd_episodes.py" "$SNAP/edit24.json" "$RUN" "$PASSES" "$fails"
echo "WINDOWPIN end $(date -Is)"
