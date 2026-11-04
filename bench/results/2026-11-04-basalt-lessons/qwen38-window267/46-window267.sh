#!/bin/bash
# #267 reproduction. Stage only; run in a coordinator-owned GPU/benchmark slot.
# No build, sysctl changes, drop_caches, or unrelated-process termination here.
# Each dirty arm uses one temporary file: <=24 GiB footprint, <=32 GiB written,
# rate-limited to 300 MiB/s. It starts AFTER model ready and is removed afterward.
H=/home/ronald_rump/inbox/agents/claude/strata-ab
OLD=/home/ronald_rump/inbox/agents/claude/strata-q38-next/build/strata
FIX=/home/ronald_rump/inbox/agents/claude/strata-verify267/build/strata
FIX_REPO=/home/ronald_rump/inbox/agents/claude/strata-verify267
FIX_COMMIT=230a21fe18bbdeba5477ff211611e42bce98ef30
# Freeze the first failing arm's config, not a live JSON that may change later.
CFG=$H/ab/w_kvs2_kv61_agent/cfg.json
MODEL=qwen3.8-abl-iq3_s
PROMPTS=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json
REPEATS=${W267_REPEATS:-1}
cd "$H" || exit 1
ulimit -c 0  # the diagnostic-only abort must not dump the model's tens of GiB
unset STRATA_PLE_CACHE_UNLOCK STRATA_PLE_AUTO_RAM STRATA_MTP_CHAIN_CAP STRATA_PF_ROUTE_AHEAD
unset STRATA_WATCHDOG_S STRATA_WATCHDOG_IO_S
COMMON=("STRATA_PLE_CACHE_UNLOCK=" "STRATA_MTP_CHAIN_CAP=0" "STRATA_PF_ROUTE_AHEAD="
        "STRATA_VERIFY_TRACE=1" "STRATA_PIPELINE_DEBUG=1" "STRATA_PIPELINE_LOG=1"
        "STRATA_PIPELINE_TRACE=pipeline.trace" "STRATA_DECODE_TIMING=1"
        "STRATA_WATCHDOG_S=" "STRATA_WATCHDOG_IO_S=")
monitor= pressure_file= harness=
cleanup() {
    if [ -n "$harness" ]; then
        # This session contains only this arm's harness/server. The server's
        # SIGTERM handler closes the engine; never match unrelated processes.
        kill -TERM -- "-$harness" 2>/dev/null || true
        wait "$harness" 2>/dev/null || true
        harness=
    fi
    if [ -n "$monitor" ]; then
        kill -TERM "$monitor" 2>/dev/null || true
        wait "$monitor" 2>/dev/null || true
        monitor=
    fi
    if [ -n "$pressure_file" ]; then rm -f -- "$pressure_file"; pressure_file=; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# The collector runs on CPUs only. In dirty mode it also performs bounded buffered
# writes, matching a large download rather than allocating an OOM-sized VM heap.
collect() {
    exec python3 -I - "$1" "$2" "$3" "$4" <<'PY'
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

arm, exe, filename, dirty = sys.argv[1:]
root = Path(arm)
dirty = dirty == '1'
stopping = False
signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__('stopping', True))
signal.signal(signal.SIGINT, lambda *_: globals().__setitem__('stopping', True))
fd = -1
written = 0
start = None
next_sample = 0.0
limit = 32 * 1024**3
footprint = 24 * 1024**3
rate = 300 * 1024**2
chunk = b'\x5a' * (1024**2)


def read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return ''


def sample(final=False):
    mem = {}
    for line in read('/proc/meminfo').splitlines():
        key, _, value = line.partition(':')
        if key in ('MemAvailable', 'Dirty', 'Writeback', 'SwapFree', 'Unevictable', 'Mlocked'):
            mem[key + '_kib'] = int(value.split()[0])
    vm = {}
    for line in read('/proc/vmstat').splitlines():
        key, value = line.split()
        if key in ('pgmajfault', 'pswpin', 'pswpout') or key.startswith(('pgscan_', 'pgsteal_', 'allocstall_')):
            vm[key] = int(value)
    engines = []
    try:
        pids = subprocess.run(['ps', '-C', 'strata', '-o', 'pid='], capture_output=True,
                              text=True, check=False).stdout.split()
        for pid in pids:
            try:
                if os.path.realpath('/proc/' + pid + '/exe') != os.path.realpath(exe):
                    continue
                base = Path('/proc') / pid
                stat = read(base / 'stat').rpartition(') ')[2].split()
                status = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                          for line in read(base / 'status').splitlines() if ':' in line}
                waits = {}
                for task in (base / 'task').iterdir():
                    w = read(task / 'wchan').strip()
                    if w and w != '0':
                        waits[task.name] = w
                engines.append({'pid': int(pid), 'minflt': int(stat[7]), 'majflt': int(stat[9]),
                                'status': {k: status.get(k) for k in ('VmRSS', 'VmLck', 'VmPin', 'VmSwap', 'RssFile')},
                                'thread_wchan': waits})
            except (OSError, IndexError, ValueError):
                pass
    except OSError:
        pass
    print(json.dumps({'kind': 'final' if final else 'sample', 'time': time.time(),
                      'elapsed': 0 if start is None else time.monotonic() - start,
                      'dirty_arm': dirty, 'written_bytes': written, 'mem': mem, 'vm': vm,
                      'memory_psi': read('/proc/pressure/memory'), 'io_psi': read('/proc/pressure/io'),
                      'engines': engines}), flush=True)


try:
    deadline = time.monotonic() + 1500
    while not stopping and time.monotonic() < deadline:
        if 'ready:' in read(root / 'server.out'):
            start = time.monotonic()
            break
        time.sleep(0.25)
    if start is None:
        raise RuntimeError('server never reached ready before pressure/collection')
    if dirty:
        fd = os.open(filename, os.O_RDWR)
        os.posix_fallocate(fd, 0, footprint)
    print(json.dumps({'kind': 'started', 'dirty_arm': dirty, 'time': time.time(),
                      'max_written_bytes': limit, 'file_bytes': footprint, 'rate_bytes_s': rate}), flush=True)
    while not stopping and time.monotonic() - start < 600:
        now = time.monotonic()
        if now >= next_sample:
            sample()
            next_sample = now + 1
        if not dirty or written >= limit:
            time.sleep(0.2)
            continue
        offset = written % footprint
        n = min(len(chunk), limit - written, footprint - offset)
        view = memoryview(chunk)[:n]
        while view and not stopping:
            count = os.pwrite(fd, view, offset)
            if count <= 0:
                raise OSError('buffered writer made no progress')
            written += count
            offset += count
            view = view[count:]
        delay = written / rate - (time.monotonic() - start)
        if delay > 0:
            time.sleep(min(delay, 0.2))
finally:
    sample(final=True)
    if fd >= 0:
        os.fsync(fd)  # only our file, after the request batch; never sync unrelated data
        os.close(fd)
PY
}

run_arm() {
    local name=$1 exe=$2 dirty=$3 rc mrc
    shift 3
    local D=$H/ab/$name
    if [ -e "$D" ]; then echo "GATE FAIL: refusing existing arm $D"; return 1; fi
    mkdir -p "$D" || return 1
    if [ "$dirty" = 1 ]; then
        local free_bytes
        free_bytes=$(df -B1 --output=avail /srv/models | tail -1 | tr -d ' ')
        if [ "${free_bytes:-0}" -lt 34359738368 ]; then
            echo 'GATE FAIL: pressure file needs 24 GiB plus 8 GiB disk headroom'
            return 1
        fi
        pressure_file=$(mktemp /srv/models/.strata-267.XXXXXX) || return 1
    fi
    collect "$D" "$exe" "$pressure_file" "$dirty" > "$D/host-pressure.jsonl" 2> "$D/host-pressure.err" &
    monitor=$!
    setsid env CFG="$CFG" MODEL="$MODEL" PROMPTS="$PROMPTS" \
        "$H/agab2.sh" "$name" "EXE=$exe" "${COMMON[@]}" "$@" &
    harness=$!
    wait "$harness"; rc=$?
    kill -TERM -- "-$harness" 2>/dev/null || true
    harness=
    kill -TERM "$monitor" 2>/dev/null || true
    wait "$monitor"; mrc=$?
    monitor=
    if [ -n "$pressure_file" ]; then rm -f -- "$pressure_file"; pressure_file=; fi
    echo "HARNESS_STATUS $name $rc MONITOR_STATUS $mrc"
    printf '%s %s\n' "$rc" "$mrc" > "$D/status.txt"
    # Expected broken controls must not prevent the fixed arms from running.
    return 0
}

fixed_ready() {
    local exe=$1 repo=$2 commit=$3 built fixed_at tag
    if [ ! -x "$exe" ]; then
        echo "WINDOW267 SKIPPED: fixed binary missing/not executable: $exe; the slot build may have failed"
        return 2
    fi
    fixed_at=$(git -C "$repo" show -s --format=%ct "$commit" 2>/dev/null) || {
        echo "WINDOW267 SKIPPED: cannot verify required fix commit $commit in $repo"; return 2;
    }
    built=$(stat -c %Y "$exe") || return 2
    if [ "$built" -lt "$fixed_at" ]; then
        echo "WINDOW267 SKIPPED: fixed binary predates fix commit $commit; rebuild in a build slot"
        return 2
    fi
    # Timestamps alone cannot distinguish a later build of an older tree. These
    # three new diagnostic strings cover residency, broadcast release, and timeout
    # logging; inspect the binary as bytes, never execute it during preflight.
    for tag in 'STRATA_PLE_CACHE_UNLOCK' 'RELEASE-SIGNAL' 'strata verify: completion timeout on CUDA'; do
        if ! LC_ALL=C grep -aFq "$tag" "$exe"; then
            echo "WINDOW267 SKIPPED: fixed binary lacks fix marker '$tag' from $commit; stale/wrong build"
            return 2
        fi
    done
}

echo "window267 start $(date --iso-8601=seconds)"
(
    trap cleanup EXIT
    if ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || [ "$REPEATS" -gt 5 ]; then
        echo 'W267_REPEATS must be 1..5'; exit 1
    fi
    test -x "$OLD" || { echo "WINDOW267 SKIPPED: old control binary missing: $OLD"; exit 2; }
    fixed_ready "$FIX" "$FIX_REPO" "$FIX_COMMIT" || exit 2
    for ((r=1; r<=REPEATS; ++r)); do
        for profile in auto direct; do
            for arm in old_quiet fix_quiet old_dirty fix_dirty; do
                test ! -e "$H/ab/w267_${profile}_${arm}_$r" || { echo 'GATE FAIL: stale arm directory'; exit 1; }
            done
        done
        test ! -e "$H/ab/w267_ram_old_dirty_$r" || { echo 'GATE FAIL: stale ram arm'; exit 1; }
    done
    test ! -e "$H/ab/w267_direct_old_earlydiag" || { echo 'GATE FAIL: stale diagnostic arm'; exit 1; }
    sha256sum "$OLD" "$FIX"
    echo 'Host dirty limits (read only)'
    for key in dirty_bytes dirty_background_bytes dirty_ratio dirty_background_ratio; do
        printf '%s ' "$key"; cat "/proc/sys/vm/$key"
    done
    for ((r=1; r<=REPEATS; ++r)); do
        for profile in auto direct; do
            if [ "$profile" = auto ]; then
                PROFILE=("STRATA_PLE_AUTO_RAM=1" "STRATA_KV_PF_SLOTS=0" "ARG=--kv-resident 61440")
            else
                PROFILE=("STRATA_PLE_AUTO_RAM=0" "STRATA_KV_PF_SLOTS=1" "ARG=--kv-resident 65536" "ARG=--ple-io direct")
            fi
            run_arm "w267_${profile}_old_quiet_$r" "$OLD" 0 "${PROFILE[@]}"
            run_arm "w267_${profile}_fix_quiet_$r" "$FIX" 0 "${PROFILE[@]}"
            run_arm "w267_${profile}_old_dirty_$r" "$OLD" 1 "${PROFILE[@]}"
            run_arm "w267_${profile}_fix_dirty_$r" "$FIX" 1 "${PROFILE[@]}"
        done
        # Existing-option workaround: same old binary, but explicit ram forbids auto-unlock.
        run_arm "w267_ram_old_dirty_$r" "$OLD" 1 "ARG=--ple-io ram" \
            "STRATA_KV_PF_SLOTS=0" "ARG=--kv-resident 61440"
    done
    # Diagnostic-only arm. The old binary does not dump all GPU breadcrumbs in
    # done() before its 20 s timeout. Its EXISTING fatal watchdog at 12 s does:
    # report two snapshots of flags, events, GPU breadcrumbs, disk-wait threads,
    # memory and faults BEFORE release. This deliberately changes failure timing;
    # it is NOT counted as a natural #267 reproduction or as a throughput result.
    run_arm w267_direct_old_earlydiag "$OLD" 1 "ARG=--ple-io direct" \
        "STRATA_PLE_AUTO_RAM=0" "STRATA_KV_PF_SLOTS=1" "ARG=--kv-resident 65536" \
        "STRATA_WATCHDOG_S=12" "STRATA_WATCHDOG_IO_S=0"
    python3 -I - "$H" "$PROMPTS" "$REPEATS" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]) / 'ab'
cases = [c['id'] for c in json.loads(Path(sys.argv[2]).read_text())['cases']]
failures = []
inconclusive = []
reproduced = 0


def inspect(name, reference=None, required=True):
    folder = root / name
    log = (folder / 'engine.log').read_text(errors='replace') if (folder / 'engine.log').exists() else ''
    natural267 = any('verify:' in line and '#267' in line for line in log.splitlines())
    print(f'OUTCOME {name}: natural_verify267={natural267}')
    texts = {}
    bad = []
    try:
        host = [json.loads(line) for line in (folder / 'host-pressure.jsonl').read_text().splitlines()]
        samples = [x for x in host if x['kind'] in ('sample', 'final')]
        if not samples or not any(x['kind'] == 'started' for x in host):
            raise ValueError('collector/pressure did not start')
        written = max(x['written_bytes'] for x in samples)
        print(f'HOST {name}: written={written / 1024**3:.2f} GiB, '
              f'min_available={min(x["mem"].get("MemAvailable_kib", 0) for x in samples) / 1024**2:.2f} GiB, '
              f'max_dirty={max(x["mem"].get("Dirty_kib", 0) for x in samples) / 1024:.0f} MiB')
        if '_dirty_' in name and written < 4 * 1024**3:
            inconclusive.append(f'{name}: under 4 GiB written during requests')
        status = [int(x) for x in (folder / 'status.txt').read_text().split()]
        if status != [0, 0]:
            bad.append(f'harness/monitor status {status}')
    except (OSError, KeyError, ValueError) as exc:
        bad.append(f'host collector/status: {exc}')
    for cid in cases:
        try:
            doc = json.loads((folder / f'out_{cid}.json').read_text())
            text = doc['choices'][0]['message']['content']
            if not isinstance(text, str) or doc['timings']['predicted_n'] <= 0:
                raise ValueError('invalid response or empty generation')
            texts[cid] = text
            print(f'MD5 {name} {cid} {hashlib.md5(text.encode()).hexdigest()}')
            if reference is not None and text != reference.get(cid):
                bad.append(f'{cid}: full text differs from quiet old control')
        except (OSError, KeyError, IndexError, TypeError, ValueError) as exc:
            bad.append(f'{cid}: {exc}')
    if required and (natural267 or bad):
        failures.append(f'{name}: #267={natural267}, ' + '; '.join(bad))
    elif bad:
        print(f'CONTROL FAILURE {name}: ' + '; '.join(bad))
    return texts, natural267, log


for r in range(1, int(sys.argv[3]) + 1):
    for profile in ('auto', 'direct'):
        ref, _, old_log = inspect(f'w267_{profile}_old_quiet_{r}')
        _, _, fixed_log = inspect(f'w267_{profile}_fix_quiet_{r}', ref)
        if profile == 'auto':
            if 'ram chosen by the residency policy' not in old_log:
                inconclusive.append(f'auto old quiet {r}: mmap-auto was not admitted; this did not test unlocked mmap')
            if 'retaining the ' not in fixed_log:
                inconclusive.append(f'auto fix quiet {r}: PLE lock was not retained/admitted')
        _, hit, _ = inspect(f'w267_{profile}_old_dirty_{r}', ref, required=False)
        reproduced += int(hit)
        inspect(f'w267_{profile}_fix_dirty_{r}', ref)
    # This is a separate workaround probe, not a condition for the direct-mode fix.
    inspect(f'w267_ram_old_dirty_{r}', required=False)
diag = root / 'w267_direct_old_earlydiag' / 'engine.log'
if diag.exists():
    text = diag.read_text(errors='replace')
    print('EARLY_DIAG: stall_report=', 'stall report' in text, 'GPU_breadcrumbs=', 'GPU breadcrumbs' in text)
else:
    inconclusive.append('early diagnostic arm has no engine log')
if not reproduced:
    inconclusive.append('no natural baseline #267 reproduced: no robustness claim is established')
for message in failures:
    print('GATE FAIL:', message)
for message in inconclusive:
    print('INCONCLUSIVE:', message)
print('WINDOW267', 'FAILED' if failures else 'INCONCLUSIVE' if inconclusive else 'REPRODUCED: fixed arms survived with matching text')
sys.exit(1 if failures else 2 if inconclusive else 0)
PY
) > "$H/window267.out" 2>&1
status=$?
if [ "$status" -eq 2 ]; then grep '^WINDOW267 SKIPPED:' "$H/window267.out" || true; fi
echo WINDOW267 DONE
exit "$status"
