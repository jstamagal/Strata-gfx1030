#!/bin/bash
# Window STALL: where do the 0.3-4 s single-window stalls go? Live IQ3_S config, main build-hip,
# 4 agent + 4 edit passes with pipeline traces, while stall.bt logs strata off-CPU intervals >= 100 ms
# (kernel stacks) and block I/O > 50 ms, and a sampler logs vmstat/PSI deltas every 100 ms.
H=/home/ronald_rump/inbox/agents/claude/strata-ab
E=/home/ronald_rump/Strata_donttouch/build-hip/strata
CFG=$H/qwen-iq3s.json
MODEL=qwen3.8-abl-iq3_s
AGENT=/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json
EDIT=$H/edit-prompts.json
OUT=$H/stall
cd "$H" || exit 1
COMMON=("EXE=$E" "STRATA_PIPELINE_DEBUG=1" "STRATA_PIPELINE_TRACE=pipeline.trace")
ARMS=(ws_agent_1 ws_edit_1 ws_agent_2 ws_edit_2 ws_agent_3 ws_edit_3 ws_agent_4 ws_edit_4)
bt= sampler=
cleanup() {
    if [ -n "$bt" ]; then sudo -n kill -INT "$bt" 2>/dev/null; wait "$bt" 2>/dev/null; bt=; fi
    if [ -n "$sampler" ]; then kill -TERM "$sampler" 2>/dev/null; wait "$sampler" 2>/dev/null; sampler=; fi
}
trap cleanup EXIT
echo "window STALL start $(date +%H:%M)"
(
    trap cleanup EXIT  # a subshell does not inherit the EXIT trap
    for arm in "${ARMS[@]}"; do
        if [ -e "$H/ab/$arm" ]; then echo "GATE FAIL: $H/ab/$arm exists"; exit 1; fi
    done
    if [ -e "$OUT" ]; then echo "GATE FAIL: $OUT exists"; exit 1; fi
    mkdir -p "$OUT"
    sha256sum "$E"
    git -C /home/ronald_rump/Strata_donttouch rev-parse HEAD
    sudo -n bpftrace "$H/stall.bt" > "$OUT/stall.bt.log" 2>&1 &
    bt=$!
    python3 -I - "$OUT/sampler.log" <<'PY' &
import signal, sys, time
out = open(sys.argv[1], 'w', buffering=1)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
keys = ('compact_stall', 'allocstall_normal', 'allocstall_movable', 'pgmajfault', 'pgscan_direct')
def vm():
    d = {}
    for line in open('/proc/vmstat'):
        k, v = line.split()
        if k in keys: d[k] = int(v)
    return d
def psi(name):
    some = open(f'/proc/pressure/{name}').readline().split()
    return int(some[-1].split('=')[1])
out.write('mono_ms ' + ' '.join(keys) + ' psi_mem_us psi_io_us\n')
prev = (vm(), psi('memory'), psi('io'))
while True:
    time.sleep(0.1)
    cur = (vm(), psi('memory'), psi('io'))
    out.write(f'{time.monotonic_ns() // 1000000} ' +
              ' '.join(str(cur[0][k] - prev[0][k]) for k in keys) +
              f' {cur[1] - prev[1]} {cur[2] - prev[2]}\n')
    prev = cur
PY
    sampler=$!
    sleep 5
    grep -q STALLBT "$OUT/stall.bt.log" || { echo 'GATE FAIL: stall.bt did not attach'; cat "$OUT/stall.bt.log"; exit 1; }
    fails=0
    for arm in "${ARMS[@]}"; do
        case $arm in ws_agent_*) p=$AGENT ;; *) p=$EDIT ;; esac
        echo "ARM $arm start mono_ms=$(python3 -I -c 'import time; print(time.monotonic_ns() // 1000000)')"
        env CFG="$CFG" MODEL="$MODEL" PROMPTS="$p" "$H/agab2.sh" "$arm" "${COMMON[@]}"
        rc=$?
        echo "ARM $arm end mono_ms=$(python3 -I -c 'import time; print(time.monotonic_ns() // 1000000)') rc=$rc"
        [ "$rc" -eq 0 ] || fails=$((fails + 1))
    done
    cleanup
    python3 -I - "$H/ab" "$OUT" "$fails" <<'PY'
import glob, hashlib, json, sys
from pathlib import Path
ab, out, fails = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
bad = []
if fails: bad.append(f'{fails} harness run(s) failed')
for group in ('agent', 'edit'):
    arms = [f'ws_{group}_{i}' for i in range(1, 5)]
    ref = {}
    for arm in arms:
        for f in sorted((ab / arm).glob('out_*.json')):
            try:
                text = json.loads(f.read_text())['choices'][0]['message']['content']
            except Exception as exc:
                bad.append(f'{arm} {f.name}: {exc}'); continue
            h = hashlib.md5(text.encode()).hexdigest()
            if ref.setdefault(f.name, h) != h: bad.append(f'{arm} {f.name}: text differs from ws_{group}_1')
gaps = 0
for tr in sorted(ab.glob('ws_*/pipeline.trace')):
    prev, req = None, -1
    for line in open(tr, errors='replace'):
        if line.startswith('#'): req += 1; prev = None; continue
        p = line.split()
        try: t = float(p[0])
        except (ValueError, IndexError): continue
        if prev and req > 0 and t - prev[0] > 300:
            gaps += 1
            print(f'GAP {tr.parent.name} req {req} {t - prev[0]:.0f} ms | {prev[1]} -> {line.strip()}')
        prev = (t, line.strip())
offs = [l for l in open(out / 'stall.bt.log', errors='replace') if l.startswith('OFF ')]
bios = [l for l in open(out / 'stall.bt.log', errors='replace') if l.startswith('BIO ')]
print(f'SUMMARY gaps>300ms={gaps} off_cpu>=100ms={len(offs)} slow_bio={len(bios)}')
for b in bad: print('GATE FAIL:', b)
print('WINDOWSTALL ' + ('TEXT/HARNESS CHECK FAILED' if bad else 'COLLECTED: texts deterministic across passes'))
PY
) > "$H/windowstall.out" 2>&1
status=$?
echo WINDOWSTALL DONE
exit "$status"
