#!/usr/bin/env python3
"""Window PIN: raw per-node eviction counts, episodes, case identity, gaps and rate."""
import collections, hashlib, json, re, subprocess, sys
from pathlib import Path
ab, kep, prompts, run, passes, failures = sys.argv[1:]
ab = Path(ab); passes = int(passes)
expected = {c['id'] for c in json.load(open(prompts))['cases']}
bad = []
if int(failures): bad.append(f'{failures} harness failures')
rows = {}
for i in range(1, passes + 1):
    for arm in ('m', 'b0', 'b1'):
        name = f'{run}_{arm}_e{i}'; d = ab / name
        cases = {}; nt = 0; seconds = 0.0
        for f in sorted(d.glob('out_*.json')):
            case = f.stem[4:]
            try:
                j = json.loads(f.read_text()); t = j['timings']; text = j['choices'][0]['message']['content']
                cases[case] = hashlib.sha256(text.encode('utf-8')).hexdigest()
                nt += t['predicted_n']; seconds += t['predicted_n'] / t['predicted_per_second']
            except (KeyError, TypeError, ValueError, ZeroDivisionError, OSError) as e:
                bad.append(f'{name}/{case}: invalid response {e}')
        if set(cases) != expected: bad.append(f'{name}: missing/extra cases {sorted(set(cases) ^ expected)}')
        gaps = []; prev = None; request = -1
        trace = d / 'pipeline.trace'
        if not trace.exists(): bad.append(f'{name}: missing pipeline trace')
        else:
            for line in trace.read_text(errors='replace').splitlines():
                if line.startswith('#'): request += 1; prev = None; continue
                try: ms = float(line.split()[0])
                except (ValueError, IndexError): continue
                if request > 0 and prev is not None and ms - prev > 300: gaps.append(round(ms - prev, 3))
                prev = ms
        pids = set(); available = []; memory_peaks = {}
        if (d / 'mem.log').exists():
            for line in (d / 'mem.log').read_text().splitlines():
                f = line.split()
                if len(f) < 2: continue
                available.append(int(f[1]))
                for token in f[2:]:
                    if ':' not in token:
                        pids.update(int(p) for p in token.split(',') if p.isdigit())
                    else:
                        card, gtt, vram = token.rsplit(':', 2)
                        old = memory_peaks.get(card, [0, 0])
                        memory_peaks[card] = [max(old[0], int(gtt)), max(old[1], int(vram))]
        if not pids: bad.append(f'{name}: no engine PID sampled')
        try: lo, hi = map(int, (d / 'span').read_text().split())
        except (OSError, ValueError): lo, hi = 0, 2**63; bad.append(f'{name}: invalid timing span')
        counts = collections.Counter(); klog = d / 'kfd.log'; episodes = ''
        if not klog.exists() or not klog.stat().st_size: bad.append(f'{name}: missing KFD log')
        elif pids:
            for line in klog.read_text(errors='replace').splitlines():
                f = line.split()
                if len(f) < 7 or f[2] != 'QUEUE_EVICTION': continue
                try: ns, pid, trigger = int(f[3]), -int(f[4]), int(f[6])
                except ValueError: continue
                if lo <= ns <= hi and pid in pids: counts[{0:'svm',1:'userptr',2:'ttm'}.get(trigger, str(trigger))] += 1
            p = subprocess.run([sys.executable, '-I', kep, str(klog), '--pids', ','.join(map(str, sorted(pids))),
                                '--from-ns', str(lo), '--to-ns', str(hi), '--list', '--min-ms', '300'],
                               text=True, capture_output=True)
            episodes = p.stdout
            (d / 'kfd-episodes.txt').write_text(episodes + p.stderr)
            if p.returncode: bad.append(f'{name}: episode parser failed')
        log = (d / 'engine.log').read_text(errors='replace') if (d / 'engine.log').exists() else ''
        if arm == 'b1' and 'STRATA_HOST_IMMUNE=1: host allocations use GTT' not in log:
            bad.append(f'{name}: missing immune bootstrap marker')
        if arm == 'b1' and 'STRATA_HOST_IMMUNE: reserved' not in log:
            bad.append(f'{name}: missing persistent checkpoint reservation')
        if '100% of the experts resident' not in log: bad.append(f'{name}: resident gate missing')
        r = dict(name=name, cases=cases, tokens=nt, seconds=seconds, tps=nt/seconds if seconds else 0,
                 gaps_ms=gaps, node_evictions=dict(counts), engine_pids=sorted(pids),
                 available_floor_GiB=min(available, default=0)/1048576, memory_peak_bytes=memory_peaks,
                 episodes=episodes, gtt_policy_marker='host allocations use GTT' in log)
        rows[name] = r
        print(f'{name}: {len(cases)}/{len(expected)} cases; {r["tps"]:.2f} tok/s; gaps>300ms={gaps}; '
              f'raw node evictions SVM/USERPTR/TTM={counts["svm"]}/{counts["userptr"]}/{counts["ttm"]}; '
              f'MemAvailable floor={r["available_floor_GiB"]:.2f} GiB; pids={sorted(pids)}')
        print(episodes.rstrip())
        print('  global card peaks (GTT, VRAM bytes):', memory_peaks)
for i in range(1, passes + 1):
    for ref_arm, arm in (('m','b0'), ('m','b1'), ('b0','b1')):
        ref, cur = rows[f'{run}_{ref_arm}_e{i}'], rows[f'{run}_{arm}_e{i}']
        for case in sorted(expected):
            equal = case in cur['cases'] and case in ref['cases'] and cur['cases'][case] == ref['cases'][case]
            print(f'IDENTITY pass={i} {ref_arm}/{arm} case={case} {"SAME" if equal else "FAIL"}')
            if not equal: bad.append(f'pass {i} {ref_arm}/{arm} {case}: text differs or missing')
for arm in ('m','b0','b1'):
    rr = [rows[f'{run}_{arm}_e{i}'] for i in range(1, passes+1)]
    nt = sum(r['tokens'] for r in rr); tt = sum(r['seconds'] for r in rr)
    counts = {k:sum(r['node_evictions'].get(k,0) for r in rr) for k in ('svm','userptr','ttm')}
    print(f'SET {arm}: {nt/tt if tt else 0:.2f} tok/s; gaps={sum(len(r["gaps_ms"]) for r in rr)}; node evictions={counts}')
(ab / f'{run}-summary.json').write_text(json.dumps(dict(arms=rows, failures=bad),indent=2)+'\n')
for message in bad: print('GATE FAIL:',message)
print('WINDOWPIN ' + ('CHECK FAILED' if bad else 'COLLECTED'))
sys.exit(bool(bad))
