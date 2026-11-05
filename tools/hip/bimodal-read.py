#!/usr/bin/env python3
"""Read-only, CPU-only window BIM readout. Never starts an engine or queries a GPU."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


def trace(path):
    reqs = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('# request'):
            if 'ended in an error' in line:
                raise ValueError('trace request ended in an engine error')
            reqs.append([])
        elif line and not line.startswith('#'):
            f = line.split()
            if not reqs or len(f) != 5:
                raise ValueError(f'malformed trace event: {line!r}')
            time = float(f[0])
            if not math.isfinite(time) or time < 0 or (reqs[-1] and time < reqs[-1][-1][0]) or f[1] == 'ENDED':
                raise ValueError('invalid trace clock or ended request')
            reqs[-1].append((time, f[1], *map(int, f[2:])))
    return reqs


def outputs(path):
    out = {}
    for f in sorted(Path(path).glob('out_*.json')):
        j = json.loads(f.read_text())
        t = j['timings']
        n, rate = int(t['predicted_n']), float(t['predicted_per_second'])
        if n <= 0 or not math.isfinite(rate) or rate <= 0:
            raise ValueError(f'{f.name}: invalid generated count/rate')
        text = j['choices'][0]['message']['content']
        if not isinstance(text, str):
            raise ValueError(f'{f.name}: content is not text')
        out[f.stem[4:]] = {'text': text, 'n': n, 'seconds': n / rate}
    if not out:
        raise ValueError('no response JSON files')
    return out


def summarize(path):
    path = Path(path)
    out = outputs(path)
    reqs = trace(path / 'pipeline.trace')
    if len(reqs) != len(out) + 1:
        raise ValueError(f'{len(reqs)} trace requests for {len(out)} cases plus warmup')
    schedule, front, widths, classes = {}, {}, collections.Counter(), collections.Counter()
    waits = {'launch_finish': [], 'launch_service': [], 'service_finish': []}
    pk = l0 = verdicts = served = 0
    for case, events in zip(sorted(out), reqs[1:]):
        pending, by_key = collections.deque(), collections.defaultdict(collections.deque)
        rows, launches, offset, width, last_finish = [], [], 0, None, -math.inf
        for time, event, seq, a, b in events:
            if event == 'L0':
                if b not in (0, 10):
                    raise ValueError('window BIM expects exactly two stages')
                w = {'launch': time, 'seq': seq, 'width': a, 'spec': int(b >= 10), 'service': None}
                pending.append(w)
                launches.append((seq, a, w['spec']))
                by_key[(seq, w['spec'])].append(w)
                l0 += 1
            elif event == 'S0':
                if a != 0:
                    raise ValueError('unexpected front stage in S0')
                candidates = [w for w in by_key[(seq, b)] if w['service'] is None]
                if not candidates:
                    raise ValueError(f'orphan S0 at seq {seq}')
                candidates[0]['service'] = time
            elif event == 'F0':
                if not pending:
                    raise ValueError('orphan F0')
                w = pending.popleft()
                if (w['seq'], w['spec']) != (seq, b):
                    raise ValueError('F0 not paired with its FIFO launch')
                by_key[(seq, b)].popleft()
                start = max(w['launch'], last_finish)
                if time < start:
                    raise ValueError('negative stage duration')
                waits['launch_finish'].append(time - start)
                if w['service'] is not None:
                    if w['service'] > time:
                        raise ValueError('host service after F0')
                    waits['launch_service'].append(max(0.0, w['service'] - start))
                    waits['service_finish'].append(time - max(start, w['service']))
                    served += 1
                last_finish = time
            elif event == 'L1':
                width = a
                widths[a] += 1
            elif event == 'PK':
                pk += 1
            elif event == 'V':
                if width is None or not 0 <= a < width or b not in (-1, 0, 1):
                    raise ValueError('invalid verdict')
                rows.append((offset, width, a, b))
                offset += a + 1
                classes[b] += 1
                verdicts += 1
                width = None
        if pending or width is not None:
            raise ValueError('request did not drain')
        if offset != out[case]['n']:
            raise ValueError(f'{case}: trace tokens {offset} != response tokens {out[case]["n"]}')
        schedule[case], front[case] = rows, launches
    n = sum(x['n'] for x in out.values())
    seconds = sum(x['seconds'] for x in out.values())
    canonical = {'verdicts': schedule, 'front_launches': front}
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'name': path.name, 'out': out, 'schedule': schedule, 'front': front, 'fingerprint': fingerprint,
            'widths': widths, 'classes': classes, 'pk': pk, 'l0': l0, 'verdicts': verdicts,
            'served': served, 'waits': waits, 'n': n, 'seconds': seconds, 'tps': n / seconds}


def first_difference(got, ref):
    for case in sorted(set(got) | set(ref)):
        a, b = got.get(case, []), ref.get(case, [])
        for i in range(max(len(a), len(b))):
            x, y = a[i] if i < len(a) else None, b[i] if i < len(b) else None
            if x != y:
                return case, i, x, y
    return None


def divergence_text(got, ref):
    d = first_difference(got, ref)
    if d is None:
        return 'identical'
    case, i, a, b = d
    return f'{case} window={i} got={a} ref={b} (offset,T,accepted,class)'


def at_offset(rows, offset):
    return next((r for r in rows if r[0] == offset), None)


def state(row, fast, slow):
    if row['fingerprint'] == fast['fingerprint']:
        return 'FAST', 'fast'
    if row['fingerprint'] == slow['fingerprint']:
        return 'SLOW', 'slow'
    d = first_difference(slow['schedule'], fast['schedule'])
    if d is None:
        raise ValueError('fast and slow reference schedules are identical')
    case, _, sr, fr = d
    observed = at_offset(row['schedule'].get(case, []), fr[0])
    plan = 'json_parallel_tool_plan'
    count8 = lambda x: sum(r[1] == 8 for r in x['schedule'].get(plan, []))
    midpoint = (count8(fast) + count8(slow)) / 2
    # Structural families, NOT speed bins: first split plus the later lookup-heavy case. Exact fingerprints
    # are reported separately; a new deterministic schedule is never called the reference fast schedule.
    if observed and observed[1] == fr[1] and count8(row) < midpoint:
        return 'FAST_LIKE', 'fast'
    if observed and observed[1] == sr[1] and count8(row) >= midpoint:
        return 'SLOW_LIKE', 'slow'
    return 'MIXED_OR_OTHER', 'other'


def stat_text(values):
    if not values:
        return 'n/a'
    xs = sorted(values)
    p95 = xs[max(0, math.ceil(0.95 * len(xs)) - 1)]
    return f'n={len(xs)} p50={statistics.median(xs):.3f} p95={p95:.3f} max={xs[-1]:.3f}ms'


def pooled(rows):
    return sum(r['n'] for r in rows) / sum(r['seconds'] for r in rows)


def read(ab, manifest_path, fast_path, slow_path, edit_path):
    ab = Path(ab)
    manifest = json.loads(Path(manifest_path).read_text())
    bad, rows = [], []
    fast, slow, edit = map(summarize, (fast_path, slow_path, edit_path))
    print('STATE_RULE exact FAST/SLOW fingerprints include all L0 widths/placements and V decisions; '
          '*_LIKE families use the first split and later parallel-plan T8 count, never tok/s')
    print(f'REFERENCE_FAST {fast["name"]} {fast["tps"]:.3f} tok/s schedule={fast["fingerprint"][:12]}')
    print(f'REFERENCE_SLOW {slow["name"]} {slow["tps"]:.3f} tok/s schedule={slow["fingerprint"][:12]}')
    status = {}
    for line in (ab / 'bim_harness.tsv').read_text().splitlines():
        name, rc = line.split('\t')
        if name in status:
            bad.append(f'duplicate harness status: {name}')
        status[name] = int(rc)
    names = [x['name'] for x in manifest['loads']]
    expected = {f'bim_{a}_agent_p{p}' for a in 'mufc' for p in (1, 2, 3)} | {f'bim_{a}_edit24_p1' for a in 'mufc'}
    if len(names) != 16 or set(names) != expected:
        bad.append('manifest differs from the required 12 agent + 4 edit24 loads')
    if set(status) != set(names):
        bad.append('harness status set differs from the planned 16 loads')
    for item in manifest['loads']:
        name, arm, kind = item['name'], item['arm'], item['kind']
        if status.get(name) != 0:
            bad.append(f'{name}: harness exit {status.get(name)}')
        try:
            d = ab / name
            r = summarize(d)
            r.update(arm=arm, kind=kind, pass_n=item['pass'])
            cfg = json.loads((d / 'cfg.json').read_text())
            env = cfg.get('env', {})
            if cfg.get('exe') != manifest['main_exe' if arm == 'm' else 'branch_exe']:
                bad.append(f'{name}: executable differs from its arm')
            if cfg.get('gpu') != [0, 1] or str(cfg.get('layer_split')) != '25':
                bad.append(f'{name}: both-card layer split gate')
            args = cfg.get('args', [])
            if '--pipeline-windows' not in args or args[args.index('--pipeline-windows') + 1] != '2':
                bad.append(f'{name}: pipeline-windows gate')
            if env.get('STRATA_PIPELINE_DEBUG') != '1' or env.get('STRATA_PIPELINE_TRACE') != str(d / 'pipeline.trace'):
                bad.append(f'{name}: trace/debug gate')
            want_fixed = manifest['profile'] if arm == 'f' else None
            want_clip = manifest['clip'] if arm == 'c' else None
            if env.get('STRATA_PL_POLICY_COSTS') != want_fixed or env.get('STRATA_PL_POLICY_CLIP') != want_clip:
                bad.append(f'{name}: fixed/clip arm gate')
            log = (d / 'engine.log').read_text(errors='replace')
            resident = re.findall(r'(\d+)% of the experts resident', log)
            if not resident or set(resident) != {'100'} or log.count('zero-doorbell graph') < 4:
                bad.append(f'{name}: resident/graph gate')
            has_fixed = 'strata serve: STRATA_PL_POLICY_COSTS=' in log
            has_clip = 'strata serve: STRATA_PL_POLICY_CLIP=' in log
            if has_fixed != (arm == 'f') or has_clip != (arm == 'c'):
                bad.append(f'{name}: startup flag acknowledgement gate')
            if not (d / 'harness.out').is_file():
                bad.append(f'{name}: missing harness log')
            if arm != 'm' and (r['pk'] == 0 or r['served'] != r['l0']):
                bad.append(f'{name}: missing PK/S0 instrumentation')
            ref = fast if kind == 'agent' else edit
            if set(r['out']) != set(ref['out']):
                bad.append(f'{name}: case set differs from reference')
            r['state'], r['family'] = state(r, fast, slow) if kind == 'agent' else ('EDIT_REFERENCE', 'edit')
            rows.append(r)
            cls = r['classes']
            print(f'LOAD {name} {r["tps"]:.3f} tok/s hit/plc/miss {cls[1]}/{cls[0]}/{cls[-1]} '
                  f'L0/V {r["l0"] / r["verdicts"]:.6f} T8/T3 {r["widths"][8]}/{r["widths"][3]} '
                  f'PK {r["pk"]} state={r["state"]} schedule={r["fingerprint"][:12]}')
            print('  FIRST_DIFF V:', divergence_text(r['schedule'], ref['schedule']))
            front_diff = first_difference(r['front'], ref['front'])
            if front_diff is None:
                print('  FIRST_DIFF L0: identical')
            else:
                case, index, got, wanted = front_diff
                print(f'  FIRST_DIFF L0: {case} launch={index} got={got} ref={wanted} (seq,T,spec)')
            print('  WAITS L0->F0', stat_text(r['waits']['launch_finish']),
                  '| L0->S0', stat_text(r['waits']['launch_service']),
                  '| S0->F0', stat_text(r['waits']['service_finish']))
        except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as e:
            bad.append(f'{name}: {e}')
    for kind in ('agent', 'edit24'):
        mains = [r for r in rows if r['arm'] == 'm' and r['kind'] == kind]
        ref = mains[0]['out'] if mains else {}
        for arm in ('m', 'u', 'f', 'c'):
            rr = [r for r in rows if r['arm'] == arm and r['kind'] == kind]
            same = total = 0
            for r in rr:
                if set(r['out']) != set(ref):
                    bad.append(f'{r["name"]}: case set differs from main')
                for case, out in r['out'].items():
                    total += 1
                    if case in ref and out['text'].encode() == ref[case]['text'].encode():
                        same += 1
                    else:
                        bad.append(f'{r["name"]}/{case}: text differs from main')
            if rr:
                print(f'ARM {kind} {arm}: {pooled(rr):.3f} tok/s states={dict(collections.Counter(r["state"] for r in rr))} '
                      f'exact_schedules={len(set(r["fingerprint"] for r in rr))} text_vs_m={same}/{total}')
        for family in ('fast', 'slow', 'other') if kind == 'agent' else ('edit',):
            mm = [r for r in mains if r['family'] == family]
            for arm in ('u', 'f', 'c'):
                aa = [r for r in rows if r['kind'] == kind and r['arm'] == arm and r['family'] == family]
                if mm and aa:
                    print(f'WITHIN_STATE {kind} {family} {arm}/m n={len(aa)}/{len(mm)} '
                          f'{pooled(aa):.3f}/{pooled(mm):.3f} tok/s {100 * (pooled(aa) / pooled(mm) - 1):+.3f}%')
                else:
                    print(f'WITHIN_STATE {kind} {family} {arm}/m not_comparable n={len(aa)}/{len(mm)}')
    main_fast = [r for r in rows if r['kind'] == 'agent' and r['arm'] == 'm' and r['family'] == 'fast']
    if not main_fast:
        bad.append('no fresh main fast-state load: speed verdict is inconclusive')
    for arm in ('f', 'c'):
        rr = [r for r in rows if r['kind'] == 'agent' and r['arm'] == arm]
        stable = len(rr) == 3 and len(set(r['fingerprint'] for r in rr)) == 1
        speed = bool(rr and main_fast and pooled(rr) >= pooled(main_fast))
        every = bool(rr and main_fast and min(r['tps'] for r in rr) >= max(r['tps'] for r in main_fast))
        print(f'VERDICT {arm}: one_exact_agent_schedule={stable} pooled_ge_main_fast={speed} '
              f'every_load_ge_fastest_main={every} '
              f'control={pooled(rr) if rr else 0:.3f} main_fast={pooled(main_fast) if main_fast else 0:.3f} tok/s')
        if not stable:
            bad.append(f'{arm}: the 3 agent loads did not land in one exact schedule')
        if not speed:
            bad.append(f'{arm}: pooled throughput did not establish >= fresh main fast-state throughput')
    for msg in bad:
        print('GATE FAIL:', msg)
    print('WINDOWBIMODAL ' + ('CHECK FAILED' if bad else 'COLLECTED'))
    return int(bool(bad))


if __name__ == '__main__':
    if len(sys.argv) != 7 or sys.argv[1] != 'read':
        raise SystemExit('usage: bimodal-read.py read AB MANIFEST FAST_AGENT SLOW_AGENT EDIT_REFERENCE')
    try:
        sys.exit(read(*sys.argv[2:]))
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        print('GATE FAIL: reader inputs:', exc)
        print('WINDOWBIMODAL CHECK FAILED')
        sys.exit(1)
