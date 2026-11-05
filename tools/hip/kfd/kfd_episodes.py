#!/usr/bin/env python3
"""Queue-eviction episodes per process from a kfd_events.py log.

An episode runs from a process's first QUEUE_EVICTION on any GPU node to the QUEUE_RESTORE that leaves no node evicted.
Evictions nest (an SVM and a userptr eviction at once: KFD counts them per node and the queues run only after every one
is restored); a rescheduled restore ('R': the restore failed and was queued again) restores nothing.

usage: kfd_episodes.py LOG [--pids P1,P2] [--from-ns T] [--to-ns T] [--list] [--min-ms 0]
  one 'KFDSUM pid=... episodes=... frozen_ms=... span_ms=... longest_ms=... p90_ms=... svm=... userptr=... ttm=...
  retries=... unmaps=...' line per process, then (--list) the episodes of --min-ms or more.
"""
import argparse, collections

TRIG = {0: 'svm', 1: 'userptr', 2: 'ttm', 3: 'suspend', 4: 'criu', 5: 'criu'}


def parse(path, lo, hi, pids):
    out = []
    for ln in open(path, errors='replace'):
        if ln.startswith('#'):
            continue
        f = ln.split()
        if len(f) < 5 or not f[0].isdigit():
            continue
        kind = f[2]
        try:
            ns = int(f[3])
        except ValueError:
            continue
        pid = int(f[4][1:]) if f[4].startswith('-') and f[4][1:].isdigit() else None
        if pid is None or (pids and pid not in pids) or ns < lo or ns > hi:
            continue
        if kind == 'QUEUE_EVICTION' and len(f) >= 7:
            out.append((ns, pid, 'E', f[5], int(f[6]) if f[6].lstrip('-').isdigit() else -1))
        elif kind == 'QUEUE_RESTORE' and len(f) >= 7:
            out.append((ns, pid, 'R' if f[6] == 'R' else 'D', f[5], None))
        elif kind == 'UNMAP_FROM_GPU':
            out.append((ns, pid, 'U', f[-2] if len(f) >= 8 else '?', None))
    out.sort()
    return out


def episodes(evs):
    per = collections.defaultdict(lambda: {'cnt': collections.Counter(), 'open': None, 'eps': [], 'unmaps': 0,
                                           't0': None, 't1': None})
    for ns, pid, k, node, trig in evs:
        s = per[pid]
        s['t0'] = ns if s['t0'] is None else s['t0']
        s['t1'] = ns
        if k == 'U':
            s['unmaps'] += 1
        elif k == 'E':
            s['cnt'][node] += 1
            if s['open'] is None:
                s['open'] = {'begin': ns, 'trig': trig, 'retries': 0, 'evicts': 0}
            s['open']['evicts'] += 1
        elif k == 'R':
            if s['open'] is not None:
                s['open']['retries'] += 1
        elif k == 'D':
            if s['open'] is None or s['cnt'][node] == 0:
                continue
            s['cnt'][node] -= 1
            if sum(s['cnt'].values()) == 0:
                s['open']['end'] = ns
                s['eps'].append(s['open'])
                s['open'] = None
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('--pids', default='')
    ap.add_argument('--from-ns', type=int, default=0)
    ap.add_argument('--to-ns', type=int, default=1 << 62)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--min-ms', type=float, default=0.0)
    a = ap.parse_args()
    pids = {int(x) for x in a.pids.split(',') if x.strip().isdigit()}
    per = episodes(parse(a.log, a.from_ns, a.to_ns, pids))
    if not per:
        print('KFDSUM pid=- episodes=0 frozen_ms=0 span_ms=0 longest_ms=0 p90_ms=0 svm=0 userptr=0 ttm=0 retries=0 unmaps=0')
    for pid, s in sorted(per.items()):
        eps = list(s['eps'])
        if s['open'] is not None:   # still evicted when the log ends: counted up to the last event
            o = dict(s['open']); o['end'] = s['t1']; o['open'] = True; eps.append(o)
        d = sorted((e['end'] - e['begin']) / 1e6 for e in eps)
        tr = collections.Counter(TRIG.get(e['trig'], '?') for e in eps)
        span = (s['t1'] - s['t0']) / 1e6 if s['t0'] is not None else 0
        print(f"KFDSUM pid={pid} episodes={len(eps)} frozen_ms={sum(d):.0f} span_ms={span:.0f} "
              f"longest_ms={(d[-1] if d else 0):.0f} p90_ms={(d[int(0.9 * (len(d) - 1))] if d else 0):.0f} "
              f"svm={tr['svm']} userptr={tr['userptr']} ttm={tr['ttm']} retries={sum(e['retries'] for e in eps)} "
              f"unmaps={s['unmaps']}" + (' OPEN-AT-END' if s['open'] is not None else ''))
        if a.list:
            for e in eps:
                ms = (e['end'] - e['begin']) / 1e6
                if ms >= a.min_ms:
                    print(f"  pid {pid} +{(e['begin'] - s['t0']) / 1e9:9.3f} s {ms:9.1f} ms "
                          f"{TRIG.get(e['trig'], '?'):8s} evicts {e['evicts']:3d} failed restores {e['retries']:5d}"
                          + (' (open at end)' if e.get('open') else ''))


if __name__ == '__main__':
    main()
