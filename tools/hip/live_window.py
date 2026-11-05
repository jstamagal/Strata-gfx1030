#!/usr/bin/env python3
"""Queue-only live switch window; --read is CPU-only. No production config is changed."""
import argparse
import collections
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location('live_build', Path(__file__).with_name('live_build.py'))
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
W, R, H = build.W, build.R, build.H
ARMS = ('m', 'L0', 'L')
AGENT = Path('/home/ronald_rump/inbox/agents/claude/strata-q38-dflash2/bench/results/2026-11-04-dflash2-agent-prompts.json')
EDIT = Path('/home/ronald_rump/inbox/agents/claude/mtpq/prompts/edit24.json')
COPY = re.compile(r'STRATA_HC_REQ8 on a layer split: CUDA(\d+) layers (\d+)-(\d+): (\d+) hyper-connection projections get an int8 \+ (fp32/32|fp16/16) copy for the verify window \((\d+) MiB;.*?\) in ([\d.]+) s; ([\d.]+) GiB free')
FEATURES = (*build.L_ENV, 'STRATA_PL_POLICY_COSTS', 'STRATA_HC_Q8')
STAGES = tuple(f's{s}_T{t}' for s in (0, 1) for t in (1, 2, 3))


def plan():
    rows = []
    for pa in (1, 2):
        for arm in ARMS if pa == 1 else reversed(ARMS):
            for kind in ('agent', 'edit24') if pa == 1 else ('edit24', 'agent'):
                rows.append(dict(name=f'live_{arm}_{kind}_p{pa}', arm=arm, kind=kind, pass_n=pa))
    for pa in (1, 2, 3):
        for arm in ARMS if pa != 2 else reversed(ARMS):
            rows.append(dict(name=f'live_{arm}_read36_p{pa}', arm=arm, kind='read36', pass_n=pa))
    return rows


def reserve(ab):
    if list(ab.glob('live_*')):
        raise RuntimeError('ab/live_* exists; refusing ALL output reuse/overwrite')
    (ab / 'live_lock').mkdir()  # atomic claim, retained as evidence; no cleanup needed to finish


def config(live, arm, folder):
    d = copy.deepcopy(live)
    d.update(exe=str(R / 'build-hip/strata' if arm == 'm' else W / 'build-hip/strata'),
             host='127.0.0.1', log=str(folder / 'engine.log'))
    # m and L0 preserve the live env, apart from the SAME measurement-only trace settings.
    d['env'].update(STRATA_PIPELINE_DEBUG='1', STRATA_PIPELINE_TRACE=str(folder / 'pipeline.trace'))
    if arm == 'L':
        d['env'].update(build.L_ENV)
    return d


def gate(log, arm):
    bad = []
    slots = re.findall(r'CUDA1 runs layers \S+, expert cache (\d+) slots', log)
    resident = re.findall(r'(\d+)% of the experts resident', log)
    zd = log.count('zero-doorbell graph')
    free = {str(c): x[-1] for c in (0, 1) if (x := re.findall(rf'CUDA{c}: (\d+) MiB of VRAM free with everything loaded', log))}
    if x := re.findall(r'strata serve: (\d+) MiB of VRAM free with everything loaded', log):
        free['0'] = x[-1]
    if slots != ['11776'] or not resident or set(resident) != {'100'}:
        bad.append(f'RESIDENCY FAIL: card1 slots={slots}, resident={resident}; need 11776/100%')
    if zd != 4:
        bad.append(f'zero-doorbell graphs {zd}, expected 4')
    if set(free) != {'0', '1'}:
        bad.append('missing per-card free MiB')
    copies = COPY.findall(log)
    want = {('0', '0', '24', '100', 'fp32/32', '351'), ('1', '25', '47', '92', 'fp32/32', '323')}
    if arm == 'L' and {c[:6] for c in copies} != want:
        bad.append('HC fp32/32 copy table differs: ' + str([c[:6] for c in copies]))
    if arm != 'L' and copies:
        bad.append('HC copies present with feature unset')
    diet = log.count('STRATA_T1_DIET 0xc: HC norm in the down launch off, one-row QSA batched on, DPP routing on')
    if (arm == 'L' and diet != 4) or (arm != 'L' and 'STRATA_T1_DIET 0x' in log):
        bad.append(f'diet engagement mismatch ({diet}/4)')
    clip = re.findall(r'strata serve: STRATA_PL_POLICY_CLIP=([\d.]+):', log)
    if (arm == 'L' and (len(clip) != 1 or float(clip[0]) != 2.10)) or (arm != 'L' and clip):
        bad.append('policy clip acknowledgement mismatch: ' + str(clip))
    if 'strata serve: STRATA_PL_POLICY_COSTS=' in log:
        bad.append('fixed policy costs must remain OFF')
    return dict(slots=slots, resident=resident, zero_doorbell=zd, free_MiB=free,
                HC_copies=copies, clip=clip, failures=bad)


def response(j):
    t = j['timings']
    n, ms = int(t['predicted_n']), float(t['predicted_ms'])
    text = j['choices'][0]['message']['content']
    if n <= 0 or not math.isfinite(ms) or ms <= 0 or not isinstance(text, str):
        raise ValueError('invalid response text/count/timing')
    return dict(text=text, n=n, seconds=ms / 1000, tps=n * 1000 / ms,
                accepted=int(t['draft_n_accepted']), drafts=int(t['draft_n']))


def traces(path, outputs, skip=1):
    reqs = []
    with path.open() as f:
        for line in f:
            if line.startswith('# request'):
                if 'error' in line:
                    raise ValueError('trace request ended in an error')
                reqs.append([])
            elif line.strip() and not line.startswith('#'):
                fields = line.split()
                if len(fields) != 5 or not reqs:
                    raise ValueError('malformed trace')
                ms, event, seq, a, b = fields
                ms = float(ms)
                if not math.isfinite(ms) or ms < 0 or (reqs[-1] and ms < reqs[-1][-1][0]):
                    raise ValueError('invalid trace clock')
                reqs[-1].append((ms, event, int(seq), int(a), int(b)))
    if len(reqs) != len(outputs) + skip:
        raise ValueError(f'trace request count {len(reqs)}, expected {len(outputs) + skip}')
    durations = {k: [] for k in STAGES}
    hit = miss = placing = 0
    for cid, events in zip(sorted(outputs), reqs[skip:]):
        pending, finish = {0: collections.deque(), 1: collections.deque()}, {0: -math.inf, 1: -math.inf}
        tokens = 0
        for ms, event, seq, a, b in events:
            if event in ('L0', 'L1'):
                pending[int(event[1])].append((ms, a, b))
            elif event in ('F0', 'F1'):
                s = int(event[1])
                if not pending[s]:
                    raise ValueError('orphan trace finish')
                start, t, speculative = pending[s].popleft()
                duration = ms - max(start, finish[s])
                finish[s] = ms
                if duration < 0:
                    raise ValueError('negative trace duration')
                if t in (1, 2, 3) and not (s == 0 and speculative >= 10):
                    durations[f's{s}_T{t}'].append(duration)
            elif event == 'V':
                tokens += a + 1
                hit += b == 1
                miss += b == -1
                placing += b == 0
        if any(pending.values()) or tokens != outputs[cid]['n']:
            raise ValueError(f'{cid}: trace did not drain or tokens {tokens} != response {outputs[cid]["n"]}')
    return dict(samples=durations, hit=hit, miss=miss, placing=placing,
                stage_ms={k: statistics.median(v) if len(v) >= 5 else None for k, v in durations.items()})


def speed(rows):
    return sum(r['n'] for r in rows) / sum(r['seconds'] for r in rows)


def case_medians(rows, ref, ids):
    out = {}
    for cid in ids:
        a = statistics.median(r['cases'][cid]['tps'] for r in rows)
        m = statistics.median(r['cases'][cid]['tps'] for r in ref)
        out[cid] = dict(tps=a, main_tps=m, delta_pct=100 * (a / m - 1))
    return dict(median_pct=statistics.median(v['delta_pct'] for v in out.values()), cases=out)


def classify(rows, main):
    # Never label quantization gains as a fast state just because tok/s rose.
    anchors = {'fast': (1898, 997), 'slow': (1750, 1030)}
    for arm in ARMS:
        runs = [r for r in rows if r['arm'] == arm]
        exact = bool(main) and all(all(r['cases'][c]['text'] == main[0]['cases'][c]['text'] for c in r['cases']) for r in runs)
        for r in runs:
            r.update(state='unknown', state_method='unresolved', hit_share=r['hit'] / (r['hit'] + r['miss']) if r['hit'] + r['miss'] else None)
        if exact:
            for r in runs:
                distance = {s: math.hypot((r['hit'] - h) / 148, (r['miss'] - m) / 33) for s, (h, m) in anchors.items()}
                nearest = min(distance, key=distance.get)
                r.update(state=nearest if distance[nearest] <= 0.5 else 'unknown', state_method='same-text hit/miss anchors')
        elif len(runs) == 2:
            lo, hi = sorted(runs, key=lambda r: r['tps'])
            repeat = all(lo['cases'][c]['text'] == hi['cases'][c]['text'] for c in lo['cases'])
            if repeat and hi['tps'] / lo['tps'] >= 1.01 and lo['hit_share'] is not None and hi['hit_share'] - lo['hit_share'] >= .005:
                lo.update(state='slow', state_method='INFERRED own-repeat speed/counter clusters')
                hi.update(state='fast', state_method='INFERRED own-repeat speed/counter clusters')
            else:
                for r in runs:
                    r['state_method'] = 'UNRESOLVED changed text: one/discordant cluster or repeat text drift'


def agent_gate_metric(metrics, states):
    matched = states.get('L', {}).get('matched', {})
    for state in ('fast', 'slow'):
        if state in matched:
            return matched[state]['median_pct'], 'within-' + state + ' per-case median'
    return metrics.get('agent/L', {}).get('median_pct'), 'all-pass per-case median fallback; STATE-CONFOUNDED, no proven state overlap'


def verdict(errors, identical, stage, agent, edit, prompt):
    reasons = list(errors)
    if not identical:
        reasons.append('L0 text is not byte-identical to m')
    for k, pair in stage.items():
        a, m = pair
        if a is None or m is None or not math.isfinite(a) or not math.isfinite(m) or a >= m:
            reasons.append(f'L stage {k} did not beat m ({a} vs {m})')
    if len(stage) != 12:
        reasons.append('missing agent/edit T1-T3 per-card stage cells')
    for name, value, floor in (('agent median', agent, 3.0), ('edit median', edit, 0.0), ('36K median', prompt, -1.0)):
        if value is None or not math.isfinite(value) or value < floor:
            reasons.append(f'{name} {value}%, needs >= {floor}%')
    return not reasons, reasons


def readout(ab, failures=()):
    meta = json.loads((ab / 'live_manifest.json').read_text())
    live = json.loads((ab / 'live_cfg.json').read_text())
    bad, rows, identity = list(failures), [], True
    if meta.get('commit') and meta['commit'] != build.git('rev-parse', 'HEAD'):
        bad.append('live branch HEAD changed during/after the measurement')
    for p, digest in meta['hashes'].items():
        if build.sha(p) != digest:
            bad.append('input hash changed: ' + p)
    if meta['loads'] != plan():
        bad.append('manifest differs from the required 12 decode + 9 fresh-read loads')
    for item in plan():
        d = ab / item['name']
        try:
            if (d / 'status').read_text().strip() != '0':
                raise ValueError('load incomplete')
            if json.loads((d / 'cfg.json').read_text()) != config(live, item['arm'], d):
                raise ValueError('arm config differs from snapshot + exact arm env')
            gates = gate((d / 'engine.log').read_text(errors='replace'), item['arm'])
            print('GATES', item['name'], json.dumps(gates), flush=True)
            bad.extend(item['name'] + ': ' + x for x in gates['failures'])
            row = dict(item)
            if item['kind'] == 'read36':
                j = json.loads((d / 'read36.json').read_text())
                t = j['timings']
                if not 34000 <= t['prompt_n'] <= 39000 or t.get('cache_n') != 0 or t['prompt_ms'] <= 0:
                    raise ValueError('36K prompt is not a fresh full read')
                row.update(prompt_tps=1000 * t['prompt_n'] / t['prompt_ms'], text=j['choices'][0]['message']['content'])
            else:
                files = {f.stem[4:]: f for f in d.glob('out_*.json')}
                ids = meta['ids'][item['kind']]
                if set(files) != set(ids):
                    raise ValueError('missing/extra response cases')
                cases = {c: response(json.loads(files[c].read_text())) for c in ids}
                tr = traces(d / 'pipeline.trace', cases)
                row.update(cases=cases, n=sum(x['n'] for x in cases.values()), seconds=sum(x['seconds'] for x in cases.values()), **tr)
                row['tps'] = row['n'] / row['seconds']
                print('STAGE_RUN', item['name'], json.dumps(row['stage_ms']), flush=True)
            rows.append(row)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            bad.append(item['name'] + ': ' + str(e))
    metrics, stage_pairs, states = {}, {}, {}
    for kind in ('agent', 'edit24'):
        ref = [r for r in rows if r['kind'] == kind and r['arm'] == 'm']
        use = [r for r in rows if r['kind'] == kind]
        if len(ref) != 2:
            bad.append(kind + ': missing main passes')
        if kind == 'agent':
            classify(use, ref)
            for r in use:
                print('LOAD_STATE', r['name'], r['tps'], 'hit/plc/miss', r['hit'], r['placing'], r['miss'], r['state'], r['state_method'], flush=True)
        for arm in ARMS:
            arm_rows = [r for r in use if r['arm'] == arm]
            if len(arm_rows) != 2 or not ref:
                bad.append(f'{arm}/{kind}: incomplete paired loads')
                continue
            ids = meta['ids'][kind]
            med = case_medians(arm_rows, ref, ids)
            stage = {k: statistics.median(v) if len(v := [x for r in arm_rows for x in r['samples'][k]]) >= 5 else None for k in STAGES}
            same = sum(r['cases'][c]['text'].encode() == ref[0]['cases'][c]['text'].encode() for r in arm_rows for c in ids)
            if arm == 'm' and same != 2 * len(ids):
                bad.append(kind + ': main text changed across passes; default-off identity proof is not stable')
            if arm == 'L0' and same != 2 * len(ids):
                identity = False
            label = 'STATE-CONFOUNDED pooled; no stalled sample dropped' if kind == 'agent' else 'all samples; no stalled sample dropped'
            print('SET', kind, arm, speed(arm_rows), 'tok/s median_vs_m_pct', med['median_pct'], 'text_vs_m1', same, '/', 2 * len(ids), label, flush=True)
            for c, value in med['cases'].items():
                print('CASE', kind, arm, c, json.dumps(value), flush=True)
            print('STAGE', kind, arm, json.dumps(stage), flush=True)
            metrics[kind + '/' + arm] = dict(pooled_tps=speed(arm_rows), **med, stage_ms=stage, text_same=same, note=label)
            if arm == 'L':
                base = metrics.get(kind + '/m', {}).get('stage_ms', {})
                stage_pairs.update({kind + '/' + k: (stage[k], base.get(k)) for k in STAGES})
            if kind == 'agent':
                matched = {}
                for state in ('fast', 'slow'):
                    aa = [r for r in arm_rows if r['state'] == state]
                    mm = [r for r in ref if r['state'] == state]
                    if aa and mm:
                        matched[state] = dict(arm_passes=[r['pass_n'] for r in aa], main_passes=[r['pass_n'] for r in mm],
                            tps=speed(aa), main_tps=speed(mm), **case_medians(aa, mm, ids))
                        print('WITHIN_STATE', arm, state, json.dumps(matched[state]), flush=True)
                if not matched:
                    print('STATE_WARNING NO_SHARED_STATE', arm, 'vs m; within-state comparison inconclusive', flush=True)
                states[arm] = dict(matched=matched, loads=[{k: r[k] for k in ('name', 'hit', 'miss', 'placing', 'state', 'state_method', 'tps')} for r in arm_rows])
    prompt = {}
    for arm in ARMS:
        rr = [r for r in rows if r['kind'] == 'read36' and r['arm'] == arm]
        if len(rr) != 3:
            bad.append(arm + ': need three separate fresh 36K loads')
        prompt[arm] = statistics.median(r['prompt_tps'] for r in rr) if len(rr) == 3 else None
        print('PROMPT36', arm, 'per_load', [r['prompt_tps'] for r in rr], 'median', prompt[arm], flush=True)
    for pa in (1, 2, 3):
        a = next((r for r in rows if r['arm'] == 'L0' and r['kind'] == 'read36' and r['pass_n'] == pa), None)
        m = next((r for r in rows if r['arm'] == 'm' and r['kind'] == 'read36' and r['pass_n'] == pa), None)
        if not a or not m or a['text'].encode() != m['text'].encode():
            identity = False
    pp = 100 * (prompt['L'] / prompt['m'] - 1) if prompt['L'] and prompt['m'] else None
    # Predeclared: compare within fast (else slow) where possible; otherwise show the explicitly labelled
    # all-pass BUNDLE1 per-case median. Never force a quantization-changed schedule into an absolute tok/s bin.
    agent, agent_basis = agent_gate_metric(metrics, states)
    edit = metrics.get('edit24/L', {}).get('median_pct')
    print('GO_AGENT_METRIC', agent, agent_basis, flush=True)
    go, reasons = verdict(bad, identity, stage_pairs, agent, edit, pp)
    result = dict(go=go, reasons=reasons, metrics=metrics, states=states, prompt_medians=prompt, prompt_delta_pct=pp,
                  L0_identical=identity, agent_gate_basis=agent_basis, agent_gate_pct=agent)
    (ab / 'live_results.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = f'all required gates pass; agent {agent:+.3f}% ({agent_basis}), edit {edit:+.3f}%, fresh36K {pp:+.3f}%; stages improve 12/12' if go else '; '.join(reasons)
    print('LIVE_SWITCH ' + ('GO' if go else 'HOLD') + ' | ' + summary, flush=True)
    return 0 if go else 1


def call(body):
    req = Request('http://127.0.0.1:8091/v1/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urlopen(req, timeout=1800) as response:
        j = json.load(response)
    if 'timings' not in j or not j.get('choices'):
        raise RuntimeError('bad engine response: ' + str(j)[:400])
    return j


def run_load(ab, live, item, cases, long_prompt):
    d = ab / item['name']
    d.mkdir()
    cfg = config(live, item['arm'], d)
    (d / 'cfg.json').write_text(json.dumps(cfg, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('STRATA_', 'HIP_VISIBLE', 'ROCR_VISIBLE', 'CUDA_VISIBLE', 'GPU_DEVICE_ORDINAL'))}
    proc = None
    try:
        with (d / 'server.out').open('w') as log:
            proc = subprocess.Popen([R / '.venv/bin/python', R / 'serve/server.py', '--engine', 'strata', '--config', d / 'cfg.json', '--port', '8091'],
                                    cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            end = time.monotonic() + 1500
            while 'ready:' not in (d / 'server.out').read_text(errors='replace'):
                if proc.poll() is not None or time.monotonic() > end:
                    raise RuntimeError('server stopped or timed out during load')
                time.sleep(1)
            info = gate((d / 'engine.log').read_text(errors='replace'), item['arm'])
            print('LOAD_GATES', item['name'], json.dumps(info), flush=True)
            if info['failures']:
                raise RuntimeError('; '.join(info['failures']))
            body = dict(model=cfg['model_name'], max_tokens=32, temperature=0, chat_template_kwargs={'enable_thinking': False})
            if item['kind'] == 'read36':
                messages = copy.deepcopy(long_prompt['messages'])
                messages[0]['content'] = f'Live fresh read {item["pass_n"]}. ' + messages[0]['content']
                body['messages'] = messages
                (d / 'req_read36.json').write_text(json.dumps(body))
                j = call(body)
                (d / 'read36.json').write_text(json.dumps(j, indent=1) + '\n')
                print('FRESH36', item['name'], json.dumps(j['timings']), flush=True)
            else:
                body.update(messages=[dict(role='user', content='Say hi.')], max_tokens=16)
                (d / 'warm.json').write_text(json.dumps(call(body)))
                for c in sorted(cases, key=lambda c: c['id']):
                    body.update(messages=c['messages'], max_tokens=1200)
                    (d / ('req_' + c['id'] + '.json')).write_text(json.dumps(body))
                    j = call(body)
                    (d / ('out_' + c['id'] + '.json')).write_text(json.dumps(j, indent=1) + '\n')
                    print('RESPONSE', item['name'], c['id'], json.dumps(j['timings']), flush=True)
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
    (d / 'status').write_text('0\n')


def run():
    ab = H / 'ab'
    if list(ab.glob('live_*')):
        raise RuntimeError('ab/live_* exists; refusing to start')
    receipt = build.verified_build()
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', 8091)) == 0:
            raise RuntimeError('port 8091 is in use')
    cfg_path = R / 'strata-sc117-IQ3_S-qwen-abliterated.json'
    print('LIVE_CONFIG_SHA256', build.sha(cfg_path), cfg_path, flush=True)
    live = json.loads(cfg_path.read_text())
    if build.sha(cfg_path) != receipt['inputs'][str(cfg_path)]:
        raise RuntimeError('live config changed after build receipt verification')
    if live['gpu'] != [0, 1] or str(live['layer_split']) != '25' or '--resident-experts' not in live['args']:
        raise RuntimeError('live both-card resident layer split changed')
    if live['args'][live['args'].index('--mtp') + 1] != '/home/ronald_rump/Strata-data/mtp/q38-s2-gptq-a':
        raise RuntimeError('live drafter changed; q4n is NOT part of this candidate')
    for key in FEATURES:
        if str(live['env'].get(key, '0')) not in ('', '0'):
            raise RuntimeError('live config already enables ' + key)
    cases = {k: json.loads(p.read_text())['cases'] for k, p in (('agent', AGENT), ('edit24', EDIT))}
    if len(cases['agent']) != 6 or len(cases['edit24']) != 24:
        raise RuntimeError('workload shape changed')
    long_prompt = json.loads((H / 'long-prompt.json').read_text())
    reserve(ab)
    (ab / 'live_cfg.json').write_text(json.dumps(live, indent=2) + '\n')
    inputs = [cfg_path, R / 'build-hip/strata', W / 'build-hip/strata', R / 'build-hip/CMakeCache.txt', W / 'build-hip/CMakeCache.txt',
              H / 'staging/live-build.json', H / 'staging/live-build.log', AGENT, EDIT, H / 'long-prompt.json',
              ab / 'live_cfg.json', Path(__file__), Path(build.__file__)]
    meta = dict(loads=plan(), ids={k: sorted(c['id'] for c in cs) for k, cs in cases.items()},
                commit=build.git('rev-parse', 'HEAD'), hashes={str(p): build.sha(p) for p in inputs}, env=build.L_ENV,
                gate_policy='agent matched-fast (else matched-slow, else labelled all-pass fallback) per-case median>=3%; edit per-case median>=0%; median-of-three fresh36K>=-1%; all 12 T1-T3 stage cells must improve')
    (ab / 'live_manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    failed = []
    for item in plan():
        print('ARM', item['name'], time.strftime('%Y-%m-%dT%H:%M:%S%z'), flush=True)
        try:
            run_load(ab, live, item, cases.get(item['kind'], []), long_prompt)
        except Exception as e:
            failed.append(item['name'] + ': ' + str(e))
            break
    build.source_gate()
    return readout(ab, failed)


if __name__ == '__main__':
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser()
    p.add_argument('--read', type=Path, help='CPU-only readout of an ab directory containing live_*')
    a = p.parse_args()
    try:
        sys.exit(readout(a.read) if a.read else run())
    except Exception as e:
        print('LIVE_SWITCH HOLD | ' + str(e), flush=True)
        sys.exit(1)
