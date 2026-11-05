#!/usr/bin/env python3
"""CPU-only reader gate tests. Fixtures are marked synthetic; no engine is executed, no queue data is written."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import uuid

MODULE = Path(__file__).with_name('bimodal-read.py')
spec = importlib.util.spec_from_file_location('bimodal_reader', MODULE)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def marked_trace(text):
    lines = []
    for line in text.splitlines():
        lines.append(line)
        f = line.split()
        if len(f) == 5 and f[1] == 'L0':
            lines.append(f'{f[0]} S0 {f[2]} 0 {int(int(f[4]) >= 10)}')
            if int(f[2]) > 0:
                lines.append(f'{f[0]} PK {f[2]} {f[3]} 0')
    return '\n'.join(lines) + '\n'


def main(ab, scratch):
    ab, scratch = Path(ab), Path(scratch)
    root = scratch / ('reader-fixtures-' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True)
    (root / 'SYNTHETIC.txt').write_text('CPU gate-test fixtures only. No GPU measurements were produced here.\n')
    fast, slow, edit = ab / 'prp2_m_agent_2', ab / 'prp2_m_agent_1', ab / 'prp2_m_edit_2'
    manifest = {'main_exe': '/fixture/main', 'branch_exe': '/fixture/bimodal',
                'profile': '14.3505,15.5044,17.9436,24.1993,28.9211,33.6429,38.3647,24.9153', 'clip': '2.10', 'loads': []}
    for p, arms in ((1, 'mufc'), (2, 'cfum'), (3, 'mufc')):
        manifest['loads'] += [dict(name=f'bim_{a}_agent_p{p}', arm=a, kind='agent', **{'pass': p}) for a in arms]
    manifest['loads'] += [dict(name=f'bim_{a}_edit24_p1', arm=a, kind='edit24', **{'pass': 1}) for a in 'mufc']
    mp = root / 'bim_manifest.json'
    mp.write_text(json.dumps(manifest))
    status = ''.join(x['name'] + '\t0\n' for x in manifest['loads'])
    (root / 'bim_harness.tsv').write_text(status)
    for item in manifest['loads']:
        d = root / item['name']
        d.mkdir()
        source = fast if item['kind'] == 'agent' else edit
        t = (source / 'pipeline.trace').read_text()
        (d / 'pipeline.trace').write_text(t if item['arm'] == 'm' else marked_trace(t))
        env = {'STRATA_PIPELINE_DEBUG': '1', 'STRATA_PIPELINE_TRACE': str(d / 'pipeline.trace')}
        if item['arm'] == 'f':
            env['STRATA_PL_POLICY_COSTS'] = manifest['profile']
        if item['arm'] == 'c':
            env['STRATA_PL_POLICY_CLIP'] = manifest['clip']
        cfg = dict(exe=manifest['main_exe' if item['arm'] == 'm' else 'branch_exe'], env=env,
                   args=['--pipeline-windows', '2'], gpu=[0, 1], layer_split='25')
        (d / 'cfg.json').write_text(json.dumps(cfg))
        log = '100% of the experts resident\n' + 'zero-doorbell graph\n' * 4
        if item['arm'] == 'f':
            log += 'strata serve: STRATA_PL_POLICY_COSTS=synthetic\n'
        if item['arm'] == 'c':
            log += 'strata serve: STRATA_PL_POLICY_CLIP=synthetic\n'
        (d / 'engine.log').write_text(log)
        (d / 'harness.out').write_text('Synthetic CPU fixture; no engine executed.\n')
        for f in source.glob('out_*.json'):
            j = json.loads(f.read_text())
            if item['arm'] != 'm':
                j['timings']['predicted_per_second'] = 200 if item['kind'] == 'agent' else 400
            (d / f.name).write_text(json.dumps(j))
    def check(label, expected):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = r.read(root, mp, fast, slow, edit)
        assert rc == expected, (label, buf.getvalue())
        (root / (label + '.out')).write_text(buf.getvalue())
        print('SELFTEST', label, 'PASSED', flush=True)
    check('clean', 0)
    parsed = r.summarize(root / 'bim_c_agent_p1')
    assert parsed['served'] == parsed['l0'] and max(parsed['waits']['launch_service']) == 0
    assert parsed['waits']['service_finish'] == parsed['waits']['launch_finish']
    d = r.first_difference(r.summarize(slow)['schedule'], r.summarize(fast)['schedule'])
    assert d[0] == 'code_async_batcher' and d[1] == 269 and d[2][0] == 568
    print('SELFTEST wait-pairing-and-reference-divergence PASSED')
    f = next((root / 'bim_u_agent_p1').glob('out_*.json'))
    old = f.read_text(); j = json.loads(old); j['choices'][0]['message']['content'] += ' changed'
    f.write_text(json.dumps(j)); check('text-gate', 1); f.write_text(old)
    f = root / 'bim_c_agent_p1/cfg.json'
    old = f.read_text(); j = json.loads(old); j['env'].pop('STRATA_PL_POLICY_CLIP')
    f.write_text(json.dumps(j)); check('flag-gate', 1); f.write_text(old)
    f = root / 'bim_c_agent_p1/pipeline.trace'
    old = f.read_text(); f.write_text('\n'.join(l for l in old.splitlines() if ' S0 ' not in l) + '\n')
    check('missing-service-marker', 1); f.write_text(old)
    f = root / 'bim_f_agent_p2/pipeline.trace'
    old = f.read_text(); f.write_text(marked_trace((slow / 'pipeline.trace').read_text()))
    check('unstable-schedule', 1); f.write_text(old)
    lines, request = old.splitlines(), -1
    for i, line in enumerate(lines):
        if line.startswith('# request'):
            request += 1
        fields = line.split()
        if request == 1 and len(fields) == 5 and fields[1] == 'L0' and fields[4] == '10':
            fields[3] = '2' if fields[3] != '2' else '3'
            lines[i] = ' '.join(fields)
            break
    f.write_text('\n'.join(lines) + '\n')
    assert r.summarize(root / 'bim_f_agent_p2')['schedule'] == r.summarize(fast)['schedule']
    check('speculative-launch-only-drift', 1); f.write_text(old)
    saved = []
    for p in (1, 2, 3):
        f = root / f'bim_m_agent_p{p}/pipeline.trace'; saved.append((f, f.read_text()))
        f.write_text((slow / 'pipeline.trace').read_text())
    check('no-fresh-main-fast-state', 1)
    for f, old in saved:
        f.write_text(old)
    (root / 'bim_harness.tsv').write_text(status.replace('bim_c_agent_p1\t0', 'bim_c_agent_p1\t1'))
    check('harness-failure', 1)
    print('READER SELFTEST PASSED; synthetic fixtures:', root)


if __name__ == '__main__':
    main(*sys.argv[1:])
